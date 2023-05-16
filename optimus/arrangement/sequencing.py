"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Any, List, Optional

import numpy as np
from sortedcontainers import SortedSet

from optimus.activity import BaseActivityManager
from optimus.elements.inventory import Inventory
from optimus.arrangement.process_handler import ProcessHandler
from optimus.utils.constants import MaterialType, CAMode
from optimus.utils.general import multidict
from optimus.machines.machine import Machine
from optimus.machines.event import MachineState
from optimus.utils.constants import InfoSpreading, ResourceType
from optimus.utils.general import compare_linking_op, close_subtract
from optimus.arrangement.task import Task
from optimus.arrangement.chain import SoloChain, Chain
from optimus.arrangement.tools import TaskGraph, ATI
from optimus.machines.feedback import ProcessInfo

import logging

logger = logging.getLogger(__name__)


class Sequencer:
    INITIAL = "Initial"
    SPAWNED = "Spawned"
    URGENT = "Urgent"

    def __init__(
        self,
        task_graph: TaskGraph,
        process_handler: ProcessHandler,
        inventory: Inventory,
        products,
        config: Dict[str, Any],
        activity_manager: BaseActivityManager,
        random_state: int = None,
    ) -> None:
        self.task_graph = task_graph
        self.process_handler = process_handler
        self.inventory = inventory
        self.products = products
        self.activity_manager = activity_manager
        self.info_spreading = config["scheduling_params"]["info_spreading"]
        self.feedback_threshold = config["scheduling_params"]["feedback_threshold"]
        self.bind_nextop = config["scheduling_params"]["bind_nextop"]
        self.bind_resource_type = config["scheduling_params"]["bind_resource_type"]
        self.use_machine_feedback = config["scheduling_params"]["use_machine_feedback"]
        self.consume_only_one = config["consume_only_one"]

        # Information
        self.information = {}
        self.metadata = {"batch_times_per_room": multidict(2, dict)}

        # Available task ids
        self.ati = ATI()

        # Locked task ids
        self.lti = defaultdict(lambda: SortedSet(key=self.__lti_sort_key))

        # Completed task ids
        self.cti = {"last": set(), "others": set()}

        self.active_chains = []
        self.initialization_chains = []
        self.released_type = {}

        # Initialize
        self.tasks = self.task_graph.get_tasks()
        self.__initialize()
        logger.debug(f"Total number of tasks initial: {len(self.tasks)}")

    def __lti_sort_key(self, task_id: int):
        task = self.tasks[task_id]
        return (
            task.ca_mode != CAMode.CAV.value,
            -task.quantity * task.product.count_factor,
            -task.priority,
        )

    def __initialize(self):
        """Initialize task lists"""
        for task in self.tasks.values():
            # Add to initialization chains
            if task.is_initialised:
                self.initialization_chains.append(
                    (SoloChain(task=task), Sequencer.INITIAL)
                )

            # Add to LTI if initially locked else ATI
            if task.is_initially_locked:
                self.lti[(task.product.material_id, task.batch_size, task.alt_bom)].add(
                    task.task_id
                )
            else:
                self.ati.add_task(task)

    def empty(self) -> bool:
        return self.ati.length() == 0

    def bind_tasks_from_feedback(self, machine_feedback) -> None:
        # Not configured
        if not self.use_machine_feedback:
            return

        # Campaign length got completed
        if (
            machine_feedback["curr_campaign_length"]
            >= machine_feedback["max_campaign_length"]
        ):
            return

        # Create urgent task from the feedback
        machine = machine_feedback["machine"]
        picked_tasks = [
            {
                "task": None,
                "available_at": 0,
                "estimated_availability": np.inf,
                "demand_priority": 0,
            }
        ]
        for task in self.ati.get_tasks(family_id=machine_feedback["product"].family_id):
            # Check machine
            is_machine_possible = False
            if task.alt_recipe is None:
                is_machine_possible = machine in task.get_possible_machines()
            else:
                is_machine_possible = machine in task.get_approved_machines()

            if not is_machine_possible:
                continue

            # Create duplicate task
            duplicate_task = self.task_graph.duplicate_task(task)

            # Fake process here and skip if room changeover B found
            process_info, _ = self.process_handler.fake_process_with_machine(
                task=duplicate_task, machine=machine
            )
            changeover_found = False
            for changeover_trigger in process_info.changeover_triggers:
                if changeover_trigger["state"] in [MachineState.ROOM_CHANGEOVER_B]:
                    changeover_found = True
                    break

            if changeover_found:
                continue

            # Update condition
            curr_item = {
                "task": task,
                "available_at": task.available_at,
                "estimated_availability": task.estimate_availability(),
                "demand_priority": task.demand_priority,
            }

            if (
                curr_item["estimated_availability"]
                == picked_tasks[0]["estimated_availability"]
            ):
                if curr_item["demand_priority"] == picked_tasks[0]["demand_priority"]:
                    if curr_item["available_at"] == picked_tasks[0]["available_at"]:
                        picked_tasks.append(curr_item)

                    elif curr_item["available_at"] > picked_tasks[0]["available_at"]:
                        picked_tasks = [curr_item]

                elif curr_item["demand_priority"] > picked_tasks[0]["demand_priority"]:
                    picked_tasks = [curr_item]

            elif (
                curr_item["estimated_availability"]
                < picked_tasks[0]["estimated_availability"]
            ):
                picked_tasks = [curr_item]

        picked_task = np.random.choice(picked_tasks)["task"]
        if picked_task is not None:
            self.active_chains.append((SoloChain(task=picked_task), Sequencer.URGENT))
            picked_task.preferences["machines"].discard(machine)
            picked_task.preferences["machines"].add(machine)

    def bind_next_operation(self, next_task: Optional[Task], machine: Machine):
        # Cannot bind condition
        if not self.bind_nextop or next_task is None:
            return

        if (
            (
                self.bind_resource_type == ResourceType.block.value
                and machine.block_id
                in [
                    approved_machine.block_id
                    for approved_machine in next_task.get_approved_machines()
                ]
            )
            or (
                self.bind_resource_type == ResourceType.room.value
                and machine.room_id
                in [
                    approved_machine.room_id
                    for approved_machine in next_task.get_approved_machines()
                ]
            )
            or (
                self.bind_resource_type == ResourceType.machine.value
                and machine.machine_id
                in [
                    approved_machine.machine_id
                    for approved_machine in next_task.get_approved_machines()
                ]
            )
        ):
            self.active_chains.append((SoloChain(task=next_task), Sequencer.URGENT))
            next_task.preferences["machines"].discard(machine)
            next_task.preferences["machines"].add(machine)

    def make_information_key(self, task: Task):
        if self.info_spreading == InfoSpreading.vertical.value:
            return task.task_id

        elif self.info_spreading == InfoSpreading.horizontal.value:
            return (
                task.product.material_id,
                task.batch_size,
                task.sequence,
            )

        else:
            if self.info_spreading is not None:
                raise ValueError(f"Invalid info spreading mode: {self.info_spreading}")

    def attach_information(self, task: Task) -> None:
        if self.info_spreading == InfoSpreading.vertical.value:
            information_key = self.make_information_key(task)
            if information_key in self.information:
                for machine in self.information[information_key]:
                    task.preferences["machines"].discard(machine)
                    task.preferences["machines"].add(machine)

        elif self.info_spreading == InfoSpreading.horizontal.value:
            information_key = self.make_information_key(task)
            if (
                information_key in self.information
                and self.information[information_key][0] > 0
            ):
                task.preferences["machines"].discard(
                    self.information[information_key][1]
                )
                task.preferences["machines"].add(self.information[information_key][1])
                self.information[information_key][0] -= 1

    def detach_information(self, task: Task, machine: Machine):
        if self.info_spreading == InfoSpreading.vertical.value:
            batch_id = task.batch_id
            while not task.is_last_sequence():
                for out_node in task.out_nodes:
                    if out_node.batch_id == batch_id:
                        assert out_node.sequence > task.sequence
                        task = out_node
                        information_key = self.make_information_key(task)

                        if information_key in self.information:
                            self.information[information_key].append(machine)
                        else:
                            self.information[information_key] = [machine]

        elif self.info_spreading == InfoSpreading.horizontal.value:
            information_key = self.make_information_key(task)
            if information_key not in self.information:
                campaign_length = 5
                self.information[information_key] = [campaign_length - 1, machine]

            elif self.information[information_key][0] == 0:
                assert self.information[information_key][1] is not None
                del self.information[information_key]

        elif self.info_spreading == InfoSpreading.plus.value:
            raise NotImplementedError()

        else:
            if self.info_spreading is not None:
                # if None, then solo chain does necessary single node info transfer
                raise ValueError(f"Invalid info spreading mode: {self.info_spreading}")

    def prune_chains(self, chains_list: List[Chain]) -> List[Chain]:
        # Prune the chains list
        while len(chains_list) > 0:
            if chains_list[-1][0].empty():
                chains_list.pop()
            else:
                # Check whether the chain got completed already
                chains_list[-1][0].prune()
                if not chains_list[-1][0].empty():
                    break

        return chains_list

    def obtain_task(self) -> Task:
        if self.empty():
            raise IndexError("Empty sequence cannot give a task")

        # Send all initialization tasks first
        self.initialization_chains = self.prune_chains(self.initialization_chains)
        if len(self.initialization_chains) > 0:
            task = self.initialization_chains[-1][0].obtain_task()
            self.released_type[task.task_id] = self.initialization_chains[-1][1]

        else:
            # Create a new core chain if required
            self.active_chains = self.prune_chains(self.active_chains)
            if len(self.active_chains) == 0:
                chain = self.activity_manager.create_core_chain(
                    ati=self.ati, metadata=self.metadata
                )
                self.active_chains.append((chain, Sequencer.SPAWNED))

            # Extract task from the chain
            task = self.active_chains[-1][0].obtain_task()
            self.released_type[task.task_id] = self.active_chains[-1][1]

        # Attach information
        # if self.active_chains[-1][1] == "Spawned":
        #     self.attach_information(task)

        return task

    def complete_task(
        self,
        task: Task,
        assigned_machine: Machine,
        process_info: ProcessInfo,
        feedback: Dict[str, Any],
    ) -> None:
        # Complete task in the active chain
        if self.released_type[task.task_id] == Sequencer.INITIAL:
            self.initialization_chains[-1][0].complete_task(
                task=task,
                machine=assigned_machine,
                process_info=process_info,
            )

        else:
            self.active_chains[-1][0].complete_task(
                task=task,
                machine=assigned_machine,
                process_info=process_info,
            )

        # Create its next operation task
        next_task = self.task_graph.obtain_next_sequence_task(task)
        if next_task is not None:
            self.tasks[next_task.task_id] = next_task
            self.ati.add_task(next_task)

        # Spread info
        # self.detach_information(task=task, machine=assigned_machine)

        # Update task lists
        is_curr_last_sequence = task.is_last_sequence()
        self.ati.remove_task(task)
        if is_curr_last_sequence:
            self.cti["last"].add(task.task_id)
        else:
            self.cti["others"].add(task.task_id)

        # Update metadata
        if (
            task.batch_id
            in self.metadata["batch_times_per_room"][assigned_machine.room_id]
        ):
            curr_end_time = self.metadata["batch_times_per_room"][
                assigned_machine.room_id
            ][task.batch_id]["end_time"]
            self.metadata["batch_times_per_room"][assigned_machine.room_id][
                task.batch_id
            ]["end_time"] = max(curr_end_time, task.task_end)
        else:
            self.metadata["batch_times_per_room"][assigned_machine.room_id][
                task.batch_id
            ] = {
                "material_id": task.product.material_id,
                "ca_mode": task.ca_mode,
                "end_time": task.task_end,
            }

        # Use machine feedback
        self.bind_tasks_from_feedback(machine_feedback=feedback)

        # Bind next operation
        self.bind_next_operation(next_task=next_task, machine=assigned_machine)

        # Check if restricted batches can be released
        if is_curr_last_sequence and task.ca_mode == CAMode.CAI.value:
            unlocked_any = False

            # Iterating over possible impacted keys
            for parent_key in task.product.get_inverse_bom(
                task.batch_size, task.alt_bom
            ):
                (
                    parent_material_id,
                    parent_batch_size,
                    parent_alt_bom,
                    parent_comp_group,
                ) = parent_key
                lti_to_be_removed = []

                # Iterate over possible locked task in given key
                for locked_task_id in self.lti[
                    (parent_material_id, parent_batch_size, parent_alt_bom)
                ]:
                    locked_task = self.tasks[locked_task_id]

                    # Iterate over all component groups because
                    # need to check the fulfilment of each component group anyways
                    unlocked = True
                    useful_inventory = multidict(2, float)
                    useful_completed_tasks = multidict(2, float)
                    for (
                        component_group,
                        component,
                    ) in locked_task.product.get_indirect_composition(
                        locked_task.batch_size, locked_task.alt_bom
                    ).items():
                        reqd_qnty = component["quantity"] * (
                            locked_task.quantity / locked_task.batch_size
                        )

                        # First try n complete through inventory
                        for component_id in component["component_ids"]:
                            # skip component if zero inventory
                            available_qnty = self.inventory.get_quantity(component_id)
                            if available_qnty <= 0:
                                continue

                            # fulfill inventory condition
                            if available_qnty >= reqd_qnty:
                                useful_inventory[component_group][
                                    component_id
                                ] = reqd_qnty
                                reqd_qnty = 0
                                break

                            else:
                                if not self.consume_only_one:
                                    useful_inventory[component_group][
                                        component_id
                                    ] = available_qnty
                                    reqd_qnty = close_subtract(
                                        reqd_qnty, available_qnty
                                    )

                        if np.isclose(reqd_qnty, 0):
                            break

                        # Iterate over all completed task IDs
                        for completed_task_id in self.cti["last"]:
                            completed_task = self.tasks[completed_task_id]

                            # Skip if product ID is out of scope
                            if (
                                completed_task.product.material_id
                                not in component["component_ids"]
                            ):
                                continue

                            # Skip if batch size does not meet indirect mode
                            if not compare_linking_op(
                                available=completed_task.batch_size,
                                needed=component["quantity"],
                                mode=component["indirect"],
                            ):
                                continue

                            # Skip if only batch can be consumed and qnty does not meet the requirement
                            consumed_qnty = min(
                                reqd_qnty,
                                completed_task.quantity_remaining
                                - sum(
                                    [
                                        x[completed_task_id]
                                        for x in useful_completed_tasks.values()
                                    ]
                                ),
                            )
                            if self.consume_only_one and not np.isclose(
                                reqd_qnty - consumed_qnty, 0
                            ):
                                continue

                            # Hit reqd qnty
                            reqd_qnty = close_subtract(reqd_qnty, consumed_qnty)
                            useful_completed_tasks[component_group][
                                completed_task_id
                            ] = consumed_qnty

                            # Stop searching completed tasks if requirement is fulfilled
                            if np.isclose(reqd_qnty, 0):
                                break

                        # If unable to fulfill group requirement, then the task cant be unlocked
                        # Otherwise continue to next component group
                        if not np.isclose(reqd_qnty, 0):
                            unlocked = False
                            break

                    # Unlock the current locked task
                    if unlocked:
                        unlocked_any = True
                        for (
                            component_group,
                            component,
                        ) in locked_task.product.get_indirect_composition(
                            locked_task.batch_size, locked_task.alt_bom
                        ).items():
                            # Update consumed variable by inventory
                            for (
                                component_id,
                                consumed_qnty,
                            ) in useful_inventory[component_group].items():
                                locked_task.consume_inventory(
                                    component_group, component_id, consumed_qnty
                                )
                                self.inventory.decrease(component_id, consumed_qnty)

                            # Update consumed variable by batches
                            for (
                                completed_task_id,
                                consumed_qnty,
                            ) in useful_completed_tasks[component_group].items():
                                completed_task = self.tasks[completed_task_id]
                                locked_task.consume_batch(
                                    component_group,
                                    completed_task.batch_id,
                                    consumed_qnty,
                                )
                                completed_task.quantity_remaining -= consumed_qnty
                                if np.isclose(completed_task.quantity_remaining, 0):
                                    self.cti["last"].remove(completed_task.task_id)

                        # Release task and set information
                        locked_task.set_availability(task.task_finish)

                        self.ati.add_task(locked_task)
                        lti_to_be_removed.append(locked_task_id)

                for locked_task_id in lti_to_be_removed:
                    self.lti[
                        (parent_material_id, parent_batch_size, parent_alt_bom)
                    ].remove(locked_task_id)

            if not unlocked_any:
                logger.debug(
                    (
                        task.product.material_id,
                        task.batch_size,
                        task.quantity_remaining,
                        task.alt_bom,
                        task.ca_mode,
                    )
                )

    def log_remaining(self):
        wasted_total = {}
        wasted_25 = {}
        for t in self.cti["last"]:
            task = self.tasks[t]
            material_type = task.product.get_material_type()

            if material_type != MaterialType.fg.value:
                if material_type not in wasted_total:
                    wasted_total[material_type] = {
                        "qnty": 0,
                        "batches": 0,
                        "full_batches": 0,
                    }
                if material_type not in wasted_25:
                    wasted_25[material_type] = {
                        "qnty": 0,
                        "batches": 0,
                        "full_batches": 0,
                    }

                wasted_total[material_type]["qnty"] += task.quantity_remaining
                wasted_total[material_type]["batches"] += 1
                wasted_total[material_type]["full_batches"] += (
                    1 if np.isclose(task.quantity_remaining, task.batch_size) else 0
                )

                if task.task_end <= 600:
                    wasted_25[material_type]["qnty"] += task.quantity_remaining
                    wasted_25[material_type]["batches"] += 1
                    wasted_25[material_type]["full_batches"] += (
                        1 if np.isclose(task.quantity_remaining, task.batch_size) else 0
                    )

        logger.debug(f"Wasted total: {wasted_total}")
        logger.debug(f"Wasted 25: {wasted_25}")

        # Locked
        lti_length = 0
        locked_mios = 0
        num_locked_fgs = 0
        for task_ids in self.lti.values():
            for t in task_ids:
                task = self.tasks[t]
                material_type = task.product.get_material_type()
                logger.debug(
                    (
                        task.product.material_id,
                        task.batch_size,
                        task.quantity,
                        task.alt_bom,
                    )
                )

                lti_length += 1
                if material_type == MaterialType.fg.value:
                    num_locked_fgs += 1
                    locked_mios += (
                        task.quantity
                        * self.products[task.product.material_id].count_factor
                    )

        logger.debug(f"LTI length: {lti_length}")
        logger.debug(f"LTI FG length: {num_locked_fgs}")
        logger.debug(f"LTI mios pending: {locked_mios}")
