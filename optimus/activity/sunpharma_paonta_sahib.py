"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
from optimus.activity.base import BaseActivityManager
import random
import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING
from optimus.arrangement.chain import SoloChain, StraightChain
from optimus.utils.constants import MaterialType
from optimus.utils.structs import DemandST

if TYPE_CHECKING:
    from optimus.arrangement.tools import ATI
    from optimus.arrangement.task import Task
    from optimus.machines.machine import Machine

import logging

logger = logging.getLogger(__name__)


class SunPharmaPaontaSahibActivityManager(BaseActivityManager):
    def __init__(self) -> None:
        super().__init__()
        self.alt_recipe_cached = {}

    def compare_op_order(self, op_order):
        return op_order.str[-2:].astype(np.uint32)

    def select_alt_recipe(self, task: Task, change_state: bool = True):
        if (task.product.material_id, task.batch_size) in self.alt_recipe_cached:
            return self.alt_recipe_cached[(task.product.material_id, task.batch_size)]

        # Select the best recipe
        best_alt_recipe = None
        best_val = np.inf
        for alt_recipe in task.product.recipe[task.batch_size]:
            total_val = 0
            num_ops = 0
            for sequence, op in enumerate(
                task.product.get_recipe(task.batch_size, alt_recipe)
            ):
                val = 0
                for machine in op["machines"]:
                    val += machine.get_setuptime(
                        product=task.product,
                        batch_size=task.batch_size,
                        alt_recipe=alt_recipe,
                        sequence=sequence,
                    )
                    val += machine.get_runtime(
                        product=task.product,
                        batch_size=task.batch_size,
                        alt_recipe=alt_recipe,
                        sequence=sequence,
                    )
                val /= op["machines"].get_length() ** 2
                total_val += val
                num_ops += 1

            total_val /= num_ops
            if total_val <= best_val:
                best_val = total_val
                best_alt_recipe = alt_recipe

        # Cache results and return
        self.alt_recipe_cached[
            (task.product.material_id, task.batch_size)
        ] = best_alt_recipe
        return best_alt_recipe

    def choose_machine(self, task: Task) -> Machine:
        # Get approved machines
        approved_machines = task.get_approved_machines()

        # Try task preferences first - machine, room and then block
        chosen_machine = None
        good_found = False
        if task.preferences is not None and "machines" in task.preferences:
            for preferred_machine in reversed(task.preferences["machines"]):
                for machine in approved_machines:
                    if preferred_machine.block_id == machine.block_id:
                        chosen_machine = machine
                        if preferred_machine.room_id == machine.room_id:
                            good_found = True
                    if good_found:
                        break
                if good_found:
                    break

        # Try dedicated blocks
        # if chosen_machine is None:
        #     dedicated_blocks = task.product.get_dedicated_blocks(
        #         batch_size=task.batch_size, alt_recipe=task.alt_recipe
        #     )
        #     good_machines = RandomizedSet()
        #     for block_id in dedicated_blocks:
        #         for machine in approved_machines:
        #             if machine.block_id == block_id:
        #                 good_machines.add(machine)
        #     if good_machines.get_length() > 0:
        #         chosen_machine = good_machines.get_random()
        #         for machine in good_machines:
        #             if (
        #                 self.rooms[machine.room_id].get_next_availability()
        #                 < self.rooms[chosen_machine.room_id].get_next_availability()
        #             ):
        #                 chosen_machine = machine

        # Try nearest available
        if chosen_machine is None:
            chosen_machine = approved_machines.get_random()
            for machine in approved_machines:
                if (
                    self.rooms[machine.room_id].get_next_availability()
                    < self.rooms[chosen_machine.room_id].get_next_availability()
                ):
                    chosen_machine = machine

        return chosen_machine

    def __pick_fifo_task(self, ati: ATI):
        # Get minimum task in ati
        picked_tasks = [
            {
                "task": None,
                "estimated_availability": np.inf,
                "priority": 0,
                "demand_priority": 0,
            }
        ]

        for task in ati.get_tasks():
            estimated_availability = task.estimate_availability() // 24

            priority = task.priority
            demand_priority = task.demand_priority

            curr_item = {
                "task": task,
                "estimated_availability": estimated_availability,
                "priority": priority,
                "demand_priority": demand_priority,
            }

            if estimated_availability == picked_tasks[0]["estimated_availability"]:
                if demand_priority == picked_tasks[0]["demand_priority"]:
                    if priority == picked_tasks[0]["priority"]:
                        picked_tasks.append(curr_item)

                    elif priority > picked_tasks[0]["priority"]:
                        picked_tasks = [curr_item]

                elif demand_priority > picked_tasks[0]["demand_priority"]:
                    picked_tasks = [curr_item]

            elif estimated_availability < picked_tasks[0]["estimated_availability"]:
                picked_tasks = [curr_item]

        final_picked_task = np.random.choice(picked_tasks)["task"]
        return final_picked_task

    def __pick_tasks(self, ati: ATI, k: int, master_task: Task):
        picked_tasks = []
        for task in ati.get_tasks():
            if master_task.product.family_id == task.product.family_id:
                picked_tasks.append(task)
        picked_tasks = picked_tasks[:k]
        return picked_tasks

    def create_core_chain(self, ati: ATI, metadata: Dict[str, Any] = None):
        # Pick a task required to build the core chain
        picked_task = self.__pick_fifo_task(ati=ati)

        # Build the core chain
        chain = SoloChain(task=picked_task)

        # if picked_task.product.get_material_type() != MaterialType.fg.value:
        #     chain = SoloChain(task=picked_task)

        # else:
        #     # TODO: Obtain campaign length (+ default)
        #     campaign_length = 5
        #     if campaign_length is None:
        #         campaign_length = 5

        #     picked_tasks = self.__pick_tasks(
        #         available_task_ids=available_task_ids,
        #         k=campaign_length,
        #         master_task=picked_task,
        #     )

        #     chain = StraightChain(tasks=picked_tasks)

        return chain
