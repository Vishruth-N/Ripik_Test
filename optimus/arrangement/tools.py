"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Iterable

from sortedcontainers import SortedSet

from optimus.arrangement.task import Task
from optimus.preschedule.batching import Batch
from optimus.elements.inventory import Inventory


class TaskGraph:
    def __init__(
        self,
        feasible_batches: Dict[str, Batch],
        inventory: Inventory,
        initial_start_time: float = 0,
    ) -> None:
        self.inventory = inventory
        self.initial_start_time = initial_start_time
        self._tasks = self._make_task_graph(feasible_batches=feasible_batches)

    def _make_task_graph(self, feasible_batches: Dict[str, Batch]) -> Dict[str, Task]:
        """
        Make task graph given feasible batches
        """
        tasks = {}
        for batch_id, batch in feasible_batches.items():
            is_initially_locked = batch.is_locked()

            # Set available at
            available_at = None
            if not is_initially_locked:
                available_at = self.initial_start_time

            # Set sequence
            sequence = 0
            if batch.initialised:
                sequence = batch.sequence
                if batch.start_time is not None:
                    available_at = batch.start_time

            # Create the first task
            task = Task(
                product=batch.product,
                batch_id=batch_id,
                batch_size=batch.batch_size,
                alt_bom=batch.alt_bom,
                ca_mode=batch.ca_mode,
                quantity=batch.quantity,
                priority=batch.priority,
                due_date=batch.due_date,
                demand_contri=batch.demand_contri,
                sequence=sequence,
                is_initially_locked=is_initially_locked,
                is_initialised=batch.initialised,
                available_at=available_at if not is_initially_locked else None,
            )
            tasks[task.task_id] = task

            if batch.initialised:
                # Set machine preference and alt recipe
                task.set_alt_recipe(batch.alt_recipe)
                if batch.machine_id is not None:
                    for approved_machine in task.get_approved_machines():
                        if approved_machine.machine_id == batch.machine_id:
                            task.preferences["machines"].discard(approved_machine)
                            task.preferences["machines"].add(approved_machine)
                            break

            else:
                if not is_initially_locked:
                    # Consume inventory
                    for component_group, component in batch.requires.items():
                        assert len(component["consumes"]["batches"]) == 0
                        for component_id, qnty in component["consumes"]["inventory"]:
                            task.consume_inventory(component_group, component_id, qnty)
                            self.inventory.decrease(component_id, qnty)

        return tasks

    def get_tasks(self) -> Dict[str, Task]:
        return self._tasks

    def obtain_next_sequence_task(
        self, task: Task, start_time: Optional[float] = None
    ) -> Optional[Task]:
        if not task.is_last_sequence():
            # Find attribute values
            if start_time is not None:
                available_at = start_time
            else:
                available_at = task.task_finish

            quantity = task.quantity_processed
            if quantity is None:
                quantity = task.quantity

            # Create task
            next_task = Task(
                product=task.product,
                batch_id=task.batch_id,
                batch_size=task.batch_size,
                alt_bom=task.alt_bom,
                ca_mode=task.ca_mode,
                quantity=quantity,
                priority=task.priority,
                due_date=task.due_date,
                demand_contri=task.demand_contri,
                sequence=task.sequence + 1,
                available_at=available_at,
                is_initially_locked=task.is_initially_locked,
                consumed=task.consumed,
            )

            # Pass necessary information
            next_task.set_alt_recipe(task.alt_recipe)

            return next_task

    def duplicate_task(self, task: Task) -> Task:
        duplicate_task = Task(
            product=task.product,
            batch_id=task.batch_id,
            batch_size=task.batch_size,
            alt_bom=task.alt_bom,
            ca_mode=task.ca_mode,
            quantity=task.quantity,
            priority=task.priority,
            due_date=task.due_date,
            demand_contri=task.demand_contri,
            sequence=task.sequence,
            available_at=task.available_at,
            is_initially_locked=task.is_initially_locked,
            is_initialised=task.is_initialised,
            consumed=task.consumed,
        )
        duplicate_task.set_alt_recipe(task.alt_recipe)

        return duplicate_task


class ATI:
    def __init__(self) -> None:
        self._core_list = SortedSet(key=self.__core_sort_key)
        self._family_grouped = defaultdict(set)

    def __core_sort_key(self, task: Task):
        return task.demand_priority

    def length(self):
        return len(self._core_list)

    def add_task(self, task: Task) -> None:
        assert task.task_end is None, f"Already processed task: {task.task_id}"

        # Add to core list
        self._core_list.add(task)

        # Add to family sequence grouped
        self._family_grouped[task.product.family_id].add(task)

    def remove_task(self, task: Task) -> None:
        # Remove from core list
        self._core_list.remove(task)

        # Remove from family sequence grouped
        self._family_grouped[task.product.family_id].remove(task)
        if len(self._family_grouped[task.product.family_id]) == 0:
            del self._family_grouped[task.product.family_id]

    def get_tasks(self, family_id: str = None) -> Iterable[Task]:
        if family_id is None:
            return self._core_list

        else:
            return self._family_grouped[family_id]
