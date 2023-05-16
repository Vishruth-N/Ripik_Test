"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
from ordered_set import OrderedSet
from typing import TypeVar, Optional, Dict, List, Tuple

import numpy as np

from optimus.utils.general import multidict, RandomizedSet
from optimus.elements.product import Material
from optimus.utils.structs import DemandST

SelfTask = TypeVar("SelfTask", bound="Task")


class Task:
    # To track task id
    NUM_TASKS = 0

    def __init__(
        self,
        product: Material,
        batch_id: str,
        batch_size: float,
        alt_bom: str,
        ca_mode: str,
        quantity: float,
        priority: float,
        due_date: float,
        demand_contri: List[Tuple[str, float]],
        sequence: int = 0,
        available_at: Optional[float] = None,
        is_initially_locked: bool = True,
        is_initialised: bool = False,
        consumed: Dict[str, Dict[str, float]] = None,
    ) -> None:
        self.task_id = Task.NUM_TASKS
        self.product = product
        self.batch_id = batch_id
        self.batch_size = batch_size
        self.alt_bom = alt_bom
        self.ca_mode = ca_mode
        self.quantity = quantity
        self.sequence = sequence
        self.priority = priority
        self.due_date = due_date
        self.demand_contri = demand_contri

        self.quantity_remaining = quantity
        self.is_initially_locked = is_initially_locked
        self.is_initialised = is_initialised
        self.available_at = available_at

        self.consumed = multidict(2, list)
        if consumed is not None:
            self.consumed = consumed

        self.in_nodes = []
        self.out_nodes = []
        self.preferences = {"machines": OrderedSet()}

        # During completion
        self.alt_recipe = None

        # After completion
        self.task_start = None
        self.task_end = None
        self.quantity_processed = None

        # Update static variables
        Task.NUM_TASKS += 1

    ######################### MAGIC FUNCTIONS #########################
    def __eq__(self, __o: SelfTask) -> bool:
        return self.task_id == __o.task_id

    def __hash__(self):
        return hash(self.task_id)

    ######################### CONSUMED HANDLING #########################
    def consume_inventory(self, component_group: str, component_id: str, qnty: float):
        self.consumed["inventory"][component_group].append((component_id, qnty))

    def consume_batch(self, component_group: str, batch_id: str, qnty: float):
        self.consumed["batches"][component_group].append((batch_id, qnty))

    def get_consumed_inventory(self):
        return self.consumed["inventory"]

    def get_consumed_batches(self):
        return self.consumed["batches"]

    ######################### HELPER FUNCTIONS #########################
    @property
    def demand_priority(self):
        cols = DemandST.hilo_priority_cols()
        min_index = len(cols)
        for demand_col, qnty in self.demand_contri:
            min_index = min(min_index, cols.index(demand_col))
        return len(cols) - min_index

    def ready(self) -> bool:
        return self.available_at is not None

    def is_last_sequence(self) -> bool:
        assert self.alt_recipe is not None, "Cannot be determined (no alt recipe)"
        return (
            self.sequence
            == len(self.product.get_recipe(self.batch_size, self.alt_recipe)) - 1
        )

    ######################### MACHINE RELATIONS #########################
    def get_approved_machines(self) -> RandomizedSet:
        assert (
            self.alt_recipe is not None
        ), "Requesting approved machines without setting alt recipe"
        return self.product.approved_machines_of(
            batch_size=self.batch_size,
            alt_recipe=self.alt_recipe,
            sequence=self.sequence,
        )

    def get_possible_machines(self) -> RandomizedSet:
        possible_machines = RandomizedSet()
        if self.alt_recipe is None:
            alt_recipes = [
                alt_recipe
                for _, alt_recipe in self.product.iterate_recipes(
                    batch_size=self.batch_size
                )
            ]
        else:
            alt_recipes = [self.alt_recipe]

        for alt_recipe in alt_recipes:
            for machine in self.product.approved_machines_of(
                batch_size=self.batch_size,
                alt_recipe=alt_recipe,
                sequence=self.sequence,
            ):
                possible_machines.add(machine)
        return possible_machines

    def estimate_availability(self) -> float:
        estimated_availability = np.inf
        possible_rooms = {}
        for machine in self.get_possible_machines():
            if machine.room_id not in possible_rooms:
                possible_rooms[machine.room_id] = machine.get_room()

        for room in possible_rooms.values():
            room_availability = room.get_next_availability()
            estimated_availability = min(estimated_availability, room_availability)
        assert not np.isinf(estimated_availability)
        estimated_availability = max(estimated_availability, self.available_at)
        return estimated_availability

    ######################### CORE FUNCTIONS #########################
    def set_availability(self, available_at: float) -> None:
        if self.available_at is None:
            self.available_at = available_at
        else:
            self.available_at = max(self.available_at, available_at)

    def set_alt_recipe(self, alt_recipe: str) -> None:
        self.alt_recipe = alt_recipe

    def process_task(
        self,
        task_start: float,
        task_end: float,
        quantity_processed: float,
        task_finish: float,
    ) -> None:
        self.task_start = task_start
        self.task_end = task_end
        self.quantity_processed = quantity_processed
        self.task_finish = task_finish
