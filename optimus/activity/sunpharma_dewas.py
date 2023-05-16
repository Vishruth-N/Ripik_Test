"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
from copy import deepcopy
from typing import List, Dict, Any, TYPE_CHECKING

import numpy as np

from optimus.activity.base import BaseActivityManager
from optimus.arrangement.chain import SoloChain, StraightChain
from optimus.utils.structs import DemandST

if TYPE_CHECKING:
    from optimus.arrangement.tools import ATI
    from optimus.utils.general import RandomizedSet
    from optimus.arrangement.task import Task
    from optimus.machines.machine import Machine


class SunPharmaDewasActivityManager(BaseActivityManager):
    def __init__(self) -> None:
        super().__init__()
        self.alt_recipe_storage = {}
        self.alt_recipe_fake_storage = {}
        self.is_fake_started = False

    def compare_op_order(self, op_order):
        return op_order.str[-3:].astype(np.uint32)

    def __select_alt_recipe(
        self, task: Task, alt_recipe_storage: Dict[str, Dict[str, str]]
    ) -> str:
        # Get all alt recipes
        alt_recipes = set(
            [
                alt_recipe
                for _, alt_recipe in task.product.iterate_recipes(
                    batch_size=task.batch_size
                )
            ]
        )

        # Create if does not exist
        if task.product.material_id not in alt_recipe_storage:
            alt_recipe_storage[task.product.material_id] = {
                "batch_id": task.batch_id,
                "alt_recipe": np.random.choice(list(alt_recipes)),
            }

        # Change route if batch ID does not match
        prev_alt_recipe = alt_recipe_storage[task.product.material_id]["alt_recipe"]
        if task.batch_id == alt_recipe_storage[task.product.material_id]["batch_id"]:
            return prev_alt_recipe

        else:
            if len(alt_recipes) > 1:
                alt_recipes = alt_recipes - set([prev_alt_recipe])
            alt_recipe = np.random.choice(list(alt_recipes))

            alt_recipe_storage[task.product.material_id]["alt_recipe"] = alt_recipe
            alt_recipe_storage[task.product.material_id]["batch_id"] = task.batch_id

            return alt_recipe

    def select_alt_recipe(self, task: Task, change_state: bool = True) -> str:
        if change_state:
            self.is_fake_started = False
            return self.__select_alt_recipe(
                task=task, alt_recipe_storage=self.alt_recipe_storage
            )

        else:
            if not self.is_fake_started:
                self.alt_recipe_fake_storage = deepcopy(self.alt_recipe_storage)
                self.is_fake_started = True

            return self.__select_alt_recipe(
                task=task, alt_recipe_storage=self.alt_recipe_fake_storage
            )

    def choose_machine(self, task: Task):
        # Get approved machines
        approved_machines = task.get_approved_machines()

        # Choose random machine
        chosen_machine = approved_machines.get_random()
        return chosen_machine

    def __pick_fifo_task(self, ati: ATI, metadata: Dict[str, Any] = None):
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
            estimated_availability = task.estimate_availability()

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
        picked_task = self.__pick_fifo_task(ati=ati, metadata=metadata)

        # Build the core chain
        chain = SoloChain(task=picked_task)

        # campaign_length = 12
        # if campaign_length is None:
        #     campaign_length = 12

        # picked_tasks = self.__pick_tasks(
        #     available_task_ids=available_task_ids,
        #     k=campaign_length,
        #     master_task=picked_task,
        # )

        # chain = StraightChain(tasks=picked_tasks)
        return chain
