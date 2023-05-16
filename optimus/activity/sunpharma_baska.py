"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
from optimus.activity.base import BaseActivityManager
import random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, TYPE_CHECKING
from optimus.arrangement.chain import SoloChain, StraightChain
from optimus.utils.structs import DemandST

if TYPE_CHECKING:
    from optimus.utils.general import RandomizedSet
    from optimus.arrangement.task import Task
    from optimus.machines.machine import Machine
    from optimus.arrangement.tools import ATI


class SunPharmaBaskaActivityManager(BaseActivityManager):
    def __init__(self) -> None:
        super().__init__()
        self.alt_recipes = {}

    def compare_op_order(self, op_order):
        return op_order.str[-3:].astype(np.uint32)

    def select_alt_recipe(self, task: Task, change_state: bool = True):
        # Get all alt recipes
        alt_recipes = set(task.product.recipe[task.batch_size].keys())

        # Create if does not exist
        if task.product.material_id not in self.alt_recipes:
            self.alt_recipes[task.product.material_id] = {
                "batch_id": task.batch_id,
                "alt_recipe": np.random.choice(list(alt_recipes)),
            }

        # Change route if batch ID does not match
        prev_alt_recipe = self.alt_recipes[task.product.material_id]["alt_recipe"]
        if task.batch_id == self.alt_recipes[task.product.material_id]["batch_id"]:
            return prev_alt_recipe

        else:
            if len(alt_recipes) > 1:
                alt_recipes = alt_recipes - set([prev_alt_recipe])
            alt_recipe = np.random.choice(list(alt_recipes))

            self.alt_recipes[task.product.material_id]["alt_recipe"] = alt_recipe
            self.alt_recipes[task.product.material_id]["batch_id"] = task.batch_id

            return alt_recipe

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
                "priority": False,
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

    def create_core_chain(self, ati: ATI, metadata: Dict[str, Any] = None):
        # Pick a task required to build the core chain
        picked_task = self.__pick_fifo_task(ati=ati, metadata=metadata)

        # Build the core chain
        chain = SoloChain(task=picked_task)

        return chain
