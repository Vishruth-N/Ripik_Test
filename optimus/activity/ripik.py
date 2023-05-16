"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
from optimus.activity.base import BaseActivityManager
import numpy as np
from typing import Dict, Any, TYPE_CHECKING
from optimus.arrangement.chain import SoloChain

if TYPE_CHECKING:
    from optimus.arrangement.task import Task
    from optimus.machines.machine import Machine
    from optimus.arrangement.tools import ATI


class RipikActivityManager(BaseActivityManager):
    def __init__(self) -> None:
        super().__init__()

    def compare_op_order(self, op_order):
        return op_order.astype(np.uint32)

    def select_alt_recipe(self, task: Task, change_state: bool = True):
        # Randomly select a recipe
        alt_recipe = np.random.choice(list(task.product.recipe[task.batch_size].keys()))
        return alt_recipe

    def choose_machine(self, task: Task):
        # Get approved machines
        approved_machines = task.get_approved_machines()

        # Choose random machine
        chosen_machine = approved_machines.get_random()
        return chosen_machine

    def __pick_random_task(self, ati: ATI):
        # Choose random task
        picked_task = np.random.choice(ati)
        return picked_task

    def create_core_chain(self, ati: ATI, metadata: Dict[str, Any] = None):
        # Pick a task required to build the core chain
        picked_task = self.__pick_random_task(ati=ati)

        # Build the core chain
        chain = SoloChain(task=picked_task)
        return chain
