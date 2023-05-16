"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
from typing import Dict, Tuple, Any

from optimus.activity import BaseActivityManager
from optimus.arrangement.task import Task
from optimus.arrangement.tools import TaskGraph
from optimus.constraints.inrecipe_changeovers import IR0_RoomChangeoverB
from optimus.constraints.holdtime import HoldTime
from optimus.machines.machine import Machine


class ProcessHandler:
    def __init__(
        self,
        task_graph: TaskGraph,
        config: Dict[str, Any],
        activity_manager: BaseActivityManager,
    ) -> None:
        self.task_graph = task_graph
        self.activity_manager = activity_manager

        # Build constraints
        self.constraints = []
        if config["IR0_changeoverB"]:
            self.constraints.append(
                IR0_RoomChangeoverB(
                    task_graph=self.task_graph,
                    activity_manager=self.activity_manager,
                )
            )
        if config["holdtime"]:
            self.constraints.append(
                HoldTime(
                    task_graph=self.task_graph,
                    activity_manager=self.activity_manager,
                    df_holdtime=None,
                )
            )

    def set_alt_recipe(self, task: Task, execute: bool = True) -> None:
        """Sets alt recipe only if none"""
        if task.alt_recipe is None:
            assert (
                task.sequence == 0
            ), "Other operations must have alt recipe known from first operation"
            # Select alt recipe
            alt_recipe = self.activity_manager.select_alt_recipe(
                task=task, change_state=execute
            )

            # Set alt recipe to the task
            task.set_alt_recipe(alt_recipe=alt_recipe)

    def assign_machine(self, task: Task):
        machine = self.activity_manager.choose_machine(task)
        return machine

    def __process(self, task: Task, machine: Machine = None, execute: bool = True):
        """
        Assign a machine to the given task and run it
        Params
        -------------------------
        """
        assert task.task_end is None, "Already processed this task"

        # Sets alt recipe
        self.set_alt_recipe(task=task, execute=execute)

        # Assign best machine
        if machine is None:
            machine = self.assign_machine(task)

        # Update resources by constraints
        best_start_time = task.available_at
        for constraint in self.constraints:
            good_start_time = constraint.good_start_time(
                task=task, start_time=best_start_time, machine=machine
            )
            best_start_time = max(best_start_time, good_start_time)

        # Process
        if execute:
            (process_info, feedback,) = machine.get_room().process(
                task=task,
                machine=machine,
                start_time=best_start_time,
            )
        else:
            (process_info, feedback,) = machine.get_room().fake_process(
                task=task,
                machine=machine,
                start_time=best_start_time,
            )

        return (
            machine,
            process_info,
            feedback,
        )

    def fake_process(self, task: Task):
        return self.__process(task=task, execute=False)

    def fake_process_with_machine(self, task: Task, machine: Machine):
        _, process_info, feedback = self.__process(
            task=task, machine=machine, execute=False
        )
        return process_info, feedback

    def process(self, task: Task):
        return self.__process(task=task, execute=True)

    def process_with_machine(self, task: Task, machine: Machine):
        _, process_info, feedback = self.__process(
            task=task, machine=machine, execute=True
        )
        return process_info, feedback
