"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
from typing import Dict, List

from optimus.activity import BaseActivityManager
from optimus.arrangement.task import Task
from optimus.arrangement.tools import TaskGraph
from optimus.constraints.constraint import Constraint
from optimus.machines.event import MachineState
from optimus.machines.machine import Machine


class IR0_Changeover(Constraint):
    """
    IR0 - In-recipe zero
    Constraint only holds if all the subsequent processes of this task
    can be performed without triggering changeover anywhere
    """

    def __init__(
        self,
        task_graph: TaskGraph,
        activity_manager: BaseActivityManager,
        changeover_types: List[MachineState],
    ) -> None:
        super().__init__(task_graph, activity_manager)
        self.changeover_types = changeover_types

    def good_start_time(self, task: Task, start_time: float, machine: Machine) -> float:
        # Might not get a free state so limit iterations
        max_iterations = 10
        changeover_found = False
        for iter_no in range(max_iterations):
            curr_task = task
            curr_start_time = start_time
            changeover_found = False
            while curr_task is not None:
                curr_machine = self.activity_manager.choose_machine(curr_task)

                # Fake process
                process_info, _ = curr_machine.get_room().fake_process(
                    task=curr_task, machine=curr_machine, start_time=curr_start_time
                )

                # Update start time if changeover found
                for changeover_trigger in process_info.changeover_triggers:
                    if changeover_trigger["state"] in self.changeover_types:
                        changeover_found = True
                        start_time = changeover_trigger["end_time"]
                        break
                if changeover_found:
                    break

                curr_start_time = process_info.process_finish_time
                curr_task = self.task_graph.obtain_next_sequence_task(
                    curr_task, start_time=curr_start_time
                )

            if not changeover_found:
                break

        return start_time


class IR0_RoomChangeoverB(IR0_Changeover):
    """In-recipe zero Changeover B"""

    def __init__(
        self,
        task_graph: TaskGraph,
        activity_manager: BaseActivityManager,
    ) -> None:
        super().__init__(
            task_graph,
            activity_manager,
            changeover_types=[MachineState.ROOM_CHANGEOVER_B],
        )

    def good_start_time(self, task: Task, start_time: float, machine: Machine) -> float:
        return super().good_start_time(task, start_time, machine)
