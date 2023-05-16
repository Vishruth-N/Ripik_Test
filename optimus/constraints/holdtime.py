"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
from typing import Dict, List

import pandas as pd

from optimus.activity import BaseActivityManager
from optimus.arrangement.task import Task
from optimus.arrangement.tools import TaskGraph
from optimus.constraints.constraint import Constraint
from optimus.machines.machine import Machine


class HoldTime(Constraint):
    """
    IR0 - In-recipe zero
    Constraint only holds if all the subsequent processes of this task
    can be performed without triggering changeover anywhere
    """

    def __init__(
        self,
        task_graph: TaskGraph,
        activity_manager: BaseActivityManager,
        df_holdtime: pd.Dataframe,
    ) -> None:
        super().__init__(task_graph, activity_manager)

    def good_start_time(self, task: Task, start_time: float, machine: Machine) -> float:
        # Only applicable to sequence 0
        if task.sequence != 0:
            return task.available_at

        # Might not get a free state so limit iterations
        max_iterations = 20
        for iter_no in range(max_iterations):
            curr_task = task
            curr_start_time = start_time
            process_times = []
            while curr_task is not None:
                curr_machine = self.activity_manager.choose_machine(curr_task)

                # Fake process
                process_info, _ = curr_machine.get_room().fake_process(
                    task=curr_task, machine=curr_machine, start_time=curr_start_time
                )

                # Store
                process_times.append(
                    (
                        process_info.process_start_time,
                        process_info.process_end_time,
                    )
                )

                curr_start_time = process_info.process_finish_time
                curr_task = self.task_graph.obtain_next_sequence_task(
                    curr_task, start_time=curr_start_time
                )

            # TODO: Use holdtime data here and change sequence to op_order, currently hardcoded
            gaps = [0]
            for sequence_a, sequence_b, max_holdtime in [
                (1, 8, 24),  # (0, 1, 24),
                (5, 10, 48),  # (3, 4, 48),
                (4, 5, 0),
                (7, 20, 48),  # (6, 7, 48),
                (8, 9, 0),
                (11, 12, 0),
                (13, 14, 0),
                (16, 17, 0),
                (17, 18, 0),
                (21, 22, 0),
                (22, 23, 0),
                (23, 24, 0),
                (24, 25, 0),
            ]:
                gaps.append(
                    process_times[sequence_b][0]
                    - process_times[sequence_a][1]
                    - max_holdtime
                )
            reqd_shift_time = max(gaps)
            if reqd_shift_time == 0:
                break
            start_time += reqd_shift_time

        return start_time
