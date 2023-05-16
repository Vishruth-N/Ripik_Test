"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations

from optimus.activity import BaseActivityManager
from optimus.arrangement.task import Task
from optimus.arrangement.tools import TaskGraph
from optimus.machines.machine import Machine


class Constraint:
    def __init__(
        self,
        task_graph: TaskGraph,
        activity_manager: BaseActivityManager,
    ) -> None:
        self.task_graph = task_graph
        self.activity_manager = activity_manager

    def good_start_time(self, task: Task, start_time: float, machine: Machine) -> float:
        raise NotImplementedError("Must be overrided by child class")
