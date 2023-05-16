"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from optimus.arrangement.task import Task
    from optimus.machines.machine import Machine
    from optimus.machines.feedback import ProcessInfo


class Chain:
    def obtain_task(self) -> Task:
        raise NotImplementedError()

    def prune(self) -> None:
        raise NotImplementedError()

    def complete_task(
        self,
        task: Task,
        machine: Machine,
        process_info: ProcessInfo,
    ) -> None:
        raise NotImplementedError()

    def empty(self) -> bool:
        raise NotImplementedError()


class SoloChain(Chain):
    def __init__(self, task: Task) -> None:
        self.task = task

    def obtain_task(self) -> Task:
        assert self.task.ready()
        return self.task

    def prune(self) -> None:
        if self.task.task_end is not None:
            self.task = None

    def complete_task(
        self, task: Task, machine: Machine, process_info: ProcessInfo
    ) -> None:
        if self.task == task:
            # Process the task
            task.process_task(
                task_start=process_info.process_start_time,
                task_end=process_info.process_end_time,
                quantity_processed=process_info.quantity_processed,
                task_finish=process_info.process_finish_time,
            )
            # Remove the task
            self.task = None

        else:
            raise ValueError("Task does not exist in the chain")

    def empty(self) -> bool:
        return self.task is None


class BatchChain(Chain):
    def __init__(self, task: Task) -> None:
        self.chains = [SoloChain(task)]
        batch_id = task.batch_id
        while not task.is_last_sequence():
            for out_node in task.out_nodes:
                if out_node.batch_id == batch_id:
                    assert out_node.sequence > task.sequence
                    task = out_node
            self.chains.append(SoloChain(task))
        self.chains.reverse()

    def obtain_task(self) -> Task:
        return self.chains[-1].obtain_task()

    def prune(self) -> None:
        i = len(self.chains) - 1
        while i >= 0:
            self.chains[i].prune()
            if self.chains[i].empty():
                del self.chains[i]
            i -= 1

    def complete_task(
        self, task: Task, machine: Machine, process_info: ProcessInfo
    ) -> None:
        # Process the task
        self.chains[-1].complete_task(
            task=task,
            machine=machine,
            process_info=process_info,
        )
        # Remove the task
        if self.chains[-1].empty():
            self.chains.pop()

    def empty(self) -> bool:
        return len(self.chains) == 0


class StraightChain(Chain):
    def __init__(self, tasks: List[Task]) -> None:
        self.chains = [SoloChain(task) for task in tasks]
        self.chains.reverse()

    def obtain_task(self) -> Task:
        return self.chains[-1].obtain_task()

    def prune(self) -> None:
        i = len(self.chains) - 1
        while i >= 0:
            self.chains[i].prune()
            if self.chains[i].empty():
                del self.chains[i]
            i -= 1

    def complete_task(
        self, task: Task, machine: Machine, process_info: ProcessInfo
    ) -> None:
        # Process the task
        self.chains[-1].complete_task(
            task=task,
            machine=machine,
            process_info=process_info,
        )
        # Remove the task
        if self.chains[-1].empty():
            self.chains.pop()

    def empty(self) -> bool:
        return len(self.chains) == 0


class CampaignChain(Chain):
    def __init__(self, tasks: List[Task]) -> None:
        self.chains = [BatchChain(task) for task in tasks]

    def obtain_task(self) -> Task:
        return self.chains[-1].obtain_task()

    def prune(self) -> None:
        i = len(self.chains) - 1
        while i >= 0:
            self.chains[i].prune()
            if self.chains[i].empty():
                del self.chains[i]
            i -= 1

    def complete_task(
        self, task: Task, machine: Machine, process_info: ProcessInfo
    ) -> None:
        self.chains[-1].complete_task(
            task=task, machine=machine, process_info=process_info
        )
        if self.chains[-1].empty():
            self.chains.pop()

    def empty(self) -> bool:
        return len(self.chains) == 0
