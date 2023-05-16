"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
import pandas as pd
from typing import Tuple, Optional

from optimus.machines.event import MachineState
from optimus.arrangement.task import Task

# Forward declarations
class Machine:
    pass


class Room:
    registry = {}

    def __init__(
        self,
        room_id: str,
        block_id: str,
        changeover: Optional[pd.DataFrame] = None,
    ) -> None:
        self.room_id = room_id
        self.block_id = block_id
        self.changeover = changeover

        self.machines = {}
        self.last_changeoverB_end_time = 0
        self.num_stinged_processes = 0

        # Update registry
        self.registry[self.room_id] = self

    def add_machine(self, machine: Machine) -> None:
        """Adds the given machine"""
        self.machines[machine.machine_id] = machine

    def get_changeover_time(self, changeover_type: Optional[str]):
        """Get changeover time"""
        if self.changeover is None:
            return 0
        if changeover_type is None:
            return 0

        return self.changeover[changeover_type].iloc[0]

    def get_next_availability(self) -> float:
        """Returns time when all machines in room are available"""
        last_availability = 0
        for machine in self.machines.values():
            last_availability = max(last_availability, machine.get_availability())
        return last_availability

    def do_changeover(
        self,
        changeover_type: Optional[str],
        changeover_start: float,
        msg: Optional[str] = None,
        execute: bool = True,
    ) -> Tuple:
        """Perform cleanover the room"""
        # Find changeover duration
        changeover_duration = self.get_changeover_time(changeover_type)
        if changeover_duration == 0:
            return None, []

        # update room state
        changeover_end = changeover_start + changeover_duration
        if changeover_type == "type_A":
            machine_state = MachineState.ROOM_CHANGEOVER_A
        elif changeover_type == "type_B":
            machine_state = MachineState.ROOM_CHANGEOVER_B
        else:
            raise ValueError(f"Unsupported room changeover type: {changeover_type}")

        # Then clean all the machines
        changeover_end = changeover_start
        for machine in self.machines.values():
            changeover_event = machine.create_event(
                start_time=changeover_start,
                duration=changeover_duration,
                state=machine_state,
                note=msg,
            )
            changeover_start = changeover_event.start_time
            changeover_end = changeover_event.end_time

            if execute:
                machine.update_schedule(changeover_event)

        if execute and machine_state == MachineState.ROOM_CHANGEOVER_B:
            self.last_changeoverB_end_time = changeover_end
            self.num_stinged_processes = 0

        changeover_triggers = [
            {
                "state": machine_state,
                "start_time": changeover_event.start_time,
                "end_time": changeover_event.end_time,
            }
        ]

        return changeover_end, changeover_triggers

    def get_max_campaign_length(self) -> Tuple[float, float]:
        raise NotImplementedError()

    def apply_room_constraints(
        self,
        machine: Machine,
        task: Task,
        start_time: float,
        execute: bool = True,
    ) -> None:
        raise NotImplementedError()

    def __process(
        self,
        task: Task,
        machine: Machine,
        start_time: Optional[float] = None,
        execute: bool = True,
    ):
        """Process task in the given machine"""
        # Room constraint
        changeover_end, changeover_triggers = self.apply_room_constraints(
            machine=machine,
            task=task,
            start_time=start_time,
            execute=execute,
        )

        best_machine_setuptime = machine.get_setuptime(
            product=task.product,
            batch_size=task.batch_size,
            alt_recipe=task.alt_recipe,
            sequence=task.sequence,
        )
        if changeover_end is None:
            start_time = start_time - best_machine_setuptime
        else:
            start_time = changeover_end

        # Run a machine normally without machine constraints
        if execute:
            process_info = machine.process(
                task=task,
                start_time=start_time,
                fit_method="last",
            )
        else:
            process_info = machine.fake_process(
                task=task,
                start_time=start_time,
                fit_method="last",
            )

        # Update room params
        self.num_stinged_processes += 1
        for changeover_trigger in changeover_triggers:
            process_info.add_changeover_trigger(changeover_trigger)

        # Find busy states after changeover B
        approved_machines = list(
            filter(
                lambda approved_machine: approved_machine.machine_id in self.machines,
                task.get_approved_machines(),
            )
        )
        curr_campaign_length = 0
        for approved_machine in approved_machines:
            curr_campaign_length += len(approved_machine.batches_after_choB)

        # Feedback
        feedback = {
            "machine": machine,
            "product": task.product,
            "ideal_availability": process_info.process_end_time
            + max(
                0,
                self.get_changeover_time("type_B")
                - self.get_changeover_time("type_A")
                + best_machine_setuptime,
            ),
            "curr_campaign_length": curr_campaign_length,
            "max_campaign_length": self.get_max_campaign_length()[0],
        }

        return (
            process_info,
            feedback,
        )

    def process(self, task: Task, machine: Machine, start_time: float):
        """Process task in the given machine"""
        return self.__process(
            task=task,
            machine=machine,
            start_time=start_time,
            execute=True,
        )

    def fake_process(self, task: Task, machine: Machine, start_time: float):
        """Process task in the given machine but don't execute in schedule"""
        return self.__process(
            task=task,
            machine=machine,
            start_time=start_time,
            execute=False,
        )
