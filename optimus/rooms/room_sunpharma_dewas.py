"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
import pandas as pd

from optimus.machines.event import MachineState
from optimus.utils.constants import MachineType
from optimus.rooms.room import Room
from optimus.arrangement.task import Task
from optimus.utils.structs import *

# Forward declarations
class Machine:
    pass


class SunPharmaDewasRoom(Room):
    def __init__(
        self,
        room_id: str,
        block_id: str,
        changeover: pd.DataFrame,
    ) -> None:
        super().__init__(room_id, block_id, changeover)

        if self.room_id == "S01":
            self.last_changeoverB_end_time = 24 * 7
        elif self.room_id == "S02":
            self.last_changeoverB_end_time = 24 * 7
        elif self.room_id == "S03":
            self.last_changeoverB_end_time = 24 * 8
        elif self.room_id == "S03_B":
            self.last_changeoverB_end_time = 24 * 9

    def get_max_campaign_length(self) -> Tuple[float, float]:
        if self.room_id == "S01":
            changeover_by_batch = 12
            changeover_by_days = 14
        elif self.room_id == "S02":
            changeover_by_batch = 1
            changeover_by_days = np.inf
        elif self.room_id == "S03":
            changeover_by_batch = 8
            changeover_by_days = 13
        elif self.room_id == "S03_B":
            changeover_by_batch = 5
            changeover_by_days = 6
        else:
            changeover_by_batch = 5
            changeover_by_days = 6

        return changeover_by_batch, changeover_by_days

    def determine_changeover_type(
        self,
        machine: Machine,
        task: Task,
        start_time: float,
    ):
        # Infinite machine doesnt have any changeovers
        changeover_type, msg = None, ""
        if machine.machine_type == MachineType.infinite.value:
            return changeover_type, msg

        # Get the latest process in the room
        latest_event = None
        for curr_machine in self.machines.values():
            curr_machine_last_event = curr_machine.get_last_event()
            if curr_machine_last_event is None:
                continue

            # Because only one thing can happen at a time, no machine intersects
            if (
                latest_event is None
                or latest_event.end_time < curr_machine_last_event.end_time
            ):
                latest_event = curr_machine_last_event

        # Changeover conditions
        if latest_event is None or latest_event.state != MachineState.BUSY:
            return changeover_type, msg

        prev_batch_id = latest_event.task.batch_id
        prev_family_id = latest_event.task.product.family_id

        if prev_family_id != task.product.family_id:
            changeover_type = "type_B"
            msg = f"Family changed from {prev_family_id} to {task.product.family_id}"

        elif prev_batch_id != task.batch_id:
            # Check for changeover by days condition
            changeover_by_batch, changeover_by_days = self.get_max_campaign_length()
            changeoverB_start = max(
                self.get_next_availability(),
                start_time,
            )

            # Find num batches after changeover B
            machine_events = machine.get_events_till_state(
                state=MachineState.ROOM_CHANGEOVER_B
            )
            num_batches_after_choB = 0
            for machine_event in machine_events:
                if (
                    machine_event.state == MachineState.BUSY
                    and machine_event.task.product.family_id == task.product.family_id
                    and machine_event.task.sequence == task.sequence
                ):
                    num_batches_after_choB += 1

            if (
                changeoverB_start - self.last_changeoverB_end_time
                > changeover_by_days * 24
            ):
                changeover_type = "type_B"
                msg = f"Campaign exceeded by {(changeoverB_start - self.last_changeoverB_end_time)/24:.2f} days"

            elif num_batches_after_choB == changeover_by_batch:
                changeover_type = "type_B"
                msg = f"Campaign length maxed out for {task.product.material_id}"

        return changeover_type, msg

    def apply_room_constraints(
        self,
        machine: Machine,
        task: Task,
        start_time: float,
        execute: bool = True,
    ) -> None:
        """
        Apply room constraints
        """
        # Determine what changeover
        changeover_type, msg = self.determine_changeover_type(
            machine=machine,
            task=task,
            start_time=start_time,
        )

        # Find optimal start time
        if changeover_type is None:
            changeover_end = start_time - machine.get_setuptime(
                product=task.product,
                batch_size=task.batch_size,
                alt_recipe=task.alt_recipe,
                sequence=task.sequence,
            )
            changeover_triggers = []

        else:
            changeover_start = min(
                max(
                    self.get_next_availability(),
                    start_time - self.get_changeover_time(changeover_type),
                ),
                max(
                    self.get_next_availability(),
                    start_time
                    - self.get_changeover_time(changeover_type)
                    - machine.get_setuptime(
                        product=task.product,
                        batch_size=task.batch_size,
                        alt_recipe=task.alt_recipe,
                        sequence=task.sequence,
                    ),
                ),
            )

            # Apply changeover
            changeover_end, changeover_triggers = self.do_changeover(
                changeover_type=changeover_type,
                changeover_start=changeover_start,
                msg=msg,
                execute=execute,
            )

        return changeover_end, changeover_triggers
