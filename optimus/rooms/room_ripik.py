"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
import pandas as pd

from optimus.machines.event import MachineState
from optimus.utils.constants import MachineType, MaterialType
from optimus.rooms.room import Room
from optimus.arrangement.task import Task
from optimus.utils.structs import *

# Forward declarations
class Machine:
    pass


class RipikRoom(Room):
    def __init__(
        self,
        room_id: str,
        block_id: str,
        changeover: pd.DataFrame,
    ) -> None:
        super().__init__(room_id, block_id, changeover)

    def get_max_campaign_length(self) -> Tuple[float, float]:
        return 5, 6

    def determine_changeover_type(
        self,
        machine: Machine,
        task: Task,
        start_time: float,
    ):
        changeover_type, msg = None, ""
        if machine.machine_type == MachineType.infinite.value:
            return changeover_type, msg

        # Get the latest process in the room
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
        if latest_event is None:
            return changeover_type, msg

        changeover_by_batch, changeover_by_days = self.get_max_campaign_length()

        if latest_event.state == MachineState.BUSY:
            prev_batch_id = latest_event.task.batch_id
            prev_material_id = latest_event.task.product.material_id

            if prev_material_id != task.product.material_id:
                changeover_type = "type_B"
                msg = f"Material changed from {prev_material_id} to {task.product.material_id}"

            else:
                if prev_batch_id != task.batch_id:
                    changeover_type = "type_A"
                    msg = f"Different batch but same family"

                    if task.product.get_material_type() != MaterialType.fg.value:
                        changeoverB_start = max(
                            self.get_next_availability(),
                            start_time - self.get_changeover_time("type_B"),
                        )
                        if (
                            changeoverB_start - self.last_changeoverB_end_time
                            > changeover_by_days * 24
                        ):
                            changeover_type = "type_B"
                            msg = f"Campaign exceeded by {(changeoverB_start - self.last_changeoverB_end_time)/24:.2f} days"

                        else:
                            # See machine's campaign length
                            last_events = machine.get_last_events(
                                k=5,
                                filter_states=[
                                    MachineState.BUSY,
                                    MachineState.ROOM_CHANGEOVER_B,
                                ],
                            )
                            if len(last_events) == changeover_by_batch:
                                campaign_length_crossed = True
                                for event in last_events:
                                    if event.state == MachineState.ROOM_CHANGEOVER_B:
                                        campaign_length_crossed = False
                                        break

                                if campaign_length_crossed:
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
