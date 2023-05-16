"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
import pandas as pd

from optimus.rooms.room import Room
from optimus.arrangement.task import Task
from optimus.utils.structs import *

# Forward declarations
class Machine:
    pass


class SunPharmaBaskaRoom(Room):
    def __init__(
        self,
        room_id: str,
        block_id: str,
        changeover: pd.DataFrame,
    ) -> None:
        super().__init__(room_id, block_id, changeover)

    def get_max_campaign_length(self) -> Tuple[float, float]:
        changeover_by_batch = np.inf
        changeover_by_days = np.inf
        return changeover_by_batch, changeover_by_days

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
        # Only one machine at a time
        latest_event = None
        latest_machine = None
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
                latest_machine = curr_machine

        if latest_machine is not None and (
            latest_machine.machine_id.startswith("CIPSIP")
            or machine.machine_id.startswith("CIPSIP")
        ):
            start_time = max(start_time, latest_event.end_time)

        # Assuming no changeovers, get the changeover start time i.e. equal to end time
        changeover_end = start_time - machine.get_setuptime(
            product=task.product,
            batch_size=task.batch_size,
            alt_recipe=task.alt_recipe,
            sequence=task.sequence,
        )
        changeover_triggers = []

        return changeover_end, changeover_triggers
