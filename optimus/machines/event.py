"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""


from enum import Enum, auto
from typing import Dict, Any, TypeVar, Optional

from optimus.arrangement.task import Task

SelfEvent = TypeVar("SelfEvent", bound="Event")

# Machine processing state
class MachineState(Enum):
    BUSY = auto()
    MACHINE_CHANGEOVER_A = auto()
    ROOM_CHANGEOVER_A = auto()
    ROOM_CHANGEOVER_B = auto()
    UNAVAILABLE = auto()


class Event:
    def __init__(
        self,
        state: MachineState,
        start_time: float,
        duration: float,
        task: Task = None,
        note: str = "",
        metadata: Dict[str, Any] = None,
    ) -> None:
        # Checks
        if state == MachineState.BUSY and task is None:
            raise ValueError(f"task cannot be None in {state.name} state")

        self.state = state
        self.start_time = start_time
        self.duration = duration
        self.task = task
        self.note = note
        self.metadata = metadata

    def post_machine_status(self):
        raise NotImplementedError()

    # PROPERTIES
    @property
    def end_time(self) -> float:
        return self.start_time + self.duration
