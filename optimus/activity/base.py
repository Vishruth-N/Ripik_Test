"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from typing import List, Dict, Any

if TYPE_CHECKING:
    from optimus.arrangement.tools import ATI
    from optimus.arrangement.task import Task
    from optimus.machines.machine import Machine
    from optimus.rooms.room import Room


class BaseActivityManager:
    def __init__(self) -> None:
        self.rooms = None

    ######################### GETTERS AND SETTERS #########################
    def set_rooms(self, rooms: Dict[str, Room]):
        self.rooms = rooms

    ######################### CORE FUNCTIONS #########################
    def compare_op_order(self):
        raise NotImplementedError()

    def select_alt_recipe(self, task: Task, change_state: bool = True):
        raise NotImplementedError()

    def choose_machine(self, task: Task) -> Machine:
        raise NotImplementedError()

    def create_core_chain(self, ati: ATI, metadata: Dict[str, Any] = None):
        raise NotImplementedError()
