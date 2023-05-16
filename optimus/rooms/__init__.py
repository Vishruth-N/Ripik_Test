"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
from typing import List, Dict

import pandas as pd

from optimus.utils.structs import RoomChangeoverST
from optimus.rooms.room import Room
from optimus.rooms.room_ripik import RipikRoom
from optimus.rooms.room_sunpharma_dewas import SunPharmaDewasRoom
from optimus.rooms.room_sunpharma_paonta_sahib import SunPharmaPaontaSahibRoom
from optimus.rooms.room_sunpharma_baska import SunPharmaBaskaRoom

# Forward declarations
class Machine:
    pass


def initialize_rooms(
    client_id: str,
    machines: List[Machine],
    changeover: pd.DataFrame,
) -> Dict[str, Room]:
    """
    Given all the machine objects, initialize rooms

    Parameters
    -------------------------
    machines: List of all the machine objects
    changeover: Changeover time B per transition per room
    """
    rooms = {}
    for machine in machines.values():
        if machine.room_id not in rooms:
            # Build changeover data
            curr_changeover = None
            if changeover is not None:
                curr_changeover = changeover[
                    changeover[RoomChangeoverST.cols.room_id] == machine.room_id
                ]

            # Get room params
            room_params = dict(
                room_id=machine.room_id,
                block_id=machine.block_id,
                changeover=curr_changeover,
            )

            # Create room based on client id
            if client_id == "ripik":
                room = RipikRoom(**room_params)

            elif client_id == "sunpharma_paonta_sahib":
                room = SunPharmaPaontaSahibRoom(**room_params)

            elif client_id == "sunpharma_dewas":
                room = SunPharmaDewasRoom(**room_params)

            elif client_id == "sunpharma_baska":
                room = SunPharmaBaskaRoom(**room_params)

            else:
                raise ValueError(f"Invalid client id: {client_id}")

            # Add room
            rooms[machine.room_id] = room

        # Add machine to the room
        rooms[machine.room_id].add_machine(machine)

    return rooms
