"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from typing import Dict

from optimus.machines.event import MachineState
from optimus.machines.machine import Machine


def room_capacity(machines: Dict[str, Machine]):
    """
    Insights with respect to room capacity
    """
    output = {}
    for machine in machines.values():
        for event in machine.get_schedule():
            if event.state == MachineState.BUSY:
                for machine in event.task.get_approved_machines():
                    room_id = machine.room_id
                    if room_id not in output:
                        output[room_id] = {
                            "performed": 0,
                            "possible": 0,
                            "other_rooms": [],
                        }
                    output[room_id]["possible"] += 1
                    if room_id != machine.room_id:
                        output[room_id]["other_rooms"].append(machine.room_id)
                output[machine.room_id]["performed"] += 1

    return output
