"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import inspect
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sortedcontainers import SortedList

from optimus.rooms.room import Room
from optimus.machines.event import Event, MachineState
from optimus.machines.feedback import ProcessInfo
from optimus.machines.utils import get_next_sunday
from optimus.utils.constants import MachineType, ResourceType
from optimus.elements.product import Material
from optimus.arrangement.task import Task
from optimus.utils.structs import *


class Machine:
    DEFAULT_AVAILABILTY = -1e10

    def __init__(
        self,
        machine_id: str,
        room_id: str,
        block_id: str,
        operation: str,
        machine_type: str,
        power: Dict[str, Dict[str, Any]],
        execution_start: datetime,
        changeover: Optional[pd.DataFrame] = None,
        machine_availability: Optional[pd.DataFrame] = None,
        is_sunday_off: bool = False,
    ) -> None:
        self.machine_id = machine_id
        self.room_id = room_id
        self.block_id = block_id
        self.operation = operation
        self.machine_type = machine_type
        self.power = power
        self.execution_start = execution_start
        self.changeover = changeover
        self.machine_availability = machine_availability
        self.is_sunday_off = is_sunday_off

        # Preprocess
        if self.machine_availability is not None:
            self.machine_availability = self.__preprocess_machine_availability(
                self.machine_availability
            )

        # Machine schedule
        self.batches_after_choB = set()
        self._schedule = SortedList(key=self.__schedule_sort_key)
        self.commit()

    ######################### MAGIC FUNCTIONS #########################
    def __preprocess_machine_availability(self, machine_availability):
        return (
            machine_availability.drop(MachineAvailabilityST.cols.machine_id, axis=1)
            .sort_values(MachineAvailabilityST.cols.start_datetime, ascending=False)
            .values.tolist()
        )

    def __schedule_sort_key(self, event):
        return event.start_time

    ######################### MAGIC FUNCTIONS #########################
    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Machine):
            return self.machine_id == __o.machine_id
        else:
            raise ValueError(f"Cannot compare Machine with {type(__o)}")

    def __hash__(self) -> int:
        return hash(self.machine_id)

    ######################### TRANSACTION FUNCTIONS #########################
    def commit(self) -> None:
        self.committed_index = len(self._schedule)

    def rollback(self) -> None:
        self._schedule = SortedList(
            self._schedule.islice(0, self.committed_index),
            key=self.__schedule_sort_key,
        )

    def dump_schedule(self) -> SortedList:
        return self._schedule

    def load_schedule(self, schedule: SortedList) -> None:
        self._schedule = schedule

    ######################### GETTERS AND SETTERS #########################
    def _get_local_power(
        self, product: Material, batch_size: float, alt_recipe: str, sequence: int
    ):
        return self.power[
            (
                product.material_id,
                batch_size,
                product.op_order_of(batch_size, alt_recipe, sequence),
            )
        ]

    def get_setuptime(
        self, product: Material, batch_size: float, alt_recipe: str, sequence: int
    ):
        curr_power = self._get_local_power(
            product=product,
            batch_size=batch_size,
            alt_recipe=alt_recipe,
            sequence=sequence,
        )
        return curr_power["setuptime_per_batch"]

    def get_runtime(
        self, product: Material, batch_size: float, alt_recipe: str, sequence: int
    ):
        curr_power = self._get_local_power(
            product=product,
            batch_size=batch_size,
            alt_recipe=alt_recipe,
            sequence=sequence,
        )
        return curr_power["runtime_per_batch"]

    def get_wait_time(
        self, product: Material, batch_size: float, alt_recipe: str, sequence: int
    ):
        curr_power = self._get_local_power(
            product=product,
            batch_size=batch_size,
            alt_recipe=alt_recipe,
            sequence=sequence,
        )
        return curr_power["min_wait_time"]

    def get_changeover_time(self, material_id: str) -> float:
        """Get machine changeover time of given transition"""
        df = self.changeover[
            (self.changeover[MachineChangeoverST.cols.material_id] == material_id)
        ]

        if len(df) == 0:
            raise ValueError(
                f"Machine changeover not defined for product {material_id} in {self.machine_id}"
            )

        return df[MachineChangeoverST.cols.type_A].iloc[0]

    ######################### SCHEDULE FUNCTIONS #########################
    def get_schedule(self):
        """Get formatted schedule for printing purposes"""
        # Convert to simple list
        formatted_schedule = []
        for event in self._schedule:
            formatted_schedule.append(event)

        return formatted_schedule

    def update_schedule(self, event: Event):
        # Update schedule list
        self._schedule.add(event)

        # Update num busy states
        if event.state == MachineState.ROOM_CHANGEOVER_B:
            self.batches_after_choB = set()
        elif event.state == MachineState.BUSY:
            self.batches_after_choB.add(event.task.batch_id)

    ######################### UTILITY FUNCTIONS #########################
    def get_room(self) -> Room:
        return Room.registry.get(self.room_id)

    def get_events_till_state(self, state: MachineState):
        """
        Get events till we find given event state
        """
        output = []
        i = len(self._schedule) - 1
        while i >= 0:
            if self._schedule[i].state == state:
                break
            output.append(self._schedule[i])
            i -= 1
        return output

    def get_last_events(self, k: int = 1, filter_states: List[MachineState] = None):
        """
        Get last k states
        """
        output = []
        i = len(self._schedule) - 1
        while i >= 0 and k > 0:
            if filter_states is None or self._schedule[i].state in filter_states:
                output.append(self._schedule[i])
                k -= 1
            i -= 1
        return output

    def get_last_event(self) -> Optional[Event]:
        if len(self._schedule) > 0:
            return self._schedule[-1]

    def get_last_working_event(self) -> Optional[Event]:
        i = len(self._schedule) - 1
        while i >= 0:
            event = self._schedule[i]
            if event.state != MachineState.UNAVAILABLE:
                return event

    def get_availability(self) -> float:
        # Get last end time
        last_event = self.get_last_event()
        if last_event is None:
            return Machine.DEFAULT_AVAILABILTY
        return self._schedule[-1].end_time

    ######################### EVENT FUNCTIONS #########################
    def __get_sensible_time(self, start_time: float):
        """
        Returns sensible time such that it does not overlap with last event

        Params
        -------------------------
        start_time: wished start time
        lower_limit: lower limit of the sensible time
        """
        # Find min start time
        last_event = self.get_last_event()
        min_start_time = (
            last_event.end_time
            if last_event is not None
            else Machine.DEFAULT_AVAILABILTY
        )

        return max(min_start_time, start_time)

    def __calibrate_time(self, start_time: float, duration: float) -> float:
        # Get sensible start and end date
        calibrated_start_time = self.__get_sensible_time(start_time=start_time)
        start_date = self.execution_start + timedelta(hours=calibrated_start_time)
        end_date = start_date + timedelta(hours=duration)

        # Maintenance downtime
        if self.machine_availability is not None:
            i = len(self.machine_availability) - 1
            while i >= 0:
                unavailable_start, unavailable_end, reason = self.machine_availability[
                    i
                ]

                # No overlap
                if end_date <= unavailable_start:
                    break

                # Shift to the end of unavailability if overlaps
                if start_date < unavailable_end and end_date > unavailable_start:
                    calibrated_start_time = (
                        unavailable_end - self.execution_start
                    ).total_seconds() / 3600
                    start_date = self.execution_start + timedelta(
                        hours=calibrated_start_time
                    )
                    end_date = start_date + timedelta(hours=duration)

                # Add the unavailable period to schedule and move to next one
                i -= 1

        # Periodic downtime
        if self.is_sunday_off:
            next_sunday_start = get_next_sunday(start_date, first_shift_hour=6)
            next_sunday_end = next_sunday_start + timedelta(days=1)

            # Shift to next available shift i.e. Monday 6AM if sunday overlaps
            if end_date > next_sunday_start:
                calibrated_start_time = (
                    next_sunday_end - self.execution_start
                ).total_seconds() / 3600

        return calibrated_start_time

    def create_event(
        self,
        start_time: float,
        duration: float,
        state: MachineState,
        task: Task = None,
        note: str = "",
        metadata: Dict[str, Any] = None,
    ) -> Event:
        # Calibrate start time
        calibrated_start_time = self.__calibrate_time(
            start_time=start_time, duration=duration
        )

        # Initialize new event object
        event = Event(
            state=state,
            start_time=calibrated_start_time,
            duration=duration,
            task=task,
            note=note,
            metadata=metadata,
        )

        return event

    ######################### MAIN FUNCTIONS #########################
    def machine_changeover_check(self, task: Task) -> bool:
        """Checks whether machine changeover condition is true"""
        if self.changeover is None:
            return False

        # Last working event state must be busy
        last_event = self.get_last_working_event()
        if last_event is None or last_event.state != MachineState.BUSY:
            return False

        # Family must match
        if last_event.task.product.family_id != task.product.family_id:
            return False

        # Batch ID must be different
        if last_event.task.batch_id == task.batch_id:
            return False

        # Changeover time must be positive
        changeover_duration = self.get_changeover_time(
            material_id=task.product.material_id
        )
        if changeover_duration <= 0:
            return False

        return True

    def __fit_algo(
        self,
        task: Task,
        start_time: float,
        execute: bool = True,
        fit_method: str = "nearest",
    ) -> ProcessInfo:
        if (
            task.product.operation_of(task.batch_size, task.alt_recipe, task.sequence)
            != self.operation
        ):
            raise ValueError("Machine have been assigned more than one operation")

        if fit_method == "nearest":
            raise NotImplementedError()

        elif fit_method == "last":
            # Calculate setup, runtime and waittime
            qnty_scale_factor = task.quantity / task.batch_size
            machine_setuptime = self.get_setuptime(
                product=task.product,
                batch_size=task.batch_size,
                alt_recipe=task.alt_recipe,
                sequence=task.sequence,
            )
            machine_runtime = (
                self.get_runtime(
                    product=task.product,
                    batch_size=task.batch_size,
                    alt_recipe=task.alt_recipe,
                    sequence=task.sequence,
                )
                * qnty_scale_factor
            )
            machine_wait_time = (
                self.get_wait_time(
                    product=task.product,
                    batch_size=task.batch_size,
                    alt_recipe=task.alt_recipe,
                    sequence=task.sequence,
                )
                * qnty_scale_factor
            )

            # Calculate process duration and start time
            process_duration = machine_setuptime + machine_runtime
            process_start_time = self.__calibrate_time(
                start_time=start_time, duration=process_duration
            )

            # Create changeover event if machine changeover condition is passed
            changeover_triggers = []
            if self.machine_changeover_check(task=task):
                changeover_duration = self.get_changeover_time(
                    material_id=task.product.material_id
                )
                changeover_event = self.create_event(
                    start_time=start_time - changeover_duration,
                    duration=changeover_duration,
                    state=MachineState.MACHINE_CHANGEOVER_A,
                    note=f"Triggered for {task.product.material_id} ({task.product.material_name})",
                    metadata={"process_request_time": start_time},
                )
                changeover_triggers.append(
                    {
                        "state": MachineState.MACHINE_CHANGEOVER_A,
                        "start_time": changeover_event.start_time,
                        "end_time": changeover_event.end_time,
                    }
                )

                # Change process start time
                process_start_time = self.__calibrate_time(
                    start_time=changeover_event.end_time, duration=process_duration
                )

                # Execute machine changeover
                if execute:
                    self.update_schedule(changeover_event)

            # Quantity processed
            quantity_processed = task.quantity

            # Process
            process_end_time = process_start_time + process_duration
            process_finish_time = (
                process_start_time + machine_setuptime + machine_wait_time
            )

            # Execute processing
            if execute:
                busy_event = self.create_event(
                    start_time=process_start_time,
                    duration=process_duration,
                    state=MachineState.BUSY,
                    task=task,
                    note=f"Processing {task.product.material_id} ({task.product.material_name})",
                    metadata={"process_request_time": start_time},
                )
                self.update_schedule(busy_event)

            process_info = ProcessInfo(
                process_start_time=process_start_time,
                process_end_time=process_end_time,
                process_finish_time=process_finish_time,
                quantity_processed=quantity_processed,
            )
            for changeover_trigger in changeover_triggers:
                process_info.add_changeover_trigger(changeover_trigger)

            return process_info

        else:
            raise ValueError(f"Invalid fitting method provided: {fit_method}")

    def fake_process(
        self,
        task: Task,
        start_time: float,
        fit_method: str = "last",
    ) -> ProcessInfo:
        return self.__fit_algo(
            task=task,
            start_time=start_time,
            execute=False,
            fit_method=fit_method,
        )

    def process(
        self,
        task: Task,
        start_time: float,
        fit_method: str = "last",
    ) -> ProcessInfo:
        return self.__fit_algo(
            task=task,
            start_time=start_time,
            execute=True,
            fit_method=fit_method,
        )

    def remove_batches(self, batches):
        """Remove slots of given batches"""
        i = 0
        while i < len(self._schedule):
            process = self._schedule[i]
            if process["details"] and process["details"]["batch_id"] in batches:
                self._schedule.pop(i)
            else:
                i += 1

    def trim_schedule_till(self, end_time: float):
        """Returns set of removed batches that crosses end time"""
        stopping_idx = len(self._schedule)
        for i, process in enumerate(self._schedule):
            if process["end_time"] > end_time:
                stopping_idx = i
                break

        batches_removed = set()
        for i in range(stopping_idx, len(self._schedule)):
            process = self._schedule[i]
            if process["details"]:
                batches_removed.add(process["details"]["batch_id"])

        self._schedule = SortedList(
            self._schedule.islice(0, stopping_idx), key=lambda x: x["start_time"]
        )
        return batches_removed


class InfinitePowerMachine(Machine):
    def __init__(
        self,
        machine_id: str,
        room_id: str,
        block_id: str,
        operation: str,
        machine_type: str,
        power: Dict[str, Dict[str, Any]],
        execution_start: datetime,
        changeover: Optional[pd.DataFrame] = None,
        machine_availability: Optional[pd.DataFrame] = None,
        is_sunday_off: bool = False,
    ) -> None:
        super().__init__(
            machine_id,
            room_id,
            block_id,
            operation,
            machine_type,
            power,
            execution_start,
            changeover,
            machine_availability,
            is_sunday_off,
        )

    def get_availability(self) -> float:
        return 0

    def process(
        self,
        task: Task,
        start_time: float,
        *args,
        **kwargs,
    ) -> ProcessInfo:
        # Calculate setup, runtime and waittime
        machine_setuptime = self.get_setuptime(
            product=task.product,
            batch_size=task.batch_size,
            alt_recipe=task.alt_recipe,
            sequence=task.sequence,
        )
        machine_runtime = self.get_runtime(
            product=task.product,
            batch_size=task.batch_size,
            alt_recipe=task.alt_recipe,
            sequence=task.sequence,
        )
        machine_wait_time = self.get_wait_time(
            product=task.product,
            batch_size=task.batch_size,
            alt_recipe=task.alt_recipe,
            sequence=task.sequence,
        )

        # Quantity processed
        quantity_processed = task.quantity

        # Process
        process_duration = machine_setuptime + machine_runtime
        process_start_time = start_time
        process_end_time = process_start_time + process_duration
        process_finish_time = process_start_time + machine_setuptime + machine_wait_time

        # Execute processing
        busy_event = Event(
            state=MachineState.BUSY,
            start_time=process_start_time,
            duration=process_duration,
            task=task,
            note=f"Parallel processing...",
            metadata={"process_request_time": start_time},
        )
        self.update_schedule(busy_event)

        process_info = ProcessInfo(
            process_start_time=process_start_time,
            process_end_time=process_end_time,
            process_finish_time=process_finish_time,
            quantity_processed=quantity_processed,
        )

        return process_info


def initialize_machines(
    plant_map: pd.DataFrame,
    recipes: pd.DataFrame,
    changeover: pd.DataFrame,
    machine_availability: pd.DataFrame,
    execution_start: datetime,
    is_sunday_off: bool,
) -> Dict[str, Machine]:
    """
    Given all the machine data, initialize all machines

    Parameters
    -------------------------
    plant_map: Plant map, Room to machine to operation mapping
    recipes: Base quantity and machine hours reqd each product per resource
    changeover: Changeover time (type A) per transition per machine
    machine_availability: Non availability schedule per machine
    """
    # Create a machine object for each
    machines = {}

    # Get all machine ids
    for row in plant_map.itertuples():
        row = row._asdict()
        machine_id = row[PlantMapST.cols.machine_id]
        machine_type = row[PlantMapST.cols.machine_type]
        room_id = row[PlantMapST.cols.room_id]
        block_id = row[PlantMapST.cols.block_id]
        operation = row[PlantMapST.cols.operation]

        # Build power
        power = {}
        for row in recipes[
            (recipes[RecipeST.cols.operation] == operation)
            & (
                (
                    (recipes[RecipeST.cols.resource_id] == room_id)
                    & (recipes[RecipeST.cols.resource_type] == ResourceType.room.value)
                )
                | (
                    (recipes[RecipeST.cols.resource_id] == machine_id)
                    & (
                        recipes[RecipeST.cols.resource_type]
                        == ResourceType.machine.value
                    )
                )
            )
        ].itertuples():
            row = row._asdict()
            key = (
                row[RecipeST.cols.material_id],
                row[RecipeST.cols.batch_size],
                row[RecipeST.cols.op_order],
            )
            power[key] = {
                "min_lot_size": row[RecipeST.cols.min_lot_size],
                "max_lot_size": row[RecipeST.cols.max_lot_size],
                "setuptime_per_batch": row[RecipeST.cols.setuptime_per_batch],
                "min_wait_time": row[RecipeST.cols.min_wait_time],
                "runtime_per_batch": row[RecipeST.cols.runtime_per_batch],
            }

        # Build changeover data
        curr_changeover = None
        if changeover is not None:
            curr_changeover = changeover[
                changeover[MachineChangeoverST.cols.machine_id] == machine_id
            ]

        # Build machine availability data
        curr_machine_availability = None
        if machine_availability is not None:
            curr_machine_availability = machine_availability[
                machine_availability[MachineAvailabilityST.cols.machine_id]
                == machine_id
            ]

        # Create machine
        if machine_type == MachineType.infinite.value:
            machine = InfinitePowerMachine(
                machine_id=machine_id,
                room_id=room_id,
                block_id=block_id,
                operation=operation,
                machine_type=machine_type,
                power=power,
                execution_start=execution_start,
                changeover=curr_changeover,
                machine_availability=curr_machine_availability,
                is_sunday_off=is_sunday_off,
            )
        else:
            machine = Machine(
                machine_id=machine_id,
                room_id=room_id,
                block_id=block_id,
                operation=operation,
                machine_type=machine_type,
                power=power,
                execution_start=execution_start,
                changeover=curr_changeover,
                machine_availability=curr_machine_availability,
                is_sunday_off=is_sunday_off,
            )

        # Not available
        # for row in machine_non_availability[
        #     machine_non_availability["machine_id"] == machine_id
        # ].itertuples():
        #     machine.update_schedule(
        #         MachineState.UNAVAILABLE,
        #         row.start_time,
        #         row.end_time,
        #         "prescheduled blockage",
        #     )

        # Set
        machines[machine_id] = machine

    return machines
