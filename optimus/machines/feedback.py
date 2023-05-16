"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from optimus.machines.event import Event


class ProcessInfo:
    def __init__(
        self,
        process_start_time: float,
        process_end_time: float,
        quantity_processed: float,
        process_finish_time: float,
    ) -> None:
        self.process_start_time = process_start_time
        self.process_end_time = process_end_time
        self.process_finish_time = process_finish_time
        self.quantity_processed = quantity_processed
        self.changeover_triggers = []

    def add_changeover_trigger(self, changeover_trigger: Event) -> None:
        self.changeover_triggers.append(changeover_trigger)
