"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import time
import numpy as np
from typing import Dict, Any

from optimus.activity import BaseActivityManager
from optimus.machines.machine import Machine
from optimus.rooms.room import Room
from optimus.elements.product import Material
from optimus.elements.inventory import Inventory
from optimus.metrics.objective import Objective
from optimus.arrangement.sequencing import Sequencer
from optimus.arrangement.tools import TaskGraph
from optimus.arrangement.process_handler import ProcessHandler

import logging

logger = logging.getLogger(__name__)
progress_logger = logging.getLogger("progressLogger")


def schedule_single(
    feasible_batches: Dict[str, float],
    inventory: Inventory,
    products: Dict[str, Material],
    rooms: Dict[str, Room],
    config: Dict[str, Any],
    activity_manager: BaseActivityManager,
    initial_start_time: float = 0,
):
    """
    Params
    -------------------------
    """
    # Copy inventory
    inventory_copy = inventory.copy()

    # Make task graph
    task_graph = TaskGraph(
        feasible_batches=feasible_batches,
        inventory=inventory_copy,
        initial_start_time=initial_start_time,
    )

    # Initialize process handler
    process_handler = ProcessHandler(
        task_graph=task_graph,
        config=config,
        activity_manager=activity_manager,
    )

    # Make sequencer
    sequencer = Sequencer(
        task_graph=task_graph,
        process_handler=process_handler,
        inventory=inventory_copy,
        products=products,
        config=config,
        activity_manager=activity_manager,
    )

    num_tasks_completed = 0
    while not sequencer.empty():
        # Obtain task from the sequence
        task = sequencer.obtain_task()

        # Assign a resource to this task
        (assigned_machine, process_info, feedback,) = process_handler.process(
            task=task,
        )

        # Finish the task
        sequencer.complete_task(
            task=task,
            assigned_machine=assigned_machine,
            process_info=process_info,
            feedback=feedback,
        )

        num_tasks_completed += 1

    logger.debug(f"Number of tasks completed: {num_tasks_completed}")
    logger.debug(f"Total number of tasks final: {len(sequencer.tasks)}")
    sequencer.log_remaining()


def optimise_single(
    feasible_batches: Dict[str, float],
    inventory: Inventory,
    products: Dict[str, Material],
    machines: Dict[str, Machine],
    rooms: Dict[str, Room],
    config: Dict[str, Any],
    activity_manager: BaseActivityManager,
    objective: Objective,
    num_iterations: int = 100,
    initial_start_time: float = 0,
) -> float:
    # Initialize best vars
    best_value = -np.inf
    best_schedule_states = {}
    for machine_id, machine in machines.items():
        best_schedule_states[machine_id] = machine.dump_schedule()

    # Initialise debug vars
    time_taken_per_schedule = []

    for iter_no in range(num_iterations):
        progress_logger.info(f"Running iteration no {iter_no+1}...")

        # Commit at the start of the iteration
        for machine in machines.values():
            machine.commit()

        # Schedule it
        start_time = time.time()
        schedule_single(
            feasible_batches=feasible_batches,
            inventory=inventory,
            products=products,
            rooms=rooms,
            config=config,
            activity_manager=activity_manager,
            initial_start_time=initial_start_time,
        )
        time_taken_per_schedule.append(time.time() - start_time)

        # Find metrics
        objective_value, component_values = objective(
            machines=machines,
            feasible_batches=feasible_batches,
            config=config,
        )
        logger.debug(
            f"Total obj value:{objective_value}, Components:{component_values}"
        )

        if objective_value > best_value:
            best_value = objective_value
            for machine_id, machine in machines.items():
                best_schedule_states[machine_id] = machine.dump_schedule()

        # Rollback for the next iteration
        for machine in machines.values():
            machine.rollback()

    logger.debug(
        f"Average time taken per schedule: {np.mean(time_taken_per_schedule):.2f} seconds"
    )

    # Load best machine states
    for machine_id, machine in machines.items():
        machine.load_schedule(best_schedule_states[machine_id])

    return best_value
