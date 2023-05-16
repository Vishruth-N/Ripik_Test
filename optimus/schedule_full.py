"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from datetime import timedelta
from typing import Any, Dict, Optional, List, Tuple, Union

import pandas as pd

from optimus.utils.state import initialize_state
from optimus.preschedule.feasible import preschedule_demand
from optimus.schedule_single import optimise_single
from optimus.machines.machine import Machine
from optimus.elements.product import Material
from optimus.elements.inventory import Inventory
from optimus.utils.constants import PrescheduleMode
from optimus.metrics.objective import Objective

import logging

logger = logging.getLogger(__name__)
progress_logger = logging.getLogger("progressLogger")


def schedule_full(
    data: Dict[str, Optional[pd.DataFrame]],
    config: Dict[str, Any],
    inner_optimisation_params: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Dict[str, Material],
    Dict[str, Machine],
    Dict[str, Union[str, float, List[str]]],
    pd.DataFrame,
]:
    # Handle input params
    if inner_optimisation_params is None:
        inner_optimisation_params = {"num_iterations": 1}

    # Initialize inventory
    progress_logger.info("Initializing inventory...")
    inventory = Inventory(
        df_inventory=data["df_inventory"],
        df_phantom_items=data["df_phantom_items"],
        df_procurement_plan=data["df_procurement_plan"],
    )

    # Initialize state
    progress_logger.info("Initializing state...")
    products, machines, rooms, activity_manager = initialize_state(
        config=config,
        df_products_desc=data["df_products_desc"],
        df_bom=data["df_bom"],
        df_recipe=data["df_recipe"],
        df_plant_map=data["df_plant_map"],
        df_machine_changeover=data["df_machine_changeover"],
        df_room_changeover=data["df_room_changeover"],
        df_machine_availability=data["df_machine_availability"],
    )

    # Add procured quantity before execution to inventory
    procured_quantity_before_execution = inventory.get_procured_quantity(
        end_date=config["execution_start"] + timedelta(days=10)
    )
    for material_id, quantity in procured_quantity_before_execution.items():
        inventory.add(material_id, quantity)

    # Find future inventory if preschedule mode is feedbackward
    if config["preschedule_mode"] == PrescheduleMode.feedforward.value:
        future_inventory = None

    elif config["preschedule_mode"] == PrescheduleMode.feedbackward.value:
        future_inventory = inventory.copy()
        procured_quantity_till_end = future_inventory.get_procured_quantity(
            start_date=config["execution_start"],
            end_date=config["execution_end"],
        )
        for material_id, quantity in procured_quantity_till_end.items():
            future_inventory.add(material_id, quantity)

    else:
        raise ValueError(f"Invalid preschedule mode: {config['preschedule_mode']}")

    # Define inner objective function
    inner_objective = Objective(
        objective_coeffs=config["objective_coeffs"],
        df_crossblock_penalties=data["df_crossblock_penalties"],
    )

    # Preschedule demand
    progress_logger.info("Prescheduling...")
    feasible_batches, normalised_demand, dropped_reasons = preschedule_demand(
        df_forecasted_demand=data["df_forecasted_demand"],
        df_initial_state=data["df_initial_state"],
        products=products,
        config=config,
        current_inventory=inventory,
        future_inventory=future_inventory,
    )

    # Make valid schedule
    progress_logger.info("Starting scheduling...")
    inner_objective_value = optimise_single(
        feasible_batches=feasible_batches,
        inventory=inventory,
        products=products,
        machines=machines,
        rooms=rooms,
        initial_start_time=0,
        config=config,
        activity_manager=activity_manager,
        objective=inner_objective,
        **inner_optimisation_params,
    )

    return (
        products,
        machines,
        normalised_demand,
        dropped_reasons,
        feasible_batches,
    )


def optimise_full(
    inner_optimisation_params: Optional[Dict[str, Any]] = None,
    outer_optimisation_params: Optional[Dict[str, Any]] = None,
    *args,
    **kwargs,
):
    raise NotImplementedError()
