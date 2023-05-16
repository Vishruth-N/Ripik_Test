"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from typing import Tuple, Dict, Any

import pandas as pd

from optimus.elements.product import Material
from optimus.machines.machine import Machine
from optimus.activity import BaseActivityManager, get_activity_manager
from optimus.machines.machine import initialize_machines
from optimus.elements.product import initialize_products
from optimus.rooms import Room, initialize_rooms
from optimus.utils.structs import *


def initialize_state(
    config: Dict[str, Any],
    df_products_desc: pd.DataFrame,
    df_bom: pd.DataFrame,
    df_recipe: pd.DataFrame,
    df_plant_map: pd.DataFrame,
    df_machine_changeover: pd.DataFrame,
    df_room_changeover: pd.DataFrame,
    df_machine_availability: pd.DataFrame,
) -> Tuple[
    Dict[str, Material],
    Dict[str, Machine],
    Dict[str, Room],
    BaseActivityManager,
]:
    # Get activity manager
    activity_manager = get_activity_manager(client_id=config["client_id"])

    # Intiialize machines
    machines = initialize_machines(
        plant_map=df_plant_map,
        recipes=df_recipe,
        machine_availability=df_machine_availability,
        changeover=df_machine_changeover,
        execution_start=config["execution_start"],
        is_sunday_off=config["is_sunday_off"],
    )

    # Initialize products
    products = initialize_products(
        df_products_desc=df_products_desc,
        df_bom=df_bom,
        df_recipe=df_recipe,
        df_plant_map=df_plant_map,
        machines=machines,
        activity_manager=activity_manager,
    )

    # Initialize rooms
    rooms = initialize_rooms(
        client_id=config["client_id"],
        machines=machines,
        changeover=df_room_changeover,
    )

    # Set activity mananger
    activity_manager.set_rooms(rooms=rooms)

    return (products, machines, rooms, activity_manager)
