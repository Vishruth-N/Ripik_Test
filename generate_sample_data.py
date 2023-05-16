"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import os
import string
import random
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from optimus.utils.structs import *
from optimus.utils.constants import (
    BatchingMode,
    ClubbingMode,
    MachineType,
    MaterialType,
    ResourceType,
    CAMode,
    ProductionStrategy,
)

# coremanf => core manufacturing


def list_ids(n: int, prefix: str, start_id: int = 0) -> List[str]:
    """Returns n number of unique ids each prefixed by prefix parameter"""
    assert n > 0
    len_digit_code = max(len(str(start_id + n - 1)), 3)
    all_ids = []

    for i in range(start_id, start_id + n):
        prefixed_id = prefix + str(i).zfill(len_digit_code)
        all_ids.append(prefixed_id)

    return all_ids


def generate_coremanf_demand(
    fgs: List[str],
    max_q: float = 1000.0,
    least_p: int = 1,
    highest_p: int = 6,
    min_dd: float = 240.0,
    max_dd: float = 600.0,
    dd_prob: float = 0.25,
) -> pd.DataFrame:
    """
    Params
    -------------------------
    fgs: List of all fgs
    max_q: Maximum quantity of each product
    least_p: Least priority value
    highest_p: Highest priority value
    min_dd: Minimum due date (in hrs)
    max_dd: Maximum due date (in hrs)
    dd_prob: Due date probability
    """
    assert 0 < least_p <= highest_p and max_q >= 0
    assert 0 <= dd_prob <= 1
    n = len(fgs)

    due_dates = []
    for _ in range(n):
        if np.random.random() < dd_prob:
            due_dates.append(np.random.uniform(min_dd, max_dd))
        else:
            due_dates.append(np.nan)

    coremanf_demand = pd.DataFrame(
        {
            DemandST.cols.material_id: fgs,
            DemandST.cols.m0_demand: np.random.rand(n) * max_q,
            DemandST.cols.m0_commit: np.random.rand(n) * max_q,
            DemandST.cols.m1_crit: np.random.rand(n) * max_q,
            DemandST.cols.m1_std: np.random.rand(n) * max_q,
            DemandST.cols.m2_crit: np.random.rand(n) * max_q,
            DemandST.cols.m2_std: np.random.rand(n) * max_q,
            DemandST.cols.m3_crit: np.random.rand(n) * max_q,
            DemandST.cols.m3_std: np.random.rand(n) * max_q,
            DemandST.cols.priority: np.random.randint(
                least_p, highest_p + 1, size=n, dtype=np.int32
            ),
            DemandST.cols.due_date: due_dates,
            DemandST.cols.production_strategy: ProductionStrategy.MTO.value,
            DemandST.cols.ca_mode: CAMode.CAD.value,
        }
    )

    coremanf_demand = coremanf_demand.round(
        {
            col: 1
            for col in DemandST.cols._fields
            if col not in [DemandST.cols.material_id, DemandST.cols.priority]
        }
    )
    return coremanf_demand


def generate_inventory(
    fgs: List[str],
    sfgs: List[str],
    raw_materials: List[str],
    packing_materials: List[str],
    common_blends: List[str],
    max_q: float = 1000.0,
) -> pd.DataFrame:
    """
    Params
    -------------------------
    fgs: List of all fgs
    sfgs: List of all sfgs
    raw_materials: List of all raw materials
    packing_materials: List of all packing materials
    common_blends: List of all common blends
    max_q: Maximum quantity of each product
    """
    assert max_q >= 0
    n1 = len(fgs)
    n2 = len(sfgs)
    n3 = len(raw_materials)
    n4 = len(packing_materials)
    n5 = len(common_blends)

    inventory = pd.DataFrame(
        {
            InventoryST.cols.material_id: fgs
            + sfgs
            + raw_materials
            + packing_materials
            + common_blends,
            InventoryST.cols.material_type: [MaterialType.fg.value] * n1
            + [MaterialType.sfg.value] * n2
            + [MaterialType.rm.value] * n3
            + [MaterialType.pm.value] * n4
            + [MaterialType.cb.value] * n5,
            InventoryST.cols.quantity: max_q
            * np.concatenate(
                [
                    np.random.rand(n1),
                    np.random.rand(n2),
                    np.random.rand(n3),
                    np.random.rand(n4),
                    np.random.rand(n5),
                ]
            ),
        }
    )

    inventory = inventory.round({col: 1 for col in [InventoryST.cols.quantity]})
    return inventory


def generate_plant_map(
    machines: List[str], blocks: List[str], rooms: List[str], operations: List[str]
) -> pd.DataFrame:
    """
    Parameters
    -------------------------
    machines: List of all the machines
    blocks: List of all the blocks
    rooms: List of all the rooms
    operations: List of all the operations
    """
    n = len(machines)
    assert n >= len(operations), "Atleast one machine per operation is required"

    machine_types = {op: random.choice(list(MachineType)).value for op in operations}
    operations_arr = np.concatenate(
        [
            operations,
            np.random.choice(operations, size=n - len(operations), replace=True),
        ]
    )
    np.random.shuffle(operations_arr)
    blocks = {room: np.random.choice(blocks) for room in rooms}
    rooms_arr = np.random.choice(rooms, size=len(machines), replace=True)

    plant_map = {
        PlantMapST.cols.machine_id: machines,
        PlantMapST.cols.machine_type: [machine_types[op] for op in operations_arr],
        PlantMapST.cols.block_id: [blocks[room] for room in rooms_arr],
        PlantMapST.cols.room_id: rooms_arr,
        PlantMapST.cols.operation: operations_arr,
    }

    return pd.DataFrame(plant_map)


def generate_products_desc(
    fgs: List[str],
    sfgs: List[str],
    units: List[str],
    common_blends: List[str],
    max_batches: int = 2,
    min_batch_qnty: float = 500.0,
    max_batch_qnty: float = 1000.0,
    min_pack_size: int = 2,
    max_pack_size: int = 30000,
) -> pd.DataFrame:
    """
    Parameters
    -------------------------
    fgs: List of all the fgs
    sfgs: List of all the sfgs
    common_blends: List of all the common blends
    max_batches: Maximum number of batches per product
    min_batch_qnty: Minimum batch quantity
    max_batch_qnty: Maximum batch quantity
    """
    assert 0 < min_batch_qnty <= max_batch_qnty
    n1 = len(sfgs)
    n2 = len(common_blends)
    n3 = len(fgs)

    fg_unit_to_count_factor = {
        unit: np.random.randint(min_pack_size, max_pack_size + 1) for unit in units
    }

    products_desc = defaultdict(list)
    for i in range(n1 + n2 + n3):
        clubbing_mode = ClubbingMode.clubbing.value
        batching_mode = BatchingMode.tint.value

        if i < n1:
            material = sfgs[i]
            material_type = MaterialType.sfg.value
            unit = units[0]
        elif i < n1 + n2:
            material = common_blends[i - n1]
            material_type = MaterialType.cb.value
            unit = units[1]
        else:
            material = fgs[i - n1 - n2]
            material_type = MaterialType.fg.value
            unit = random.choice(units[2:])
            batching_mode = BatchingMode.treal.value

        num_batches = np.random.randint(1, max_batches + 1)

        products_desc[ProductDescST.cols.material_id].extend([material] * num_batches)
        products_desc[ProductDescST.cols.family_id].extend([material] * num_batches)
        products_desc[ProductDescST.cols.material_type].extend(
            [material_type] * num_batches
        )
        products_desc[ProductDescST.cols.material_name].extend(
            ["x" + material + "x"] * num_batches
        )
        products_desc[ProductDescST.cols.batch_size].extend(
            np.random.uniform(min_batch_qnty, max_batch_qnty, size=num_batches)
        )
        products_desc[ProductDescST.cols.material_unit].extend([unit] * num_batches)
        products_desc[ProductDescST.cols.count_factor].extend(
            [fg_unit_to_count_factor[unit]] * num_batches
        )
        products_desc[ProductDescST.cols.clubbing_mode].extend(
            [clubbing_mode] * num_batches
        )
        products_desc[ProductDescST.cols.batching_mode].extend(
            [batching_mode] * num_batches
        )

    products_desc = pd.DataFrame(products_desc).round(
        {col: 1 for col in [ProductDescST.cols.batch_size]}
    )
    return products_desc


def generate_bom(
    products_desc: pd.DataFrame,
    raw_materials: List[str],
    packing_materials: List[str],
    common_blends: List[str],
    sfgs: List[str],
    min_alt_boms: int = 1,
    max_alt_boms: int = 2,
    min_items: int = 3,
    max_items: int = 7,
) -> pd.DataFrame:
    """
    Parameters
    -------------------------
    products_desc: Description of all products
    raw_materials: List of all raw materials
    packing_materials: List of all packing materials
    min_items: Minimum number of raw materials required to make a product
    max_items: Maximum number of raw materials required to make a product
    """
    assert 0 < min_items <= max_items <= len(raw_materials)

    bom = defaultdict(list)
    for row in products_desc.itertuples():
        row = row._asdict()

        material_id = row[ProductDescST.cols.material_id]
        material_type = row[ProductDescST.cols.material_type]
        material_quantity = row[ProductDescST.cols.batch_size]

        alt_boms = np.random.randint(min_alt_boms, max_alt_boms + 1)
        for alt_bom in range(alt_boms):
            num_items = np.random.randint(min_items, max_items + 1)
            if material_type == MaterialType.cb.value:
                component_ids = random.sample(raw_materials, k=num_items)
                component_types = [MaterialType.rm.value] * num_items
                indirects = [0] * num_items
            elif material_type == MaterialType.sfg.value:
                component_ids = random.sample(
                    raw_materials, k=num_items - 1
                ) + random.sample(common_blends, k=1)
                component_types = [MaterialType.rm.value] * (num_items - 1) + [
                    MaterialType.cb.value
                ]
                indirects = [0] * (num_items - 1) + [6]
            elif material_type == MaterialType.fg.value:
                component_ids = random.sample(
                    packing_materials, k=num_items - 1
                ) + random.sample(sfgs, k=1)
                component_types = [MaterialType.pm.value] * (num_items - 1) + [
                    MaterialType.sfg.value
                ]
                indirects = [0] * (num_items - 1) + [6]
            else:
                raise ValueError("Invalid material type for BOM")

            component_quantities = (
                np.random.random(size=(num_items,))
                * (material_quantity / num_items)
                * 2
            )

            bom[BOMST.cols.material_id].extend([material_id] * num_items)
            bom[BOMST.cols.material_type].extend([material_type] * num_items)
            bom[BOMST.cols.material_quantity].extend([material_quantity] * num_items)
            bom[BOMST.cols.alt_bom].extend([alt_bom] * num_items)
            bom[BOMST.cols.component_group].extend(component_ids)
            bom[BOMST.cols.component_id].extend(component_ids)
            bom[BOMST.cols.component_type].extend(component_types)
            bom[BOMST.cols.component_quantity].extend(component_quantities)
            bom[BOMST.cols.indirect].extend(indirects)

    bom = pd.DataFrame(bom).round(
        {
            col: 1
            for col in [BOMST.cols.material_quantity, BOMST.cols.component_quantity]
        }
    )
    return bom


def generate_recipes(
    products_desc: pd.DataFrame,
    plant_map: pd.DataFrame,
    min_alt_recipes: int = 1,
    max_alt_recipes: int = 2,
    min_processes: int = 2,
    max_processes: int = 4,
    min_lot_size: float = 1000,
    max_lot_size: float = 100000,
    min_setuptime_hrs: float = 2.0,
    max_setuptime_hrs: float = 15.0,
    min_runtime_hrs: float = 6.0,
    max_runtime_hrs: float = 48.0,
    approved_resource_type: str = "machine",
) -> pd.DataFrame:
    """
    Generate recipes of each product

    Parameters
    -------------------------
    products_desc: Description of all products
    plant_map: Plant Map
    min_processes: Minimum number of processes involved per product
    max_processes: Maximum number of processes involved per product
    min_lot_size: Minimum capacity of the machine
    max_lot_size: Maximum capacity of the machine
    min_setuptime_hrs: Minimum setup operation hours per batch
    max_setuptime_hrs: Maximum setup operation hours per batch
    min_runtime_hrs: Minimum runtime operation hours per batch
    max_runtime_hrs: Maximum runtime operation hours per batch
    approved_resource_type: product operation approval to either of [machine, room]
    """
    # For the (material_id,machine_id) combination -> lot size attributes must be same
    machine_power = {}
    for material_id in products_desc[ProductDescST.cols.material_id].unique():
        if approved_resource_type == ResourceType.room.value:
            resource_col_name = PlantMapST.cols.room_id
        elif approved_resource_type == ResourceType.machine.value:
            resource_col_name = PlantMapST.cols.machine_id
        else:
            raise ValueError(f"Invalid resource type: {approved_resource_type}")

        for resource_id in plant_map[resource_col_name].unique():
            if material_id not in machine_power:
                machine_power[material_id] = {}
            min_lot_size_ = np.random.uniform(0, min_lot_size)
            max_lot_size_ = np.random.uniform(min_lot_size_, max_lot_size)
            machine_power[material_id][resource_id] = {
                "min_lot_size": min_lot_size_,
                "max_lot_size": max_lot_size_,
            }

    recipes = defaultdict(list)
    for row in products_desc.itertuples():
        row = row._asdict()

        batch_size = row[ProductDescST.cols.batch_size]
        assert batch_size >= min_lot_size

        alt_recipes = np.random.randint(min_alt_recipes, max_alt_recipes + 1)
        for alt_recipe in range(alt_recipes):
            all_operations = plant_map[PlantMapST.cols.operation].unique()
            num_processes = np.random.randint(min_processes, max_processes + 1)
            recipe = np.random.choice(all_operations, size=num_processes, replace=True)

            for i, operation in enumerate(recipe):
                available_machines = plant_map[
                    plant_map[PlantMapST.cols.operation] == operation
                ][PlantMapST.cols.machine_id]
                num_approved_machines = np.random.randint(
                    1, len(available_machines) + 1
                )

                # Get approved machines
                approved_equipments = np.random.choice(
                    available_machines, size=num_approved_machines, replace=False
                ).tolist()
                if approved_resource_type == ResourceType.room.value:
                    approved_equipments = (
                        plant_map[
                            plant_map[PlantMapST.cols.machine_id].isin(
                                approved_equipments
                            )
                        ][PlantMapST.cols.room_id]
                        .unique()
                        .tolist()
                    )
                elif approved_resource_type != "machine":
                    raise ValueError(
                        f"Invalid approved_resource_type provided: {approved_resource_type}"
                    )

                num_approved_equipments = len(approved_equipments)
                recipes[RecipeST.cols.material_id].extend(
                    [row[ProductDescST.cols.material_id]] * num_approved_equipments
                )
                recipes[RecipeST.cols.operation].extend(
                    [operation] * num_approved_equipments
                )
                recipes[RecipeST.cols.op_order].extend(
                    [i + 1] * num_approved_equipments
                )
                recipes[RecipeST.cols.alt_recipe].extend(
                    [alt_recipe] * num_approved_equipments
                )
                recipes[RecipeST.cols.resource_id].extend(approved_equipments)
                recipes[RecipeST.cols.resource_type].extend(
                    [approved_resource_type] * num_approved_equipments
                )
                recipes[RecipeST.cols.min_lot_size].extend(
                    [
                        machine_power[row[ProductDescST.cols.material_id]][resource_id][
                            "min_lot_size"
                        ]
                        for resource_id in approved_equipments
                    ]
                )
                recipes[RecipeST.cols.max_lot_size].extend(
                    [
                        machine_power[row[ProductDescST.cols.material_id]][resource_id][
                            "max_lot_size"
                        ]
                        for resource_id in approved_equipments
                    ]
                )
                recipes[RecipeST.cols.batch_size].extend(
                    [batch_size] * num_approved_equipments
                )
                recipes[RecipeST.cols.setuptime_per_batch].extend(
                    np.random.uniform(
                        min_setuptime_hrs,
                        max_setuptime_hrs,
                        size=(num_approved_equipments,),
                    )
                )
                recipes[RecipeST.cols.runtime_per_batch].extend(
                    np.random.uniform(
                        min_runtime_hrs,
                        max_runtime_hrs,
                        size=(num_approved_equipments,),
                    )
                )

    recipes = pd.DataFrame(recipes).round(
        {
            col: 1
            for col in [
                RecipeST.cols.min_lot_size,
                RecipeST.cols.max_lot_size,
                RecipeST.cols.setuptime_per_batch,
                RecipeST.cols.runtime_per_batch,
            ]
        }
    )

    recipes[RecipeST.cols.min_wait_time] = recipes[RecipeST.cols.runtime_per_batch]

    return recipes


def generate_machine_changeover(
    plant_map: pd.DataFrame,
    operation_to_products: Dict[str, str],
    time_bounds: List[Tuple[float, float]] = [(1.0, 3.0)],
) -> pd.DataFrame:
    """
    Generate changeover time per product transition per machine

    Parameters
    -------------------------
    plant_map: Dataframe plant map
    operation_to_products: operation to product mapping
    time_bounds: List of time bounds for multiple changeover values
    """
    if len(time_bounds) > 26:
        raise NotImplementedError("Modify column names")

    changeover_data = []
    for operation, products in operation_to_products.items():
        # for all the machines which have that operation
        machines = plant_map[plant_map["operation"] == operation]["machine_id"]
        for machine_id in machines:
            for from_product in products:
                for to_product in products:
                    changeover_times = []
                    for min_ct, max_ct in time_bounds:
                        changeover_times.append(np.random.uniform(min_ct, max_ct))
                    changeover_data.append(
                        [machine_id, from_product, to_product] + changeover_times
                    )

    changeover_cols = ["type_" + chr(ord("A") + i) for i in range(len(time_bounds))]
    changeover_data = pd.DataFrame(
        changeover_data,
        columns=["machine_id", "from_product_id", "to_product_id"] + changeover_cols,
    )

    changeover_data = changeover_data.round({k: 1 for k in changeover_cols})
    return changeover_data


def generate_room_changeover(
    rooms: List[str], time_bounds: List[Tuple[float, float]] = [(1.0, 3.0), (4.0, 18.0)]
) -> pd.DataFrame:
    """
    Generate changeover time per product transition per machine

    Parameters
    -------------------------
    plant_map: Dataframe plant map
    operation_to_products: operation to product mapping
    time_bounds: List of time bounds for multiple changeover values
    """
    assert len(time_bounds) == 2
    changeover_data = []
    for room_id in rooms:
        changeover_times = []
        for min_ct, max_ct in time_bounds:
            changeover_times.append(np.random.uniform(min_ct, max_ct))
        changeover_data.append([room_id] + changeover_times)

    changeover_data = pd.DataFrame(
        changeover_data,
        columns=RoomChangeoverST.cols._asdict().values(),
    )

    changeover_data = changeover_data.round(
        {k: 1 for k in [RoomChangeoverST.cols.type_A, RoomChangeoverST.cols.type_B]}
    )
    return changeover_data


def generate_campaign_constraints(
    products_desc: pd.DataFrame,
    families: List[str],
    min_campaign_length: int,
    max_campaign_length: int,
):
    assert 0 < min_campaign_length <= max_campaign_length

    campaign_constraints = defaultdict(list)
    for row in products_desc.itertuples():
        row = row._asdict()

        material_id = row[ProductDescST.cols.material_id]
        material_type = row[ProductDescST.cols.material_type]
        batch_size = row[ProductDescST.cols.batch_size]
        family_id = np.random.choice(families)
        campaign_length = np.random.randint(
            min_campaign_length, max_campaign_length + 1
        )

        campaign_constraints[CampaignConstST.cols.material_id].append(material_id)
        campaign_constraints[CampaignConstST.cols.material_type].append(material_type)
        campaign_constraints[CampaignConstST.cols.batch_size].append(batch_size)
        campaign_constraints[CampaignConstST.cols.family_id].append(family_id)
        campaign_constraints[CampaignConstST.cols.campaign_length].append(
            campaign_length
        )

    return pd.DataFrame(campaign_constraints)


def generate_crossblock_penalty(blocks: List[str]):
    """
    Generate crossblock penalty

    Params
    -------------------------
    blocks: List of all the blocks
    """
    crossblock_penalties = defaultdict(list)
    # Penalize between 0 and 1, and then scale later
    for block_a in blocks:
        for block_b in blocks:
            penalty = np.random.random() if block_a != block_b else 0.0
            crossblock_penalties[CrossBlockST.cols.from_block_id].append(block_a)
            crossblock_penalties[CrossBlockST.cols.to_block_id].append(block_b)
            crossblock_penalties[CrossBlockST.cols.penalty].append(penalty)

    crossblock_penalties = pd.DataFrame(crossblock_penalties).round(
        {k: 2 for k in [CrossBlockST.cols.penalty]}
    )
    return crossblock_penalties


def generate_machine_availability(
    machines: List[str],
    max_stops: int,
    max_time_range: float,
    min_stop_duration: float,
    max_stop_duration: float,
) -> pd.DataFrame:
    """
    Generate non availability (prior schedule) schedule of machines

    Parameters
    -------------------------
    machines: List of all machines
    max_stops: Max number of stops
    max_time_range: Interval limits cannot exceed this range
    min_stop_duration: Minimum break duration
    max_stop_duration: Maximum break duration
    """
    base_datetime = datetime(year=2023, month=5, day=1)
    machine_availability = []
    for machine_id in machines:
        num_stops = np.random.randint(0, max_stops + 1)
        for stop_no in range(1, num_stops + 1):
            min_start_time = (stop_no - 1) * (max_time_range / num_stops)
            max_end_time = stop_no * (max_time_range / num_stops)
            duration = np.random.uniform(min_stop_duration, max_stop_duration)
            assert max_end_time - duration > 0, "reduce duration or increase time range"

            start_time = np.random.uniform(min_start_time, max_end_time - duration)
            start_datetime = base_datetime + timedelta(hours=start_time)
            machine_availability.append(
                [
                    machine_id,
                    start_datetime,
                    start_datetime + timedelta(hours=duration),
                    "Maintenance...",
                ]
            )

    machine_availability = pd.DataFrame(
        machine_availability,
        columns=[
            MachineAvailabilityST.cols.machine_id,
            MachineAvailabilityST.cols.start_datetime,
            MachineAvailabilityST.cols.end_datetime,
            MachineAvailabilityST.cols.reason,
        ],
    )

    return machine_availability


def generate_procurement_plan(
    raw_materials: List[str],
    procurement_plan_end: float = 2160,
    min_orders: int = 30,
    max_orders: int = 50,
    min_order_quantity: float = 50,
    max_order_quantity: float = 100,
) -> pd.DataFrame:
    """
    Params
    -------------------------
    raw_materials: List of all raw materials
    procurement_plan_end: Procurement plan for next how many number of days
    min_orders: Minimum number of orders
    max_orders: Maximum number of orders
    min_order_quantity: Minimum quantity per order
    max_order_quantity: Maximum quantity per order
    """
    assert min_order_quantity < max_order_quantity
    assert min_orders <= max_orders

    num_orders = np.random.randint(min_orders, max_orders + 1)
    procurement_plan = pd.DataFrame(
        {
            ProcurementST.cols.material_id: random.choices(raw_materials, k=num_orders),
            ProcurementST.cols.quantity: np.random.uniform(
                min_order_quantity, max_order_quantity, size=num_orders
            ),
            ProcurementST.cols.available_at: np.random.uniform(
                0, procurement_plan_end, size=num_orders
            ),
        }
    )

    procurement_plan = procurement_plan.round(
        {k: 1 for k in [ProcurementST.cols.quantity, ProcurementST.cols.available_at]}
    )
    return procurement_plan


def generate_phantom_items(
    raw_materials: List[str], packing_materials: List[str], phantom_prob: float = 0.02
):
    phantom_items = defaultdict(list)
    for material_id in raw_materials:
        if np.random.random() < phantom_prob:
            phantom_items[PhantomST.cols.material_id].append(material_id)
            phantom_items[PhantomST.cols.material_type].append(MaterialType.rm.value)
    for material_id in packing_materials:
        if np.random.random() < phantom_prob:
            phantom_items[PhantomST.cols.material_id].append(material_id)
            phantom_items[PhantomST.cols.material_type].append(MaterialType.pm.value)

    return pd.DataFrame(phantom_items)


if __name__ == "__main__":
    # Parameters
    num_fgs = 20  # 500
    num_sfgs = 15  # 300
    num_common_blends = 10  # 250
    num_raw_materials = 50  # 100
    num_packing_materials = 50  # 100
    num_families = 3  # 40
    num_machines = 20  # 400
    num_blocks = 2  # 8
    num_rooms = 5  # 50
    num_operations = 8  # 20

    # Demand
    max_quantity_demanded = 1000
    min_priority, max_priority = 1, 3
    min_due_date, max_due_date = 240.0, 600.0
    due_date_prob = 0.25

    # Inventory
    max_inventory_capacity = 10000

    # Material
    max_batches = 2
    min_batch_qnty, max_batch_qnty = 500, 1000

    # BOM
    min_alt_boms, max_alt_boms = 1, 2
    min_bom_length, max_bom_length = 3, 7

    # Procurement plan
    procurement_plan_end = 2160
    min_orders, max_orders = 30, 50
    min_order_quantity, max_order_quantity = 50, 100

    # Recipe
    min_alt_recipes, max_alt_recipes = 1, 2
    min_operations_pr, max_operations_pr = 3, 6
    approved_resource_type = "machine"

    # Machine power
    min_setuptime_hrs, max_setuptime_hrs = 0.0, 2.0
    min_runtime_hrs, max_runtime_hrs = 6.0, 48.0
    min_lot_size, max_lot_size = 500, 100000

    # Changeover
    min_mach_cho_a, max_mach_cho_a = 1.0, 6.0
    min_room_cho_a, max_room_cho_a = 1.0, 3.0
    min_room_cho_b, max_room_cho_b = 4.0, 18.0

    # Constraints
    min_campaign_length, max_campaign_length = 4, 7

    # Machine non availability
    max_stops = 2
    max_time_range = 720.0
    min_stop_duration = 5.0
    max_stop_duration = 24.0

    # Phantom items
    phantom_prob = 0.03

    # Units
    num_units = 50
    min_pack_size = 2
    max_pack_size = 30000

    # Make list of all components
    fgs = list_ids(num_fgs, "FG")
    sfgs = list_ids(num_sfgs, "SFG")
    raw_materials = list_ids(num_raw_materials, "RM")
    packing_materials = list_ids(num_packing_materials, "PM")
    common_blends = list_ids(num_common_blends, "CB")
    families = list_ids(num_machines, "Family")
    blocks = list_ids(num_blocks, "Block")
    rooms = list_ids(num_rooms, "Room")
    operations = list_ids(num_operations, "Function")
    machines = list_ids(num_machines, "Machine")
    units = list_ids(num_units, "Un")

    # GENERATE DEMAND
    coremanf_demand = generate_coremanf_demand(
        fgs=fgs,
        max_q=max_quantity_demanded,
        least_p=min_priority,
        highest_p=max_priority,
        min_dd=min_due_date,
        max_dd=max_due_date,
        dd_prob=due_date_prob,
    )

    # GENERATE INVENTORY
    inventory = generate_inventory(
        fgs=fgs,
        sfgs=sfgs,
        raw_materials=raw_materials,
        packing_materials=packing_materials,
        common_blends=common_blends,
        max_q=max_inventory_capacity,
    )
    procurement_plan = generate_procurement_plan(
        raw_materials=raw_materials,
        procurement_plan_end=procurement_plan_end,
        min_orders=min_orders,
        max_orders=max_orders,
        min_order_quantity=min_order_quantity,
        max_order_quantity=max_order_quantity,
    )
    phantom_items = generate_phantom_items(
        raw_materials=raw_materials,
        packing_materials=packing_materials,
        phantom_prob=phantom_prob,
    )

    # GENERATE MACHINES
    plant_map = generate_plant_map(
        machines=machines, blocks=blocks, rooms=rooms, operations=operations
    )

    # GENERATE PRODUCTS
    products_desc = generate_products_desc(
        fgs=fgs,
        sfgs=sfgs,
        units=units,
        common_blends=common_blends,
        max_batches=max_batches,
        min_batch_qnty=min_batch_qnty,
        max_batch_qnty=max_batch_qnty,
        min_pack_size=min_pack_size,
        max_pack_size=max_pack_size,
    )
    bom = generate_bom(
        products_desc=products_desc,
        raw_materials=raw_materials,
        packing_materials=packing_materials,
        common_blends=common_blends,
        sfgs=sfgs,
        min_alt_boms=min_alt_boms,
        max_alt_boms=max_alt_boms,
        min_items=min_bom_length,
        max_items=max_bom_length,
    )
    recipes = generate_recipes(
        products_desc=products_desc,
        plant_map=plant_map,
        min_alt_recipes=min_alt_recipes,
        max_alt_recipes=max_alt_recipes,
        min_processes=min_operations_pr,
        max_processes=max_operations_pr,
        min_lot_size=min_lot_size,
        max_lot_size=max_lot_size,
        min_setuptime_hrs=min_setuptime_hrs,
        max_setuptime_hrs=max_setuptime_hrs,
        min_runtime_hrs=min_runtime_hrs,
        max_runtime_hrs=max_runtime_hrs,
        approved_resource_type=approved_resource_type,
    )

    # Get useful mapping for machine changeover constraint
    operation_to_products = defaultdict(set)
    for row in recipes.itertuples():
        row = row._asdict()
        operation_to_products[row[RecipeST.cols.operation]].add(
            row[RecipeST.cols.material_id]
        )

    # GENERATE CONSTRAINTS
    machine_changeover = generate_machine_changeover(
        plant_map=plant_map,
        operation_to_products=operation_to_products,
        time_bounds=[(min_mach_cho_a, max_mach_cho_a)],
    )
    room_changeover = generate_room_changeover(
        rooms=rooms,
        time_bounds=[
            (min_room_cho_a, max_room_cho_a),
            (min_room_cho_b, max_room_cho_b),
        ],
    )
    # campaign_constraints = generate_campaign_constraints(
    #     products_desc=products_desc,
    #     families=families,
    #     min_campaign_length=min_campaign_length,
    #     max_campaign_length=max_campaign_length,
    # )
    crossblock_penalties = generate_crossblock_penalty(blocks=blocks)
    machine_availability = generate_machine_availability(
        machines=machines,
        max_stops=max_stops,
        max_time_range=max_time_range,
        min_stop_duration=min_stop_duration,
        max_stop_duration=max_stop_duration,
    )

    # Save all data
    output_dir = "input/generated/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    coremanf_demand.to_csv(os.path.join(output_dir, "coremanf_demand.csv"), index=False)
    inventory.to_csv(os.path.join(output_dir, "inventory.csv"), index=False)
    phantom_items.to_csv(os.path.join(output_dir, "phantom_items.csv"), index=False)
    products_desc.to_csv(os.path.join(output_dir, "products_desc.csv"), index=False)
    bom.to_csv(os.path.join(output_dir, "bom.csv"), index=False)
    recipes.to_csv(os.path.join(output_dir, "recipes.csv"), index=False)
    plant_map.to_csv(os.path.join(output_dir, "plant_map.csv"), index=False)
    room_changeover.to_csv(os.path.join(output_dir, "room_changeover.csv"), index=False)
    crossblock_penalties.to_csv(
        os.path.join(output_dir, "crossblock_penalties.csv"), index=False
    )
    machine_changeover.to_csv(
        os.path.join(output_dir, "machine_changeover.csv"), index=False
    )
    # campaign_constraints.to_csv(
    #     os.path.join(output_dir, "campaign_constraints.csv"), index=False
    # )
    machine_availability.to_csv(
        os.path.join(output_dir, "machine_availability.csv"), index=False
    )
    procurement_plan.to_csv(
        os.path.join(output_dir, "procurement_plan.csv"), index=False
    )
