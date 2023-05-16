"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Any

from optimus.utils.constants import BatchingMode, DroppedReason
from optimus.elements.product import Material
from optimus.elements.inventory import Inventory
from optimus.utils.structs import DemandST


def filter_zero_demand(
    demand: pd.DataFrame,
    pulling_months: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Check for empty demand first
    if len(demand) == 0:
        return demand, {}

    # Aggregate the quantity for same group identified by ["material_id", "priority", "due_date"]
    demand = demand.groupby(
        [
            DemandST.cols.material_id,
            DemandST.cols.production_strategy,
            DemandST.cols.priority,
            DemandST.cols.due_date,
            DemandST.cols.ca_mode,
        ],
        dropna=False,
        as_index=False,
    ).sum()

    # Add backlog
    demand["backlog"] = (
        demand[DemandST.cols.m0_demand] - demand[DemandST.cols.m0_commit]
    )

    for row in demand.itertuples():
        row = row._asdict()

        left = row["backlog"]
        for col in DemandST.hilo_priority_cols():
            if demand.loc[row["Index"], col] + left >= 0:
                demand.loc[row["Index"], col] += left
                break
            left += demand.loc[row["Index"], col]
            demand.loc[row["Index"], col] = 0

    # Filter out the negative demand and remove 'backlog' column
    filtered_demand = demand[demand[pulling_months].sum(axis=1) > 0].drop(
        "backlog", axis=1
    )

    # Dropped info
    dropped_info = {
        idx: DroppedReason.NO_DEMAND.value
        for idx in set(demand["material_id"]) - set(filtered_demand["material_id"])
    }

    return filtered_demand, dropped_info


def filter_covered_demand(
    demand: pd.DataFrame,
    pulling_months: List[str],
    inventory: Inventory,
):
    # Check for empty demand first
    if len(demand) == 0:
        return demand, {}

    # Sway subtract inventory
    for row in demand.itertuples():
        row = row._asdict()
        material_id = row[DemandST.cols.material_id]

        old_left = -inventory.get_quantity(material_id)
        left = old_left
        for col in DemandST.hilo_priority_cols():
            if demand.loc[row["Index"], col] + left >= 0:
                demand.loc[row["Index"], col] += left
                break
            left += demand.loc[row["Index"], col]
            demand.loc[row["Index"], col] = 0
        inventory.decrease(material_id, left - old_left)

    # Filter out the negative demand and remove 'backlog' column
    filtered_demand = demand[demand[pulling_months].sum(axis=1) > 0]

    # Dropped info
    dropped_info = {
        idx: DroppedReason.COVERED_DEMAND.value
        for idx in set(demand["material_id"]) - set(filtered_demand["material_id"])
    }

    return filtered_demand, dropped_info


def filter_low_demand(
    demand: pd.DataFrame,
    pulling_months: List[str],
    mts_demand_buffer: float,
    products: Dict[str, Material],
):
    # Check for empty demand first
    if len(demand) == 0:
        return demand, {}

    # Create useful columns
    demand["min_batch_size"] = demand[DemandST.cols.material_id].apply(
        lambda x: min(products[x].get_batch_sizes())
    )
    demand["batching_mode"] = demand[DemandST.cols.material_id].apply(
        lambda x: products[x].batching_mode
    )
    demand["diff"] = demand["min_batch_size"] - demand[pulling_months].sum(axis=1)

    # Filter out SKus which meet buffer condition
    filtered_demand = demand[
        (demand["batching_mode"] == BatchingMode.treal.value)
        | (
            demand[pulling_months].sum(axis=1) * (1 + mts_demand_buffer)
            >= demand["min_batch_size"]
        )
    ]

    # Pump up low demand to the min batch size
    mask = (filtered_demand["batching_mode"] == BatchingMode.tint.value) & (
        filtered_demand["diff"] > 0
    )
    n = len(pulling_months)
    N = n * (n + 1) / 2
    for i, col in enumerate(pulling_months):
        filtered_demand.loc[mask, col] += (
            filtered_demand.loc[mask, "diff"] * (i + 1) / N
        )

    # Drop derived columns
    filtered_demand = filtered_demand.drop(
        ["min_batch_size", "batching_mode", "diff"], axis=1
    )

    # Dropped info
    dropped_info = {
        idx: DroppedReason.LOW_DEMAND.value
        for idx in set(demand["material_id"]) - set(filtered_demand["material_id"])
    }

    return filtered_demand, dropped_info


def filter_missing_info(
    demand: pd.DataFrame, products: Dict[str, Material]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Check for empty demand first
    if len(demand) == 0:
        return demand, {}

    # Filter out no information
    dropped_info = {}
    dropped_index = []
    for row in demand.itertuples():
        row = row._asdict()
        if row[DemandST.cols.material_id] not in products:
            dropped_info[row["material_id"]] = DroppedReason.MISSING_INFO.value
            dropped_index.append(row["Index"])

    filtered_demand = demand.drop(dropped_index)

    return filtered_demand, dropped_info


def _recursive_inventory_check(
    products: Dict[str, Material],
    inventory: Inventory,
    material_id: str,
    quantity: float,
    level: int,
    batching: bool,
    max_level: int,
    local_stack: Dict[str, int],
) -> Tuple[Dict[str, float], bool, frozenset]:
    if level >= max_level:
        # Assume it to be available from somewhere
        return {}, True, frozenset()

    product = products[material_id]

    missing_elements = []
    for batch_size, alt_bom in product.iterate_boms():
        my_consumed = defaultdict(float)
        my_feasible = True
        bom = product.get_bom(batch_size=batch_size, alt_bom=alt_bom)
        for component_group, component in bom.items():
            qnty_required = component["quantity"] * (quantity / batch_size)

            if batching:
                if product.batching_mode == BatchingMode.tint.value:
                    scale_factor = quantity / batch_size
                    if not np.isclose(scale_factor, np.floor(scale_factor)):
                        scale_factor = np.ceil(scale_factor)

                    qnty_required = component["quantity"] * scale_factor

                elif not product.batching_mode == BatchingMode.treal.value:
                    raise ValueError(f"Invalid batching mode: {product.batching_mode}")

            if component["indirect"]:
                # Already fulfilled by the inventory itself
                qnty_in_inventory = 0
                for component_id in component["component_ids"]:
                    qnty_in_inventory += inventory.get_quantity(component_id)

                if qnty_in_inventory >= qnty_required:
                    continue

                # Check for cyclic BOMs
                good_component_ids = []
                cyclic = True
                for component_id in component["component_ids"]:
                    if not (
                        component_id in local_stack
                        and level > local_stack[component_id]
                    ):
                        cyclic = False
                        good_component_ids.append(component_id)

                if cyclic:
                    my_feasible = False
                    missing_elements.append((level, component_group, "Cycle"))
                    break

                else:
                    for component_id in good_component_ids:
                        local_stack[component_id] = level

                # Recursive check all components
                consumed = None
                missing_info = []
                feasible = False
                for component_id in good_component_ids:
                    (
                        local_consumed,
                        local_feasible,
                        local_missing_info,
                    ) = _recursive_inventory_check(
                        products=products,
                        inventory=inventory,
                        material_id=component_id,
                        quantity=qnty_required - qnty_in_inventory,
                        level=level + 1,
                        batching=batching,
                        max_level=max_level,
                        local_stack=local_stack,
                    )

                    del local_stack[component_id]

                    if not local_feasible:
                        missing_info.append(local_missing_info)
                    else:
                        consumed = local_consumed
                        feasible = True
                        break

                if not feasible:
                    my_feasible = False
                    missing_elements.append(
                        (level, component_group, frozenset(missing_info))
                    )
                    break

                for intm_consumed_key, qnty in consumed.items():
                    my_consumed[intm_consumed_key] += qnty

            else:
                my_consumed[
                    (material_id, batch_size, alt_bom, component_group)
                ] += qnty_required

        if not my_feasible:
            continue

        for consumed_key, reqd_qnty in my_consumed.items():
            (
                mat_material_id,
                mat_batch_size,
                mat_alt_bom,
                mat_component_group,
            ) = consumed_key
            qnty_in_inventory = 0
            for component_id in products[mat_material_id].get_bom(
                batch_size=mat_batch_size, alt_bom=mat_alt_bom
            )[mat_component_group]["component_ids"]:
                qnty_in_inventory += inventory.get_quantity(component_id)

            if qnty_in_inventory < reqd_qnty:
                my_feasible = False
                missing_elements.append((level, consumed_key))
                break

        if my_feasible:
            return my_consumed, True, frozenset()

    return {}, False, frozenset(missing_elements)


def recursive_inventory_check(
    products: Dict[str, Material],
    inventory: Inventory,
    material_id: str,
    batching: bool,
    max_level: int,
    quantity: float = 1,
) -> Tuple[Dict[str, float], bool, frozenset]:
    level = 0
    local_stack = {material_id: level}

    consumed, feasible, missing_info = _recursive_inventory_check(
        products=products,
        inventory=inventory,
        material_id=material_id,
        quantity=quantity,
        level=level,
        batching=batching,
        max_level=max_level,
        local_stack=local_stack,
    )

    return consumed, feasible, missing_info


def filter_missing_inventory(
    demand: pd.DataFrame,
    products: Dict[str, Material],
    inventory: Inventory,
    batching: bool,
    max_level: int,
):
    # Check for empty demand first
    if len(demand) == 0:
        return demand, {}

    dropped_info = {}
    dropped_index = []
    for row in demand.itertuples():
        row = row._asdict()

        _, feasible, missing_info = recursive_inventory_check(
            products=products,
            inventory=inventory,
            material_id=row[DemandST.cols.material_id],
            batching=batching,
            max_level=max_level,
            quantity=1,
        )

        if not feasible:
            dropped_info[row["material_id"]] = missing_info
            dropped_index.append(row["Index"])

    filtered_demand = demand.drop(dropped_index)

    return filtered_demand, dropped_info
