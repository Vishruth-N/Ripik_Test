"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from optimus.machines.event import MachineState
from optimus.utils.constants import MaterialType
from optimus.utils.general import merge_intervals


def quantity_stats(
    production_view: pd.DataFrame,
    period: str,
) -> Dict[str, Any]:
    # Basic insights
    output = {}
    output["mios"] = float(
        production_view.loc[
            production_view["material_type"] == MaterialType.fg.value,
            f"quantity_counted_({period})",
        ].sum()
        / 1e6
    )
    output["cb_batches"] = int(
        production_view.loc[
            production_view["material_type"] == MaterialType.cb.value, f"nob_({period})"
        ].sum()
    )
    output["sfg_batches"] = int(
        production_view.loc[
            production_view["material_type"] == MaterialType.sfg.value,
            f"nob_({period})",
        ].sum()
    )
    output["fg_batches"] = int(
        production_view.loc[
            production_view["material_type"] == MaterialType.fg.value, f"nob_({period})"
        ].sum()
    )
    output["unique_cb"] = int(
        production_view.loc[
            (production_view[f"nob_({period})"] > 0)
            & (production_view["material_type"] == MaterialType.cb.value),
            "material_id",
        ].nunique()
    )
    output["unique_sfg"] = int(
        production_view.loc[
            (production_view[f"nob_({period})"] > 0)
            & (production_view["material_type"] == MaterialType.sfg.value),
            "material_id",
        ].nunique()
    )
    output["unique_fg"] = int(
        production_view.loc[
            (production_view[f"nob_({period})"] > 0)
            & (production_view["material_type"] == MaterialType.fg.value),
            "material_id",
        ].nunique()
    )
    output["sfg_quantity"] = float(
        production_view.loc[
            production_view["material_type"] == MaterialType.sfg.value,
            f"quantity_processed_({period})",
        ].sum()
    )
    if output["sfg_batches"] > 0:
        output["sfg_avg_batch_size"] = output["sfg_quantity"] / output["sfg_batches"]
    else:
        output["sfg_avg_batch_size"] = 0

    return output


def utility_stats(
    machine_view: pd.DataFrame, start_period: float, end_period: float
) -> Dict[str, Any]:
    # Create time filtered machine view
    tf_machine_view = machine_view[
        (machine_view["rel_end_time"] >= start_period)
        & (machine_view["rel_end_time"] < end_period)
    ]
    total_time = end_period - start_period

    # Calculate all states time in the rooms
    roomsi = {}
    for name, group in tf_machine_view.groupby(["room_id", "state"]):
        room_id, state = name
        if room_id not in roomsi:
            roomsi[room_id] = {
                "block_id": group["block_id"].iloc[0],
                "utilization_time": 0,
                "quantity_processed_across_machines": 0,
                "curr_intervals": [],
            }

            for machine_state in MachineState:
                roomsi[room_id][machine_state.name] = 0

        curr_intervals = []
        for row in group.itertuples():
            curr_intervals.append((row.rel_start_time, row.rel_end_time))

        if curr_intervals:
            curr_intervals = merge_intervals(curr_intervals)
            for s, e in curr_intervals:
                roomsi[room_id][state] += e - s
            roomsi[room_id]["curr_intervals"].extend(curr_intervals)

    # Calculate all states time in the machines
    machinesi = {}
    for group_name, group in tf_machine_view.groupby(["machine_id", "state"]):
        machine_id, state = group_name
        room_id = group["room_id"].iloc[0]
        block_id = group["block_id"].iloc[0]
        operation = group["operation"].iloc[0]

        if machine_id not in machinesi:
            machinesi[machine_id] = {
                "room_id": room_id,
                "block_id": block_id,
                "operation": operation,
                "quantity_processed": 0,
                "utilization_time": 0,
            }

            for machine_state in MachineState:
                machinesi[machine_id][machine_state.name] = 0

        curr_intervals = []
        for row in group.itertuples():
            curr_intervals.append((row.rel_start_time, row.rel_end_time))

        if curr_intervals:
            curr_intervals = merge_intervals(curr_intervals)
            for s, e in curr_intervals:
                machinesi[machine_id][state] += e - s
            machinesi[machine_id]["utilization_time"] += machinesi[machine_id][state]

        if state == MachineState.BUSY.name:
            for row in group.itertuples():
                all_info_items = row.info.split(";\n")
                data_info = {}
                for item in all_info_items:
                    key, value = item.split(": ", maxsplit=1)
                    data_info[key] = value
                machinesi[machine_id]["quantity_processed"] += float(
                    data_info["Quantity processed"]
                )

        # Add quantity to room
        roomsi[room_id]["quantity_processed_across_machines"] += machinesi[machine_id][
            "quantity_processed"
        ]

    # Calculate average utility
    average_room_utilization = []
    average_machine_utilization = []
    for room_id in roomsi:
        curr_intervals = merge_intervals(roomsi[room_id]["curr_intervals"])
        for s, e in curr_intervals:
            roomsi[room_id]["utilization_time"] += e - s

        roomsi[room_id]["utilization"] = (
            roomsi[room_id]["utilization_time"] / total_time
        )
        average_room_utilization.append(roomsi[room_id]["utilization"])

    for machine_id in machinesi:
        machinesi[machine_id]["utilization"] = (
            machinesi[machine_id]["utilization_time"] / total_time
        )
        average_machine_utilization.append(machinesi[machine_id]["utilization"])

    # Remove curr intervals for clarity
    for room_id in list(roomsi.keys()):
        del roomsi[room_id]["curr_intervals"]

    output = {
        "average_room_utilization": np.mean(average_room_utilization)
        if len(average_room_utilization) > 0
        else 0,
        "average_machine_utilization": np.mean(average_machine_utilization)
        if len(average_machine_utilization) > 0
        else 0,
        "num_rooms": len(roomsi),
        "num_machines": len(machinesi),
        "rooms": roomsi,
        "machines": machinesi,
    }

    return output


def block_stats(
    product_view: pd.DataFrame, start_period: float, end_period: float
) -> Dict[str, Any]:
    # Create time filtered product view
    tf_product_view = product_view[
        (product_view["rel_end_time"] >= start_period)
        & (product_view["rel_end_time"] < end_period)
    ]

    # Create block- view
    block_view = tf_product_view.groupby(["block_id", "material_type"]).agg(
        num_operations=("batch_id", "count"),
        unique_batches=("batch_id", "nunique"),
    )

    # Split by index and create output
    output = {}
    for row in block_view.itertuples():
        block_id, material_type = row.Index
        if block_id not in output:
            output[block_id] = {}

        output[block_id][material_type] = {
            "num_operations": row.num_operations,
            "unique_batches": row.unique_batches,
        }

    return output
