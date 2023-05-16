"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import inspect
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Any

from optimus.machines.event import MachineState
from optimus.utils.constants import CAMode, DroppedReason
from optimus.machines.machine import Machine
from optimus.utils.structs import DemandST
from optimus.preschedule.batching import Batch


def view_by_machine(machines: Dict[str, Machine], execution_start: datetime = None):
    """
    View by machine
    """
    df = defaultdict(list)
    for machine in machines.values():
        for event in machine.get_schedule():
            df["block_id"].append(machine.block_id)
            df["room_id"].append(machine.room_id)
            df["machine_id"].append(machine.machine_id)
            df["machine_type"].append(machine.machine_type)
            df["operation"].append(machine.operation)
            df["state"].append(event.state.name)
            df["rel_start_time"].append(event.start_time)
            df["rel_end_time"].append(event.end_time)
            df["remarks"].append(event.note)
            if event.state == MachineState.BUSY:
                step_description = event.task.product.step_description_of(
                    event.task.batch_size,
                    event.task.alt_recipe,
                    event.task.sequence,
                )
                info = inspect.cleandoc(
                    f"""
                    Material ID: {event.task.product.material_id};
                    Material Name: {event.task.product.material_name};
                    Material Type: {event.task.product.get_material_type()};
                    Batch ID: {event.task.batch_id};
                    Step description: {step_description};
                    Sequence: {event.task.sequence};
                    Quantity processed: {event.task.quantity_processed};
                    Batch size: {event.task.batch_size};
                    Alt BOM: {event.task.alt_bom};
                    Alt recipe: {event.task.alt_recipe};
                    Task available at: {event.task.available_at};
                    Alt machines: {[machine.machine_id for machine in event.task.get_approved_machines()]};
                    Alt rooms: {[machine.room_id for machine in event.task.get_approved_machines()]};
                """
                )
                df["info"].append(info)
            else:
                df["info"].append(np.nan)

    df = pd.DataFrame(df)
    if not df.empty and execution_start is not None:
        df["start_time"] = df["rel_start_time"].apply(
            lambda x: execution_start + timedelta(hours=x)
        )
        df["end_time"] = df["rel_end_time"].apply(
            lambda x: execution_start + timedelta(hours=x)
        )
        df.sort_values("room_id", inplace=True)

    return df


def view_by_product(machines: Dict[str, Machine], execution_start: datetime = None):
    """
    View by product
    """
    df = defaultdict(list)
    for machine in machines.values():
        for event in machine.get_schedule():
            if event.state == MachineState.BUSY:
                df["machine_id"].append(machine.machine_id)
                df["room_id"].append(machine.room_id)
                df["block_id"].append(machine.block_id)
                df["operation"].append(
                    event.task.product.step_description_of(
                        event.task.batch_size,
                        event.task.alt_recipe,
                        event.task.sequence,
                    )
                )
                df["product_id"].append(event.task.product.material_id)
                df["material_name"].append(event.task.product.material_name)
                df["material_type"].append(event.task.product.get_material_type())
                df["consumed_as"].append(event.task.ca_mode)
                df["batch_id"].append(event.task.batch_id)
                df["rel_start_time"].append(event.start_time)
                df["rel_end_time"].append(event.end_time)
                df["op_order"].append(
                    event.task.product.op_order_of(
                        event.task.batch_size,
                        event.task.alt_recipe,
                        event.task.sequence,
                    )
                )
                df["sequence"].append(event.task.sequence)
                df["batch_size"].append(event.task.batch_size)
                df["quantity_processed"].append(event.task.quantity_processed)
                df["alt_bom"].append(event.task.alt_bom)

    df = pd.DataFrame(df)
    if not df.empty and execution_start is not None:
        df["start_time"] = df["rel_start_time"].apply(
            lambda x: execution_start + timedelta(hours=x)
        )
        df["end_time"] = df["rel_end_time"].apply(
            lambda x: execution_start + timedelta(hours=x)
        )
        df.sort_values(
            by=["product_id", "batch_id", "sequence", "start_time"], inplace=True
        )

    return df


def view_by_quantity(
    machines: Dict[str, Machine],
    execution_start: datetime = None,
):
    """
    View by quantity
    """
    data = {}
    for machine in machines.values():
        for event in machine.get_schedule():
            if event.state == MachineState.BUSY:
                batch_id = event.task.batch_id
                if batch_id not in data:
                    data[batch_id] = {
                        "rel_start_time": np.nan,
                        "rel_end_time": np.nan,
                    }

                if (
                    event.task.sequence
                    == len(
                        event.task.product.get_recipe(
                            event.task.batch_size, event.task.alt_recipe
                        )
                    )
                    - 1
                ):
                    data[batch_id]["rel_end_time"] = event.end_time

                if event.task.sequence == 0:
                    data[batch_id]["rel_start_time"] = event.start_time

                # Store task as well
                data[batch_id]["task"] = event.task

    df = defaultdict(list)
    for batch_id, info in data.items():
        task = info["task"]
        consumed_data_items = 0
        for component_group, consumed in task.get_consumed_inventory().items():
            for component_id, qnty in consumed:
                df["consumption_type"].append("Inventory")
                df["component_group"].append(component_group)
                df["component_batch_id"].append(None)
                df["component_id"].append(component_id)
                df["component_qnty"].append(qnty)
                consumed_data_items += 1

        for component_group, consumed in task.get_consumed_batches().items():
            for component_batch_id, qnty in consumed:
                df["consumption_type"].append("INTM")
                df["component_group"].append(component_group)
                df["component_batch_id"].append(component_batch_id)
                df["component_id"].append(
                    data[component_batch_id]["task"].product.material_id
                )
                df["component_qnty"].append(qnty)
                consumed_data_items += 1

        if consumed_data_items == 0:
            df["consumption_type"].append(None)
            df["component_group"].append(None)
            df["component_batch_id"].append(None)
            df["component_id"].append(None)
            df["component_qnty"].append(None)
            consumed_data_items += 1

        df["material_id"].extend([task.product.material_id] * consumed_data_items)
        df["material_name"].extend([task.product.material_name] * consumed_data_items)
        df["material_type"].extend(
            [task.product.get_material_type()] * consumed_data_items
        )
        df["batch_id"].extend([batch_id] * consumed_data_items)
        df["consumed_as"].extend([task.ca_mode] * consumed_data_items)
        df["rel_start_time"].extend([info["rel_start_time"]] * consumed_data_items)
        df["rel_end_time"].extend([info["rel_end_time"]] * consumed_data_items)
        df["quantity_processed"].extend([task.quantity_processed] * consumed_data_items)
        df["batch_size"].extend([task.batch_size] * consumed_data_items)
        df["priority"].extend([task.priority] * consumed_data_items)
        df["due_date"].extend([task.due_date] * consumed_data_items)
        df["demand_contri"].extend([task.demand_contri] * consumed_data_items)

    # Add dates and rearrange dataframe columns
    df = pd.DataFrame(df)
    if not df.empty and execution_start is not None:
        df["start_time"] = df["rel_start_time"].apply(
            lambda x: np.nan if np.isnan(x) else execution_start + timedelta(hours=x)
        )
        df["end_time"] = df["rel_end_time"].apply(
            lambda x: np.nan if np.isnan(x) else execution_start + timedelta(hours=x)
        )
        df.sort_values(by=["batch_id"]).reset_index(drop=True, inplace=True)

    # Rearrange columns
    cols_sequence = [
        "batch_id",
        "material_id",
        "material_name",
        "material_type",
        "batch_size",
        "quantity_processed",
        "rel_start_time",
        "rel_end_time",
    ]
    if execution_start is not None:
        cols_sequence += ["start_time", "end_time"]
    cols_sequence += [
        "consumed_as",
        "consumption_type",
        "component_group",
        "component_batch_id",
        "component_id",
        "component_qnty",
        "priority",
        "due_date",
        "demand_contri",
    ]
    df = df[cols_sequence]

    return df


def view_by_production(machines: Dict[str, Machine], config: Dict[str, Any] = None):
    """
    View by production
    """
    data = {}
    for machine in machines.values():
        for event in machine.get_schedule():
            if event.state == MachineState.BUSY:
                material_id = event.task.product.material_id
                if material_id not in data:
                    data[material_id] = {
                        "material_type": event.task.product.get_material_type(),
                        "material_name": event.task.product.material_name,
                        "periods": {},
                    }
                    for period in config["periods"]:
                        data[material_id]["periods"][period] = {
                            "batch_ids": set(),
                            "quantity_processed": 0,
                            "quantity_counted": 0,
                        }

                if (
                    event.task.sequence
                    == len(
                        event.task.product.get_recipe(
                            event.task.batch_size, event.task.alt_recipe
                        )
                    )
                    - 1
                ):
                    for period, (start_period, end_period) in config["periods"].items():
                        if start_period <= event.end_time < end_period:
                            data[material_id]["periods"][period]["batch_ids"].add(
                                event.task.batch_id
                            )
                            data[material_id]["periods"][period][
                                "quantity_processed"
                            ] += event.task.quantity_processed
                            data[material_id]["periods"][period][
                                "quantity_counted"
                            ] += (
                                event.task.quantity_processed
                                * event.task.product.count_factor
                            )

    df = defaultdict(list)
    for material_id in data:
        df["material_id"].append(material_id)
        df["material_name"].append(data[material_id]["material_name"])
        df["material_type"].append(data[material_id]["material_type"])
        for period in config["periods"]:
            df[f"nob_({period})"].append(
                len(data[material_id]["periods"][period]["batch_ids"])
            )
            df[f"quantity_processed_({period})"].append(
                data[material_id]["periods"][period]["quantity_processed"]
            )
            df[f"quantity_counted_({period})"].append(
                data[material_id]["periods"][period]["quantity_counted"]
            )

    df = pd.DataFrame(df)
    return df


def view_by_commit(
    machines: Dict[str, Machine],
    actual_demand: pd.DataFrame,
    normalised_demand: pd.DataFrame,
    feasible_batches: Dict[str, Batch],
    dropped_reasons: pd.DataFrame = None,
    config: Dict[str, Any] = None,
):
    """
    View by commit
    """
    # Quantity view
    quantity_view = view_by_quantity(
        machines=machines,
    )

    # Merge normalised demand with actual demand
    actual_demand = pd.merge(
        actual_demand,
        normalised_demand,
        how="outer",
        on=DemandST.cols.material_id,
        suffixes=("", "_adjusted"),
    )
    actual_demand = actual_demand[
        actual_demand[DemandST.cols.ca_mode] == CAMode.CAD.value
    ]

    # Demand view sum on material id
    m1_cols_adjusted = [col + "_adjusted" for col in DemandST.get_m1_cols()]
    m123_cols_adjusted = [col + "_adjusted" for col in config["pulling_months"]]

    final_commit = defaultdict(list)
    for material_id in actual_demand["material_id"].unique():
        # Quantity demanded actual
        curr_qnty_demanded = float(
            actual_demand.loc[
                actual_demand["material_id"] == material_id, DemandST.get_m1_cols()
            ]
            .to_numpy()
            .sum()
        )
        total_qnty_demanded = float(
            actual_demand.loc[
                actual_demand["material_id"] == material_id, config["pulling_months"]
            ]
            .to_numpy()
            .sum()
        )

        # Quantity demanded adjusted
        curr_qnty_adjusted = float(
            actual_demand.loc[
                actual_demand["material_id"] == material_id, m1_cols_adjusted
            ]
            .to_numpy()
            .sum()
        )
        total_qnty_adjusted = float(
            actual_demand.loc[
                actual_demand["material_id"] == material_id, m123_cols_adjusted
            ]
            .to_numpy()
            .sum()
        )

        # Quantity feasible
        qnty_planned = 0
        for batch in feasible_batches.values():
            if (
                batch.ca_mode == CAMode.CAD.value
                and batch.product.material_id == material_id
            ):
                qnty_planned += batch.quantity

        # Quantity scheduled
        qnty_produced = {}
        for period in config["periods"]:
            qnty_produced[period] = 0

        for _, group in quantity_view[
            (quantity_view["material_id"] == material_id)
            & (quantity_view["consumed_as"] == CAMode.CAD.value)
        ].groupby("batch_id"):
            for period, (start_period, end_period) in config["periods"].items():
                if start_period <= group["rel_end_time"].iloc[0] < end_period:
                    qnty_produced[period] += group["quantity_processed"].iloc[0]

        final_commit["material_id"].append(material_id)
        final_commit["total_qnty_demanded"].append(total_qnty_demanded)
        final_commit["curr_qnty_demanded"].append(curr_qnty_demanded)
        final_commit["total_qnty_adjusted"].append(total_qnty_adjusted)
        final_commit["curr_qnty_adjusted"].append(curr_qnty_adjusted)
        final_commit["qnty_feasible"].append(qnty_planned)
        for period, qnty in qnty_produced.items():
            final_commit[f"qnty_scheduled_({period})"].append(qnty)

    final_commit = pd.DataFrame(final_commit)
    if dropped_reasons is not None:
        # Merge reason
        final_commit = final_commit.merge(dropped_reasons, on="material_id", how="left")

        # TODO: Give reason to feasible elements before
        final_commit.loc[
            (final_commit["qnty_feasible"] <= 0)
            & (final_commit["reason_category"].isna()),
            "reason_category",
        ] = DroppedReason.DISTRIBUTED_TO_OTHERS.name
        final_commit.loc[
            (final_commit["qnty_feasible"] <= 0)
            & (final_commit["reason_description"].isna()),
            "reason_description",
        ] = DroppedReason.DISTRIBUTED_TO_OTHERS.value

    return final_commit
