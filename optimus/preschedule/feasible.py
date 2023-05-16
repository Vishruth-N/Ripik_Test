"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, Any

from optimus.utils.constants import DroppedReason, CAMode
from optimus.preschedule.batching import create_batch_graph, discard_no_impact_intms
from optimus.preschedule.distribute import (
    distribute_inventory,
    decode_lp_output,
    get_feasible_columns,
    find_perfect_feasible,
)
from optimus.preschedule.study_lp import (
    inventory_and_coefficients,
    get_constraints_data,
    get_inventory_status,
    feasible_vs_demand,
    feasible_contri_aggregate,
)
from optimus.data.filters import (
    filter_zero_demand,
    filter_missing_inventory,
    filter_missing_info,
    filter_covered_demand,
    filter_low_demand,
)
from optimus.data.logs import (
    log_demand_description,
    log_mios_in_demand,
    log_mios_in_feasible,
)
from optimus.elements.product import Material
from optimus.elements.inventory import Inventory
from optimus.utils.general import close_subtract
from optimus.utils.structs import *

import logging

logger = logging.getLogger(__name__)
progress_logger = logging.getLogger("progressLogger")


def normalise_demand(
    df_forecasted_demand: pd.DataFrame,
    products: Dict[str, Material],
    config: Dict[str, Any],
    inventory: Inventory,
) -> Tuple[Dict, Any]:
    # Utility for adding dropped reasons
    dropped_reason_category = []
    dropped_reason_description = []
    dropped_reason_details = []
    dropped_reason_index = []

    def _add_dropped_to_df(dropped_info: Dict[str, Any], dropped_reason: DroppedReason):
        dropped_reason_category.extend([dropped_reason.name] * len(dropped_info))
        dropped_reason_description.extend([dropped_reason.value] * len(dropped_info))
        dropped_reason_details.extend(list(dropped_info.values()))
        dropped_reason_index.extend(list(dropped_info.keys()))

    # Normalise demand
    normalised_demand = df_forecasted_demand.copy()

    logger.info("Before normalising demand...")
    log_demand_description(demand=normalised_demand)

    # Filter zero M1 demand
    normalised_demand, dropped_info1 = filter_zero_demand(
        demand=normalised_demand,
        pulling_months=config["pulling_months"],
    )

    _add_dropped_to_df(dropped_info1, DroppedReason.NO_DEMAND)
    logger.info("After filtering zero demand...")
    log_demand_description(demand=normalised_demand)

    # Filter already made products that cover demand
    normalised_demand, dropped_info2 = filter_covered_demand(
        demand=normalised_demand,
        pulling_months=config["pulling_months"],
        inventory=inventory,
    )

    _add_dropped_to_df(dropped_info2, DroppedReason.COVERED_DEMAND)
    logger.info("After filtering already covered demand...")
    log_demand_description(demand=normalised_demand)

    # Filter missing info
    normalised_demand, dropped_info3 = filter_missing_info(
        demand=normalised_demand, products=products
    )

    _add_dropped_to_df(dropped_info3, DroppedReason.MISSING_INFO)
    logger.info("After filtering missing info...")
    log_demand_description(demand=normalised_demand)
    log_mios_in_demand(
        df_demand=normalised_demand,
        products=products,
        pulling_months=config["pulling_months"],
    )

    # Filter low demand
    normalised_demand, dropped_info4 = filter_low_demand(
        demand=normalised_demand,
        pulling_months=config["pulling_months"],
        mts_demand_buffer=config["mts_demand_buffer"],
        products=products,
    )

    _add_dropped_to_df(dropped_info4, DroppedReason.LOW_DEMAND)
    logger.info("After filtering low demand...")
    log_demand_description(demand=normalised_demand)
    log_mios_in_demand(
        df_demand=normalised_demand,
        products=products,
        pulling_months=config["pulling_months"],
    )

    if config["rmpm_inventory_check"]:
        # Filter missing inventory (assuming no batching)
        normalised_demand, dropped_info5 = filter_missing_inventory(
            demand=normalised_demand,
            products=products,
            inventory=inventory,
            batching=False,
            max_level=config["max_dependancy_levels"],
        )

        _add_dropped_to_df(dropped_info5, DroppedReason.LOW_RM_PM_CONT)
        logger.info("After filtering low RM/PM inventory assuming no batching...")
        log_demand_description(demand=normalised_demand)
        log_mios_in_demand(
            df_demand=normalised_demand,
            products=products,
            pulling_months=config["pulling_months"],
        )

        # Filter missing inventory (considering batching)
        normalised_demand, dropped_info6 = filter_missing_inventory(
            demand=normalised_demand,
            products=products,
            inventory=inventory,
            batching=True,
            max_level=config["max_dependancy_levels"],
        )

        _add_dropped_to_df(dropped_info6, DroppedReason.LOW_RM_PM_BATCH)
        logger.info("After filtering low RM/PM considering batching...")
        log_demand_description(demand=normalised_demand)
        log_mios_in_demand(
            df_demand=normalised_demand,
            products=products,
            pulling_months=config["pulling_months"],
        )

    # Reset index
    normalised_demand = normalised_demand.reset_index().rename(
        {"index": "true_index"}, axis=1
    )

    # Create dropped reasons
    dropped_reasons = pd.DataFrame(
        {
            "reason_category": dropped_reason_category,
            "reason_description": dropped_reason_description,
            "reason_details": dropped_reason_details,
        },
        index=pd.Series(
            dropped_reason_index,
            name="material_id",
            dtype="object",
        ),
    )

    if config["DEBUG"]:
        # Create debug path
        others_path = os.path.join(config["debug_dir"], "others/")
        if not os.path.exists(others_path):
            os.makedirs(others_path)

        # Dropped reasons
        dropped_reasons.to_csv(os.path.join(others_path, "dropped_reasons.csv"))

        # Processed RFC
        normalised_demand.to_csv(
            os.path.join(others_path, "processed_rfc.csv"), index=False
        )

    return normalised_demand, dropped_reasons


def preschedule_demand(
    df_forecasted_demand: pd.DataFrame,
    df_initial_state: pd.DataFrame,
    products: Dict[str, Material],
    config: Dict[str, Any],
    current_inventory: Inventory,
    future_inventory: Inventory = None,
) -> Tuple[Dict, Any]:
    """
    Parameters
    -------------------------
    """
    # Normalise demand and find dropped reasons
    normalised_demand, dropped_reasons = normalise_demand(
        df_forecasted_demand=df_forecasted_demand,
        products=products,
        config=config,
        inventory=current_inventory,
    )

    # Check if demand is empty
    if len(normalised_demand) == 0:
        raise ValueError(f"Cannot preschedule an empty demand")

    if future_inventory is None:
        curr_iteration_no = 0
        max_iterations_feasible = 5
        flex_demand = normalised_demand.copy()
        flex_inventory = current_inventory.copy()
        feasible_batches = {}

        for col in config["pulling_months"]:
            flex_demand[f"fixed_{col}"] = flex_demand[col]

        while (
            curr_iteration_no < max_iterations_feasible
            and (flex_demand[config["pulling_months"]].sum(axis=1) > 0).any()
        ):
            progress_logger.info(
                f"Running feasible iteration: {curr_iteration_no + 1} / {max_iterations_feasible}"
            )

            if config["DEBUG"]:
                # Create debug path
                others_feasible_path = os.path.join(
                    config["debug_dir"], "others/feasible/"
                )
                if not os.path.exists(others_feasible_path):
                    os.makedirs(others_feasible_path)
                flex_demand.to_csv(
                    os.path.join(
                        others_feasible_path, f"flex_demand{curr_iteration_no}.csv"
                    ),
                    index=False,
                )
                with open(
                    os.path.join(
                        others_feasible_path, f"flex_inventory{curr_iteration_no}.json"
                    ),
                    "w",
                ) as f:
                    json.dump(flex_inventory.to_dict(), f)

            # Definite batches
            definite_batch_count = {}
            if df_initial_state is not None and curr_iteration_no == 0:
                definite_batch_count = (
                    df_initial_state.groupby(
                        [
                            InitialST.cols.material_id,
                            InitialST.cols.batch_size,
                            InitialST.cols.alt_bom,
                            InitialST.cols.ca_mode,
                        ]
                    )[InitialST.cols.sequence]
                    .count()
                    .to_dict()
                )

            # Find optimized distribution quantity
            lp_output, lp_model, feasible_graph = distribute_inventory(
                demand=flex_demand,
                products=products,
                inventory=flex_inventory,
                definite_batch_count=definite_batch_count,
                pulling_months=config["pulling_months"],
                crit_multiplier=config["crit_multiplier"],
                max_dependancy_levels=config["max_dependancy_levels"],
                rmpm_inventory_check=config["rmpm_inventory_check"],
                intm_inventory_check=config["intm_inventory_check"],
                solve_time_limit=config["lp_time_limit"],
            )

            if config["DEBUG"]:
                # Debug save path
                others_feasible_path = os.path.join(
                    config["debug_dir"], "others/feasible/"
                )

                # Save LP model and output
                with open(
                    os.path.join(
                        others_feasible_path, f"lp_model{curr_iteration_no}.json"
                    ),
                    "w",
                ) as f:
                    json.dump(lp_model.to_dict(), f)
                with open(
                    os.path.join(
                        others_feasible_path, f"lp_output{curr_iteration_no}.json"
                    ),
                    "w",
                ) as f:
                    json.dump(lp_output, f)

                # Constraints data
                constraints_data = get_constraints_data(
                    lp_model=lp_model,
                    lp_output=lp_output,
                )
                constraints_data.to_csv(
                    os.path.join(
                        others_feasible_path, f"lp_constraints{curr_iteration_no}.csv"
                    ),
                    index=False,
                )

            # Decode LP output to human-readable feasible format
            logger.info("Decoding feasible data...")
            feasible = decode_lp_output(
                demand=flex_demand,
                lp_output=lp_output,
                feasible_graph=feasible_graph,
                products=products,
            )

            # Create batch graph
            logger.info("Creating batches...")
            filled_batches, unfilled_batches = create_batch_graph(
                df_initial_state=df_initial_state if curr_iteration_no == 0 else None,
                demand=flex_demand,
                feasible=feasible,
                products=products,
                inventory=flex_inventory,
                consume_only_one=config["consume_only_one"],
            )

            # Log number of batches by CA Mode
            total_filled, total_unfilled = 0, 0
            for ca_mode in CAMode:
                filled, unfilled = 0, 0
                for batches in filled_batches[ca_mode.value].values():
                    filled += len(batches)
                for batches in unfilled_batches[ca_mode.value].values():
                    unfilled += len(batches)

                total_filled += filled
                total_unfilled += unfilled
                logger.debug(f"Total filled {ca_mode.name} batches: {filled}")
                logger.debug(f"Total unfilled {ca_mode.name} batches: {unfilled}")

            # Update feasible batches
            for ca_mode in filled_batches:
                for batch_key in filled_batches[ca_mode]:
                    for batch in filled_batches[ca_mode][batch_key]:
                        feasible_batches[batch.batch_id] = batch

            # Find feasible FGs
            df_feasible_vs_demand = feasible_vs_demand(
                feasible_batches=feasible_batches,
                df_demand=normalised_demand,
                products=products,
            )
            log_mios_in_feasible(df_feasible_vs_demand=df_feasible_vs_demand)

            if config["DEBUG"]:
                # Debug save path
                others_feasible_path = os.path.join(
                    config["debug_dir"], "others/feasible/"
                )
                # Save feasible FG
                feasible.to_csv(
                    os.path.join(
                        others_feasible_path, f"feasible{curr_iteration_no}.csv"
                    ),
                    index=False,
                )
                df_feasible_vs_demand.to_csv(
                    os.path.join(
                        others_feasible_path,
                        f"feasible_vs_demand{curr_iteration_no}.csv",
                    ),
                    index=False,
                )

            # No fulfilled or unfulfilled batches then break
            if total_unfilled == 0 or total_filled == 0:
                break

            # Update inventory
            for ca_mode in filled_batches:
                for batch_key in filled_batches[ca_mode]:
                    for batch in filled_batches[ca_mode][batch_key]:
                        for component_group in batch.requires:
                            for component_id, qnty in batch.requires[component_group][
                                "consumes"
                            ]["inventory"]:
                                flex_inventory.decrease(component_id, qnty)

                        bom = batch.product.get_bom(
                            batch_size=batch.batch_size, alt_bom=batch.alt_bom
                        )
                        for _, component in bom.items():
                            if component["indirect"] <= 0:
                                qnty_consumed = component["quantity"] * (
                                    batch.quantity / batch.batch_size
                                )
                                for component_id in component["component_ids"]:
                                    min_qnty = min(
                                        flex_inventory.get_quantity(component_id),
                                        qnty_consumed,
                                    )
                                    qnty_consumed -= min_qnty
                                    flex_inventory.decrease(component_id, min_qnty)

            # Update demand
            cad_dict = defaultdict(float)
            for ca_mode in filled_batches:
                for batch_key in filled_batches[ca_mode]:
                    for batch in filled_batches[ca_mode][batch_key]:
                        if batch.ca_mode == CAMode.CAD.value:
                            cad_dict[batch.product.material_id] += batch.quantity

            for row in flex_demand[
                flex_demand[DemandST.cols.material_id].isin(list(cad_dict.keys()))
            ].itertuples():
                row = row._asdict()
                left = cad_dict[row[DemandST.cols.material_id]]
                for col in DemandST.hilo_priority_cols():
                    if flex_demand.loc[row["Index"], col] - left >= 0:
                        flex_demand.loc[row["Index"], col] -= left
                        break
                    left = left - flex_demand.loc[row["Index"], col]
                    flex_demand.loc[row["Index"], col] = 0

            # Set quantity precision to 1
            flex_demand[config["pulling_months"]] = np.floor(
                flex_demand[config["pulling_months"]]
            )
            flex_demand = flex_demand[
                flex_demand[config["pulling_months"]].sum(axis=1) > 0
            ]

            if config["rmpm_inventory_check"]:
                # Filter missing inventory (considering batching)
                flex_demand, _ = filter_missing_inventory(
                    demand=flex_demand,
                    products=products,
                    inventory=flex_inventory,
                    batching=True,
                    max_level=config["max_dependancy_levels"],
                )

            curr_iteration_no += 1

        df_feasible_contri_aggregate = feasible_contri_aggregate(
            feasible_batches=feasible_batches
        )

        if config["DEBUG"]:
            # Create debug path
            others_path = os.path.join(config["debug_dir"], "others/")
            if not os.path.exists(others_path):
                os.makedirs(others_path)
            df_feasible_contri_aggregate.to_csv(
                os.path.join(others_path, "feasible_contri_aggregate.csv"), index=False
            )
        # Prune here
        # perfect_batches = []
        # imperfect_batches = []
        # good_batches = []
        # for batch in feasible_batches.values():
        #     if (batch.ca_mode == CAMode.CAD.value or batch.ca_mode == CAMode.CAV.value):
        #         if batch.quantity < batch.batch_size * 0.01:
        #             imperfect_batches.append(batch)
        #         else:
        #             good_batches.append(batch)
        #     else:
        #         perfect_batches.append(batch)

        # perfect_batches, _ = discard_no_impact_intms(
        #     perfect_batches=perfect_batches,
        #     imperfect_batches=imperfect_batches,
        #     fulfilled_batches=good_batches,
        # )

        # del feasible_batches
        # feasible_batches = {}
        # for batch in perfect_batches:
        #     feasible_batches[batch.batch_id] = batch
        # for batch in good_batches:
        #     feasible_batches[batch.batch_id] = batch

        # Inventory status
        # inventory_status = get_inventory_status(
        #     lp_model=lp_model,
        #     lp_output=lp_output,
        #     inventory_initial=current_inventory.copy().to_dict(),
        # )
        # inventory_status.to_csv(
        #     os.path.join(others_path, "lp_inventory_status.csv"), index=False
        # )

        # # Study LP
        # if config["rmpm_inventory_check"]:
        #     coeff_knowledge = inventory_and_coefficients(
        #         lp_model=lp_model,
        #         lp_output=lp_output,
        #         products=products,
        #         batching=config["batching"],
        #         max_dependancy_levels=config["max_dependancy_levels"],
        #         demand=normalised_demand,
        #         inventory_status=inventory_status,
        #     )
        #     coeff_knowledge.to_csv(
        #         os.path.join(others_path, "lp_coeff_knowledge.csv"), index=False
        #     )

        return feasible_batches, normalised_demand, dropped_reasons

    else:
        raise NotImplementedError("Feedbackward pipeline broken")
