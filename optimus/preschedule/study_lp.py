"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import pulp
import numpy as np
import pandas as pd
from typing import Dict, List
from collections import defaultdict

from optimus.utils.constants import CAMode
from optimus.preschedule.utils import split_pids
from optimus.utils.general import multidict
from optimus.data.filters import recursive_inventory_check
from optimus.utils.structs import DemandST
from optimus.elements.product import Material
from optimus.preschedule.batching import Batch


def get_constraints_data(
    lp_model: pulp.LpProblem, lp_output: Dict[str, float]
) -> pd.DataFrame:
    # Use dictionary version for easy indexing
    lp_model = lp_model.to_dict()

    # Calculate lhs and rhs and store constraint variables
    data = []
    for constraint in lp_model["constraints"]:
        name = constraint["name"]
        constant = constraint["constant"]
        sense = constraint["sense"]
        value = 0
        for coefficient in constraint["coefficients"]:
            value += lp_output[coefficient["name"]] * coefficient["value"]
        data.append([name, constant, sense, value])

    # Return dataframe
    df = pd.DataFrame(data, columns=["name", "constant", "sense", "value"])
    return df


def get_inventory_status(
    lp_model: pulp.LpProblem,
    lp_output: Dict[str, float],
    inventory_initial: pd.DataFrame,
) -> pd.DataFrame:
    # Use dictionary version for easy indexing
    lp_model = lp_model.to_dict()

    # Calculate inventory consumed by inventory constraints
    inventory_after_consumed = {}
    inventory_after_used = {}
    for constraint in filter(
        lambda x: x["name"].startswith("inventory"), lp_model["constraints"]
    ):
        rmpm_id = constraint["name"].split("_", maxsplit=1)[1]
        total = constraint["constant"] * constraint["sense"]
        for coefficient in constraint["coefficients"]:
            total += (
                lp_output[coefficient["name"]]
                * coefficient["value"]
                * constraint["sense"]
            )

        if constraint["name"].startswith("inventoryConsumed_"):
            inventory_after_consumed[rmpm_id] = total
        elif constraint["name"].startswith("inventoryUsed_"):
            inventory_after_used[rmpm_id] = total

    # Create dataframe
    all_material_ids = inventory_initial.keys()
    inventory_status = pd.DataFrame(
        {
            "material_id": all_material_ids,
            "initial": [inventory_initial[m] for m in all_material_ids],
            "after_consumption": [
                inventory_after_consumed.get(m, inventory_initial[m])
                for m in all_material_ids
            ],
            "after_used": [
                inventory_after_used.get(m, inventory_initial[m])
                for m in all_material_ids
            ],
        }
    )

    # Derive other columns
    inventory_status["consumed"] = (
        inventory_status["initial"] - inventory_status["after_consumption"]
    )
    inventory_status["used"] = (
        inventory_status["initial"] - inventory_status["after_used"]
    )
    inventory_status["unused"] = inventory_status["consumed"] - inventory_status["used"]

    return inventory_status


def feasible_vs_demand(
    feasible_batches: Dict[str, Batch],
    df_demand: pd.DataFrame,
    products: Dict[str, Material],
) -> pd.DataFrame:
    # Feasible by restriction
    feasible_by_restriction = multidict(2, float)
    for batch in feasible_batches.values():
        if batch.ca_mode == CAMode.CAD.value:
            if batch.is_locked():
                feasible_by_restriction[batch.product.material_id][
                    "restricted"
                ] += batch.quantity
            else:
                feasible_by_restriction[batch.product.material_id][
                    "unrestricted"
                ] += batch.quantity

    # Find commit against the demand
    output = defaultdict(list)
    for row in df_demand.itertuples():
        row = row._asdict()
        material_id = row[DemandST.cols.material_id]
        product = products[material_id]

        output["material_id"].append(material_id)
        output["count_factor"].append(product.count_factor)
        output["total_demand"].append(
            sum(row[col] for col in DemandST.hilo_priority_cols())
        )

        m1_demand = sum(row[col] for col in DemandST.get_m1_cols())
        unrestricted_qnty_feasible = feasible_by_restriction[material_id][
            "unrestricted"
        ]
        restricted_qnty_feasible = feasible_by_restriction[material_id]["restricted"]
        total_feasible = unrestricted_qnty_feasible + restricted_qnty_feasible

        output["m1_demand"].append(m1_demand)
        output["unrestricted_feasible"].append(unrestricted_qnty_feasible)
        output["restricted_feasible"].append(restricted_qnty_feasible)
        output["total_feasible"].append(total_feasible)
        output["m1_feasible"].append(min(m1_demand, total_feasible))

    # Find feasible FGs
    output = pd.DataFrame(output)
    return output


def feasible_contri_aggregate(feasible_batches: Dict[str, Batch]) -> pd.DataFrame:
    demand_cols = DemandST.hilo_priority_cols()
    nob_dict = multidict(2, float)
    for batch in feasible_batches.values():
        nob_key = (
            batch.product.material_id,
            batch.batch_size,
            batch.alt_bom,
            batch.ca_mode,
        )

        nob_dict[nob_key]["total_nob"] += batch.quantity / batch.batch_size
        nob_dict[nob_key]["nob_to_be_scheduled"] += 1
        for demand_col, quantity in batch.demand_contri:
            nob_dict[nob_key][demand_col] += quantity / batch.batch_size

    df = defaultdict(list)
    for nob_key in nob_dict:
        df["material_id"].append(nob_key[0])
        df["batch_size"].append(nob_key[1])
        df["alt_bom"].append(nob_key[2])
        df["ca_mode"].append(nob_key[3])
        df["total_nob"].append(nob_dict[nob_key]["total_nob"])
        df["nob_to_be_scheduled"].append(nob_dict[nob_key]["nob_to_be_scheduled"])
        for col in demand_cols:
            df[col].append(nob_dict[nob_key][col])

    return pd.DataFrame(df)


def inventory_and_coefficients(
    lp_model: pulp.LpProblem,
    lp_output: Dict[str, float],
    products: Dict[str, Material],
    batching: List[str],
    max_dependancy_levels: int,
    demand: pd.DataFrame,
    inventory_status: pd.DataFrame,
) -> pd.DataFrame:
    # Use dictionary version for easy indexing
    lp_model = lp_model.to_dict()

    # Calculate inventory consumed by each index_*_batch_* variables
    consumption_by_key = multidict(2, float)
    for constraint in filter(
        lambda x: x["name"].startswith("inventoryUsed_"), lp_model["constraints"]
    ):
        rmpm_id = constraint["name"].split("_", maxsplit=1)[1]
        for coefficient in constraint["coefficients"]:
            splitted_name = split_pids(coefficient["name"], maxsplit=-1)
            demand_index = splitted_name[0]
            batch_size = splitted_name[2]
            consumption_by_key[(demand_index, batch_size)][rmpm_id] += (
                lp_output[coefficient["name"]] * coefficient["value"]
            )

    # Sort by coefficient value and group by aux_index_batch_* variables
    obj_coefficients = sorted(
        filter(
            lambda x: x["name"].startswith("aux_"),
            lp_model["objective"]["coefficients"],
        ),
        key=lambda x: x["value"],
        reverse=True,
    )
    aux_values = multidict(2, float)
    for obj_coefficient in obj_coefficients:
        _, demand_index, batch_size, col_name = split_pids(
            obj_coefficient["name"], maxsplit=3
        )
        aux_values[(demand_index, batch_size)][col_name] = lp_output[
            obj_coefficient["name"]
        ]

    # Calculate inventory consumed by aux_index_batch_* variables
    running_inventory = defaultdict(
        float, dict(zip(inventory_status["material_id"], inventory_status["initial"]))
    )
    cumulative_feasibility = defaultdict(list)
    final_feasibility = defaultdict(list)
    zero_feasibility_vars = []

    for obj_coefficient in obj_coefficients:
        # Split to get param values
        _, demand_index, batch_size, col_name = split_pids(
            obj_coefficient["name"], maxsplit=3
        )
        curr_demand = demand.loc[demand["true_index"] == int(demand_index)]

        # Calculate scale factor
        curr_var_value = aux_values[(demand_index, batch_size)][col_name]
        curr_var_sum = sum(aux_values[(demand_index, batch_size)].values())

        # Add basic parameters to the feasibility check
        material_id = curr_demand[DemandST.cols.material_id].iloc[0]
        curr_demand_value = curr_demand[col_name].iloc[0]

        cumulative_feasibility["var_name"].append(obj_coefficient["name"])
        cumulative_feasibility["material_id"].append(material_id)
        cumulative_feasibility["batch_size"].append(float(batch_size))
        cumulative_feasibility["col_name"].append(col_name)
        cumulative_feasibility["col_demand"].append(curr_demand_value)
        cumulative_feasibility["coefficient"].append(obj_coefficient["value"])
        cumulative_feasibility["var_value"].append(curr_var_value)

        if curr_var_sum > 0 and curr_var_value > 0:
            # Calculate current consumption
            curr_consumtion = {}
            for rmpm_id, qnty in consumption_by_key[(demand_index, batch_size)].items():
                curr_consumtion[rmpm_id] = qnty * (curr_var_value / curr_var_sum)

            # Subtract from final inventory
            for rmpm_id, qnty in curr_consumtion.items():
                running_inventory[rmpm_id] -= qnty

            cumulative_feasibility["consumption"].append(curr_consumtion)
            cumulative_feasibility["feasible"].append(np.nan)
            cumulative_feasibility["missing"].append(np.nan)

        else:
            if curr_demand_value > 0:
                # Check whether we can produce a single unit of current aux_ variable
                consumed, feasible, missing_info = recursive_inventory_check(
                    products=products,
                    inventory=running_inventory,
                    material_id=material_id,
                    max_level=max_dependancy_levels,
                    quantity=1,
                    batching=batching,
                )

                zero_feasibility_vars.append(
                    [obj_coefficient["name"], material_id, batch_size]
                )

                # Label it
                if feasible:
                    missing_info = np.nan
                else:
                    consumed = np.nan

                cumulative_feasibility["consumption"].append(consumed)
                cumulative_feasibility["feasible"].append(feasible)
                cumulative_feasibility["missing"].append(missing_info)

            else:
                cumulative_feasibility["consumption"].append(np.nan)
                cumulative_feasibility["feasible"].append(np.nan)
                cumulative_feasibility["missing"].append(np.nan)

    # Calculate by final consumption
    final_feasibility = []
    for var_name, material_id, batch_size in zero_feasibility_vars:
        # Check whether we can produce a single unit of current aux_ variable
        final_inventory = defaultdict(
            float,
            dict(zip(inventory_status["material_id"], inventory_status["after_used"])),
        )
        consumed, feasible, missing_info = recursive_inventory_check(
            products=products,
            inventory=final_inventory,
            material_id=material_id,
            max_level=max_dependancy_levels,
            quantity=1,
            batching=batching,
        )

        # Label it
        if feasible:
            missing_info = np.nan
        else:
            consumed = np.nan

        final_feasibility.append([var_name, consumed, feasible, missing_info])

    # Coefficient values
    df1 = pd.DataFrame(cumulative_feasibility)
    df2 = pd.DataFrame(
        final_feasibility, columns=["var_name", "consumption", "feasible", "missing"]
    )

    df = pd.merge(
        df1, df2, how="left", on="var_name", suffixes=("_by_cumu", "_by_final")
    )

    # Give reason if demand is empty
    df["reason"] = np.nan
    df.loc[df["col_demand"] <= 0, "reason"] = "No demand"

    # Give reason if demand is fulfilled
    def _fulfil_demand(g):
        total_feaasible_qnty = (g["batch_size"] * g["var_value"]).sum()
        m1_demand = (
            g.loc[g["col_name"] == DemandST.cols.m1_crit, "col_demand"].iloc[0]
            + g.loc[g["col_name"] == DemandST.cols.m1_std, "col_demand"].iloc[0]
        )

        if np.isclose(total_feaasible_qnty, m1_demand):
            g.loc[g["reason"].isna(), "reason"] = "Demand fulfilled"
        elif total_feaasible_qnty > m1_demand:
            g.loc[g["reason"].isna(), "reason"] = "Over commit"
        elif 0 < total_feaasible_qnty < m1_demand:
            g.loc[g["reason"].isna(), "reason"] = "Under commit"

        return g

    df = df.groupby("material_id", group_keys=True).apply(_fulfil_demand)

    return df
