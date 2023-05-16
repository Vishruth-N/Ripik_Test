"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import pulp
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List

from optimus.utils.constants import BatchingMode, CAMode, ClubbingMode
from optimus.preschedule.utils import (
    FeasibleGraph,
    PerfectBatch,
    combine_pids,
    split_pids,
    MAIN_PREFIX,
    UNKNOWN_TOKEN,
)
from optimus.utils.general import multidict, compare_linking_op
from optimus.elements.product import Material
from optimus.elements.inventory import Inventory
from optimus.utils.structs import *

import logging

logger = logging.getLogger(__name__)


def distribute_inventory(
    demand: pd.DataFrame,
    products: Dict[str, Material],
    inventory: Inventory,
    definite_batch_count: Dict[Tuple[str, float, str, str], int],
    pulling_months: List[str],
    crit_multiplier: float,
    max_dependancy_levels: int,
    rmpm_inventory_check: bool,
    intm_inventory_check: bool,
    solve_time_limit: int = 180,
):
    """
    Takes ground-up approach of making LP model
    """
    # Make model
    model = pulp.LpProblem(name="PSD", sense=pulp.LpMaximize)

    # Bottom up approach to create level dependancies
    restricted_material_ids = set()
    levelUB_constraints = multidict(3, set)
    possible_edges = defaultdict(set)
    lp_variables = defaultdict(set)
    for demand_row in demand.itertuples():
        # Make the demand row as the starting point of level descent
        demand_row = demand_row._asdict()
        vari_names = [
            combine_pids(
                demand_row["true_index"], "DCG", demand_row[DemandST.cols.material_id]
            )
        ]

        # Move level by level
        for level in range(1, max_dependancy_levels + 1):
            new_vari_names = []
            # Iterate over material and batch size combination
            for vari_name in vari_names:
                prev_var_name, material_group, material_id = split_pids(
                    vari_name, maxsplit=2
                )
                prev_var_name_splitted = split_pids(prev_var_name, maxsplit=-1)
                demand_true_index = prev_var_name_splitted[0]
                prev_level = len(prev_var_name_splitted) // 5
                prev_lp_key = None
                if prev_level > 0:
                    prev_material_id = prev_var_name_splitted[5 * prev_level - 3]
                    prev_batch_size = float(prev_var_name_splitted[5 * prev_level - 2])
                    prev_alt_bom = prev_var_name_splitted[5 * prev_level - 1]
                    prev_ca_mode = prev_var_name_splitted[5 * prev_level]
                    prev_lp_key = (
                        prev_material_id,
                        prev_batch_size,
                        prev_alt_bom,
                        prev_ca_mode,
                    )

                # When we need to know relations between level use this commented loop
                # while level:
                #     level -= 1

                # Obtain relevant composition
                ran = False
                for batch_size, alt_bom in products[material_id].iterate_boms():
                    # Check if batch size linking is compatible
                    if level > 1:
                        component = products[prev_lp_key[0]].get_bom(
                            float(prev_lp_key[1]), prev_lp_key[2]
                        )[material_group]

                        if not compare_linking_op(
                            available=batch_size,
                            needed=component["quantity"],
                            mode=component["indirect"],
                        ):
                            continue

                    # Found required BOM
                    ran = True

                    # Add LP key
                    ca_mode = (
                        demand_row[DemandST.cols.ca_mode]
                        if level == 1
                        else CAMode.CAI.value
                    )
                    curr_lp_key = (material_id, batch_size, alt_bom, ca_mode)
                    lp_variables[curr_lp_key].add(demand_true_index)
                    var_name = combine_pids(vari_name, batch_size, alt_bom, ca_mode)

                    # Make constraints to previous level
                    if level > 1:
                        levelUB_constraints[prev_lp_key][material_group][
                            demand_true_index
                        ].add(curr_lp_key)
                        possible_edges[prev_lp_key].add(curr_lp_key)

                    # Iterate over the composition to prepare for the next level
                    bom = products[material_id].get_bom(
                        batch_size=batch_size, alt_bom=alt_bom
                    )
                    for component_group, component in bom.items():
                        for component_id in component["component_ids"]:
                            if (
                                component["indirect"] > 0
                                and level < max_dependancy_levels
                                and not products[material_id].is_primitive(
                                    batch_size=batch_size, alt_bom=alt_bom
                                )
                            ):
                                new_vari_names.append(
                                    combine_pids(
                                        var_name,
                                        component_group,
                                        component_id,
                                    )
                                )
                                restricted_material_ids.add(component_id)

                if not ran:
                    raise ValueError(
                        f"{vari_name} didn't extrude below {level} levels. Fix in preprocessing!"
                    )

            # Update var names
            vari_names = new_vari_names

    # Define all regularization penalties here
    MAIN_ALPHA = -0.1
    AUX_ALPHA = -0.1

    # Add batching constraints
    objective_coeffs = []
    mainvars = multidict(4, list)
    rmpm_resource_constraints = defaultdict(list)
    commits = multidict(3, list)
    mainvar_constraints = []

    def _distribute_main_routine(
        product: Material, batch_size: float, alt_bom: str, demand_true_index_str: str
    ):
        if product.batching_mode == BatchingMode.tint.value:
            cat_type = pulp.LpInteger
        else:
            cat_type = pulp.LpContinuous

        # Create main variable
        mainvar_name = combine_pids(
            MAIN_PREFIX,
            demand_true_index_str,
            product.material_id,
            batch_size,
            alt_bom,
            ca_mode,
        )
        definite_count = definite_batch_count.get(
            (product.material_id, batch_size, alt_bom, ca_mode), 0
        )
        mainvar = pulp.LpVariable(
            mainvar_name, lowBound=definite_count, upBound=None, cat=cat_type
        )
        mainvar.setInitialValue(definite_count)

        # Add commits
        if ca_mode == CAMode.CAD.value or ca_mode == CAMode.CAV.value:
            commits[product.material_id][(batch_size, ca_mode)][
                "demand_true_indices"
            ].extend(demand_true_indices)
            commits[product.material_id][(batch_size, ca_mode)]["variables"].append(
                mainvar
            )

        # Add resource consumed contraint
        if rmpm_inventory_check:
            for component_id, component in product.get_bom(batch_size, alt_bom).items():
                if component["indirect"] <= 0 and rmpm_inventory_check:
                    if inventory.get_quantity(component_id) <= 0:
                        mainvar_constraints.append(mainvar == definite_count)

                    else:
                        rmpm_resource_constraints[component_id].append(
                            component["quantity"] * (mainvar - definite_count)
                        )

        # Update mappings
        mainvars[product.material_id][batch_size][alt_bom][ca_mode].append(mainvar)
        objective_coeffs.append((mainvar, MAIN_ALPHA))

    for lp_key, demand_true_indices in lp_variables.items():
        material_id, batch_size, alt_bom, ca_mode = lp_key
        product = products[material_id]

        if product.clubbing_mode == ClubbingMode.clubbing.value:
            _distribute_main_routine(
                product=product,
                batch_size=batch_size,
                alt_bom=alt_bom,
                demand_true_index_str=";".join(demand_true_indices),
            )

        elif product.clubbing_mode == ClubbingMode.standard.value:
            # Iterate over all demand index
            for demand_true_index in demand_true_indices:
                _distribute_main_routine(
                    product=product,
                    batch_size=batch_size,
                    alt_bom=alt_bom,
                    demand_true_index_str=demand_true_index,
                )

        else:
            raise ValueError(f"Invalid clubbing mode: {product.clubbing_mode}")

    # Add main var constraints
    for constraint in mainvar_constraints:
        model += constraint

    for lp_key, definite_count in definite_batch_count.items():
        if lp_key not in lp_variables:
            material_id = lp_key[0]
            if products[material_id].batching_mode == BatchingMode.tint.value:
                cat_type = pulp.LpInteger
            else:
                cat_type = pulp.LpContinuous

            mainvar_name = combine_pids(
                MAIN_PREFIX,
                UNKNOWN_TOKEN,
                material_id,
                lp_key[1],
                lp_key[2],
                lp_key[3],
            )
            mainvar = pulp.LpVariable(
                mainvar_name, lowBound=definite_count, upBound=None, cat=cat_type
            )
            model += mainvar == definite_count

    # Add RMPM resource constraints
    if rmpm_inventory_check:
        for component_id, constraint in rmpm_resource_constraints.items():
            if constraint:
                if inventory.get_quantity(component_id) > 0:
                    # slack_var = pulp.LpVariable(
                    #     name=f"slack-{component_id}", lowBound=0
                    # )
                    # slack_var.setInitialValue(0)
                    # objective_coeffs.append((slack_var, -1e15))
                    model += (
                        pulp.lpSum(constraint) <= inventory.get_quantity(component_id),
                        f"inventoryConsumedRM_{component_id}",
                    )

    # Create objective function along with demand constraints
    objective = []
    demand_constraints = defaultdict(list)
    for material_id in commits:
        pack_size = products[material_id].count_factor

        for commit_key, commit_val in commits[material_id].items():
            batch_size, ca_mode = commit_key
            aux_commit_vars = []
            for demand_true_index in set(commit_val["demand_true_indices"]):
                demand_item = demand[demand["true_index"] == int(demand_true_index)]

                crit_bonus = 1
                for col in reversed(pulling_months):
                    # Create aux variables
                    aux_commit = pulp.LpVariable(
                        name=combine_pids(
                            "aux", demand_true_index, batch_size, ca_mode, col
                        ),
                        lowBound=0,
                        upBound=None,
                        cat=pulp.LpContinuous,
                    )
                    aux_commit.setInitialValue(0)
                    aux_commit_vars.append(aux_commit)
                    objective_coeffs.append((aux_commit, AUX_ALPHA))

                    # Add commit for demand constraints
                    demand_constraints[(demand_true_index, col)].append(aux_commit)

                    # Add priority
                    priority_coeff = max(
                        1,
                        demand_item[DemandST.cols.priority].iloc[0]
                        * crit_bonus
                        * np.log(demand_item[f"fixed_{col}"].iloc[0] * pack_size + 1),
                    )
                    objective.append(priority_coeff * aux_commit)
                    crit_bonus *= crit_multiplier

            # Aux constraint
            model += (
                pulp.lpSum(commit_val["variables"]) * batch_size
                == pulp.lpSum(aux_commit_vars),
                f"aux-constraint-{material_id}-{batch_size}-{ca_mode}",
            )

    # Add demand constraints
    for demand_constraint_key, vars in demand_constraints.items():
        demand_true_index, col = demand_constraint_key
        model += (
            pulp.lpSum(vars)
            <= demand.loc[demand["true_index"] == int(demand_true_index), col].iloc[0],
            f"maxdemand-{demand_true_index}-{col}",
        )

    # Add level constraints
    levelBU_constraints = multidict(3, set)
    for prev_lp_key in levelUB_constraints:
        for component_group in levelUB_constraints[prev_lp_key]:
            for demand_true_index, curr_keys in levelUB_constraints[prev_lp_key][
                component_group
            ].items():
                levelBU_constraints[frozenset(curr_keys)][component_group][
                    demand_true_index
                ].add(prev_lp_key)

    intm_resource_constraints = defaultdict(list)
    level_constraint_no = 0
    for component_keys in levelBU_constraints:
        # Required
        required = []
        for key in component_keys:
            for var in mainvars[key[0]][key[1]][key[2]][key[3]]:
                required.append(var * key[1])

        # Needed
        varname_to_var = {}
        needed_main_vars = set()
        needed = []
        for component_group in levelBU_constraints[component_keys]:
            for demand_true_index, material_keys in levelBU_constraints[component_keys][
                component_group
            ].items():
                for key in material_keys:
                    quantity = products[key[0]].get_bom(
                        batch_size=key[1], alt_bom=key[2]
                    )[component_group]["quantity"]
                    for var in mainvars[key[0]][key[1]][key[2]][key[3]]:
                        varname_to_var[var.name] = var
                        needed_main_vars.add((var.name, quantity))

        for var_name, quantity in needed_main_vars:
            needed.append(varname_to_var[var_name] * quantity)

        # Produced
        produced = []
        unique_component_ids = set([key[0] for key in component_keys])
        for component_id in unique_component_ids:
            produced_var = pulp.LpVariable(
                name=f"intminv-{component_id}-{level_constraint_no}",
                lowBound=0,
                upBound=None,
                cat=pulp.LpContinuous,
            )
            produced_var.setInitialValue(0)
            intm_resource_constraints[component_id].append(produced_var)
            objective_coeffs.append((produced_var, 0.1))
            produced.append(produced_var)

        model += (
            pulp.lpSum(required) + pulp.lpSum(produced) >= pulp.lpSum(needed),
            f"level-{level_constraint_no}",
        )
        level_constraint_no += 1

    # Add INTM resource constraints
    for component_id, constraint in intm_resource_constraints.items():
        assert component_id in restricted_material_ids
        if intm_inventory_check and constraint:
            model += (
                pulp.lpSum(constraint) <= inventory.get_quantity(component_id),
                f"inventoryConsumed_{component_id}",
            )

    # Add objective
    model += (
        pulp.lpSum(objective) + pulp.LpAffineExpression(objective_coeffs),
        "objective-function",
    )

    # Solve
    logger.info(f"Number of LP variables: {model.numVariables()}")
    logger.info(f"Number of LP constraints: {model.numConstraints()}")

    model.solve(
        pulp.apis.PULP_CBC_CMD(msg=True, timeLimit=solve_time_limit, gapRel=1e-6)
    )
    # model.solve(pulp.apis.PULP_CBC_CMD(msg=True, timeLimit=180, gapRel=0.0, gapAbs=0.0))

    model_output_status = pulp.LpStatus[model.status]
    logger.info(f"Feasible status: {model_output_status}")
    if model_output_status != "Optimal":
        raise TimeoutError("Could not find feasible output!!")

    for var in model.variables():
        if var.name.startswith("slack"):
            if var.varValue != 0:
                raise TimeoutError(
                    "Could not find feasible output - Slack variable not 0!!"
                )

    # Format output
    output = {v.name: v.varValue for v in model.variables()}

    # Make feasible graph
    feasible_graph = FeasibleGraph()
    for u in possible_edges:
        for v in possible_edges[u]:
            feasible_graph.add_directed_edge(u[:3], v[:3])

    return output, model, feasible_graph


def decode_lp_output(
    demand: pd.DataFrame,
    lp_output: Dict[str, float],
    feasible_graph: FeasibleGraph,
    products: Dict[str, Material],
):
    """
    Assure that demand and inventory are copied because they are gonna be modified
    """
    # Reset values
    feasible_graph.reset_node_values()

    # Decode output
    output = []
    for name, nob in lp_output.items():
        name = split_pids(name, maxsplit=-1)

        # Decoding through main variables only
        if name[0] != MAIN_PREFIX:
            continue

        material_id = name[2]
        batch_size = float(name[3])
        alt_bom = name[4]
        ca_mode = name[5]

        # Priority and due date calculation for feasible amount
        if name[1] != UNKNOWN_TOKEN:
            demand_true_indices = name[1].split(";")
            priority = np.mean(
                list(
                    map(
                        lambda i: demand.loc[
                            demand["true_index"] == int(i), DemandST.cols.priority
                        ].iloc[0],
                        demand_true_indices,
                    )
                )
            )
            due_date = np.min(
                list(
                    map(
                        lambda i: demand.loc[
                            demand["true_index"] == int(i), DemandST.cols.due_date
                        ].iloc[0],
                        demand_true_indices,
                    )
                )
            )

        else:
            priority = 1
            due_date = np.nan

        # Append output
        output.append(
            [
                material_id,
                batch_size,
                alt_bom,
                ca_mode,
                nob,
                priority,
                due_date,
            ]
        )

        # Add value to feasible graph if indegree is zero
        node_key = (material_id, batch_size, alt_bom)
        node = feasible_graph.get_node(node_key)
        if node is not None and node.in_degree == 0:
            node.value += nob * batch_size * products[material_id].count_factor

    # Format output
    output = pd.DataFrame(output, columns=get_feasible_columns())

    # Aggregate
    output = output.groupby(
        ["material_id", "batch_size", "alt_bom", "ca_mode"], as_index=False
    ).agg(
        {
            "nob": "sum",
            "priority": "mean",
            "due_date": "min",
        }
    )

    # Demand indices to quantity impact
    feasible_graph.bfs_sum()

    def qnty_impact(x):
        node_key = (
            x["material_id"],
            x["batch_size"],
            x["alt_bom"],
        )
        node = feasible_graph.get_node(node_key)
        scale_factor = 1.0 / 1e6
        if node is None:
            return scale_factor
        else:
            return node.value * scale_factor

    # Add qnty impact
    output["priority"] *= output[["material_id", "batch_size", "alt_bom"]].apply(
        qnty_impact, axis=1
    )

    # Add qnty impact per batch
    # output["priority"] /= output["nob"]

    return output


def get_feasible_columns():
    return [
        "material_id",
        "batch_size",
        "alt_bom",
        "ca_mode",
        "nob",
        "priority",
        "due_date",
    ]


def find_perfect_feasible(
    demand: pd.DataFrame,
    feasible: pd.DataFrame,
    feasible_graph: FeasibleGraph,
    products: Dict[str, Material],
) -> pd.DataFrame:
    # Reset feasible graph
    feasible_graph.reset_node_values()

    # Make individual batches for restricted quantity
    raw_requirements = defaultdict(list)
    restricted_completion = {}
    for row in feasible[feasible["restricted"] > 0].itertuples():
        product = products[row.material_id]
        bom = product.get_bom(batch_size=row.batch_size, alt_bom=row.alt_bom)

        proportionate_batch, num_whole_batches = np.modf(row.restricted)
        individual_batches = [row.batch_size for _ in range(int(num_whole_batches))]
        if not np.isclose(proportionate_batch, 0):
            individual_batches.append(proportionate_batch * row.batch_size)

        material_key = (row.material_id, row.batch_size, row.alt_bom)
        assert material_key not in restricted_completion
        restricted_completion[material_key] = {}

        for batch_no, quantity in enumerate(individual_batches):
            nob = quantity / row.batch_size
            restricted_completion[material_key][batch_no] = nob

            for component_id, component in bom.items():
                if component["indirect"] > 0:
                    raw_requirements[component_id].append(
                        {
                            "material_id": row.material_id,
                            "batch_size": row.batch_size,
                            "alt_bom": row.alt_bom,
                            "quantity": component["quantity"]
                            * (quantity / row.batch_size),
                            "priority": row.priority,
                            "batch_no": batch_no,
                        }
                    )

    # Sort raw requirements
    for component_id in raw_requirements:
        raw_requirements[component_id] = sorted(
            raw_requirements[component_id],
            key=lambda x: (x["quantity"], x["priority"]),
            reverse=True,
        )

    # Initialize perfect feasible
    perfect_feasible = feasible.copy()
    perfect_feasible["unrestricted"] = 0
    perfect_feasible["restricted"] = 0

    # Group raw requirements into batches
    for component_id in raw_requirements:
        batch_sizes = feasible.loc[
            feasible["material_id"] == component_id, "batch_size"
        ].unique()

        local_restricted_completion = defaultdict(set)
        for batch_size in sorted(batch_sizes, reverse=True):
            # Grouping
            perfect_batches = []
            for raw_requirement in raw_requirements[component_id]:
                selected_perfect_batch = None
                merged_in_existing = False

                # TODO: Replace this loop with priority queue for optimization
                for perfect_batch in perfect_batches:
                    # Check whether the current one fits
                    if perfect_batch.get_space_left() >= raw_requirement["quantity"]:
                        selected_perfect_batch = perfect_batch
                        merged_in_existing = True
                        break

                if not merged_in_existing:
                    # Create a new perfect batch
                    selected_perfect_batch = PerfectBatch(
                        component_id=component_id, batch_size=batch_size
                    )
                    perfect_batches.append(selected_perfect_batch)

                # Add to the selected batch
                selected_perfect_batch.add(raw_requirement)

            # Store it sorted by quantity left
            perfect_batches = sorted(
                perfect_batches, key=lambda x: (x.get_space_left(), x.get_value())
            )

            # Add available batches to perfect feasible
            num_batches_reqd = len(perfect_batches)
            num_batches_fulfilled = 0

            # first check unrestricted then restricted
            for col in ["unrestricted", "restricted"]:
                for row in feasible[
                    (feasible["material_id"] == component_id)
                    & (feasible["batch_size"] == batch_size)
                ].itertuples():
                    row = row._asdict()

                    if row[col] > num_batches_reqd - num_batches_fulfilled:
                        batches_taken = num_batches_reqd - num_batches_fulfilled
                    else:
                        batches_taken = np.floor(row[col])

                    perfect_feasible.loc[
                        (perfect_feasible["material_id"] == row["material_id"])
                        & (perfect_feasible["batch_size"] == row["batch_size"])
                        & (perfect_feasible["alt_bom"] == row["alt_bom"]),
                        col,
                    ] += batches_taken
                    num_batches_fulfilled += batches_taken

                    if num_batches_fulfilled >= num_batches_reqd:
                        break

                if num_batches_fulfilled >= num_batches_reqd:
                    break

            # Prune raw requirements from the perfect batches fulfilled
            fulfilled_batch_nos = set()
            for i in range(int(num_batches_fulfilled)):
                for item in perfect_batches[i].get_items():
                    fulfilled_batch_nos.add(item["batch_no"])
                    material_key = (
                        item["material_id"],
                        item["batch_size"],
                        item["alt_bom"],
                    )
                    local_restricted_completion[material_key].add(item["batch_no"])

            new_raw_requirements = []
            for raw_requirement in raw_requirements[component_id]:
                if raw_requirement["batch_no"] not in fulfilled_batch_nos:
                    new_raw_requirements.append(raw_requirement)

            raw_requirements[component_id] = new_raw_requirements

        # Update restricted completion
        for material_key in local_restricted_completion:
            remove_batch_nos = []
            for batch_no in restricted_completion[material_key]:
                if batch_no not in local_restricted_completion[material_key]:
                    remove_batch_nos.append(batch_no)

            for remove_batch_no in remove_batch_nos:
                del restricted_completion[material_key][remove_batch_no]

    # Add completed restrictions
    for material_key in restricted_completion:
        material_id, batch_size, alt_bom = material_key
        if material_id in demand[DemandST.cols.material_id].values:
            num_batches = sum(restricted_completion[material_key].values())

            perfect_feasible.loc[
                (perfect_feasible["material_id"] == material_id)
                & (perfect_feasible["batch_size"] == batch_size)
                & (perfect_feasible["alt_bom"] == alt_bom),
                "restricted",
            ] += num_batches

    # Merge feasible with perfect feasible
    perfect_feasible = perfect_feasible.merge(
        feasible,
        how="outer",
        on=["material_id", "batch_size", "alt_bom"],
        suffixes=("", "_y"),
    )

    # Add unrestricted quantity which are also in demand
    feasible_demand_mask = perfect_feasible["material_id"].isin(
        demand[DemandST.cols.material_id]
    )
    perfect_feasible.loc[feasible_demand_mask, "unrestricted"] = np.maximum(
        perfect_feasible.loc[feasible_demand_mask, "unrestricted"],
        perfect_feasible.loc[feasible_demand_mask, "unrestricted_y"].fillna(0),
    )

    # Readjust priority
    non_zero_feasible_mask = (
        perfect_feasible["unrestricted_y"] + perfect_feasible["restricted_y"] > 0
    )
    perfect_feasible.loc[non_zero_feasible_mask, "priority"] *= (
        perfect_feasible.loc[non_zero_feasible_mask, "unrestricted"]
        + perfect_feasible.loc[non_zero_feasible_mask, "restricted"]
    ) / (
        perfect_feasible.loc[non_zero_feasible_mask, "unrestricted_y"]
        + perfect_feasible.loc[non_zero_feasible_mask, "restricted_y"]
    )
    perfect_feasible["due_date"] = np.minimum(
        perfect_feasible["due_date"].fillna(np.inf),
        perfect_feasible["due_date_y"].fillna(np.inf),
    )

    perfect_feasible = perfect_feasible[get_feasible_columns()]
    return perfect_feasible
