"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from __future__ import annotations
import uuid
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, TypeVar, TYPE_CHECKING

import numpy as np
import pandas as pd

from optimus.utils.general import close_subtract, multidict
from optimus.utils.constants import CAMode, BatchingMode
from optimus.utils.structs import InitialST, DemandST

if TYPE_CHECKING:
    from optimus.machines.machine import Machine
    from optimus.elements.inventory import Inventory
    from optimus.elements.product import Material

import logging

logger = logging.getLogger(__name__)


SelfBatch = TypeVar("SelfBatch", bound="Batch")


def assign_batch_id():
    return str(uuid.uuid4())


class Batch:
    def __init__(
        self,
        product: Material,
        batch_size: float,
        alt_bom: str,
        ca_mode: str,
        quantity: float,
        priority: float,
        due_date: float,
        demand_contri: List[Tuple[str, float]] = [],
        initialised: bool = False,
        alt_recipe: str = None,
        sequence: int = 0,
        machine_id: Optional[Machine] = None,
        start_time: Optional[float] = None,
    ) -> None:
        self.product = product
        self.batch_size = batch_size
        self.alt_bom = alt_bom
        self.ca_mode = ca_mode
        self.quantity = quantity
        self.batch_id = assign_batch_id()

        # Priority data
        self.priority = priority
        self.due_date = due_date
        self.demand_contri = demand_contri

        # Initialization data
        self.initialised = initialised
        self.alt_recipe = alt_recipe
        self.sequence = sequence
        self.machine_id = machine_id
        self.start_time = start_time

        # Requires
        self.indirect_composition = self.product.get_indirect_composition(
            batch_size=self.batch_size, alt_bom=self.alt_bom
        )
        self.requires = {}
        if not self.initialised:
            for component_group, component in self.indirect_composition.items():
                self.requires[component_group] = {
                    "quantity": component["quantity"]
                    * (self.quantity / self.batch_size),
                    "consumes": {
                        "inventory": [],
                        "batches": [],
                    },
                }

        # Impacts
        self.impacts = []

        # Quantity remaining
        self.quantity_remaining = quantity

    def get_impacted_batches(self) -> List[Tuple[str, float]]:
        return self.impacts

    def clear_impacted_batches(self) -> None:
        self.impacts = []

    def get_num_impacted_batches(self) -> int:
        return len(self.impacts)

    def add_impacted_batch(self, impacted_batch_id: str, qnty: float) -> None:
        self.impacts.append((impacted_batch_id, qnty))

    def is_filled(self) -> bool:
        """Check whether the batch has no requirements left"""
        for _, component in self.requires.items():
            if not np.isclose(component["quantity"], 0):
                return False
        return True

    def remaining_requirement(self, component_group: str) -> float:
        """
        Returns quantity that is not fulfilled yet

        Params
        --------------------
        component_group: Component group (indirect)
        """
        assert (
            component_group in self.requires
        ), f"{component_group} is not required in {self.product.material_id}"
        return self.requires[component_group]["quantity"]

    def consume_inventory(
        self, component_group: str, component_id: str, quantity: float
    ) -> None:
        if self.remaining_requirement(component_group) <= 0:
            logger.warn(
                f"Feeding {quantity} of {component_id} to the feasible batch {self.batch_id} when there is no requirement"
            )
            return

        self.requires[component_group]["consumes"]["inventory"].append(
            (component_id, quantity)
        )
        self.requires[component_group]["quantity"] = close_subtract(
            self.requires[component_group]["quantity"], quantity
        )

    def consume_batch(
        self, component_group: str, batch: SelfBatch, quantity: float
    ) -> None:
        if self.remaining_requirement(component_group) <= 0:
            logger.warn(
                f"Feeding {quantity} of {batch.batch_id} to the feasible batch {self.batch_id} when there is no requirement"
            )
            return

        self.requires[component_group]["consumes"]["batches"].append(
            (batch.batch_id, quantity)
        )
        self.requires[component_group]["quantity"] = close_subtract(
            self.requires[component_group]["quantity"], quantity
        )

        batch.add_impacted_batch(self.batch_id, quantity)
        batch.quantity_remaining -= quantity

    def completion_pct(self) -> float:
        """Calculates completion percentage of the batch requirements"""
        n_items = 0
        pct_sum = 0
        for component_group, component in self.requires.items():
            pct_sum += (
                component["quantity"]
                / self.indirect_composition[component_group]["quantity"]
                * (self.quantity / self.batch_size)
            )
            n_items += 1

        if n_items == 0:
            return 100.0

        return (pct_sum / n_items) * 100

    def num_reqd_batches(self) -> int:
        ans = 0
        for _, component in self.requires.items():
            ans += len(component["consumes"]["batches"])
        return ans

    def is_locked(self):
        for _, component in self.requires.items():
            if len(component["consumes"]["batches"]) > 0:
                return True
        return False


def create_batch_graph(
    df_initial_state: Optional[pd.DataFrame],
    demand: pd.DataFrame,
    feasible: pd.DataFrame,
    products: Dict[str, Material],
    inventory: Inventory,
    consume_only_one: bool,
) -> Dict[str, List[Batch]]:
    # Make a marked column for initialization
    if df_initial_state is None:
        df_initial_state = pd.DataFrame(columns=InitialST.get_fields())
    df_initial_state["marked"] = False

    # Create data structure for demand
    demand_cols = DemandST.hilo_priority_cols()
    demand_list = defaultdict(lambda: [0] * len(demand_cols))
    for row in demand.itertuples():
        row = row._asdict()
        demand_key = (row[DemandST.cols.material_id], row[DemandST.cols.ca_mode])
        for i, col in enumerate(demand_cols):
            demand_list[demand_key][i] += row[col]

    # Create individual batches
    all_batches = {}
    for row in feasible[feasible["nob"] > 0].sort_values("batch_size").itertuples():
        row = row._asdict()

        # Calculate number of batches
        product = products[row["material_id"]]
        demand_key = (row["material_id"], row["ca_mode"])
        nob, batch_size = row["nob"], row["batch_size"]
        individuals = []
        if (
            product.batching_mode == BatchingMode.treal.value
            and consume_only_one
            and demand_key in demand_list
        ):
            # Break batch quantities by demand
            available_qnty = nob * batch_size
            for i, demand_qnty in enumerate(demand_list[demand_key]):
                if demand_qnty <= 0:
                    continue
                consume_qnty = min(demand_qnty, available_qnty)
                if consume_qnty > 0:
                    demand_list[demand_key][i] -= consume_qnty
                    available_qnty -= consume_qnty
                    individuals.extend(
                        [
                            (batch_size, demand_cols[i])
                            for _ in range(int(consume_qnty // batch_size))
                        ]
                    )
                    if consume_qnty % batch_size > 0:
                        individuals.append((consume_qnty % batch_size, demand_cols[i]))
                if available_qnty == 0:
                    break
            nob = available_qnty / batch_size

        # Break nob into individual batches
        proportionate_batch, num_whole_batches = np.modf(nob)
        individuals.extend([(batch_size, None) for _ in range(int(num_whole_batches))])
        if not np.isclose(proportionate_batch, 0, atol=1e-7):
            individuals.append((proportionate_batch * batch_size, None))

        curr_initial_state = df_initial_state[
            (df_initial_state[InitialST.cols.material_id] == product.material_id)
            & (df_initial_state[InitialST.cols.batch_size] == batch_size)
            & (df_initial_state[InitialST.cols.alt_bom] == row["alt_bom"])
            & (df_initial_state[InitialST.cols.ca_mode] == row["ca_mode"])
            & (~df_initial_state["marked"])
        ]
        marked_indices = []
        for i, individual in enumerate(individuals):
            quantity, demand_col = individual
            initialised = False
            initialisation_params = {}

            # Initialize or not?
            if i < len(curr_initial_state):
                initialised = True
                initialisation_params = {
                    "alt_recipe": curr_initial_state.iloc[i]["alt_recipe"],
                    "sequence": curr_initial_state.iloc[i]["sequence"],
                    "machine_id": curr_initial_state.iloc[i]["machine_id"],
                    "start_time": curr_initial_state.iloc[i]["start_time"],
                }
                marked_indices.append(curr_initial_state.index[i])

            # Get demand contribution of this batch
            demand_contri = []
            if demand_col is None:
                if demand_key in demand_list:
                    available_qnty = quantity
                    for i, demand_qnty in enumerate(demand_list[demand_key]):
                        consume_qnty = min(demand_qnty, available_qnty)
                        if consume_qnty > 0:
                            demand_list[demand_key][i] -= consume_qnty
                            available_qnty -= consume_qnty
                            demand_contri.append((demand_cols[i], consume_qnty))
                        if available_qnty == 0:
                            break
            else:
                demand_contri.append((demand_col, quantity))

            if (
                not initialised
                and row["ca_mode"] != CAMode.CAI.value
                and len(demand_contri) == 0
            ):
                logger.warn(
                    f"Skipping unnecessary batch: {quantity} units of {product.material_id}"
                )
                continue

            batch = Batch(
                product=product,
                batch_size=batch_size,
                alt_bom=row["alt_bom"],
                ca_mode=row["ca_mode"],
                quantity=quantity,
                priority=row["priority"],
                due_date=row["due_date"],
                demand_contri=demand_contri,
                initialised=initialised,
                **initialisation_params,
            )
            all_batches[batch.batch_id] = batch

        df_initial_state.loc[marked_indices, "marked"] = True

    # 1. Check for fulfilment (initialized or primary level intm)
    filled_batches = multidict(2, list)
    unfilled_batches = multidict(2, list)
    for batch in all_batches.values():
        if batch.is_filled():
            filled_batches[batch.ca_mode][
                (batch.product.material_id, batch.batch_size, batch.alt_bom)
            ].append(batch)
        else:
            unfilled_batches[batch.ca_mode][
                (batch.product.material_id, batch.batch_size, batch.alt_bom)
            ].append(batch)

    # 2. Fulfill by inventory
    fulfill_by_inventory(
        filled_batches=filled_batches,
        unfilled_batches=unfilled_batches,
        inventory=inventory,
        consume_only_one=consume_only_one,
    )

    # 3. Fulfill by batches
    fulfill_by_batches(
        filled_batches=filled_batches,
        unfilled_batches=unfilled_batches,
        consume_only_one=consume_only_one,
    )

    # 4. Remove extra batches
    discard_no_impact_intms(
        filled_batches=filled_batches,
    )

    # 5. Perfolate
    proliferate_to_intm(
        filled_batches=filled_batches,
    )

    return filled_batches, unfilled_batches


## HELPER FUNCTIONS


def __get_demand_col_value(batch: Batch) -> float:
    """Take weighted average"""
    demand_cols = DemandST.hilo_priority_cols()
    value = 0
    for demand_col, qnty in batch.demand_contri:
        value += (len(demand_cols) - demand_cols.index(demand_col)) * qnty
    value /= batch.quantity
    return value


def fulfill_by_inventory(
    filled_batches: Dict[str, Dict[Tuple[str, float, str], List[Batch]]],
    unfilled_batches: Dict[str, Dict[Tuple[str, float, str], List[Batch]]],
    inventory: pd.DataFrame,
    consume_only_one: bool,
) -> None:
    # Prioritise filling by CA Mode
    intm_inventory_used = defaultdict(float)
    for ca_mode in [CAMode.CAV.value, CAMode.CAD.value, CAMode.CAI.value]:
        for batch_key, batches in unfilled_batches[ca_mode].items():
            batches.sort(key=__get_demand_col_value, reverse=True)

            filled_batches_in_loop = []
            for batch in batches:
                for component_group, component in batch.requires.items():
                    # skip component group if fulfilled
                    reqd_qnty = component["quantity"]
                    if reqd_qnty <= 0:
                        continue

                    for component_id in batch.indirect_composition[component_group][
                        "component_ids"
                    ]:
                        # skip component if zero inventory
                        available_qnty = (
                            inventory.get_quantity(component_id)
                            - intm_inventory_used[component_id]
                        )
                        if available_qnty <= 0:
                            continue

                        # fulfill inventory condition
                        if available_qnty >= reqd_qnty:
                            intm_inventory_used[component_id] += reqd_qnty
                            batch.consume_inventory(
                                component_group, component_id, reqd_qnty
                            )
                            break

                        else:
                            if not consume_only_one:
                                intm_inventory_used[component_id] += available_qnty
                                batch.consume_inventory(
                                    component_group, component_id, available_qnty
                                )

                # Move from unfilled to filled
                if batch.is_filled():
                    filled_batches_in_loop.append(batch)

            for batch in filled_batches_in_loop:
                unfilled_batches[ca_mode][batch_key].remove(batch)
                filled_batches[ca_mode][batch_key].append(batch)


def fulfill_by_batches(
    filled_batches: Dict[str, Dict[Tuple[str, float, str], List[Batch]]],
    unfilled_batches: Dict[str, Dict[Tuple[str, float, str], List[Batch]]],
    consume_only_one: bool,
):
    # Use INTM batches
    feeders_queue = []
    for batches in filled_batches[CAMode.CAI.value].values():
        feeders_queue.extend(batches)

    # Prioritise INTM which has less requirement of batches
    feeders_queue.sort(key=lambda batch: batch.num_reqd_batches())

    # Run queue till all INTMs are used
    feeder_qi = 0
    while feeder_qi < len(feeders_queue):
        feeder_batch = feeders_queue[feeder_qi]
        feeder_qi += 1

        # Iterate over all possible impacted keys
        for parent_key in feeder_batch.product.get_inverse_bom(
            batch_size=feeder_batch.batch_size, alt_bom=feeder_batch.alt_bom
        ):
            # Stop searching other keys if feeder batch cant feed anymore
            if feeder_batch.quantity_remaining <= 0:
                break

            (
                feeder_material_id,
                feeder_batch_size,
                feeder_alt_bom,
                feeder_comp_group,
            ) = parent_key
            consumer_batch_key = (feeder_material_id, feeder_batch_size, feeder_alt_bom)

            # Iterate over possible consumer batches prioritised by CA Mode
            for consumer_ca_mode in [
                CAMode.CAV.value,
                CAMode.CAD.value,
                CAMode.CAI.value,
            ]:
                # Stop feeding if feeder batch cant feed anymore
                if feeder_batch.quantity_remaining <= 0:
                    break

                consumer_batches = unfilled_batches[consumer_ca_mode][
                    consumer_batch_key
                ]
                consumer_batches.sort(
                    key=lambda batch: (
                        __get_demand_col_value(batch),
                        batch.completion_pct(),
                    ),
                    reverse=True,
                )

                filled_batches_in_loop = []
                for consumer_batch in consumer_batches:
                    # Stop feeding if feeder batch cant feed anymore
                    if feeder_batch.quantity_remaining <= 0:
                        break

                    # Skip if impacted component group requirement is zero
                    reqd_qnty = consumer_batch.remaining_requirement(feeder_comp_group)
                    if reqd_qnty <= 0:
                        continue

                    # When only one batch can be consumed skip if quantity is not enough
                    if consume_only_one and feeder_batch.quantity_remaining < reqd_qnty:
                        continue

                    # Consume batch
                    consumed_qnty = min(reqd_qnty, feeder_batch.quantity_remaining)
                    consumer_batch.consume_batch(
                        feeder_comp_group, feeder_batch, consumed_qnty
                    )

                    # Update batches
                    if consumer_batch.is_filled():
                        filled_batches_in_loop.append(consumer_batch)

                        if consumer_batch.ca_mode == CAMode.CAI.value:
                            feeders_queue.append(consumer_batch)

                for batch in filled_batches_in_loop:
                    unfilled_batches[consumer_ca_mode][consumer_batch_key].remove(batch)
                    filled_batches[consumer_ca_mode][consumer_batch_key].append(batch)


def discard_no_impact_intms(
    filled_batches: Dict[str, Dict[Tuple[str, float, str], List[Batch]]],
) -> Tuple[List[Batch], List[Batch]]:
    # Obtain perfect CAI batches
    perfect_cai_batches = []
    for batch_key, batches in filled_batches[CAMode.CAI.value].items():
        removed_batches_in_loop = []
        for batch in batches:
            if batch.initialised:
                continue

            # Perfect condition
            if (
                batch.quantity_remaining < batch.quantity
                and batch.get_num_impacted_batches() > 0
            ) or batch.initialised:
                perfect_cai_batches.append(batch)

            else:
                # Remove from filled batches
                removed_batches_in_loop.append(batch)

        for batch in removed_batches_in_loop:
            filled_batches[CAMode.CAI.value][batch_key].remove(batch)

    # Fulfilled final batches
    filled_cad_cav_batches = []
    for ca_mode in filled_batches:
        if ca_mode in [CAMode.CAD.value, CAMode.CAV.value]:
            for batches in filled_batches[ca_mode].values():
                for batch in batches:
                    filled_cad_cav_batches.append(batch)

    # Prepare sets with batch_ids
    dict_batches = {}
    for batch in perfect_cai_batches + filled_cad_cav_batches:
        dict_batches[batch.batch_id] = batch
    perfect_cai_batch_ids = set(batch.batch_id for batch in perfect_cai_batches)
    filled_cad_cav_batch_ids = set(batch.batch_id for batch in filled_cad_cav_batches)

    # Iterate till there is any change in perfect batches
    change_found = True
    while change_found:
        change_found = False

        for batch_id in list(perfect_cai_batch_ids):
            batch = dict_batches[batch_id]
            batch_key = (batch.product.material_id, batch.batch_size, batch.alt_bom)

            new_impact_list = []
            for impacted_batch_id, qnty in batch.get_impacted_batches():
                if (
                    impacted_batch_id in perfect_cai_batch_ids
                    or impacted_batch_id in filled_cad_cav_batch_ids
                ):
                    new_impact_list.append((impacted_batch_id, qnty))

            # Skip good batch
            if len(new_impact_list) == batch.get_num_impacted_batches():
                continue

            # Batch is imperfect if it does not impact anything
            change_found = True
            if len(new_impact_list) == 0:
                perfect_cai_batch_ids.remove(batch_id)
                filled_batches[CAMode.CAI.value][batch_key].remove(batch)

            else:
                batch.clear_impacted_batches()
                for impacted_batch_id, qnty in new_impact_list:
                    batch.add_impacted_batch(impacted_batch_id, qnty)


def proliferate_to_intm(
    filled_batches: Dict[str, Dict[Tuple[str, float, str], List[Batch]]],
) -> Tuple[List[Batch], List[Batch]]:
    # Unmarked CAI batches
    all_batches = {}
    unmarked_cai_batches = []
    for ca_mode in filled_batches:
        for batch_key, batches in filled_batches[ca_mode].items():
            for batch in batches:
                all_batches[batch.batch_id] = batch
                if ca_mode == CAMode.CAI.value:
                    unmarked_cai_batches.append(batch)

    # Iterate till done
    while len(unmarked_cai_batches) > 0:
        new_unmarked_cai_batches = []
        for unmarked_cai_batch in unmarked_cai_batches:
            agg_demand_contri = defaultdict(float)
            canbe_updated = True
            for impacted_batch_id, qnty in unmarked_cai_batch.get_impacted_batches():
                impacted_batch = all_batches[impacted_batch_id]
                demand_contri_length = len(impacted_batch.demand_contri)
                if demand_contri_length == 0:
                    canbe_updated = False
                    break
                # uniformly distribute its parents contribution
                for demand_col, _ in impacted_batch.demand_contri:
                    agg_demand_contri[demand_col] += qnty / demand_contri_length

            if canbe_updated:
                agg_demand_contri = [(k, v) for k, v in agg_demand_contri.items()]
                unmarked_cai_batch.demand_contri = agg_demand_contri
            else:
                new_unmarked_cai_batches.append(unmarked_cai_batch)
        unmarked_cai_batches = new_unmarked_cai_batches
