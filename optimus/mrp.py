"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict

import numpy as np
import pandas as pd

from optimus.elements.product import Material
from optimus.utils.structs import BOMST, PhantomST


class MRP:
    def __init__(
        self,
        inventory: pd.DataFrame,
        products: Dict[str, Material],
        product_view: pd.DataFrame,
        execution_start: datetime,
        df_phantom_items: Optional[pd.DataFrame] = None,
    ):
        # Inventory and params
        self.inventory = inventory
        self.execution_start = execution_start

        # Initialize
        self.df_finished = self.__initialize(
            products=products,
            product_view=product_view,
            df_phantom_items=df_phantom_items,
        )

    def __initialize(
        self,
        products: Dict[str, Material],
        product_view: pd.DataFrame,
        df_phantom_items: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # Preprocess phantom items
        phantom_items = set()
        if df_phantom_items is not None:
            phantom_items = set(df_phantom_items[PhantomST.cols.material_id])

        # Preprocess inventory
        curr_qnty = defaultdict(
            float, self.inventory.groupby("material_id")["quantity"].sum().to_dict()
        )

        # TODO: Remove initialized batch IDs
        finish_df_data = []
        for row in (
            product_view[product_view["sequence"] == 0]
            .sort_values(["start_time", "end_time"])
            .itertuples()
        ):
            # Find required quantity
            product = products[row.product_id]
            for component in product.get_bom(
                batch_size=row.batch_size, alt_bom=row.alt_bom
            ).values():
                # INTM component are not considered in MRP
                if component["indirect"] > 0:
                    continue

                # Phantom item component group can be skipped
                phantom_item_found = False
                for component_id in component["component_ids"]:
                    if component_id in phantom_items:
                        phantom_item_found = True
                        break

                if phantom_item_found:
                    continue

                # Use current inventory
                reqd_qnty = component["quantity"] * (
                    row.quantity_processed / row.batch_size
                )
                for component_id in component["component_ids"]:
                    if curr_qnty[component_id] - reqd_qnty < 0:
                        reqd_qnty -= curr_qnty[component_id]
                        curr_qnty[component_id] = 0
                    else:
                        curr_qnty[component_id] -= reqd_qnty
                        reqd_qnty = 0

                    if np.isclose(reqd_qnty, 0):
                        break

                # Order material if it falls short of requirement
                if not np.isclose(reqd_qnty, 0):
                    finish_df_data.append(
                        [
                            component["component_ids"][0],
                            row.start_time,
                            reqd_qnty,
                        ]
                    )

        # Create RM/PM finished dataframe
        df_finished = pd.DataFrame(
            finish_df_data, columns=["material_id", "date_finished", "qnty_required"]
        )

        # Min qnty order
        min_qnty_order = {}
        for product in products.values():
            for batch_size, alt_bom in product.iterate_boms():
                bom = product.get_bom(batch_size=batch_size, alt_bom=alt_bom)
                for component in bom.values():
                    for component_id in component["component_ids"]:
                        if component_id not in min_qnty_order:
                            min_qnty_order[component_id] = component["quantity"]
                        else:
                            min_qnty_order[component_id] = min(
                                min_qnty_order[component_id], component["quantity"]
                            )

        if not df_finished.empty:
            df_finished["qnty_ordered"] = df_finished.apply(
                lambda row: max(
                    row["qnty_required"], min_qnty_order[row["material_id"]]
                ),
                axis=1,
            )

        return df_finished

    def get_pr_material(self, material_id: str, sustaining_days: float):
        window_start = self.execution_start
        window_end = self.execution_start + pd.Timedelta(days=sustaining_days)
        df = self.df_finished.copy()
        df = df[df["material_id"] == material_id]
        df = df[df["date_finished"] < window_end]
        df = df[df["date_finished"] >= window_start]
        if len(df) == 0:
            return ()
        else:
            return (min(df["date_finished"]), sum(df["qnty_ordered"]))

    def get_pr_all(self, sustaining_days: float):
        window_start = self.execution_start
        window_end = self.execution_start + pd.Timedelta(days=sustaining_days)

        df = (self.df_finished).copy()
        df = df[df["date_finished"] < window_end]
        df = df[df["date_finished"] >= window_start]
        if len(df) == 0:
            return {}
        mat_unq_consumed = np.unique(df["material_id"])
        df2 = pd.DataFrame(
            columns=["material_id", "date_first_finished", "total_qty_ordered"]
        )
        for x in range(len(mat_unq_consumed)):
            df2.loc[x] = [
                mat_unq_consumed[x],
                min(df[df["material_id"] == mat_unq_consumed[x]]["date_finished"]),
                sum(df[df["material_id"] == mat_unq_consumed[x]]["qnty_ordered"]),
            ]
        df2 = df2.sort_values(by=["date_first_finished"])

        out_dict = {
            str(key): (
                str(
                    df2[df2["material_id"] == key].iloc[0]["date_first_finished"].date()
                ),
                df2[df2["material_id"] == key].iloc[0]["total_qty_ordered"].round(2),
            )
            for key in df2["material_id"]
        }

        return out_dict
