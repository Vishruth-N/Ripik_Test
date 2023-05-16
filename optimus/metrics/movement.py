"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from typing import Dict, Any

import pandas as pd

from optimus.metrics._core import Metric
from optimus.utils.structs import CrossBlockST
from optimus.utils.views import view_by_product
from optimus.machines.machine import Machine


class CrossBlockPenalty(Metric):
    def __init__(self):
        super().__init__(name="cross_block_penalty", maximise=False)

    def __call__(
        self,
        df_crossblock_penalties: pd.DataFrame,
        machines: Dict[str, Machine] = None,
        product_view: pd.DataFrame = None,
        *args,
        **kwargs
    ):
        if product_view is None:
            product_view = view_by_product(machines=machines)

        if product_view.empty:
            return 0
        total_value = 0
        for _, group in product_view.sort_values(["batch_id", "sequence"]).groupby(
            "batch_id"
        ):
            prev_block = None
            for row in group.itertuples():
                block = machines[row.machine_id].block_id

                if prev_block is not None:
                    if not df_crossblock_penalties[            # <--- This is the snippet that was throwing the error for baska, added the if statement to check if the df is empty.
                        (
                            df_crossblock_penalties[CrossBlockST.cols.from_block_id]
                            == prev_block
                        )
                        & (
                            df_crossblock_penalties[CrossBlockST.cols.to_block_id]
                            == block
                        )
                    ].empty:
                        total_value += df_crossblock_penalties[
                            (
                                df_crossblock_penalties[CrossBlockST.cols.from_block_id]
                                == prev_block
                            )
                            & (
                                df_crossblock_penalties[CrossBlockST.cols.to_block_id]
                                == block
                            )
                        ][CrossBlockST.cols.penalty].iloc[0]


                prev_block = block

        return total_value
