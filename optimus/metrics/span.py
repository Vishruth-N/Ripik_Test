"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from typing import Dict, Any

import numpy as np
import pandas as pd

from optimus.metrics._core import Metric
from optimus.utils.constants import MaterialType
from optimus.utils.general import merge_intervals
from optimus.utils.views import view_by_product, view_by_quantity, view_by_production
from optimus.machines.machine import Machine
from optimus.preschedule.batching import Batch


class WaitingTime(Metric):
    def __init__(self):
        super().__init__(name="waiting_time", maximise=False)

    def __call__(
        self,
        machines: Dict[str, Machine] = None,
        product_view: pd.DataFrame = None,
        *args,
        **kwargs
    ):
        if product_view is None:
            product_view = view_by_product(machines=machines)

        waiting_intervals = []
        for _, group in product_view.sort_values(["batch_id", "sequence"]).groupby(
            "batch_id"
        ):
            prev_end_time = None
            for row in group.itertuples():
                if prev_end_time is None:
                    prev_end_time = row.rel_end_time
                    continue

                if row.rel_start_time - prev_end_time > 0:
                    waiting_intervals.append((prev_end_time, row.rel_start_time))

                prev_end_time = max(prev_end_time, row.rel_end_time)

        waiting_intervals = merge_intervals(waiting_intervals)

        total_value = sum(map(lambda x: x[1] - x[0], waiting_intervals))
        return total_value


class ProductionByTime(Metric):
    def __init__(self):
        super().__init__(name="production_by_time", maximise=True)

    def __call__(
        self,
        machines: Dict[str, Machine] = None,
        quantity_view: pd.DataFrame = None,
        *args,
        **kwargs
    ):
        if quantity_view is None:
            quantity_view = view_by_quantity(
                machines=machines,
            )

        total_value = 0
        for _, group in quantity_view.groupby("batch_id"):
            priority = group["priority"].iloc[0]
            quantity_processed = float(group["quantity_processed"].iloc[0])
            if not np.isnan(priority):
                total_value += (quantity_processed * priority) / group[
                    "rel_end_time"
                ].iloc[0]

        return total_value


class AbsoluteProduce(Metric):
    def __init__(self):
        super().__init__(name="absolute_produce", maximise=True)

    def __call__(
        self,
        machines: Dict[str, Machine] = None,
        config: Dict[str, Any] = None,
        production_view: pd.DataFrame = None,
        *args,
        **kwargs
    ):
        if production_view is None:
            production_view = view_by_production(machines=machines, config=config)

        total_value = (
            production_view.loc[
                (production_view["material_type"] == MaterialType.fg.value),
                "quantity_counted_(M1)",
            ].sum()
            / 1e6
        )

        return total_value


class DueDatePenalty(Metric):
    def __init__(self):
        super().__init__(name="due_date_penalty", maximise=False)

    def __call__(
        self,
        machines: Dict[str, Machine] = None,
        quantity_view: pd.DataFrame = None,
        *args,
        **kwargs
    ):
        if quantity_view is None:
            quantity_view = view_by_quantity(machines=machines)

        # inverse otif
        total_value = 0
        for _, group in quantity_view.groupby("batch_id"):
            due_date = group["due_date"].iloc[0]
            priority = group["priority"].iloc[0]
            end_time = group["rel_end_time"].iloc[0]
            if np.isnan(due_date) or np.isnan(priority):
                continue
            # total_value += row.priority * (row.qnty_demanded - row.qnty_produced + 0.01) * max(0, row.last_end_date - row.due_date)
            total_value += priority * (end_time - due_date)

        return total_value
