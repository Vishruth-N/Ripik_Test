"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from typing import Dict, Any

import pandas as pd

from optimus.metrics._core import Metric
from optimus.utils.views import view_by_product
from optimus.machines.machine import Machine


class TotalCompletionTime(Metric):
    def __init__(self):
        super().__init__(name="total_completion_time", maximise=False)

    def __call__(
        self,
        machines: Dict[str, Machine] = None,
        product_view: pd.DataFrame = None,
        *args,
        **kwargs
    ):
        if product_view is None:
            product_view = view_by_product(machines=machines)

        total_value = product_view["rel_end_time"].max()
        return total_value
