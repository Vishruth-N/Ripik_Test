"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from typing import List, Tuple, Any, Dict

from pandas import DataFrame

from optimus.metrics._core import Metric
from optimus.metrics.basic import TotalCompletionTime
from optimus.metrics.span import (
    WaitingTime,
    ProductionByTime,
    AbsoluteProduce,
    DueDatePenalty,
)
from optimus.metrics.movement import CrossBlockPenalty


def _initialize_metrics_dict() -> Dict[str, List[Metric]]:
    all_metrics = [
        TotalCompletionTime(),
        WaitingTime(),
        ProductionByTime(),
        AbsoluteProduce(),
        DueDatePenalty(),
        CrossBlockPenalty(),
    ]

    metrics_dict = {"gain_metrics": [], "penalty_metrics": []}
    for metric in all_metrics:
        if metric.maximise:
            metrics_dict["gain_metrics"].append(metric)
        else:
            metrics_dict["penalty_metrics"].append(metric)

    return metrics_dict


class Objective:
    _METRICS_DICT = _initialize_metrics_dict()

    def __init__(
        self,
        objective_coeffs: Dict[str, float] = {},
        df_crossblock_penalties: DataFrame = None,
    ) -> None:
        self.objective_coeffs = objective_coeffs

        # Fixed params
        self.fixed_params = {"df_crossblock_penalties": df_crossblock_penalties}

    @classmethod
    def get_metrics(cls) -> Dict[str, List[Metric]]:
        return cls._METRICS_DICT

    def __call__(
        self, **metric_params: Any
    ) -> Tuple[float, Dict[str, Dict[str, float]]]:
        # Get metrics dict
        metrics_dict = self.get_metrics()

        # Calculate total score
        component_values = {"gain_components": {}, "penalty_components": {}}
        total_score = 0

        # Metric params
        metric_params.update(self.fixed_params)

        # Gain components
        for metric in metrics_dict["gain_metrics"]:
            coeff = self.objective_coeffs.get(metric.name, 0)
            if coeff == 0:
                continue

            curr_value = metric(**metric_params)
            total_score += coeff * curr_value
            component_values["gain_components"][metric.name] = curr_value

        # Penalty components
        for metric in metrics_dict["penalty_metrics"]:
            coeff = self.objective_coeffs.get(metric.name, 0)
            if coeff == 0:
                continue

            curr_value = metric(**metric_params)
            total_score -= coeff * curr_value
            component_values["penalty_components"][metric.name] = curr_value

        return total_score, component_values
