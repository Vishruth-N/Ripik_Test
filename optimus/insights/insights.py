"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import numpy as np
import pandas as pd
from ..utils.constants import DroppedReason
from .stats import quantity_stats, utility_stats, block_stats


def obtain_insights(
    machine_view: pd.DataFrame,
    product_view: pd.DataFrame,
    production_view: pd.DataFrame,
    commit_view: pd.DataFrame,
    products: pd.DataFrame,
    config: pd.DataFrame,
):
    # Remove missing info from commit view
    commit_view = commit_view[
        ~commit_view["reason_category"].isin(
            [
                DroppedReason.MISSING_INFO.name,
                DroppedReason.COVERED_DEMAND.name,
                DroppedReason.NO_DEMAND.name,
            ]
        )
    ]

    # Modify commit view
    fg_demand = commit_view.copy()
    fg_demand.loc[:, "curr_qnty_demanded"] *= fg_demand["material_id"].apply(
        lambda x: products[x].count_factor
    )
    fg_demand.loc[:, "total_qnty_demanded"] *= fg_demand["material_id"].apply(
        lambda x: products[x].count_factor
    )

    # Demand insights
    demand_insights = {}
    demand_insights["Total Demanded"] = float(commit_view["total_qnty_demanded"].sum())
    demand_insights["Total M1 Demanded"] = float(
        commit_view["curr_qnty_demanded"].sum()
    )
    demand_insights["Total Feasible"] = float(commit_view["qnty_feasible"].sum())
    demand_insights["Last end time"] = float(product_view["rel_end_time"].max())
    demand_insights["Total_mios_demanded"] = (
        float(fg_demand["total_qnty_demanded"].sum()) / 1000000
    )
    demand_insights["M1_mios_demanded"] = (
        float(fg_demand["curr_qnty_demanded"].sum()) / 1000000
    )

    # Numeric stats
    period_insights = {}
    for period, (start_period, end_period) in config["periods"].items():
        if np.isinf(end_period):
            end_period = machine_view["rel_end_time"].max() + 1

        period_insights[period] = {
            "start_period": start_period,
            "end_period": end_period,
            "quantity_stats": quantity_stats(
                production_view=production_view,
                period=period,
            ),
            "utility_stats": utility_stats(
                machine_view=machine_view,
                start_period=start_period,
                end_period=end_period,
            ),
            "block_stats": block_stats(
                product_view=product_view,
                start_period=start_period,
                end_period=end_period,
            ),
        }

    # Return insights
    insights = {"demand_insights": demand_insights, "period_insights": period_insights}

    return insights
