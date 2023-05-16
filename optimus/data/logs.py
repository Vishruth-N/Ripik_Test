"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any

from ..utils.constants import MaterialType
from ..elements.product import Material
from ..utils.structs import DemandST

import logging

logger = logging.getLogger(__name__)


def _calculate_mios(
    df: pd.DataFrame,
    products: Dict[str, Material],
    material_col: str,
    quantity_cols: List[str],
) -> float:
    if len(df) == 0:
        return 0.0

    return (
        df[quantity_cols].sum(axis=1)
        * df[material_col].apply(lambda x: products[x].count_factor)
    ).sum() / 1000000


def log_mios_in_demand(
    df_demand: pd.DataFrame,
    products: Dict[str, Material],
    pulling_months: List[str],
) -> None:
    # Define columns and calculate mios
    material_col = DemandST.cols.material_id
    m1_mios_val = _calculate_mios(
        df_demand, products, material_col, DemandST.get_m1_cols()
    )
    all_mios_val = _calculate_mios(df_demand, products, material_col, pulling_months)

    # Log mios
    logger.debug(f"M1 mios: {m1_mios_val}")
    logger.debug(f"All mios: {all_mios_val}")


def log_demand_description(demand: pd.DataFrame) -> None:
    num_demand_items = len(demand)
    unique_codes = demand[DemandST.cols.material_id].nunique()

    logger.debug(f"Num demand items: {num_demand_items}")
    logger.debug(f"Unique codes: {unique_codes}")


def log_mios_in_feasible(df_feasible_vs_demand: pd.DataFrame) -> None:
    # Unique FGs
    num_unique_fgs = len(
        df_feasible_vs_demand.loc[
            df_feasible_vs_demand["total_feasible"] > 0, "material_id"
        ].unique()
    )
    logger.debug(f"Number of unique FGs feasible: {num_unique_fgs}")

    # M1 vs rest mios split
    m1_feasible_mios = (
        df_feasible_vs_demand["m1_feasible"] * df_feasible_vs_demand["count_factor"]
    ).sum() / 1e6
    total_feasible_mios = (
        df_feasible_vs_demand["total_feasible"] * df_feasible_vs_demand["count_factor"]
    ).sum() / 1e6

    # Unrestricted and restricted mios split
    unrestricted_feasible_mios = (
        df_feasible_vs_demand["unrestricted_feasible"]
        * df_feasible_vs_demand["count_factor"]
    ).sum() / 1e6
    restricted_feasible_mios = (
        df_feasible_vs_demand["restricted_feasible"]
        * df_feasible_vs_demand["count_factor"]
    ).sum() / 1e6

    logger.debug(f"M1 feasible mios: {m1_feasible_mios}")
    logger.debug(f"Total feasible mios: {total_feasible_mios}")
    logger.debug(f"Unrestricted feasible mios: {unrestricted_feasible_mios}")
    logger.debug(f"Restricted feasible mios: {restricted_feasible_mios}")
