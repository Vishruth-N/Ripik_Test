"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import datetime

from optimus.utils.constants import (
    BatchSizeLinking,
    BatchingMode,
    ClubbingMode,
    MaterialType,
    ProductionStrategy,
    ResourceType,
    MachineType,
    CAMode,
)
from optimus.loaders.base import BaseLoader, PandasFilePath
from optimus.utils.general import DSU
from optimus.utils.structs import *

import logging

logger = logging.getLogger(__name__)


class SunPharmaBaskaLoader(BaseLoader):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def required_keys(self) -> List[str]:
        return [
            "recipe",
            "plant_map",
            "forecasted_demand",
        ]

    @property
    def optional_keys(self) -> List[str]:
        return ["holdtime_constraint"]

    @BaseLoader.validate_keys
    def load_all(
        self, data_files: Dict[str, PandasFilePath]
    ) -> Dict[str, pd.DataFrame]:
        df_inventory = pd.DataFrame(columns=InventoryST.get_fields()).astype(
            InventoryST.get_dtypes()
        )

        df_recipe = self._load_recipe(data_files["recipe"])
        df_plant_map = self._load_plant_map(data_files["plant_map"])
        df_bom = self._create_bom(df_recipe)
        df_products_desc = self._create_product_desc(df_recipe)
        df_forecasted_demand = self._create_forecasted_demand(df_products_desc)

        df_recipe = self._filter_recipe(df_recipe)
        df_room_changeover = None
        df_crossblock_penalties = pd.DataFrame(columns=CrossBlockST.get_fields()).astype(
            CrossBlockST.get_dtypes()
        )
        df_phantom_items = None
        df_machine_changeover = None
        df_machine_availability = None
        df_procurement_plan = None

        if "holdtime_constraint" in data_files:
            df_holdtime = self._load_holdtime_constraint(
                data_files["holdtime_constraint"]
            )

        # Return output
        output = {
            "df_forecasted_demand": df_forecasted_demand,
            "df_inventory": df_inventory,
            "df_procurement_plan": df_procurement_plan,
            "df_products_desc": df_products_desc,
            "df_bom": df_bom,
            "df_recipe": df_recipe,
            "df_plant_map": df_plant_map,
            "df_room_changeover": df_room_changeover,
            "df_crossblock_penalties": df_crossblock_penalties,
            "df_phantom_items": df_phantom_items,
            "df_machine_changeover": df_machine_changeover,
            "df_machine_availability": df_machine_availability,
            "df_initial_state": None,
        }

        return output

    def _load_recipe(self, input_data: PandasFilePath) -> pd.DataFrame:
        # Read data
        df = pd.read_excel(input_data, dtype="object")

        # Rename
        df.rename(
            {
                "op_order": RecipeST.cols.op_order,
                "equipment": RecipeST.cols.resource_id,
                "setup_time": RecipeST.cols.setuptime_per_batch,
            },
            axis=1,
            inplace=True,
        )

        # Assign columns
        df[RecipeST.cols.material_id] = "1"
        df[RecipeST.cols.alt_recipe] = "1"
        df[RecipeST.cols.resource_type] = ResourceType.machine.value
        df[RecipeST.cols.batch_size] = 10
        df[RecipeST.cols.min_lot_size] = df[RecipeST.cols.batch_size]
        df[RecipeST.cols.max_lot_size] = df[RecipeST.cols.batch_size]
        df[RecipeST.cols.operation] = df[RecipeST.cols.resource_id]
        df[RecipeST.cols.step_description] = df["operation_description"]

        # Calculate runtime and min wait time
        df[RecipeST.cols.runtime_per_batch] = (
            df["end_time (in mins)"] - df["start_time (in mins)"]
        )

        min_wait_time = df.groupby(
            [
                RecipeST.cols.material_id,
                RecipeST.cols.batch_size,
                RecipeST.cols.alt_recipe,
            ],
            group_keys=False,
        ).apply(
            lambda g: pd.DataFrame(
                g["start_time (in mins)"].shift(-1) - g["start_time (in mins)"],
                index=g.index,
            )
        )
        wait_time_nan_index = min_wait_time.loc[
            pd.isnull(min_wait_time).any(axis=1)
        ].index
        min_wait_time.loc[wait_time_nan_index] = df.loc[
            wait_time_nan_index, RecipeST.cols.runtime_per_batch
        ]
        df[RecipeST.cols.min_wait_time] = min_wait_time

        # Convert to hrs
        df[RecipeST.cols.runtime_per_batch] /= 60
        df[RecipeST.cols.min_wait_time] /= 60

        return df


    def _filter_recipe(self, df_recipe: pd.DataFrame) -> pd.DataFrame:
        df_recipe = df_recipe[RecipeST.get_fields()].astype(RecipeST.get_dtypes())
        return df_recipe

    def _create_bom(self, df_recipe: pd.DataFrame) -> pd.DataFrame:
        bom_data = []
        for group_name, _ in df_recipe.groupby(
            [RecipeST.cols.material_id, RecipeST.cols.batch_size]
        ):
            material_id, batch_size = group_name

            bom_data.append(
                [
                    material_id,
                    MaterialType.fg.value,
                    batch_size,
                    "1",
                    "CG1",
                    "RM1",
                    MaterialType.rm.value,
                    10,
                    False,
                ]
            )

        df = pd.DataFrame(bom_data, columns=BOMST.get_fields()).astype(
            BOMST.get_dtypes()
        )
        return df

    def _create_product_desc(self, df_recipe: pd.DataFrame) -> pd.DataFrame:
        desc_data = []
        for group_name, group in df_recipe.groupby(
            [RecipeST.cols.material_id, RecipeST.cols.batch_size]
        ):
            material_id, batch_size = group_name
            material_name = group["material_name"].iloc[0]

            desc_data.append(
                [
                    material_id,
                    MaterialType.fg.value,
                    material_name,
                    material_id,
                    batch_size,
                    "KG",
                    1,
                    ClubbingMode.clubbing.value,
                    BatchingMode.tint.value,
                ]
            )

        df = pd.DataFrame(desc_data, columns=ProductDescST.get_fields()).astype(
            ProductDescST.get_dtypes()
        )
        return df

    def _load_plant_map(self, input_data: PandasFilePath) -> pd.DataFrame:
        # Read data
        df = pd.read_excel(input_data, dtype="object")

        # Rename
        df.rename(
            {
                "machine_id": PlantMapST.cols.machine_id,
                "room_id": PlantMapST.cols.room_id,
                "block_id": PlantMapST.cols.block_id,
                "operation": PlantMapST.cols.operation,
            }
        )
        df[PlantMapST.cols.machine_type] = MachineType.formulation.value

        df = df[PlantMapST.get_fields()].astype(PlantMapST.get_dtypes())
        return df

    def _create_forecasted_demand(self, df_product_desc: pd.DataFrame) -> pd.DataFrame:
        demand_data = []
        for material_id in df_product_desc[ProductDescST.cols.material_id].unique():
            demand_data.append(
                [
                    material_id,
                    ProductionStrategy.MTS.value,
                    CAMode.CAD.value,
                    0,
                    0,
                    400,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    np.nan,
                ]
            )

        df = pd.DataFrame(demand_data, columns=DemandST.get_fields()).astype(
            DemandST.get_dtypes()
        )
        return df

    def _load_holdtime_constraint(self, input_data: PandasFilePath) -> pd.DataFrame:
        # Read data
        df = pd.read_excel(input_data, dtype="object")

        # Preprocess
        df.rename(
            {
                "op_order_A": HoldTimeST.cols.op_order_A,
                "op_order_B": HoldTimeST.cols.op_order_B,
                "max_holdtime": HoldTimeST.cols.max_holdtime,
            },
            axis=1,
            inplace=True,
        )

        df[HoldTimeST.cols.min_holdtime] = 0

        df = df[HoldTimeST.get_fields()].astype(HoldTimeST.get_dtypes())
        return df
