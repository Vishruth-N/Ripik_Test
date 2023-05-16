"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import pandas as pd
from .base import BaseLoader, PandasFilePath
from ..utils.structs import *


class RipikLoader(BaseLoader):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def required_keys(self) -> List[str]:
        return [
            "forecasted_demand",
            "inventory",
            "procurement_plan",
            "products_description",
            "bom",
            "recipe",
            "plant_map",
            "room_changeover",
            "crossblock_penalties",
        ]

    @property
    def optional_keys(self) -> List[str]:
        return [
            "machine_changeover",
            "phantom_items",
            "machine_availability",
        ]

    @BaseLoader.validate_keys
    def load_all(
        self, data_files: Dict[str, PandasFilePath]
    ) -> Dict[str, pd.DataFrame]:
        # Load required files
        df_forecasted_demand = pd.read_csv(data_files["forecasted_demand"])[
            DemandST.get_fields()
        ].astype(DemandST.get_dtypes())
        df_inventory = pd.read_csv(data_files["inventory"])[
            InventoryST.get_fields()
        ].astype(InventoryST.get_dtypes())
        df_procurement_plan = pd.read_csv(data_files["procurement_plan"])[
            ProcurementST.get_fields()
        ].astype(ProcurementST.get_dtypes())
        df_products_desc = pd.read_csv(data_files["products_description"])[
            ProductDescST.get_fields()
        ].astype(ProductDescST.get_dtypes())
        df_bom = pd.read_csv(data_files["bom"])[BOMST.get_fields()].astype(
            BOMST.get_dtypes()
        )
        df_recipe = pd.read_csv(data_files["recipe"])[RecipeST.get_fields()].astype(
            RecipeST.get_dtypes()
        )
        df_plant_map = pd.read_csv(data_files["plant_map"])[
            PlantMapST.get_fields()
        ].astype(PlantMapST.get_dtypes())
        df_room_changeover = pd.read_csv(data_files["room_changeover"])[
            RoomChangeoverST.get_fields()
        ].astype(RoomChangeoverST.get_dtypes())
        df_crossblock_penalties = pd.read_csv(data_files["crossblock_penalties"])[
            CrossBlockST.get_fields()
        ].astype(CrossBlockST.get_dtypes())

        # Load optional files
        df_phantom_items = None
        df_machine_changeover = None
        df_machine_availability = None
        if data_files.get("phantom_items", None) is not None:
            df_phantom_items = pd.read_csv(data_files["phantom_items"])[
                PhantomST.get_fields()
            ].astype(PhantomST.get_dtypes())
        if data_files.get("machine_changeover", None) is not None:
            df_machine_changeover = pd.read_csv(data_files["machine_changeover"])
        if data_files.get("machine_availability", None) is not None:
            df_machine_availability = pd.read_csv(
                data_files["machine_availability"]
            ).astype(MachineAvailabilityST.get_dtypes())

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
