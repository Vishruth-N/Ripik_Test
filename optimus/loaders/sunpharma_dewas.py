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
    CAMode,
)
from optimus.loaders.base import BaseLoader, PandasFilePath
from optimus.utils.general import DSU
from optimus.utils.structs import *

import logging

logger = logging.getLogger(__name__)


class SunPharmaDewasLoader(BaseLoader):
    def __init__(self, execution_end: datetime, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.execution_end = execution_end

    @property
    def required_keys(self) -> List[str]:
        return [
            "forecasted_demand",
            "inventory",
            "family_mapping",
            "bom",
            "recipe",
            "plant_map",
        ]

    @property
    def optional_keys(self) -> List[str]:
        return [
            "procurement_plan",
            "phantom_items",
            "machine_availability",
        ]

    @BaseLoader.validate_keys
    def load_all(
        self, data_files: Dict[str, PandasFilePath]
    ) -> Dict[str, pd.DataFrame]:
        # Load demand
        df_forecasted_demand = self._load_forecasted_demand(
            data_files["forecasted_demand"]
        )

        # Initialize code to code
        c2c_graph = DSU()

        # Load BOM and recipe
        df_bom, df_phantom_items = self._load_bom(data_files["bom"])
        df_recipe = self._load_recipe(data_files["recipe"], df_bom)
        df_bom, df_recipe = self._merge_bom_and_recipe(df_bom, df_recipe)

        # Clean BOM
        df_bom = self._clean_bom(df_bom, c2c_graph)
        # df_bom = self._clear_no_intm_cases(df_bom, df_forecasted_demand)
        df_bom, df_recipe = self._merge_bom_and_recipe(df_bom, df_recipe)

        # Create products desc
        df_family_mapping = self._load_family_mapping(data_files["family_mapping"])
        df_products_desc = self._create_products_desc(df_recipe, df_family_mapping)
        df_machine_changeover = self._create_machine_changeover(df_recipe)
        df_recipe = self._filter_recipe(df_recipe)

        # Load inventory
        # TODO: Give appropriate material type to inventory or remove it altogether
        df_inventory = self._load_inventory(
            data_files["inventory"], MaterialType.rm.value, c2c_graph
        )

        # Load plant map
        df_plant_map = self._load_plant_map(data_files["plant_map"])
        df_room_changeover = self._create_room_changeover(df_plant_map)
        df_plant_map = self._filter_plant_map(df_plant_map)

        # Load crossblock penalties
        df_crossblock_penalties = self._create_crossblock_penalties(
            df_plant_map=df_plant_map
        )

        # Load optional files
        df_procurement_plan = None
        df_machine_availability = None
        if data_files.get("phantom_items", None) is not None:
            raise NotImplementedError()
        if data_files.get("procurement_plan", None) is not None:
            raise NotImplementedError()
        if data_files.get("machine_availability", None) is not None:
            df_machine_availability = self._load_machine_availability(
                data_files["machine_availability"], df_plant_map
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

    def _load_forecasted_demand(self, input_data: PandasFilePath) -> pd.DataFrame:
        # Read data
        df = pd.read_excel(input_data, dtype="object")

        # Preprocess
        df.rename({"usage_type": DemandST.cols.ca_mode}, axis=1, inplace=True)

        # Filter by name first
        name_mask = (
            df["molecule name"]
            .astype(str)
            .str.lower()
            .str.match(
                r"(meropenem|imipenem|cilastatin|doripenem|imi\s*cili?a blend|meropenem blend|sodium bicarbonate|sodium carbonate|sbc)"
            )
        )
        df = df[name_mask]

        # Preprocess fields
        df[DemandST.cols.m1_crit] = 0
        df[DemandST.cols.m2_crit] = 0
        df[DemandST.cols.m3_crit] = 0
        df[DemandST.cols.due_date] = np.nan
        df[DemandST.cols.production_strategy] = ProductionStrategy.MTS.value
        df[DemandST.cols.ca_mode] = df[DemandST.cols.ca_mode].map(
            {"captive demand": CAMode.CAD.value, "validation": CAMode.CAV.value}
        )

        # Filter out important cols only
        df = df[DemandST.get_fields()].astype(DemandST.get_dtypes())
        return df

    def _load_inventory(
        self, input_data: PandasFilePath, material_type: str, c2c_graph: DSU
    ) -> pd.DataFrame:
        # Read data
        df = pd.read_excel(input_data, dtype="object")

        # Use unrestricted + quality inspection as quantity
        df[InventoryST.cols.quantity] = df["Unrestricted"] + df["Quality Inspection"]

        # Preprocess fields
        df.rename(
            {
                "Material": InventoryST.cols.material_id,
            },
            axis=1,
            inplace=True,
        )

        # Apply code to code
        logger.debug(
            f"Unique {material_type}: {df[InventoryST.cols.material_id].nunique()}"
        )
        df[InventoryST.cols.material_id] = (
            df[InventoryST.cols.material_id]
            .astype(str)
            .apply(lambda x: c2c_graph.find(x))
        )
        logger.debug(
            f"Unique {material_type} after code to code: {df[InventoryST.cols.material_id].nunique()}"
        )

        # Sum up quantities
        df[InventoryST.cols.quantity] = df[InventoryST.cols.quantity].astype(np.float64)
        df = df.groupby(InventoryST.cols.material_id, as_index=False).agg(
            {InventoryST.cols.quantity: "sum"}
        )

        # Assign material type
        df[InventoryST.cols.material_type] = MaterialType.rm.value

        # Filter out important cols only
        df = df[InventoryST.get_fields()].astype(InventoryST.get_dtypes())
        return df

    def _load_bom_deprecated_v2(self, input_data: PandasFilePath) -> pd.DataFrame:
        # Read data
        df = pd.read_excel(input_data, dtype="object")

        # Rename
        df = df.rename(
            {
                "Material": BOMST.cols.material_id,
                "LSzeFrom": BOMST.cols.material_quantity,
                "AltBOM": BOMST.cols.alt_bom,
                "Component": BOMST.cols.component_id,
                "Quantity": BOMST.cols.component_quantity,
            },
            axis=1,
        ).astype(
            BOMST.get_dtypes(
                [
                    k
                    for k in BOMST.get_fields()
                    if k
                    not in [
                        BOMST.cols.material_type,
                        BOMST.cols.component_type,
                        BOMST.cols.component_group,
                        BOMST.cols.indirect,
                    ]
                ]
            )
        )

        # Remove DeID and change no column
        df.drop(["DeID", "Change No."], axis=1, inplace=True)

        # Remove NA data and duplicate rows
        df = df.dropna(
            subset=[
                BOMST.cols.material_id,
                BOMST.cols.material_quantity,
                BOMST.cols.alt_bom,
                BOMST.cols.component_id,
                BOMST.cols.component_quantity,
            ]
        ).drop_duplicates()

        # Remove repack and resample BOMs
        mask = pd.MultiIndex.from_frame(
            df.loc[
                df["Text"]
                .str.lower()
                .str.contains(r"(?:re\W?pack|cleaning|reprocess)", regex=True),
                [BOMST.cols.material_id, BOMST.cols.alt_bom],
            ]
        )
        df = df.loc[
            ~pd.MultiIndex.from_frame(
                df[[BOMST.cols.material_id, BOMST.cols.alt_bom]].astype(str)
            ).isin(mask)
        ]

        # Remove 0 batch size
        df = df[df[BOMST.cols.material_quantity] > 0]

        # Drop less than 0.3 KG components (they are treated as seed)
        df = df[
            (df["Un"] != "KG")
            | (df[BOMST.cols.component_quantity] >= 0.3)
            | (df[BOMST.cols.component_quantity] < 0)
        ]

        # Drop all 5 series code and put them into phantom items
        series5_mask = df[BOMST.cols.component_id].str.startswith("5")
        packing_materials = df.loc[series5_mask, BOMST.cols.component_id]
        df = df[~series5_mask]

        df_phantom_items = pd.DataFrame(
            {
                PhantomST.cols.material_id: packing_materials,
                PhantomST.cols.material_type: MaterialType.pm.value,
            }
        )

        # Add material type and component type
        common_ids = df.loc[
            df[BOMST.cols.material_id].isin(df[BOMST.cols.component_id]),
            BOMST.cols.material_id,
        ]
        df.loc[
            df[BOMST.cols.material_id].isin(common_ids), BOMST.cols.material_type
        ] = MaterialType.sfg.value
        df.loc[
            ~df[BOMST.cols.material_id].isin(common_ids), BOMST.cols.material_type
        ] = MaterialType.fg.value

        df[BOMST.cols.component_type] = MaterialType.rm.value
        df.loc[
            df[BOMST.cols.component_id].isin(common_ids), BOMST.cols.component_type
        ] = MaterialType.sfg.value

        # Assign indirect column
        df[BOMST.cols.indirect] = 0
        df.loc[
            df[BOMST.cols.component_id].isin(df[BOMST.cols.material_id]),
            BOMST.cols.indirect,
        ] = BatchSizeLinking.EQ.value

        # For negative component quantity, make indirect False
        df.loc[df[BOMST.cols.component_quantity] < 0, BOMST.cols.indirect] = 0

        # cant go from same series code to same series
        df.loc[
            (df[BOMST.cols.material_id].str.startswith("2"))
            & (df[BOMST.cols.component_id].str.startswith("2"))
            & (df[BOMST.cols.indirect] > 0),
            BOMST.cols.indirect,
        ] = 0
        df.loc[
            (df[BOMST.cols.material_id].str.startswith("8"))
            & (df[BOMST.cols.component_id].str.startswith("8"))
            & (df[BOMST.cols.indirect] > 0),
            BOMST.cols.indirect,
        ] = 0

        # Component group
        df[BOMST.cols.component_group] = df[BOMST.cols.component_id]

        return df, df_phantom_items

    def _load_bom(self, input_data: PandasFilePath) -> pd.DataFrame:
        # Read data
        df = pd.read_excel(input_data, dtype="object")

        # Rename
        df = df.rename(
            {
                "Material": BOMST.cols.material_id,
                "LSzeFrom": BOMST.cols.material_quantity,
                "AltBOM": BOMST.cols.alt_bom,
                "Component Group": BOMST.cols.component_group,
                "Component": BOMST.cols.component_id,
                "Quantity": BOMST.cols.component_quantity,
            },
            axis=1,
        ).astype(
            BOMST.get_dtypes(
                [
                    k
                    for k in BOMST.get_fields()
                    if k
                    not in [
                        BOMST.cols.material_type,
                        BOMST.cols.component_type,
                        BOMST.cols.indirect,
                    ]
                ]
            )
        )

        # Remove DeID, change no. and material desc column
        df.drop(["Material description"], axis=1, inplace=True)

        # Remove NA data and duplicate rows
        df = df.dropna(
            subset=[
                BOMST.cols.material_id,
                BOMST.cols.material_quantity,
                BOMST.cols.alt_bom,
                BOMST.cols.component_id,
                BOMST.cols.component_quantity,
            ]
        ).drop_duplicates()

        # Remove repack and resample BOMs
        mask = pd.MultiIndex.from_frame(
            df.loc[
                df["Text"]
                .str.lower()
                .str.contains(r"(?:re\W?pack|cleaning|reprocess)", regex=True),
                [BOMST.cols.material_id, BOMST.cols.alt_bom],
            ]
        )
        df = df.loc[
            ~pd.MultiIndex.from_frame(
                df[[BOMST.cols.material_id, BOMST.cols.alt_bom]].astype(str)
            ).isin(mask)
        ]

        # Remove 0 batch size
        df = df[df[BOMST.cols.material_quantity] > 0]

        # Drop less than 0.3 KG components (they are treated as seed)
        df = df[
            (df["Un"] != "KG")
            | (df[BOMST.cols.component_quantity] >= 0.3)
            | (df[BOMST.cols.component_quantity] < 0)
        ]

        # Drop all 5 series code and put them into phantom items
        series5_mask = df[BOMST.cols.component_id].str.startswith("5")
        packing_materials = df.loc[series5_mask, BOMST.cols.component_id]
        df = df[~series5_mask]

        df_phantom_items = pd.DataFrame(
            {
                PhantomST.cols.material_id: packing_materials,
                PhantomST.cols.material_type: MaterialType.pm.value,
            }
        )

        # Assign indirect column
        df[BOMST.cols.indirect] = 0
        mask = (df[BOMST.cols.material_id].str.startswith("2")) & (
            df[BOMST.cols.component_id].str.startswith("8")
        )
        df.loc[
            (df["Plnt"].astype(str) == df["Manufactured in plant"].astype(str))
            & (mask),
            BOMST.cols.indirect,
        ] = BatchSizeLinking.EQ.value
        df.loc[
            (df["Plnt"].astype(str) == df["Manufactured in plant"].astype(str))
            & (~mask),
            BOMST.cols.indirect,
        ] = BatchSizeLinking.NA.value

        # Add material type and component type
        common_ids = df.loc[
            (df[BOMST.cols.component_id].isin(df[BOMST.cols.material_id]))
            & (df[BOMST.cols.indirect] > 0),
            BOMST.cols.component_id,
        ]
        df.loc[
            df[BOMST.cols.material_id].isin(common_ids), BOMST.cols.material_type
        ] = MaterialType.sfg.value
        df.loc[
            ~df[BOMST.cols.material_id].isin(common_ids), BOMST.cols.material_type
        ] = MaterialType.fg.value

        df[BOMST.cols.component_type] = MaterialType.rm.value
        df.loc[
            df[BOMST.cols.component_id].isin(common_ids), BOMST.cols.component_type
        ] = MaterialType.sfg.value

        return df, df_phantom_items

    def _load_recipe_deprecated_v2(
        self, input_data: PandasFilePath, df_bom: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Read data
        df = pd.read_excel(input_data, dtype="object")

        # Rename fields
        def _rename(df):
            return df.rename(
                {
                    "Product Number": RecipeST.cols.material_id,
                    "Product Short Description": ProductDescST.cols.material_name,
                    "Operation Number": RecipeST.cols.op_order,
                    "Operation Description": RecipeST.cols.operation,
                    "Minimum Lot Size": RecipeST.cols.min_lot_size,
                    "Maximum Lot Size": RecipeST.cols.max_lot_size,
                    "Output Qty": RecipeST.cols.batch_size,
                    "Set-up time (hrs)": RecipeST.cols.setuptime_per_batch,
                    "Min wait time (hrs)": RecipeST.cols.min_wait_time,
                    "Run time (hrs)": RecipeST.cols.runtime_per_batch,
                    "Resource Name": RecipeST.cols.resource_id,
                    "Route": RecipeST.cols.alt_recipe,
                    "Product Short Description": ProductDescST.cols.material_name,
                    "Mettis UOM": ProductDescST.cols.material_unit,
                },
                axis=1,
            ).astype(
                RecipeST.get_dtypes(
                    [
                        k
                        for k in RecipeST.get_fields()
                        if k not in [RecipeST.cols.resource_type]
                    ]
                )
            )

        df = _rename(df)

        # Assign material type
        df = pd.merge(
            df,
            df_bom[
                [BOMST.cols.material_id, BOMST.cols.material_type]
            ].drop_duplicates(),
            how="left",
            left_on=RecipeST.cols.material_id,
            right_on=BOMST.cols.material_id,
        )

        # Assign resource type
        df[RecipeST.cols.resource_type] = ResourceType.machine.value

        # Drop (Code, batch size, alt recipe) if resource id or batch size or operation is not given
        mask = pd.MultiIndex.from_frame(
            df.loc[
                df[
                    [
                        RecipeST.cols.resource_id,
                        RecipeST.cols.batch_size,
                        RecipeST.cols.operation,
                    ]
                ]
                .isna()
                .any(axis=1),
                [
                    RecipeST.cols.material_id,
                    RecipeST.cols.batch_size,
                    RecipeST.cols.alt_recipe,
                ],
            ]
        )
        df = df.loc[
            ~pd.MultiIndex.from_frame(
                df[
                    [
                        RecipeST.cols.material_id,
                        RecipeST.cols.batch_size,
                        RecipeST.cols.alt_recipe,
                    ]
                ]
            ).isin(mask)
        ]

        # Explode resource ids
        df[RecipeST.cols.resource_id] = df[RecipeST.cols.resource_id].apply(
            lambda s: [x.strip() for x in s.split(",")]
        )
        df = df.explode(RecipeST.cols.resource_id).reset_index(drop=True)

        # Fix operation as their inconsistencies in the data
        df[RecipeST.cols.operation] = df.groupby(RecipeST.cols.resource_id)[
            RecipeST.cols.operation
        ].transform("first")

        return df

    def _load_recipe(
        self, input_data: PandasFilePath, df_bom: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Read data
        df = pd.read_excel(input_data, dtype="object")

        # Rename fields
        def _rename(df):
            return df.rename(
                {
                    "Product Number": RecipeST.cols.material_id,
                    "Operation Number": RecipeST.cols.op_order,
                    "Operation Description": RecipeST.cols.operation,
                    "Output Qty": RecipeST.cols.batch_size,
                    "Resource Name": RecipeST.cols.resource_id,
                    "Route": RecipeST.cols.alt_recipe,
                    "Product Short Description": ProductDescST.cols.material_name,
                    "Mettis UOM": ProductDescST.cols.material_unit,
                },
                axis=1,
            ).astype(
                RecipeST.get_dtypes(
                    [
                        RecipeST.cols.material_id,
                        RecipeST.cols.op_order,
                        RecipeST.cols.operation,
                        RecipeST.cols.batch_size,
                        RecipeST.cols.resource_id,
                        RecipeST.cols.alt_recipe,
                    ]
                )
            )

        df = _rename(df)

        # Assign material type
        df = pd.merge(
            df,
            df_bom[
                [BOMST.cols.material_id, BOMST.cols.material_type]
            ].drop_duplicates(),
            how="left",
            left_on=RecipeST.cols.material_id,
            right_on=BOMST.cols.material_id,
        )

        # Drop (Code, batch size, alt recipe) if resource id or batch size or operation is not given
        mask = pd.MultiIndex.from_frame(
            df.loc[
                df[
                    [
                        RecipeST.cols.resource_id,
                        RecipeST.cols.batch_size,
                        RecipeST.cols.operation,
                    ]
                ]
                .isna()
                .any(axis=1),
                [
                    RecipeST.cols.material_id,
                    RecipeST.cols.batch_size,
                    RecipeST.cols.alt_recipe,
                ],
            ]
        )
        df = df.loc[
            ~pd.MultiIndex.from_frame(
                df[
                    [
                        RecipeST.cols.material_id,
                        RecipeST.cols.batch_size,
                        RecipeST.cols.alt_recipe,
                    ]
                ]
            ).isin(mask)
        ]

        # Explode resource ids
        df[RecipeST.cols.resource_id] = df[RecipeST.cols.resource_id].apply(
            lambda s: [x.strip() for x in s.split(",")]
        )
        df = df.explode(RecipeST.cols.resource_id).reset_index(drop=True)

        # Assign resource type
        df[RecipeST.cols.step_description] = df[RecipeST.cols.operation]
        df[RecipeST.cols.resource_type] = ResourceType.machine.value
        df[RecipeST.cols.min_lot_size] = df[RecipeST.cols.batch_size]
        df[RecipeST.cols.max_lot_size] = df[RecipeST.cols.batch_size]

        # Find runtime
        assert (
            df["Batch 1 End Time"] - df["Batch 1 Start Time"]
            != df["Batch 2 End Time"] - df["Batch 2 Start Time"]
        ).any()

        df[RecipeST.cols.setuptime_per_batch] = 0
        df[RecipeST.cols.runtime_per_batch] = (
            df["Batch 1 End Time"] - df["Batch 1 Start Time"]
        )

        min_wait_time = df.groupby(
            [
                RecipeST.cols.material_id,
                RecipeST.cols.batch_size,
                RecipeST.cols.alt_recipe,
            ],
            group_keys=False,
        ).apply(lambda g: g["Batch 1 Start Time"].shift(-1) - g["Batch 1 Start Time"])
        wait_time_nan_index = min_wait_time[min_wait_time.isna()].index
        min_wait_time.loc[wait_time_nan_index] = df.loc[
            wait_time_nan_index, RecipeST.cols.runtime_per_batch
        ]

        df.loc[df.index, RecipeST.cols.min_wait_time] = min_wait_time[df.index]

        return df

    def _filter_recipe(self, df_recipe: pd.DataFrame) -> pd.DataFrame:
        # Filter out important cols only
        df_recipe = df_recipe[RecipeST.get_fields()].astype(RecipeST.get_dtypes())
        return df_recipe

    def _merge_bom_and_recipe(
        self,
        df_bom: pd.DataFrame,
        df_recipe: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Extract code sets
        bom_set = set(
            zip(df_bom[BOMST.cols.material_id], df_bom[BOMST.cols.material_quantity])
        )
        recipe_set = set(
            zip(
                df_recipe[RecipeST.cols.material_id],
                df_recipe[RecipeST.cols.batch_size],
            )
        )
        merged = pd.MultiIndex.from_tuples(bom_set.intersection(recipe_set))

        # Log info
        logger.debug(f"Unique codes in BOM: {len(bom_set)}")
        logger.debug(f"Unique codes in Recipe: {len(recipe_set)}")
        logger.debug(f"Unique codes in Merged (intersection): {len(merged)}")

        # Merge by code
        df_bom = df_bom[
            pd.MultiIndex.from_frame(
                df_bom[[BOMST.cols.material_id, BOMST.cols.material_quantity]]
            ).isin(merged)
        ]
        df_recipe = df_recipe[
            pd.MultiIndex.from_frame(
                df_recipe[[RecipeST.cols.material_id, RecipeST.cols.batch_size]]
            ).isin(merged)
        ]

        return df_bom, df_recipe

    def _clean_bom(self, df_bom: pd.DataFrame, c2c_graph: DSU) -> pd.DataFrame:
        # Add same string to code to code
        # for _, group in df_bom[df_bom["SortStrng"] == "C01"].groupby(
        #     [BOMST.cols.material_id, BOMST.cols.material_quantity, BOMST.cols.alt_bom]
        # ):
        #     component_ids = list(group[BOMST.cols.component_id])
        #     for i in range(1, len(component_ids)):
        #         c2c_graph.union(component_ids[i - 1], component_ids[i])

        # Apply code to code
        # df_bom[BOMST.cols.component_id] = df_bom[BOMST.cols.component_id].apply(
        #     lambda x: c2c_graph.find(x)
        # )

        # Sum up the quantity
        df_bom = df_bom.groupby(
            [
                BOMST.cols.material_id,
                BOMST.cols.material_quantity,
                BOMST.cols.alt_bom,
                BOMST.cols.component_group,
                BOMST.cols.component_id,
            ],
            as_index=False,
        ).agg(
            {
                BOMST.cols.material_type: "first",
                BOMST.cols.component_type: "first",
                BOMST.cols.indirect: "first",
                BOMST.cols.component_quantity: "sum",
            }
        )

        # Filter out important cols only
        df_bom = df_bom[BOMST.get_fields()].astype(BOMST.get_dtypes())
        return df_bom

    def _clear_no_intm_cases_deprecated_v2(
        self, df_bom: pd.DataFrame, df_forecasted_demand: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        mask = pd.MultiIndex.from_frame(
            df_bom.loc[
                (df_bom[BOMST.cols.indirect] > 0)
                & (
                    ~df_bom[BOMST.cols.component_id].isin(
                        df_bom[BOMST.cols.material_id]
                    )
                ),
                [
                    BOMST.cols.material_id,
                    BOMST.cols.material_quantity,
                    BOMST.cols.alt_bom,
                ],
            ]
        )

        df_bom.loc[
            pd.MultiIndex.from_frame(
                df_bom[
                    [
                        BOMST.cols.material_id,
                        BOMST.cols.material_quantity,
                        BOMST.cols.alt_bom,
                    ]
                ]
            ).isin(mask),
            BOMST.cols.indirect,
        ] = 0

        # TODO: Change based on max dependancy levels in config
        # Make indirect 0 for subsequent levels
        df_bom.loc[
            (
                df_bom[BOMST.cols.material_id].isin(
                    df_bom.loc[df_bom[BOMST.cols.indirect] > 0, BOMST.cols.component_id]
                )
            )
            & (
                ~df_bom[BOMST.cols.material_id].isin(
                    df_forecasted_demand[DemandST.cols.material_id]
                )
            ),
            BOMST.cols.indirect,
        ] = 0

        return df_bom

    def _load_family_mapping(self, input_data: PandasFilePath) -> pd.DataFrame:
        # Read data
        df = pd.read_excel(input_data, dtype="object")

        # Preprocess fields
        df.rename(
            {
                "Code": ProductDescST.cols.material_id,
                "Family": ProductDescST.cols.family_id,
            },
            axis=1,
            inplace=True,
        )

        df = df.astype(ProductDescST.get_dtypes(df.columns))
        return df

    def _create_products_desc(
        self, df_recipe: pd.DataFrame, df_family_mapping: pd.DataFrame
    ) -> pd.DataFrame:
        # Filter out important cols only
        df = df_recipe.rename(
            {
                RecipeST.cols.material_id: ProductDescST.cols.material_id,
                BOMST.cols.material_type: ProductDescST.cols.material_type,
                RecipeST.cols.batch_size: ProductDescST.cols.batch_size,
            },
            axis=1,
        ).drop_duplicates(
            subset=[ProductDescST.cols.material_id, ProductDescST.cols.batch_size]
        )

        # Add other columns
        df[ProductDescST.cols.material_unit] = "KG"
        df[ProductDescST.cols.count_factor] = 1
        df[ProductDescST.cols.clubbing_mode] = ClubbingMode.standard.value
        df[ProductDescST.cols.batching_mode] = BatchingMode.tint.value

        # Obtain family column
        df = df.merge(df_family_mapping, how="left", on=ProductDescST.cols.material_id)

        # Filter out important cols only
        df = df[ProductDescST.get_fields()].astype(ProductDescST.get_dtypes())

        return df

    def _load_plant_map(self, input_data: PandasFilePath) -> pd.DataFrame:
        # Read data
        df = pd.read_excel(input_data, dtype="object")

        # Preprocess fields
        df.rename(
            {
                "Stream": PlantMapST.cols.room_id,
                "Machine ID": PlantMapST.cols.machine_id,
                "Operation": PlantMapST.cols.operation,
                "Machine Type": PlantMapST.cols.machine_type,
                "Max clean (hrs)": RoomChangeoverST.cols.type_B,
            },
            axis=1,
            inplace=True,
        )

        # Add block ID same as room ID
        df[PlantMapST.cols.block_id] = df[PlantMapST.cols.room_id]

        # Remove empty machine ids
        df.dropna(subset=[PlantMapST.cols.machine_id], inplace=True)

        return df

    def _filter_plant_map(self, df_plant_map: pd.DataFrame) -> pd.DataFrame:
        # Remove fixed operation machine
        df_plant_map = df_plant_map[
            df_plant_map[PlantMapST.cols.operation] != "Fixed Machine Operation"
        ]

        # Filter out important cols only
        df_plant_map = (
            df_plant_map[PlantMapST.get_fields()]
            .reset_index()
            .astype(PlantMapST.get_dtypes())
        )
        return df_plant_map

    def _create_machine_changeover(self, df_recipe: pd.DataFrame) -> pd.DataFrame:
        # Filter out data
        df = (
            df_recipe[
                df_recipe[RecipeST.cols.resource_type] == ResourceType.machine.value
            ][
                [
                    RecipeST.cols.resource_id,
                    RecipeST.cols.material_id,
                    "Batch 1 End Time",
                    "Batch 2 Start Time",
                ]
            ]
            .drop_duplicates()
            .copy()
            .rename(
                {
                    RecipeST.cols.resource_id: MachineChangeoverST.cols.machine_id,
                    RecipeST.cols.material_id: MachineChangeoverST.cols.material_id,
                },
                axis=1,
            )
        )

        def _calc_time(x):
            s = pd.Series(
                x["Batch 2 Start Time"].min() - x["Batch 1 End Time"].max(),
                index=[MachineChangeoverST.cols.type_A],
            )
            return s

        df = df.groupby(
            [MachineChangeoverST.cols.machine_id, MachineChangeoverST.cols.material_id],
            as_index=False,
        ).apply(_calc_time)

        df.loc[
            df[MachineChangeoverST.cols.type_A] < 0, MachineChangeoverST.cols.type_A
        ] = 0

        # Filter out important cols only
        df = df[MachineChangeoverST.get_fields()].astype(
            MachineChangeoverST.get_dtypes()
        )
        return df

    def _create_room_changeover(self, df: pd.DataFrame) -> pd.DataFrame:
        # Rename cols
        df = df.rename(
            {PlantMapST.cols.room_id: RoomChangeoverST.cols.room_id}, axis=1
        )[
            [
                RoomChangeoverST.cols.room_id,
                RoomChangeoverST.cols.type_B,
            ]
        ]

        # Fill NA with 0s
        df = df.drop_duplicates().fillna(0)

        # Make type-A changeover 0
        df[RoomChangeoverST.cols.type_A] = 0

        # Filter out important cols only
        df = df[RoomChangeoverST.get_fields()].astype(RoomChangeoverST.get_dtypes())

        return df

    def _create_crossblock_penalties(self, df_plant_map: pd.DataFrame) -> pd.DataFrame:
        # Read data
        block_ids = df_plant_map[PlantMapST.cols.block_id].unique()

        # Create data
        crossblock_penalties = defaultdict(list)
        for block_id_a in block_ids:
            for block_id_b in block_ids:
                crossblock_penalties[CrossBlockST.cols.from_block_id].append(block_id_a)
                crossblock_penalties[CrossBlockST.cols.to_block_id].append(block_id_b)
                crossblock_penalties[CrossBlockST.cols.penalty].append(1)

        # Filter out important cols only
        df = pd.DataFrame(crossblock_penalties)
        df = df[CrossBlockST.get_fields()].astype(CrossBlockST.get_dtypes())
        return df

    def _load_machine_availability(
        self, input_data: PandasFilePath, df_plant_map: pd.DataFrame
    ) -> pd.DataFrame:
        # Read data
        df = pd.read_excel(input_data, dtype="object")

        # Datetime format
        df[MachineAvailabilityST.cols.start_datetime] = pd.to_datetime(
            df[MachineAvailabilityST.cols.start_datetime], format="%d.%m.%Y %H:%M:%S"
        )
        df[MachineAvailabilityST.cols.end_datetime] = pd.to_datetime(
            df[MachineAvailabilityST.cols.end_datetime], format="%d.%m.%Y %H:%M:%S"
        )

        # Add machine column
        room_to_machine = df_plant_map.groupby(PlantMapST.cols.room_id)[
            PlantMapST.cols.machine_id
        ].apply(lambda x: x.values.tolist())
        df[MachineAvailabilityST.cols.machine_id] = df["stream"].map(room_to_machine)
        df = df.explode(MachineAvailabilityST.cols.machine_id)

        # Filter cols
        df = df[MachineAvailabilityST.get_fields()].astype(
            MachineAvailabilityST.get_dtypes()
        )
        return df
