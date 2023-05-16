"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Optional
from random import choice

from optimus.utils.constants import (
    BatchSizeLinking,
    BatchingMode,
    ClubbingMode,
    MachineType,
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


class SunPharmaPaontaSahibLoader(BaseLoader):
    def __init__(
        self, execution_start: datetime, execution_end: datetime, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.execution_start = execution_start
        self.execution_end = execution_end

        # Theoritical
        self.runtime_factor = 1.0
        self.qcd_waittime_factor = 0.2
        self.whd_waittime_factor = 0

        # Experimental features
        self.is_changeoverB_equals_A = False
        self.keep_one_batch_size = True
        self.use_rmpm_qi = True

    @property
    def required_keys(self) -> List[str]:
        return [
            "forecasted_demand",
            "rm_inventory",
            "pm_inventory",
            "sfg_inventory",
            "fg_inventory",
            "procurement_plan",
            "bom",
            "recipe",
            "plant_map",
            "crossblock_penalties",
            "packsize_mapping",
        ]

    @property
    def optional_keys(self) -> List[str]:
        return [
            "code_to_code",
            "printing_xy",
            "phantom_items",
            "machine_availability",
            "sfg_underprocess",
            "fg_underprocess",
        ]

    @BaseLoader.validate_keys
    def load_all(
        self, data_files: Dict[str, PandasFilePath]
    ) -> Dict[str, pd.DataFrame]:
        # Load demand
        df_forecasted_demand = self._load_forecasted_demand(
            data_files["forecasted_demand"]
        )

        # Load pack size mapping
        packsize_mapping = self._load_packsize_mapping(data_files["packsize_mapping"])

        # Load code to code if given
        c2c_graph = DSU()
        if data_files.get("code_to_code", None) is not None:
            self._load_code_to_code(data_files["code_to_code"], c2c_graph=c2c_graph)
        if data_files.get("printing_xy", None) is not None:
            self._load_printing_xy(data_files["printing_xy"], c2c_graph=c2c_graph)

        # Load bom and recipe
        df_sfg_bom, df_fg_bom = self._load_bom(data_files["bom"])
        df_sfg_recipe, df_fg_recipe = self._load_recipe(data_files["recipe"])
        df_sfg_bom, df_fg_bom, df_sfg_recipe, df_fg_recipe = self._merge_bom_and_recipe(
            df_sfg_bom, df_fg_bom, df_sfg_recipe, df_fg_recipe
        )
        df_sfg_bom, df_fg_bom = self._clear_no_batch_size_cases(df_sfg_bom, df_fg_bom)
        df_sfg_bom, df_fg_bom = self._clear_no_intm_cases(df_sfg_bom, df_fg_bom)
        df_sfg_bom, df_fg_bom, df_sfg_recipe, df_fg_recipe = self._merge_bom_and_recipe(
            df_sfg_bom, df_fg_bom, df_sfg_recipe, df_fg_recipe
        )

        # Clean bom
        df_sfg_bom, df_fg_bom = self._clean_bom(df_sfg_bom, df_fg_bom, packsize_mapping)
        df_bom = self._concat_and_filter_bom(df_sfg_bom, df_fg_bom, c2c_graph)

        # Clean recipe and make products desc
        df_recipe = self._clean_recipe(df_sfg_recipe, df_fg_recipe, df_bom)
        df_products_desc = self._create_products_desc(df_recipe, packsize_mapping)
        df_recipe = self._filter_recipe(df_recipe)

        # Load inventory
        df_rm_inventory = self._load_rmpm_inventory(
            data_files["rm_inventory"],
            material_type=MaterialType.rm.value,
            c2c_graph=c2c_graph,
        )
        df_pm_inventory = self._load_rmpm_inventory(
            data_files["pm_inventory"],
            material_type=MaterialType.pm.value,
            c2c_graph=c2c_graph,
        )
        df_sfg_inventory = self._load_products_inventory(
            data_files["sfg_inventory"],
            material_type=None,
            df_products_desc=df_products_desc,
        )
        df_fg_inventory = self._load_products_inventory(
            data_files["fg_inventory"],
            material_type=MaterialType.fg.value,
            df_products_desc=df_products_desc,
        )

        # Load procurement plan
        df_procurement_plan = self._load_procurement_plan(
            data_files["procurement_plan"]
        )

        # Load plant map and room changeover
        df_plant_map = self._load_plant_map(
            data_files["plant_map"], df_recipe=df_recipe
        )
        df_room_changeover = self._create_room_changeover(df_plant_map)
        df_plant_map = self._filter_plant_map(df_plant_map)

        # Load crossblock penalties
        df_crossblock_penalties = self._load_crossblock_penalties(
            data_files["crossblock_penalties"]
        )

        # Load initial state
        df_initial_state = None
        if (
            data_files["sfg_underprocess"] is not None
            and data_files["fg_underprocess"] is not None
        ):
            df_sfg_underprocess, sfg_inventory_added = self._load_underprocess(
                data_files["sfg_underprocess"],
                material_type=MaterialType.sfg.value,
                datefmt="%d.%m.%Y",
            )
            df_fg_underprocess, fg_inventory_added = self._load_underprocess(
                data_files["fg_underprocess"],
                material_type=MaterialType.fg.value,
                datefmt="%d-%m-%Y",
            )
            df_initial_state = self._load_intial_state(
                df_sfg_underprocess, df_fg_underprocess
            )

            # Concat to inventory
            df_inventory = pd.concat(
                [
                    df_rm_inventory,
                    df_pm_inventory,
                    df_sfg_inventory,
                    df_fg_inventory,
                    sfg_inventory_added,
                    fg_inventory_added,
                ]
            )

        else:
            df_inventory = pd.concat(
                [df_rm_inventory, df_pm_inventory, df_sfg_inventory, df_fg_inventory]
            )

        # Load optional files
        df_phantom_items = None
        df_machine_availability = None
        if data_files.get("phantom_items", None) is not None:
            df_phantom_items = self._load_phantom_items(data_files["phantom_items"])
        if data_files.get("machine_availability", None) is not None:
            raise NotImplementedError()

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
            "df_machine_changeover": None,
            "df_machine_availability": df_machine_availability,
            "df_initial_state": df_initial_state,
        }

        return output

    def _load_forecasted_demand(self, input_data: PandasFilePath) -> pd.DataFrame:
        # Read data
        df = pd.read_excel(input_data, dtype="object")

        # Preprocess fields
        df.rename(
            {"FG Code": DemandST.cols.material_id},
            axis=1,
            inplace=True,
        )
        df[DemandST.cols.priority] = 1
        df[DemandST.cols.due_date] = np.nan
        df[DemandST.cols.production_strategy] = ProductionStrategy.MTO.value
        df[DemandST.cols.ca_mode] = CAMode.CAD.value

        # Filter out important cols only
        df = df[DemandST.get_fields()].astype(DemandST.get_dtypes())
        return df

    def _load_bom(
        self, input_data: PandasFilePath
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_sfg = pd.read_excel(input_data, sheet_name="SFG", dtype="object")
        df_fg = pd.read_excel(input_data, sheet_name="FG", dtype="object")

        # Rename fields
        def _rename(df):
            df = df.rename(
                {
                    "Material": BOMST.cols.material_id,
                    "Base quantity": BOMST.cols.material_quantity,
                    "AltBOM": BOMST.cols.alt_bom,
                    "Component": BOMST.cols.component_id,
                    "Quantity": BOMST.cols.component_quantity,
                },
                axis=1,
            ).astype(
                BOMST.get_dtypes(
                    [
                        BOMST.cols.material_id,
                        BOMST.cols.material_quantity,
                        BOMST.cols.alt_bom,
                        BOMST.cols.component_id,
                        BOMST.cols.component_quantity,
                    ]
                )
            )

            return df

        df_sfg = _rename(df_sfg)
        df_fg = _rename(df_fg)

        # Make indirect column
        df_sfg[BOMST.cols.indirect] = 0
        df_sfg.loc[
            df_sfg[BOMST.cols.component_id].isin(df_sfg[BOMST.cols.material_id]),
            BOMST.cols.indirect,
        ] = BatchSizeLinking.GE.value

        df_fg[BOMST.cols.indirect] = 0
        df_fg.loc[
            df_fg[BOMST.cols.component_id].isin(df_sfg[BOMST.cols.material_id]),
            BOMST.cols.indirect,
        ] = BatchSizeLinking.EQ.value

        # Add material & component type to SFG BOM
        df_sfg[BOMST.cols.material_type] = df_sfg["BUn"].map(
            {"EA": MaterialType.sfg.value, "KG": MaterialType.cb.value}
        )
        df_sfg[BOMST.cols.component_type] = MaterialType.rm.value

        # Drop duplicates in FG BOM
        df_fg = df_fg.drop_duplicates()

        # Add material & component type to FG BOM
        df_fg[BOMST.cols.material_type] = MaterialType.fg.value
        df_fg = pd.merge(
            df_fg,
            df_sfg[
                [BOMST.cols.material_id, BOMST.cols.material_type]
            ].drop_duplicates(),
            how="left",
            left_on=BOMST.cols.component_id,
            right_on=BOMST.cols.material_id,
            suffixes=("", "_y"),
        )
        df_fg = df_fg.drop([BOMST.cols.material_id + "_y"], axis=1)
        df_fg = df_fg.rename(
            {BOMST.cols.material_type + "_y": BOMST.cols.component_type}, axis=1
        )
        df_fg[BOMST.cols.component_type].fillna(MaterialType.pm.value, inplace=True)

        return df_sfg, df_fg

    def _load_recipe(
        self, input_data: PandasFilePath
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Read data
        df_sfg = pd.read_excel(input_data, sheet_name="SFG", dtype="object")
        df_fg = pd.read_excel(input_data, sheet_name="FG", dtype="object")

        # Rename fields
        def _rename(df):
            return df.rename(
                {
                    "Product Number": RecipeST.cols.material_id,
                    "Operation Number": RecipeST.cols.op_order,
                    "Operation": RecipeST.cols.operation,
                    "Operation Description": RecipeST.cols.step_description,
                    "Minimum Lot Size": RecipeST.cols.min_lot_size,
                    "Maximum Lot Size": RecipeST.cols.max_lot_size,
                    "Output Qty": RecipeST.cols.batch_size,
                    "Set-up time  (hrs)": RecipeST.cols.setuptime_per_batch,
                    "Run time  (hrs)": RecipeST.cols.runtime_per_batch,
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
                        if k
                        not in [
                            RecipeST.cols.resource_type,
                            RecipeST.cols.min_wait_time,
                        ]
                    ]
                )
            )

        df_sfg = _rename(df_sfg)
        df_fg = _rename(df_fg)

        return df_sfg, df_fg

    def _merge_bom_and_recipe(
        self,
        df_sfg_bom: pd.DataFrame,
        df_fg_bom: pd.DataFrame,
        df_sfg_recipe: pd.DataFrame,
        df_fg_recipe: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Merge SFG by (Code, batch size)
        sfg_bom_set = set(
            zip(
                df_sfg_bom[BOMST.cols.material_id],
                df_sfg_bom[BOMST.cols.material_quantity],
            )
        )
        sfg_recipe_set = set(
            zip(
                df_sfg_recipe[RecipeST.cols.material_id],
                df_sfg_recipe[RecipeST.cols.batch_size],
            )
        )
        sfg_merged = pd.MultiIndex.from_tuples(sfg_bom_set.intersection(sfg_recipe_set))

        df_sfg_bom = df_sfg_bom[
            pd.MultiIndex.from_frame(
                df_sfg_bom[[BOMST.cols.material_id, BOMST.cols.material_quantity]]
            ).isin(sfg_merged)
        ]
        df_sfg_recipe = df_sfg_recipe[
            pd.MultiIndex.from_frame(
                df_sfg_recipe[[RecipeST.cols.material_id, RecipeST.cols.batch_size]]
            ).isin(sfg_merged)
        ]

        # Merge FG by (Code)
        fg_bom_set = set(df_fg_bom[BOMST.cols.material_id])
        fg_recipe_set = set(df_fg_recipe[RecipeST.cols.material_id])
        fg_merged = fg_bom_set.intersection(fg_recipe_set)

        df_fg_bom = df_fg_bom[df_fg_bom[BOMST.cols.material_id].isin(fg_merged)]
        df_fg_recipe = df_fg_recipe[
            df_fg_recipe[RecipeST.cols.material_id].isin(fg_merged)
        ]

        return df_sfg_bom, df_fg_bom, df_sfg_recipe, df_fg_recipe

    def _clear_no_intm_cases(
        self, df_sfg_bom: pd.DataFrame, df_fg_bom: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        def _drop_no_intms_sfg(df_to_prune):
            mask = pd.MultiIndex.from_frame(
                df_to_prune.loc[
                    (df_to_prune[BOMST.cols.indirect] > 0)
                    & (
                        ~df_to_prune[BOMST.cols.component_id].isin(
                            df_to_prune[BOMST.cols.material_id]
                        )
                    ),
                    [
                        BOMST.cols.material_id,
                        BOMST.cols.material_quantity,
                        BOMST.cols.alt_bom,
                    ],
                ]
            )

            prev_length = len(df_to_prune)
            df_to_prune = df_to_prune.loc[
                ~pd.MultiIndex.from_frame(
                    df_to_prune[
                        [
                            BOMST.cols.material_id,
                            BOMST.cols.material_quantity,
                            BOMST.cols.alt_bom,
                        ]
                    ]
                ).isin(mask)
            ]
            new_length = len(df_to_prune)
            if prev_length == new_length:
                return df_to_prune

            return _drop_no_intms_sfg(df_to_prune)

        def _drop_no_intms_fg(df1, df2):
            mask = pd.MultiIndex.from_frame(
                df1.loc[
                    (df1[BOMST.cols.indirect] > 0)
                    & (~df1[BOMST.cols.component_id].isin(df2[BOMST.cols.material_id])),
                    [
                        BOMST.cols.material_id,
                        BOMST.cols.material_quantity,
                        BOMST.cols.alt_bom,
                    ],
                ]
            )
            return df1.loc[
                ~pd.MultiIndex.from_frame(
                    df1[
                        [
                            BOMST.cols.material_id,
                            BOMST.cols.material_quantity,
                            BOMST.cols.alt_bom,
                        ]
                    ]
                ).isin(mask)
            ]

        logger.debug(f"Before dropping no INTMS, length of SFG BOM: {len(df_sfg_bom)}")
        logger.debug(f"Before dropping no INTMS, length of FG BOM: {len(df_fg_bom)}")
        df_sfg_bom = _drop_no_intms_sfg(df_sfg_bom)
        df_fg_bom = _drop_no_intms_fg(df_fg_bom, df_sfg_bom)
        logger.debug(f"After dropping no INTMS, length of SFG BOM: {len(df_sfg_bom)}")
        logger.debug(f"After dropping no INTMS, length of FG BOM: {len(df_fg_bom)}")

        return df_sfg_bom, df_fg_bom

    def _clear_no_batch_size_cases(
        self, df_sfg_bom: pd.DataFrame, df_fg_bom: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.debug(
            f"Before dropping no batch sizes, length of SFG BOM: {len(df_sfg_bom)}"
        )

        bad_cases = []
        for row in df_sfg_bom.loc[
            df_sfg_bom[BOMST.cols.indirect] == BatchSizeLinking.GE.value
        ].itertuples():
            row = row._asdict()
            if not (
                df_sfg_bom[
                    df_sfg_bom[BOMST.cols.material_id] == row[BOMST.cols.component_id]
                ][BOMST.cols.material_quantity]
                >= row[BOMST.cols.component_quantity]
            ).any():
                bad_cases.append(
                    (
                        row[BOMST.cols.material_id],
                        row[BOMST.cols.material_quantity],
                        row[BOMST.cols.alt_bom],
                    )
                )

        if len(bad_cases) > 0:
            mask = pd.MultiIndex.from_tuples(bad_cases)
            df_sfg_bom = df_sfg_bom.loc[
                ~pd.MultiIndex.from_frame(
                    df_sfg_bom[
                        [
                            BOMST.cols.material_id,
                            BOMST.cols.material_quantity,
                            BOMST.cols.alt_bom,
                        ]
                    ]
                ).isin(mask)
            ]

        logger.debug(
            f"After dropping no batch sizes, length of SFG BOM: {len(df_sfg_bom)}"
        )

        if self.keep_one_batch_size:
            # Remove higher batch size of FG
            mask = pd.MultiIndex.from_frame(
                df_fg_bom.groupby(BOMST.cols.material_id)[BOMST.cols.material_quantity]
                .min()
                .reset_index()
            )
            df_fg_bom = df_fg_bom.loc[
                pd.MultiIndex.from_frame(
                    df_fg_bom[[BOMST.cols.material_id, BOMST.cols.material_quantity]]
                ).isin(mask)
            ]

            # Remove lower batch size of SFG
            mask = pd.MultiIndex.from_frame(
                df_sfg_bom.groupby(BOMST.cols.material_id)[BOMST.cols.material_quantity]
                .max()
                .reset_index()
            )
            df_sfg_bom = df_sfg_bom.loc[
                pd.MultiIndex.from_frame(
                    df_sfg_bom[[BOMST.cols.material_id, BOMST.cols.material_quantity]]
                ).isin(mask)
            ]

        return df_sfg_bom, df_fg_bom

    def _clean_bom(
        self,
        df_sfg_bom: pd.DataFrame,
        df_fg_bom: pd.DataFrame,
        packsize_mapping: Dict[str, int],
    ) -> pd.DataFrame:
        # Sum up the quantity
        df_sfg_bom = df_sfg_bom.groupby(
            [
                BOMST.cols.material_id,
                BOMST.cols.material_quantity,
                BOMST.cols.alt_bom,
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

        # TODO: Make a check that df_sfg have one to one mapping with material id and material type

        # Create pack size
        df_fg_bom["packsize"] = (
            df_fg_bom["BUn"].astype("string").map(lambda x: packsize_mapping.get(x, 1))
        )

        # Sum up the quantity
        df_fg_bom = df_fg_bom.groupby(
            [
                BOMST.cols.material_id,
                BOMST.cols.material_quantity,
                BOMST.cols.alt_bom,
                BOMST.cols.component_id,
            ],
            as_index=False,
        ).agg(
            {
                BOMST.cols.material_type: "first",
                BOMST.cols.component_type: "first",
                BOMST.cols.component_quantity: "sum",
                BOMST.cols.indirect: "first",
                "packsize": "first",
            }
        )

        # FG to SFG is one to one mapping so calculate batch size from there
        fg_batch_sizes = []
        for group_name, group in df_fg_bom[
            df_fg_bom[BOMST.cols.component_id].isin(df_sfg_bom[BOMST.cols.material_id])
        ].groupby(
            [BOMST.cols.material_id, BOMST.cols.material_quantity, BOMST.cols.alt_bom]
        ):
            if len(group) > 1:
                logger.warn(f"{group_name} have more than one SFG/CB in FG BOM")

            curr_pack_size = group["packsize"].iloc[0]
            for batch_size in df_sfg_bom[
                df_sfg_bom[BOMST.cols.material_id]
                == group[BOMST.cols.component_id].iloc[0]
            ][BOMST.cols.material_quantity].unique():
                fg_batch_sizes.append(
                    [
                        group_name[0],
                        group_name[1],
                        group_name[2],
                        batch_size / curr_pack_size,
                        group[BOMST.cols.component_quantity].iloc[0] / curr_pack_size,
                    ]
                )

        fg_batch_sizes = pd.DataFrame(
            fg_batch_sizes,
            columns=[
                "fg_material_id",
                "fg_material_quantity",
                "fg_alt_bom",
                "fg_batch_size",
                "fg_original_batch_size",
            ],
        ).drop_duplicates()

        df_fg_bom = pd.merge(
            df_fg_bom,
            fg_batch_sizes,
            how="left",
            left_on=[
                BOMST.cols.material_id,
                BOMST.cols.material_quantity,
                BOMST.cols.alt_bom,
            ],
            right_on=["fg_material_id", "fg_material_quantity", "fg_alt_bom"],
        )

        # Change the quantities to match each batch size in the recipe
        mask = df_fg_bom["fg_batch_size"].isna()
        df_fg_bom.loc[~mask, BOMST.cols.component_quantity] = (
            df_fg_bom.loc[~mask, BOMST.cols.component_quantity]
            / df_fg_bom.loc[~mask, "fg_original_batch_size"]
        ) * (df_fg_bom.loc[~mask, "fg_batch_size"])
        df_fg_bom.loc[~mask, BOMST.cols.material_quantity] = df_fg_bom.loc[
            ~mask, "fg_batch_size"
        ]

        return df_sfg_bom, df_fg_bom

    def _concat_and_filter_bom(
        self, df_sfg_bom: pd.DataFrame, df_fg_bom: pd.DataFrame, c2c_graph: DSU
    ) -> pd.DataFrame:
        # Concat both BOMs
        df = pd.concat([df_sfg_bom, df_fg_bom])

        # Apply code to code
        df[BOMST.cols.component_id] = df[BOMST.cols.component_id].apply(
            lambda x: c2c_graph.find(x)
        )

        # Component group is same as component ID
        df[BOMST.cols.component_group] = df[BOMST.cols.component_id]

        # Filter out important cols only
        df = df[BOMST.get_fields()].astype(BOMST.get_dtypes())
        return df

    def _clean_recipe(
        self,
        df_sfg_recipe: pd.DataFrame,
        df_fg_recipe: pd.DataFrame,
        df_bom: pd.DataFrame,
    ) -> pd.DataFrame:
        # Recreate FG with its batch sizes from BOM
        new_fg_data = []
        for bom_row in (
            df_bom.loc[
                df_bom[BOMST.cols.material_type] == MaterialType.fg.value,
                [
                    BOMST.cols.material_id,
                    BOMST.cols.material_quantity,
                ],
            ]
            .drop_duplicates()
            .itertuples()
        ):
            bom_row = bom_row._asdict()
            material_id = bom_row[BOMST.cols.material_id]
            batch_size = bom_row[BOMST.cols.material_quantity]

            recipe = df_fg_recipe[
                df_fg_recipe[RecipeST.cols.material_id] == material_id
            ]
            assert len(recipe) > 0, "Didn't merge BOM and recipe ??"

            closest_batch_size = min(
                recipe[RecipeST.cols.batch_size], key=lambda x: abs(x - batch_size)
            )
            for recipe_row in recipe[
                recipe[RecipeST.cols.batch_size] == closest_batch_size
            ].itertuples():
                recipe_row = recipe_row._asdict()
                new_fg_data.append(
                    [
                        recipe_row[RecipeST.cols.material_id],
                        recipe_row[RecipeST.cols.operation],
                        recipe_row[RecipeST.cols.op_order],
                        recipe_row[RecipeST.cols.alt_recipe],
                        recipe_row[RecipeST.cols.resource_id],
                        recipe_row[RecipeST.cols.min_lot_size],
                        recipe_row[RecipeST.cols.max_lot_size],
                        batch_size,
                        recipe_row[RecipeST.cols.setuptime_per_batch],
                        recipe_row[RecipeST.cols.runtime_per_batch]
                        * (batch_size / closest_batch_size),
                        MaterialType.fg.value,
                        recipe_row[ProductDescST.cols.material_name],
                        recipe_row[ProductDescST.cols.material_unit],
                    ]
                )

        df_fg_recipe = pd.DataFrame(
            new_fg_data,
            columns=[
                RecipeST.cols.material_id,
                RecipeST.cols.operation,
                RecipeST.cols.op_order,
                RecipeST.cols.alt_recipe,
                RecipeST.cols.resource_id,
                RecipeST.cols.min_lot_size,
                RecipeST.cols.max_lot_size,
                RecipeST.cols.batch_size,
                RecipeST.cols.setuptime_per_batch,
                RecipeST.cols.runtime_per_batch,
                ProductDescST.cols.material_type,
                ProductDescST.cols.material_name,
                ProductDescST.cols.material_unit,
            ],
        )

        # Add material type to SFG recipe as well
        df_sfg_recipe[ProductDescST.cols.material_type] = df_sfg_recipe[
            RecipeST.cols.material_id
        ].map(
            dict(zip(df_bom[BOMST.cols.material_id], df_bom[BOMST.cols.material_type]))
        )

        # Concat df
        df = pd.concat([df_sfg_recipe, df_fg_recipe])

        # Drop (Code, batch size, alt recipe) if resource id is not given
        mask = pd.MultiIndex.from_frame(
            df.loc[
                df[RecipeST.cols.resource_id].isna(),
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

        # Explode by resource ids
        def _get_resource_names(resource_names_str):
            resource_names = []
            for resource_name in resource_names_str.split(","):
                s = resource_name.split("_")[0]
                if s.startswith("W"):
                    s = s[1:]
                resource_names.append(s)
            return resource_names

        df[RecipeST.cols.resource_id] = df[RecipeST.cols.resource_id].apply(
            _get_resource_names
        )
        df = df.explode(RecipeST.cols.resource_id).reset_index(drop=True)

        # Assign resource type
        df[RecipeST.cols.resource_type] = ResourceType.room.value
        df.loc[
            df[RecipeST.cols.resource_id].str.len() > 6, RecipeST.cols.resource_type
        ] = ResourceType.machine.value

        # Reduce runtime
        df[RecipeST.cols.runtime_per_batch] *= self.runtime_factor

        # Make min wait time same as runtim
        df[RecipeST.cols.min_wait_time] = df[RecipeST.cols.runtime_per_batch] * 0.9

        # Reduce WHD and QCD wait time
        df.loc[
            df[RecipeST.cols.resource_id] == "QCD", RecipeST.cols.min_wait_time
        ] *= self.qcd_waittime_factor
        df.loc[
            df[RecipeST.cols.resource_id] == "WHD", RecipeST.cols.min_wait_time
        ] *= self.whd_waittime_factor

        return df

    def _filter_recipe(self, df_recipe: pd.DataFrame) -> pd.DataFrame:
        df_recipe = df_recipe[RecipeST.get_fields()].astype(RecipeST.get_dtypes())
        return df_recipe

    def _create_products_desc(
        self, df_recipe: pd.DataFrame, packsize_mapping: Dict[str, int]
    ):
        # Filter out important cols only
        df = df_recipe.rename(
            {
                RecipeST.cols.material_id: ProductDescST.cols.material_id,
                RecipeST.cols.batch_size: ProductDescST.cols.batch_size,
            },
            axis=1,
        ).drop_duplicates(
            subset=[ProductDescST.cols.material_id, ProductDescST.cols.batch_size]
        )

        df[ProductDescST.cols.count_factor] = (
            df[ProductDescST.cols.material_unit]
            .astype(str)
            .map(lambda x: packsize_mapping.get(x, 1))
        )

        # Assign family ID, clubbing and batching mode
        df[ProductDescST.cols.family_id] = df[ProductDescST.cols.material_id]
        df[ProductDescST.cols.clubbing_mode] = ClubbingMode.clubbing.value
        df.loc[
            df[ProductDescST.cols.material_type] == MaterialType.fg.value,
            ProductDescST.cols.batching_mode,
        ] = BatchingMode.treal.value
        df.loc[
            df[ProductDescST.cols.material_type] != MaterialType.fg.value,
            ProductDescST.cols.batching_mode,
        ] = BatchingMode.tint.value

        # Filter out important cols only
        df = df[ProductDescST.get_fields()].astype(ProductDescST.get_dtypes())

        return df

    def _load_rmpm_inventory(
        self, input_data: PandasFilePath, material_type: str, c2c_graph: DSU
    ) -> pd.DataFrame:
        # Read RM/PM data
        df = pd.read_excel(input_data, dtype="object")

        # Use unrestricted + quality inspection as quantity
        df[InventoryST.cols.quantity] = df["Unrestricted"]
        if self.use_rmpm_qi:
            df[InventoryST.cols.quantity] += df["Quality Inspection"]

        # Preprocess fields
        df.rename(
            {
                "Material": InventoryST.cols.material_id,
            },
            axis=1,
            inplace=True,
        )

        # Drop batches which have no storage location
        df.dropna(
            subset=[InventoryST.cols.material_id, "Storage location"], inplace=True
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
        df[InventoryST.cols.material_type] = material_type

        # Filter out important cols only
        df = df[InventoryST.get_fields()].astype(InventoryST.get_dtypes())

        return df

    def _load_products_inventory(
        self,
        input_data: PandasFilePath,
        material_type: Optional[str] = None,
        df_products_desc: pd.DataFrame = None,
    ) -> pd.DataFrame:
        # Read data
        df = pd.read_excel(input_data, dtype="object")

        # Preprocess fields
        df.rename(
            {
                "Material": InventoryST.cols.material_id,
                "Unrestricted": InventoryST.cols.quantity,
            },
            axis=1,
            inplace=True,
        )

        # Drop empty material ids
        df.dropna(subset=[InventoryST.cols.material_id], inplace=True)

        # TODO: Remove products which expired before execution end time

        # Sum up quantities
        df[InventoryST.cols.quantity] = df[InventoryST.cols.quantity].astype(np.float64)
        df = df.groupby(InventoryST.cols.material_id, as_index=False).agg(
            {InventoryST.cols.quantity: "sum"}
        )

        # Assign material type
        if material_type is None:
            df[InventoryST.cols.material_type] = MaterialType.sfg.value
            common_blend_ids = set(
                df_products_desc[
                    df_products_desc[ProductDescST.cols.material_type]
                    == MaterialType.cb.value
                ][ProductDescST.cols.material_id]
            )
            df.loc[
                df[InventoryST.cols.material_id].isin(common_blend_ids),
                InventoryST.cols.material_type,
            ] = MaterialType.cb.value

        else:
            df[InventoryST.cols.material_type] = material_type

        # Filter out important cols only
        df = df[InventoryST.get_fields()].astype(InventoryST.get_dtypes())

        return df

    def _load_procurement_plan(self, input_data: PandasFilePath) -> pd.DataFrame:
        # Read excel
        df_po = pd.read_excel(input_data, sheet_name="Open POs", dtype="object")
        df_pr = pd.read_excel(input_data, sheet_name="PR Detail", dtype="object")

        # Rename fields
        df_po.rename(
            {
                "Material": ProcurementST.cols.material_id,
                "Still to be delivered (qty)": ProcurementST.cols.quantity,
                "Delivery date": ProcurementST.cols.available_at,
            },
            axis=1,
            inplace=True,
        )
        df_pr.rename(
            {
                "Material": ProcurementST.cols.material_id,
                "Quantity requested": ProcurementST.cols.quantity,
                "Deliv. date(From/to)": ProcurementST.cols.available_at,
            },
            axis=1,
            inplace=True,
        )

        # Parse dates
        df_po[ProcurementST.cols.available_at] = pd.to_datetime(
            df_po[ProcurementST.cols.available_at], format="%d/%m/%Y"
        )
        df_pr[ProcurementST.cols.available_at] = pd.to_datetime(
            df_pr[ProcurementST.cols.available_at], format="%d/%m/%Y"
        )

        # Concat
        df = pd.concat(
            [
                df_po[list(ProcurementST.cols._fields)],
                df_pr[list(ProcurementST.cols._fields)],
            ]
        )

        # Filter
        df = df[df[ProcurementST.cols.available_at] <= self.execution_end]
        df = df[df[ProcurementST.cols.quantity] > 0]

        # Filter out the columns
        df = df[ProcurementST.get_fields()].astype(ProcurementST.get_dtypes())
        return df

    def _load_plant_map(
        self, input_data: PandasFilePath, df_recipe: pd.DataFrame
    ) -> pd.DataFrame:
        # Read data
        df = pd.read_excel(input_data, header=1, dtype="object")

        # Preprocess fields
        df.rename(
            {
                "Block ": PlantMapST.cols.block_id,
                "Room Code": PlantMapST.cols.room_id,
                "Equipment Code": PlantMapST.cols.machine_id,
                "Type-A (Hrs)": RoomChangeoverST.cols.type_A,
                "Type-B (Hrs)": RoomChangeoverST.cols.type_B,
            },
            axis=1,
            inplace=True,
        )

        # TODO: Change operation of fixed machines
        df[PlantMapST.cols.operation] = "Fixed Machine Operation"
        df[PlantMapST.cols.machine_type] = MachineType.formulation.value

        # Add portable machines
        unique_block_ids = df[PlantMapST.cols.block_id].unique()
        pseudo_machine_no = 0
        pseudo_room_no = 0
        recipe = df_recipe[
            [
                RecipeST.cols.resource_id,
                RecipeST.cols.resource_type,
                RecipeST.cols.operation,
            ]
        ].drop_duplicates()

        # Iterate over machines first then room
        for row in recipe.sort_values(RecipeST.cols.resource_type).itertuples():
            row = row._asdict()
            operation = row[RecipeST.cols.operation]

            to_append = False
            if row[RecipeST.cols.resource_type] == ResourceType.room.value:
                subdf = df[
                    df[PlantMapST.cols.room_id] == row[RecipeST.cols.resource_id]
                ]
                curr_room_id = row[RecipeST.cols.resource_id]
                curr_machine_type = (
                    MachineType.packing
                    if curr_room_id[2] == "P"
                    else MachineType.formulation
                )
                if (
                    curr_room_id == "WHD"
                    or curr_room_id == "QCD"
                    or operation == "PRINTING & INSPECTION"
                ):
                    curr_machine_type = MachineType.infinite

                if len(subdf) > 0:
                    if len(subdf[subdf[RecipeST.cols.operation] == operation]) == 0:
                        curr_block_id = subdf[PlantMapST.cols.block_id].iloc[0]
                        curr_machine_id = f"Machine{str(pseudo_machine_no).zfill(3)}"
                        pseudo_machine_no += 1
                        to_append = True

                else:
                    if curr_room_id == "WHD":
                        curr_block_id = "W"
                        curr_machine_type = MachineType.infinite
                    elif curr_room_id == "QCD":
                        curr_block_id = "Q"
                        curr_machine_type = MachineType.infinite
                    else:
                        if curr_room_id[1] in unique_block_ids:
                            curr_block_id = curr_room_id[1]
                        else:
                            curr_block_id = choice(unique_block_ids)

                    curr_machine_id = f"Machine{str(pseudo_machine_no).zfill(3)}"
                    pseudo_machine_no += 1
                    to_append = True

            else:
                subdf = df[
                    df[PlantMapST.cols.machine_id] == row[RecipeST.cols.resource_id]
                ]
                curr_machine_id = row[RecipeST.cols.resource_id]
                curr_machine_type = MachineType.formulation

                if len(subdf) > 0:
                    curr_block_id = subdf[PlantMapST.cols.block_id].iloc[0]
                    curr_room_id = subdf[PlantMapST.cols.room_id].iloc[0]

                    if subdf[PlantMapST.cols.operation].iloc[0] != operation:
                        # Change operation
                        df.loc[
                            df[PlantMapST.cols.machine_id] == curr_machine_id,
                            PlantMapST.cols.operation,
                        ] = operation

                else:
                    curr_block_id = choice(unique_block_ids)
                    curr_room_id = f"Room{str(pseudo_room_no).zfill(3)}"
                    pseudo_room_no += 1
                    to_append = True

            if to_append:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                PlantMapST.cols.block_id: [curr_block_id],
                                PlantMapST.cols.room_id: [curr_room_id],
                                PlantMapST.cols.machine_id: [curr_machine_id],
                                PlantMapST.cols.operation: [operation],
                                PlantMapST.cols.machine_type: [curr_machine_type.value],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

        # Handle changeover
        avg_changeover = df.groupby(PlantMapST.cols.room_id)[
            [RoomChangeoverST.cols.type_A, RoomChangeoverST.cols.type_B]
        ].mean()
        df = df.merge(avg_changeover, on=PlantMapST.cols.room_id, suffixes=("_x", ""))
        df = df.drop(
            [
                RoomChangeoverST.cols.type_A + "_x",
                RoomChangeoverST.cols.type_B + "_x",
            ],
            axis=1,
        )

        # Remove fixed operation machine
        df = df[df[PlantMapST.cols.operation] != "Fixed Machine Operation"]

        return df

    def _filter_plant_map(self, df_plant_map: pd.DataFrame) -> pd.DataFrame:
        # Filter out important cols only
        df_plant_map = (
            df_plant_map[PlantMapST.get_fields()]
            .reset_index()
            .astype(PlantMapST.get_dtypes())
        )
        return df_plant_map

    def _create_room_changeover(self, df: pd.DataFrame) -> pd.DataFrame:
        # Rename cols
        df = df.rename(
            {PlantMapST.cols.room_id: RoomChangeoverST.cols.room_id}, axis=1
        )[
            [
                RoomChangeoverST.cols.room_id,
                RoomChangeoverST.cols.type_A,
                RoomChangeoverST.cols.type_B,
            ]
        ]

        # Fill NA with 0s
        df = df.drop_duplicates().fillna(0)

        # Experiment feature
        if self.is_changeoverB_equals_A:
            df[RoomChangeoverST.cols.type_B] = df[RoomChangeoverST.cols.type_A]

        return df

    def _load_crossblock_penalties(self, input_data: PandasFilePath) -> pd.DataFrame:
        # Read data
        df = pd.read_csv(input_data)

        # Filter out important cols only
        df = df[CrossBlockST.get_fields()].astype(CrossBlockST.get_dtypes())
        return df

    def _load_packsize_mapping(self, input_data: PandasFilePath) -> pd.DataFrame:
        # Read data
        df = pd.read_csv(input_data)

        packsize_mapping = dict(
            zip(
                df["PackUnit"],
                df["PackSize"],
            )
        )

        return packsize_mapping

    def _load_phantom_items(self, input_data: PandasFilePath) -> pd.DataFrame:
        # Read excel
        df_pm = pd.read_excel(input_data, sheet_name="PM", dtype="object")
        df_rm = pd.read_excel(input_data, sheet_name="RM", dtype="object")

        # Add material type
        df_pm[PhantomST.cols.material_type] = MaterialType.pm.value
        df_rm[PhantomST.cols.material_type] = MaterialType.rm.value

        df = pd.concat([df_rm, df_pm])

        # Preprocess fields
        df.rename(
            {
                "Material": ProcurementST.cols.material_id,
            },
            axis=1,
            inplace=True,
        )

        # Filter out the columns
        df = df[PhantomST.get_fields()].astype(PhantomST.get_dtypes())
        return df

    def _load_code_to_code(self, input_data: PandasFilePath, c2c_graph: DSU) -> None:
        # Read data
        df = pd.read_excel(input_data)

        # Create code to code pairs
        code_to_code_pairs = set()

        def _add_code_batches(g):
            g = (
                g[["Material", "Quantity"]]
                .sort_values("Quantity")
                .astype({"Material": "string", "Quantity": np.float64})
                .values
            )
            assert len(g) % 2 == 0

            i = 0
            j = len(g) - 1
            while i < j:
                if float(g[i][1]) + float(g[j][1]) == 0:
                    code_to_code_pairs.add((g[i][0], g[j][0]))
                i += 1
                j -= 1

        df.groupby("Batch").apply(_add_code_batches)

        # Code to code graph
        for pair in code_to_code_pairs:
            c2c_graph.union(pair[0], pair[1])

    def _load_printing_xy(self, input_data: PandasFilePath, c2c_graph: DSU) -> None:
        # Read data
        df = pd.read_excel(input_data)

        # Add to c2c graph
        for x, y in zip(df["Plain foil code"], df["PM code"]):
            c2c_graph.union(x, y)

    def _load_underprocess(
        self, input_data: PandasFilePath, material_type: str, datefmt="%d-%m-%Y"
    ) -> pd.DataFrame:
        # Read excel
        df = pd.read_excel(input_data, dtype="object")

        # Rename
        df.rename(
            {
                "Material": InventoryST.cols.material_id,
                "Target Qty": InventoryST.cols.quantity,
            },
            axis=1,
            inplace=True,
        )

        # Convert to datetime
        df["Release"] = pd.to_datetime(df["Release"], format=datefmt)
        mask = df["Release"] + timedelta(days=4) <= self.execution_start

        # Assumed to be done in next 4 days so send to inventory
        moved_to_inventory = df.loc[
            mask, [InventoryST.cols.material_id, InventoryST.cols.quantity]
        ].copy()
        moved_to_inventory[InventoryST.cols.material_type] = material_type

        # Make probabilistic sequence for others
        return df[~mask], moved_to_inventory

    def _load_intial_state(
        self, df_sfg_underprocess: pd.DataFrame, df_fg_underprocess: pd.DataFrame
    ) -> pd.DataFrame:
        return None
