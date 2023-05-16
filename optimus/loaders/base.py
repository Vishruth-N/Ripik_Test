"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import os
from typing import Union, Dict, Callable, List
from pandas import DataFrame
from pandas._typing import FilePath, ReadCsvBuffer

PandasFilePath = Union[FilePath, ReadCsvBuffer[bytes], ReadCsvBuffer[str]]


class BaseLoader:
    def __init__(self, debug: bool = False, debug_dir: str = None) -> None:
        self.debug = debug
        self.debug_dir = debug_dir

    @property
    def required_keys(self) -> List[str]:
        raise NotImplementedError()

    @property
    def optional_keys(self) -> List[str]:
        raise NotImplementedError()

    @property
    def output_keys(self) -> List[str]:
        return [
            "df_forecasted_demand",
            "df_inventory",
            "df_procurement_plan",
            "df_products_desc",
            "df_bom",
            "df_recipe",
            "df_plant_map",
            "df_room_changeover",
            "df_crossblock_penalties",
            "df_phantom_items",
            "df_machine_changeover",
            "df_machine_availability",
            "df_initial_state",
        ]

    @classmethod
    def validate_keys(cls, func: Callable) -> Dict[str, DataFrame]:
        def inner(
            self, data_files: Dict[str, PandasFilePath], *args, **kwargs
        ) -> Dict[str, DataFrame]:
            # Validate input keys
            keys = data_files.keys()
            for required_key in self.required_keys:
                if required_key not in keys:
                    raise ValueError(
                        f"'{required_key}' is required in input data keys!"
                    )

            for key in keys:
                if key not in self.required_keys and key not in self.optional_keys:
                    raise ValueError(
                        f"'{key}' cannot be interpreted as input data keys!"
                    )

            # Load
            output = func(self, data_files, *args, **kwargs)

            # Validate output keys
            for output_key in self.output_keys:
                if output_key not in output:
                    raise ValueError(f"'{output_key}' not found. Check loader output!")

            # Save if
            if self.debug:
                cleaned_dir = os.path.join(self.debug_dir, "cleaned/")
                if not os.path.exists(cleaned_dir):
                    os.makedirs(cleaned_dir)

                for output_key, df in output.items():
                    if df is None:
                        continue
                    df.to_csv(
                        os.path.join(cleaned_dir, f"{output_key}.csv"),
                        index=False,
                    )

            return output

        return inner

    def load_all(self, data_files: Dict[str, PandasFilePath]) -> Dict[str, DataFrame]:
        raise NotImplementedError()
