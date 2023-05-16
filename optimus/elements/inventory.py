"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import bisect
import pandas as pd
from datetime import datetime
from copy import deepcopy
from collections import defaultdict
from typing import Dict, TypeVar, Any
from ..utils.structs import *

SelfInventory = TypeVar("SelfInventory", bound="Inventory")


class Inventory:
    ERROR_BUFFER = 0.1
    INFINITE_QUANTITY = 1e15

    def __init__(
        self,
        df_inventory: pd.DataFrame = None,
        df_phantom_items: pd.DataFrame = None,
        df_procurement_plan: pd.DataFrame = None,
    ) -> None:
        self._inventory = defaultdict(float)
        if df_inventory is not None:
            for row in df_inventory.itertuples():
                row = row._asdict()
                self.add(
                    row[InventoryST.cols.material_id], row[InventoryST.cols.quantity]
                )

        self._procurement_plan = pd.DataFrame(
            columns=ProcurementST.get_fields()
        ).astype(ProcurementST.get_dtypes())
        if df_procurement_plan is not None:
            self._procurement_plan = df_procurement_plan

        if df_phantom_items is not None:
            for row in df_phantom_items.itertuples():
                row = row._asdict()
                self.add(row[PhantomST.cols.material_id], Inventory.INFINITE_QUANTITY)

    def add(self, material_id: str, quantity: float) -> None:
        """Add the given quantity to the material_id stock"""
        if material_id not in self._inventory:
            self._inventory[material_id] = quantity
        else:
            self._inventory[material_id] += quantity

    def update(self, material_id: str, quantity: float) -> None:
        """Set the given quantity to the material_id stock"""
        self._inventory[material_id] = quantity

    def decrease(self, material_id: str, quantity: float) -> None:
        """Reduce the given quantity to the given material_id stock"""
        if (
            material_id not in self._inventory
            or self._inventory[material_id] + Inventory.ERROR_BUFFER < quantity
        ):
            raise ValueError(
                f"Unable to consume {quantity} of {material_id} because only {self._inventory[material_id]} left"
            )

        _initial_inventory = self._inventory[material_id]
        self._inventory[material_id] -= quantity
        if -Inventory.ERROR_BUFFER < self._inventory[material_id] < 0:
            self._inventory[material_id] = 0.0
        assert (
            self._inventory[material_id] >= 0.0
        ), f"{material_id} inventory went negative from {_initial_inventory} to {self._inventory[material_id]}"

    def get_quantity(self, material_id: str) -> float:
        return self._inventory[material_id]

    def get_procured_quantity(
        self, start_date: datetime = None, end_date: datetime = None
    ):
        """All quantity inclusive start and exlusive end date"""
        df = self._procurement_plan
        if start_date is not None:
            df = df[df[ProcurementST.cols.available_at] >= start_date]
        if end_date is not None:
            df = df[df[ProcurementST.cols.available_at] < end_date]

        procured_quantity = (
            df.groupby(ProcurementST.cols.material_id)[ProcurementST.cols.quantity]
            .sum()
            .to_dict()
        )
        return procured_quantity

    def set_inventory(self, inventory: Dict) -> None:
        """Set inventory"""
        self._inventory = inventory

    def copy(self) -> SelfInventory:
        new_inventory = Inventory()
        new_inventory.set_inventory(deepcopy(self._inventory))
        return new_inventory

    def to_dict(self) -> Dict:
        return defaultdict(float, self._inventory)
