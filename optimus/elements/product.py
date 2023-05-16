"""
 RIPIK AI
 @author    : Himanshu Mittal
"""

import pandas as pd
from typing import Dict, List, Union, Optional
from collections import defaultdict, Counter

from ..utils.constants import MaterialType, ResourceType
from ..utils.general import multidict, RandomizedSet, compare_linking_op
from ..utils.structs import *
from ..activity import BaseActivityManager


class Material:
    def __init__(
        self,
        material_id: str,
        material_name: str,
        family_id: str,
        material_type: MaterialType,
        material_unit: str,
        count_factor: int,
        clubbing_mode: int,
        batching_mode: int,
        bom: Dict[float, Dict[str, Dict[str, float]]],
        recipe: Dict[float, Dict[str, List[Dict[str, Union[str, RandomizedSet]]]]],
        inverse_bom: Dict[str, Dict[str, List[Tuple[str, float, str, str]]]],
    ) -> None:
        self.material_id = material_id
        self.material_name = material_name
        self.family_id = family_id
        self.material_type = material_type
        self.material_unit = material_unit
        self.count_factor = count_factor
        self.clubbing_mode = clubbing_mode
        self.batching_mode = batching_mode

        self.bom = bom
        self.recipe = recipe
        self.inverse_bom = inverse_bom
        self.dedicated_blocks = self._find_dedicated_blocks()

    def _find_dedicated_blocks(self) -> Dict[float, Dict[str, List[str]]]:
        dedicated_blocks = multidict(2, Counter)
        for batch_size in self.recipe:
            for alt_recipe in self.recipe[batch_size]:
                for recipe_list_item in self.recipe[batch_size][alt_recipe]:
                    curr_blocks = RandomizedSet()
                    for machine in recipe_list_item["machines"]:
                        curr_blocks.add(machine.block_id)
                    for block_id in curr_blocks:
                        dedicated_blocks[batch_size][alt_recipe][block_id] += 1
                dedicated_blocks[batch_size][alt_recipe] = [
                    p[0] for p in dedicated_blocks[batch_size][alt_recipe].most_common()
                ]
        return dedicated_blocks

    def get_batch_sizes(self) -> List[float]:
        return list(self.bom.keys())

    def get_bom(
        self, batch_size: float, alt_bom: str
    ) -> Dict[str, Dict[str, Union[float, int]]]:
        return self.bom[batch_size][alt_bom]

    def get_inverse_bom(
        self, batch_size: float, alt_bom: str
    ) -> List[Tuple[str, float, str, str]]:
        return self.inverse_bom[batch_size][alt_bom]

    def iterate_boms(
        self, batch_size: Optional[float] = None, alt_bom: Optional[str] = None
    ) -> Tuple[float, str]:
        for curr_batch_size in self.bom:
            if batch_size is not None and curr_batch_size != batch_size:
                continue
            for curr_alt_bom in self.bom[curr_batch_size]:
                if alt_bom is not None and curr_alt_bom != alt_bom:
                    continue
                yield curr_batch_size, curr_alt_bom

    def iterate_recipes(
        self, batch_size: Optional[float] = None, alt_recipe: Optional[str] = None
    ) -> Tuple[float, str]:
        for curr_batch_size in self.recipe:
            if batch_size is not None and curr_batch_size != batch_size:
                continue
            for curr_alt_recipe in self.recipe[curr_batch_size]:
                if alt_recipe is not None and curr_alt_recipe != alt_recipe:
                    continue
                yield curr_batch_size, curr_alt_recipe

    def get_indirect_composition(
        self, batch_size: float, alt_bom: str
    ) -> Dict[str, Union[float, int]]:
        return {
            component_group: component
            for component_group, component in self.get_bom(
                batch_size=batch_size, alt_bom=alt_bom
            ).items()
            if component["indirect"] > 0
        }

    def get_material_type(self) -> str:
        return self.material_type.value

    def get_recipe(self, batch_size: float, alt_recipe: str) -> Dict[str, float]:
        return self.recipe[batch_size][alt_recipe]

    def get_dedicated_blocks(self, batch_size: float, alt_recipe: str):
        return self.dedicated_blocks[batch_size][alt_recipe]

    def approved_machines_of(self, batch_size: float, alt_recipe: str, sequence: int):
        return self.recipe[batch_size][alt_recipe][sequence]["machines"]

    def operation_of(self, batch_size: float, alt_recipe: str, sequence: int):
        return self.recipe[batch_size][alt_recipe][sequence]["operation"]

    def step_description_of(self, batch_size: float, alt_recipe: str, sequence: int):
        return self.recipe[batch_size][alt_recipe][sequence]["step_description"]

    def op_order_of(self, batch_size: float, alt_recipe: str, sequence: int):
        return self.recipe[batch_size][alt_recipe][sequence]["op_order"]

    def is_primitive(self, batch_size: float, alt_bom: str) -> bool:
        for component in self.bom[batch_size][alt_bom].values():
            if component["indirect"] > 0:
                return False
        return True


def initialize_products(
    df_products_desc: pd.DataFrame,
    df_bom: pd.DataFrame,
    df_recipe: pd.DataFrame,
    df_plant_map: pd.DataFrame,
    machines,
    activity_manager: BaseActivityManager,
) -> Dict[str, Material]:
    """
    Initialize all products given all product's description

    Parameters
    -------------------------
    product_desc: Fields are product_id (pk) and other attributes
    recipes: Material to ordered sequence of functions (describes process)
    """
    # Get all approved equipments
    approved_machines = defaultdict(RandomizedSet)
    for row in df_recipe.itertuples():
        row = row._asdict()
        key = (
            row[RecipeST.cols.material_id],
            row[RecipeST.cols.batch_size],
            row[RecipeST.cols.alt_recipe],
            row[RecipeST.cols.op_order],
        )
        if row[RecipeST.cols.resource_type] == ResourceType.machine.value:
            approved_machines[key].add(machines[row[RecipeST.cols.resource_id]])

        elif row[RecipeST.cols.resource_type] == ResourceType.room.value:
            for machine_id in df_plant_map[
                (
                    df_plant_map[PlantMapST.cols.room_id]
                    == row[RecipeST.cols.resource_id]
                )
                & (
                    df_plant_map[PlantMapST.cols.operation]
                    == row[RecipeST.cols.operation]
                )
            ][PlantMapST.cols.machine_id]:
                approved_machines[key].add(machines[machine_id])

        else:
            raise ValueError(
                f"Invalid resource type: {row[RecipeST.cols.resource_type]}"
            )

    products = {}
    for material_id, descs in df_products_desc.groupby(ProductDescST.cols.material_id):
        # Get material name and material type
        material_name = descs[ProductDescST.cols.material_name].unique()
        family_id = descs[ProductDescST.cols.family_id].unique()
        material_type = descs[ProductDescST.cols.material_type].unique()
        material_unit = descs[ProductDescST.cols.material_unit].unique()
        count_factor = descs[ProductDescST.cols.count_factor].unique()
        clubbing_mode = descs[ProductDescST.cols.clubbing_mode].unique()
        batching_mode = descs[ProductDescST.cols.batching_mode].unique()
        assert (
            len(material_name) == 1
            and len(family_id) == 1
            and len(material_type) == 1
            and len(material_unit) == 1
            and len(count_factor) == 1
            and len(clubbing_mode) == 1
            and len(batching_mode) == 1
        )

        material_name = material_name[0]
        family_id = family_id[0]
        material_type = MaterialType(material_type[0])
        material_unit = material_unit[0]
        count_factor = count_factor[0]
        clubbing_mode = clubbing_mode[0]
        batching_mode = batching_mode[0]

        # Per batch make BOM and recipe
        bom = multidict(3, dict)
        recipe = multidict(2, list)
        inverse_bom = multidict(2, list)
        for desc in descs.itertuples():
            desc = desc._asdict()
            batch_size = desc[ProductDescST.cols.batch_size]

            # Build recipe
            df = df_recipe[
                (df_recipe[RecipeST.cols.material_id] == material_id)
                & (df_recipe[RecipeST.cols.batch_size] == batch_size)
            ]
            for alt_recipe, recipe_group in df.groupby(RecipeST.cols.alt_recipe):
                recipe_group = (
                    recipe_group[
                        [
                            RecipeST.cols.operation,
                            RecipeST.cols.step_description,
                            RecipeST.cols.op_order,
                        ]
                    ]
                    .drop_duplicates()
                    .sort_values(
                        RecipeST.cols.op_order, key=activity_manager.compare_op_order
                    )
                )
                recipe_list = []
                for row in recipe_group.itertuples():
                    row = row._asdict()
                    op_order = row[RecipeST.cols.op_order]
                    recipe_list_item = {
                        "operation": row[RecipeST.cols.operation],
                        "step_description": row[RecipeST.cols.step_description],
                        "op_order": op_order,
                        "machines": approved_machines[
                            (material_id, batch_size, alt_recipe, op_order)
                        ],
                    }
                    recipe_list.append(recipe_list_item)
                recipe[batch_size][alt_recipe] = recipe_list

            # Build bom
            df = df_bom[
                (df_bom[BOMST.cols.material_id] == material_id)
                & (df_bom[BOMST.cols.material_quantity] == batch_size)
            ]
            alt_boms = list(df[BOMST.cols.alt_bom].unique())
            for row in df.itertuples():
                row = row._asdict()
                alt_bom = row[BOMST.cols.alt_bom]
                component_group = row[BOMST.cols.component_group]
                component_id = row[BOMST.cols.component_id]
                component_quantity = row[BOMST.cols.component_quantity]
                indirect = row[BOMST.cols.indirect]
                if (
                    indirect > 0
                    and len(
                        df_products_desc[
                            df_products_desc[ProductDescST.cols.material_id]
                            == component_id
                        ]
                    )
                    == 0
                ):
                    # Inderect component not in master
                    raise ValueError(
                        f"BOM for indirect component {component_id} not available"
                    )

                if "indirect" not in bom[batch_size][alt_bom][component_group]:
                    bom[batch_size][alt_bom][component_group] = {
                        "quantity": 0.0,
                        "component_ids": [],
                    }
                else:
                    assert (
                        bom[batch_size][alt_bom][component_group]["indirect"]
                        == indirect
                    ), "Indirect can't be different for the component group in the BOM"

                bom[batch_size][alt_bom][component_group][
                    "quantity"
                ] += component_quantity
                bom[batch_size][alt_bom][component_group]["component_ids"].append(
                    component_id
                )
                bom[batch_size][alt_bom][component_group]["indirect"] = indirect

            # Build inverse bom
            df = df_bom[(df_bom[BOMST.cols.component_id] == material_id)]
            for row in df.itertuples():
                row = row._asdict()
                parent_material_id = row[BOMST.cols.material_id]
                parent_material_quantity = row[BOMST.cols.material_quantity]
                parent_alt_bom = row[BOMST.cols.alt_bom]
                component_group = row[BOMST.cols.component_group]
                component_quantity = row[BOMST.cols.component_quantity]
                indirect = row[BOMST.cols.indirect]

                if compare_linking_op(
                    available=batch_size, needed=component_quantity, mode=indirect
                ):
                    for alt_bom in alt_boms:
                        inverse_bom[batch_size][alt_bom].append(
                            (
                                parent_material_id,
                                parent_material_quantity,
                                parent_alt_bom,
                                component_group,
                            )
                        )

            if (
                batch_size not in bom
                or len(bom[batch_size]) == 0
                or batch_size not in recipe
                or len(recipe[batch_size]) == 0
            ):
                raise ValueError(f"BOM and recipe not merged properly")

        assert len(bom) == len(recipe)
        if len(bom) > 0 and len(recipe) > 0:
            product = Material(
                material_id=material_id,
                material_name=material_name,
                family_id=family_id,
                material_type=material_type,
                material_unit=material_unit,
                count_factor=count_factor,
                clubbing_mode=clubbing_mode,
                batching_mode=batching_mode,
                bom=bom,
                recipe=recipe,
                inverse_bom=inverse_bom,
            )
            products[material_id] = product

    return products
