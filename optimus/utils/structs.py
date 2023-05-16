"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import numpy as np
from collections import namedtuple
from typing import Dict, NamedTuple, Any, List, Tuple

######################### STRUCTURES #########################
class BaseST:
    cols = None  # Names by which we access/process/save data
    dtypes = None  # Column types (pandas-compatible)

    @staticmethod
    def fill_data(
        attributes: List[str], dtypes: List[Any]
    ) -> Tuple[NamedTuple, NamedTuple]:
        # validating attributes
        assert all([s.isidentifier() for s in attributes]), "Invalid attribute name(s)!"
        ColName = namedtuple("ColName", attributes)
        DType = namedtuple("DType", attributes)

        # return values
        cols_ = ColName(*attributes)
        dtypes_ = DType(*dtypes)

        return cols_, dtypes_

    @classmethod
    def get_fields(cls) -> List[str]:
        return list(cls.cols._fields)

    @classmethod
    def get_dtypes(cls, cols: List[str] = None) -> Dict[str, Any]:
        if cls.dtypes is None:
            raise NotImplementedError(
                "Child class must use overwrite class variables. Try fill_data() function"
            )
        if cols is None:
            cols = cls.get_fields()
        cols_dict = cls.cols._asdict()
        dtype_dict = cls.dtypes._asdict()
        return {cols_dict[k]: dtype_dict[k] for k in cols}


class DemandST(BaseST):
    cols, dtypes = BaseST.fill_data(
        attributes=[
            "material_id",
            "production_strategy",
            "ca_mode",
            "m0_demand",
            "m0_commit",
            "m1_crit",
            "m1_std",
            "m2_crit",
            "m2_std",
            "m3_crit",
            "m3_std",
            "priority",
            "due_date",
        ],
        dtypes=[
            "string",
            "category",
            "category",
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.uint64,
            np.float64,
        ],
    )

    @classmethod
    def get_m1_cols(cls) -> List[str]:
        return [DemandST.cols.m1_crit, DemandST.cols.m1_std]

    @classmethod
    def hilo_priority_cols(cls) -> List[str]:
        return [
            DemandST.cols.m1_crit,
            DemandST.cols.m1_std,
            DemandST.cols.m2_crit,
            DemandST.cols.m2_std,
            DemandST.cols.m3_crit,
            DemandST.cols.m3_std,
        ]


class InventoryST(BaseST):
    cols, dtypes = BaseST.fill_data(
        attributes=["material_id", "material_type", "quantity"],
        dtypes=["string", "category", np.float64],
    )


class ProcurementST(BaseST):
    cols, dtypes = BaseST.fill_data(
        attributes=["material_id", "quantity", "available_at"],
        dtypes=["string", np.float64, "datetime64[ns]"],
    )


class PlantMapST(BaseST):
    cols, dtypes = BaseST.fill_data(
        attributes=["machine_id", "machine_type", "block_id", "room_id", "operation"],
        dtypes=["string", "category", "string", "string", "string"],
    )


class ProductDescST(BaseST):
    cols, dtypes = BaseST.fill_data(
        attributes=[
            "material_id",
            "material_type",
            "material_name",
            "family_id",
            "batch_size",
            "material_unit",
            "count_factor",
            "clubbing_mode",
            "batching_mode",
        ],
        dtypes=[
            "string",
            "category",
            "string",
            "string",
            np.float64,
            "string",
            np.uint64,
            np.uint32,
            np.uint32,
        ],
    )


class BOMST(BaseST):
    cols, dtypes = BaseST.fill_data(
        attributes=[
            "material_id",
            "material_type",
            "material_quantity",
            "alt_bom",
            "component_group",
            "component_id",
            "component_type",
            "component_quantity",
            "indirect",
        ],
        dtypes=[
            "string",
            "category",
            np.float64,
            "string",
            "string",
            "string",
            "category",
            np.float64,
            np.uint32,
        ],
    )


class RecipeST(BaseST):
    cols, dtypes = BaseST.fill_data(
        attributes=[
            "material_id",
            "operation",
            "step_description",
            "op_order",
            "alt_recipe",
            "resource_id",
            "resource_type",
            "min_lot_size",
            "max_lot_size",
            "batch_size",
            "setuptime_per_batch",
            "min_wait_time",
            "runtime_per_batch",
        ],
        dtypes=[
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
            "category",
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
        ],
    )


class MachineChangeoverST(BaseST):
    cols, dtypes = BaseST.fill_data(
        attributes=["machine_id", "material_id", "type_A"],
        dtypes=["string", "string", np.float64],
    )


class RoomChangeoverST(BaseST):
    cols, dtypes = BaseST.fill_data(
        attributes=["room_id", "type_A", "type_B"],
        dtypes=["string", np.float64, np.float64],
    )


class CampaignConstST(BaseST):
    cols, dtypes = BaseST.fill_data(
        attributes=[
            "material_id",
            "material_type",
            "batch_size",
            "family_id",
            "campaign_length",
        ],
        dtypes=["string", "category", np.float64, "string", np.uint64],
    )


class CrossBlockST(BaseST):
    cols, dtypes = BaseST.fill_data(
        attributes=["from_block_id", "to_block_id", "penalty"],
        dtypes=["string", "string", np.float64],
    )


class PhantomST(BaseST):
    cols, dtypes = BaseST.fill_data(
        attributes=["material_id", "material_type"],
        dtypes=["string", "category"],
    )


class InitialST(BaseST):
    cols, dtypes = BaseST.fill_data(
        attributes=[
            "material_id",
            "batch_size",
            "ca_mode",
            "alt_recipe",
            "alt_bom",
            "sequence",
            "machine_id",
            "start_time",
        ],
        dtypes=[
            "string",
            np.float64,
            "category",
            "string",
            "string",
            np.uint32,
            "string",
            np.float64,
        ],
    )


class MachineAvailabilityST(BaseST):
    cols, dtypes = BaseST.fill_data(
        attributes=["machine_id", "start_datetime", "end_datetime", "reason"],
        dtypes=["string", "datetime64[ns]", "datetime64[ns]", "string"],
    )


class HoldTimeST(BaseST):
    cols, dtypes = BaseST.fill_data(
        attributes=["op_order_A", "op_order_B", "min_holdtime", "max_holdtime"],
        dtypes=["string", "string", np.float64, np.float64],
    )
