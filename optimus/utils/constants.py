"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from enum import Enum

######################### ENUMS #########################
class MaterialType(Enum):
    rm = "RawMaterial"
    pm = "PackingMaterial"
    cb = "CommonBlend"
    sfg = "SFG"
    fg = "FG"


class ClubbingMode(Enum):
    standard = 0
    clubbing = 1


class BatchingMode(Enum):
    treal = 0
    tint = 1


class MachineType(Enum):
    formulation = "formulation"
    packing = "packing"
    infinite = "infinite"


class ResourceType(Enum):
    block = "block"
    room = "room"
    machine = "machine"


class PrescheduleMode(Enum):
    feedforward = "feedforward"
    feedbackward = "feedbackward"


class InfoSpreading(Enum):
    horizontal = "horizontal"
    vertical = "vertical"
    plus = "plus"


class DroppedReason(Enum):
    NO_DEMAND = "No demand in pulling months"
    MISSING_INFO = "Missing information"
    COVERED_DEMAND = "Already covered in the inventory"
    LOW_DEMAND = "Aggregated demand less than a batch size"
    LOW_RM_PM_CONT = "Low RM/PM (no batching constraint)"
    LOW_RM_PM_BATCH = "Low RM/PM (with batching constraint)"
    DISTRIBUTED_TO_OTHERS = "RM/PM distributed to other priority products"


class BatchSizeLinking(Enum):
    GT = 1
    GE = 2
    EQ = 3
    LE = 4
    LT = 5
    NA = 6


class CAMode(Enum):
    CAI = "CAI"
    CAD = "CAD"
    CAV = "CAV"


class ProductionStrategy(Enum):
    MTS = "MTS"
    MTO = "MTO"
