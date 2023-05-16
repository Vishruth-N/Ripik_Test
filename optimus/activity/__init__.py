"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

from .base import BaseActivityManager
from .ripik import RipikActivityManager
from .sunpharma_paonta_sahib import SunPharmaPaontaSahibActivityManager
from .sunpharma_dewas import SunPharmaDewasActivityManager
from .sunpharma_baska import SunPharmaBaskaActivityManager


def get_activity_manager(client_id: str) -> BaseActivityManager:
    if client_id == "ripik":
        return RipikActivityManager()

    elif client_id == "sunpharma_paonta_sahib":
        return SunPharmaPaontaSahibActivityManager()

    elif client_id == "sunpharma_dewas":
        return SunPharmaDewasActivityManager()

    elif client_id == "sunpharma_baska":
        return SunPharmaBaskaActivityManager()

    else:
        raise ValueError(f"Invalid client id: {client_id}")
