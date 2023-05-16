from .base import BaseLoader, PandasFilePath
from .sunpharma_paonta_sahib import SunPharmaPaontaSahibLoader
from .sunpharma_dewas import SunPharmaDewasLoader
from .sunpharma_baska import SunPharmaBaskaLoader
from .ripik import RipikLoader
from datetime import datetime
from typing import Dict, Any


def get_loader(config: Dict[str, Any]) -> BaseLoader:
    # Get client ID
    client_id = config["client_id"]

    # Base arguments
    base_arguments = dict(
        debug=config["DEBUG"],
        debug_dir=config["debug_dir"],
    )

    # Return loader based on the client ID
    if client_id == "ripik":
        return RipikLoader(**base_arguments)

    elif client_id == "sunpharma_paonta_sahib":
        return SunPharmaPaontaSahibLoader(
            **base_arguments,
            execution_start=config["execution_start"],
            execution_end=config["execution_end"],
        )

    elif client_id == "sunpharma_dewas":
        return SunPharmaDewasLoader(
            **base_arguments,
            execution_end=config["execution_end"],
        )

    elif client_id == "sunpharma_baska":
        return SunPharmaBaskaLoader(
            **base_arguments,
        )

    else:
        raise ValueError(f"Invalid client id: {client_id}")
