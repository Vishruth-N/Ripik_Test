"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import os
import yaml
import logging
import logging.config
import numpy as np
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv

load_dotenv()


def setup_logging():
    with open(os.getenv("LOGGING_CONFIG_FILE")) as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)


def setup_config_params(user_parameters: Dict[str, Any]) -> Dict[str, Any]:
    assert "client_id" in user_parameters

    # Main configuration
    config = None
    with open(os.getenv("ALGO_CONFIG_FILE"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load default parameters
    default_user_config_file = os.path.join(
        os.getenv("DEFAULT_USER_CONFIG_PATH"), f"{user_parameters['client_id']}.yaml"
    )
    with open(default_user_config_file, "r") as f:
        config.update(yaml.load(f, Loader=yaml.FullLoader))

    # Override config by the user parameters
    config.update(user_parameters)

    # Parse datetimes for execution start and end
    config["execution_start"] = datetime.strptime(
        config["execution_start"], "%Y-%m-%d %H:%M:%S"
    )
    config["execution_end"] = datetime.strptime(
        config["execution_end"], "%Y-%m-%d %H:%M:%S"
    )

    # Periods handle None and type conversion to tuple
    for period in config["periods"]:
        if config["periods"][period][0] is None:
            config["periods"][period][0] = 0
        if config["periods"][period][1] is None:
            config["periods"][period][1] = np.inf
        config["periods"][period] = tuple(config["periods"][period])

    # Min consumption for production
    if "min_sfg_consumption_for_production" not in config:
        config["min_sfg_consumption_for_production"] = 0.0

    # Debug mode on or off
    config["DEBUG"] = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

    return config
