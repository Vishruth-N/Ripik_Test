"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import os
import random
import yaml, json
from typing import Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd

from optimus import Optimus

import os
os.environ["LOGGING_CONFIG_FILE"] = 'D:\\Internships\\Ripik\\Ripik Optimus\\optimus_dep\\config\\logging.yaml'
os.environ["ALGO_CONFIG_FILE"] = 'D:\\Internships\\Ripik\\Ripik Optimus\\optimus_dep\\config\\algo.yaml'
os.environ["DEFAULT_USER_CONFIG_PATH"] = 'D:\\Internships\\Ripik\\Ripik Optimus\\optimus_dep\\config\\default\\'

if __name__ == "__main__":
    # Read user parameters
    user_parameters = None
    with open("config/user.yaml", "r") as config_file:
        user_parameters = yaml.load(config_file, Loader=yaml.FullLoader)

    # Read input files
    if user_parameters["client_id"] == "ripik":
        data_files = {
            # Use your directory here for ripik files if needed
        }
        output_dir = "input/generated/output/"

    elif user_parameters["client_id"] == "sunpharma_paonta_sahib":
        base_dir = "D:/Internships/Ripik/Ripik Optimus/optimus_dep/DataFiles/paonta_may23_data/"
        data_files = {
            "forecasted_demand": base_dir + "1_forecasted_demand.xlsx",
            "rm_inventory": base_dir + "2_rm_inventory.xlsx",
            "pm_inventory": base_dir + "3_pm_inventory.xlsx",
            "sfg_inventory": base_dir + "4_sfg_inventory.xlsx",
            "fg_inventory": base_dir + "5_fg_inventory.xlsx",
            "procurement_plan": base_dir + "10_proc_plan.xlsx",
            "bom": base_dir + "6_bom.xlsx",
            "recipe": base_dir + "7_recipe.xlsx",
            "plant_map": base_dir + "8_plant_map.xlsx",
            "crossblock_penalties": base_dir + "9_cross_block_penalty.csv",
            "packsize_mapping": base_dir + "12_packsize_mapping.csv",
            "phantom_items": base_dir + "11_phantom_items.xlsx",
            "code_to_code": base_dir + "13_code_to_code.xlsx",
            "printing_xy": base_dir + "14_printing_xy.xlsx",
            "sfg_underprocess": base_dir + "15_sfg_underprocess.xlsx",
            "fg_underprocess": base_dir + "16_fg_underprocess.xlsx",
        }
        output_dir = base_dir + "output/"

    elif user_parameters["client_id"] == "sunpharma_baska":
        base_dir = "D:/Internships/Ripik/Ripik Optimus/optimus_dep/DataFiles/baska_data_15052023/"
        data_files = {
            "forecasted_demand": base_dir + "1_forecasted_demand.xlsx",
            "recipe": base_dir + "1_recipe.xlsx",
            "plant_map": base_dir + "2_plant_map.xlsx",
            "holdtime_constraint": base_dir + "3_holdtime.xlsx",
        }
        output_dir = base_dir + "output/"


    elif user_parameters["client_id"] == "sunpharma_dewas":
        base_dir = "D:/Internships/Ripik/Ripik Optimus/optimus_dep/DataFiles/dewas_may23_data/"
        data_files = {
            "forecasted_demand": base_dir + "1_demand_rfc.xlsx",
            "inventory": base_dir + "2_inventory.xlsx",
            "family_mapping": base_dir + "3_family_mapping.xlsx",
            "procurement_plan": None,
            "bom": base_dir + "5_bom.xlsx",
            "recipe": base_dir + "6_recipe.xlsx",
            "plant_map": base_dir + "7_plant_map.xlsx",
            "machine_availability": base_dir + "8_machine_availability.xlsx",
        }
        output_dir = base_dir + "output/"

    else:
        raise ValueError(f"Invalid client id!")

    # Initialize random seed
    random.seed(42)
    np.random.seed(42)

    # Create optimus object
    my_optimus = Optimus(user_parameters=user_parameters)

    # Create schedule
    my_optimus.create_schedule(data_files=data_files)

    # Create views
    machine_view = my_optimus.get_machine_view()
    product_view = my_optimus.get_product_view()
    quantity_view = my_optimus.get_quantity_view()
    production_view = my_optimus.get_production_view()
    commit_view = my_optimus.get_commit_view()
    insights = my_optimus.create_insights()

    # MRP
    mrp = my_optimus.create_mrp(sustaining_days=90)

    # Room insights
    room_insights = my_optimus.create_room_insights()

    # Save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    machine_view.to_csv(os.path.join(output_dir, "out1_machine_view.csv"), index=False)
    product_view.to_csv(os.path.join(output_dir, "out2_product_view.csv"), index=False)
    quantity_view.to_csv(
        os.path.join(output_dir, "out3_quantity_view.csv"), index=False
    )
    production_view.to_csv(
        os.path.join(output_dir, "out4_production_view.csv"), index=False
    )
    commit_view.to_csv(os.path.join(output_dir, "out5_final_commit.csv"), index=False)

    with open(os.path.join(output_dir, "out6_insights.json"), "w") as f:
        json.dump(insights, f)

    with open(os.path.join(output_dir, "out7_room_insights.json"), "w") as f:
        json.dump(room_insights, f)

    with open(os.path.join(output_dir, "out8_mrp.json"), "w") as f:
        json.dump(mrp, f)
