"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import json
import argparse
import numpy as np
import pandas as pd
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, help="/path/to/input/file.csv", required=True
    )
    parser.add_argument(
        "-om",
        "--output-machine",
        type=str,
        help="/path/to/output/file.json",
    )
    parser.add_argument(
        "-op",
        "--output-product",
        type=str,
        help="/path/to/output/file.json",
    )
    args = parser.parse_args()

    # Read file
    df = pd.read_csv(args.input)

    # Current
    # For future, take machine ID as key for its values
    unique_material_ids = set()
    machine_view = []
    for row in df.itertuples():
        if row.machine_type == "infinite":
            continue

        if row.state == "MACHINE_CHANGEOVER_A":
            continue

        data_row = {
            "machine_id": row.machine_id,
            "machine_type": row.machine_type,
            "block_id": row.block_id,
            "room_id": row.room_id,
            "operation": row.operation,
            "state": row.state,
            "rel_start_time": row.rel_start_time,
            "rel_end_time": row.rel_end_time,
            "remarks": row.remarks,
            "start_time": row.start_time,
            "end_time": row.end_time,
        }

        # Split info
        if row.info and isinstance(row.info, str):
            all_info_items = row.info.split(";\n")
            data_row["info"] = {}
            for item in all_info_items:
                key, value = item.split(": ", maxsplit=1)
                data_row["info"][key] = value
            unique_material_ids.add(data_row["info"]["Material ID"])

        machine_view.append(data_row)

    # Create machine view output
    machine_view_output = {"machine_view": machine_view}

    # Create unique product to color mapping
    product_to_color_mapping = {}
    for material_id in unique_material_ids:
        product_to_color_mapping[material_id] = "#%06x" % random.randint(0, 0xFFFFFF)

    if args.output_machine:
        with open(args.output_machine, "w") as f:
            json.dump(machine_view_output, f)

    if args.output_product:
        with open(args.output_product, "w") as f:
            json.dump(product_to_color_mapping, f)
