import pandas as pd
import ast
import argparse
import os

# Create the parser
parser = argparse.ArgumentParser(description='Process the input file path.')

# Add an argument for the input file path
parser.add_argument('input_path', type=str, help='The path to the input .csv file')

# Parse the arguments
args = parser.parse_args()

# Load the data
df = pd.read_csv(args.input_path)

# Initialize an empty dictionary to store the results
results = {}

# Iterate over the unique material_id values in the DataFrame
for material_id in df['material_id'].unique():
    # Filter the DataFrame for the current material_id
    df_material = df[df['material_id'] == material_id]

    # Get the material name (assuming all rows for the same material_id have the same material_name)
    material_name = df_material['material_name'].iloc[0]

    # Calculate the number of batches produced from INTM and Inventory
    num_INTM = len(df_material[df_material['consumed_as'] == 'INTM'])
    num_Inventory = len(df_material[df_material['consumed_as'] == 'Inventory'])

    # Calculate the total number of batches produced
    total_batches = len(df_material)

    # Initialize an empty dictionary to store the demand quantities
    demands = {}

    # Iterate over the rows in the DataFrame
    for idx, row in df_material.iterrows():
        # Parse the demand_contri column (convert from string to list of tuples)
        demand_contri = ast.literal_eval(row['demand_contri'])

        # Iterate over the tuples in demand_contri
        for demand, quantity in demand_contri:
            # If the demand is already in the dictionary, add the quantity to it
            if demand in demands:
                demands[demand] += quantity
            # If the demand is not in the dictionary, add it with the quantity
            else:
                demands[demand] = quantity

    # Store the results in the dictionary
    results[material_id] = {
        'Material Name': material_name,
        'INTM_batches': num_INTM,
        'Inventory_batches': num_Inventory,
        'Total_batches': total_batches
    }

    # Add the demand columns to the results
    for demand, total_quantity in demands.items():
        results[material_id][f'Demand_{demand}'] = total_quantity

# Convert the results dictionary to a DataFrame
results_df = pd.DataFrame(results).transpose()

# Define the output file path
output_path = os.path.join(os.path.dirname(args.input_path), 'material_insights.xlsx')

# Save the results to a new Excel file
results_df.to_excel(output_path)
