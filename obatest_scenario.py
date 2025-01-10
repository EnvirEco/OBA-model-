import pandas as pd
from obamodel import obamodel

# Load input data
facilities_data = pd.read_csv("facilities_data.csv")
abatement_cost_curve = pd.read_csv("abatement_cost_curve.csv")

# Load scenario parameters
scenario_file = "scenarios.csv"
scenario_name = "Aggressive Reduction"  # Update for the scenario to test
scenario_params = obamodel.load_scenario(scenario_file, scenario_name)
print("Loaded scenario parameters:", scenario_params)

# Initialize and run the model
start_year = 2025
end_year = 2035
model = obamodel(facilities_data, abatement_cost_curve, start_year, end_year, scenario_params)

# Run the model and save results
market_summary, facility_results = model.run_model(output_file="results.csv")

# Display results
print("\nMarket Summary:")
print(market_summary.head())
print("\nFacility Results:")
print(facility_results.head())
