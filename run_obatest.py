import pandas as pd
from obamodel import obamodel

# Paths to input data and scenarios
facilities_data_file = "facilities_data.csv"
abatement_cost_curve_file = "abatement_cost_curve.csv"
scenario_file = "scenarios.csv"
output_dir = "scenario_results"

# Load input data
facilities_data = pd.read_csv(facilities_data_file)
abatement_cost_curve = pd.read_csv(abatement_cost_curve_file)

# Model parameters
start_year = 2025
end_year = 2035

# Run all scenarios
print("Starting scenario analysis...")
try:
    obamodel().run_all_scenarios(
        scenario_file=scenario_file,
        facilities_data=facilities_data,
        abatement_cost_curve=abatement_cost_curve,
        start_year=start_year,
        end_year=end_year,
        output_dir=output_dir
    )
    print("Scenario analysis completed successfully. Results saved to:", output_dir)
except Exception as e:
    print("Error during scenario analysis:", e)

