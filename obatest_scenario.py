import pandas as pd
from obamodel import obamodel

# File paths
facilities_data_file = "facilities_data.csv"
abatement_cost_curve_file = "abatement_cost_curve.csv"
scenario_file = "scenarios.csv"
output_directory = "scenario_results"

# Load input data
facilities_data = pd.read_csv(facilities_data_file)
abatement_cost_curve = pd.read_csv(abatement_cost_curve_file)

# Run all scenarios
model.run_all_scenarios(
    scenario_file=scenario_file,
    facilities_data=facilities_data,
    abatement_cost_curve=abatement_cost_curve,
    start_year=start_year,
    end_year=end_year,
    output_dir=r"C:\Users\user\AppData\Local\Programs\Python\Python313\OBA_test\scenario_results"
)

# Combine results into a single summary
save_combined_results(output_dir=r"C:\Users\user\AppData\Local\Programs\Python\Python313\OBA_test\scenario_results")
