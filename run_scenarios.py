# run_scenarios.py
import os
import pandas as pd
from obamodel import obamodel

def run_scenario_test():
    """Run scenario analysis with the model."""
    # Define data directory path
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # Load base data with correct paths
    facilities_data = pd.read_csv(os.path.join(data_dir, "facilities_data.csv"))
    abatement_cost_curve = pd.read_csv(os.path.join(data_dir, "abatement_cost_curve.csv"))
    
    # Initialize model
    model = obamodel(facilities_data, abatement_cost_curve, start_year=2025)
    
    # Run scenarios from scenarios directory
    scenarios_path = os.path.join(os.path.dirname(__file__), 'scenarios', 'scenarios.csv')
    scenario_results = model.run_scenario_analysis(scenarios_path)
    
    print("\nScenario analysis complete. Results saved to:")
    print("- scenario_price_comparison.csv")
    print("- scenario_emissions_comparison.csv")
    print("- scenario_abatement_comparison.csv")
    print("- scenario_analysis_report.txt")
    
    return scenario_results

if __name__ == "__main__":
    run_scenario_test()
