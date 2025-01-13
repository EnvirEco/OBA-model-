# run_obatest.py
import os
import pandas as pd
from obamodel import obamodel

def run_scenario_analysis():
    """Run complete scenario analysis with proper file handling."""
    try:
        # Set up directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        scenario_dir = os.path.join(base_dir, 'scenarios')
        output_dir = os.path.join(base_dir, 'scenario_results')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load input data
        facilities_data = pd.read_csv(os.path.join(data_dir, 'facilities_data.csv'))
        abatement_cost_curve = pd.read_csv(os.path.join(data_dir, 'abatement_cost_curve.csv'))
        scenario_file = os.path.join(scenario_dir, 'scenarios.csv')
        
        print(f"Loading data from:")
        print(f"Facilities data: {os.path.join(data_dir, 'facilities_data.csv')}")
        print(f"Abatement curves: {os.path.join(data_dir, 'abatement_cost_curve.csv')}")
        print(f"Scenarios: {scenario_file}")
        
        # Model parameters
        start_year = 2025
        end_year = 2035
        
        # Load reference scenario for model initialization
        scenarios = pd.read_csv(scenario_file)
        reference_scenario = scenarios.iloc[0].to_dict()  # Use first scenario as reference
        
        # Initialize model
        model = obamodel(
            facilities_data=facilities_data,
            abatement_cost_curve=abatement_cost_curve,
            start_year=start_year,
            end_year=end_year,
            scenario_params=reference_scenario
        )
        
        # Run scenarios
        model.run_all_scenarios(
            scenario_file=scenario_file,
            facilities_data=facilities_data,
            abatement_cost_curve=abatement_cost_curve,
            start_year=start_year,
            end_year=end_year,
            output_dir=output_dir
        )
        
        print(f"\nScenario results saved to: {output_dir}")
        print("\nFiles created:")
        for file in os.listdir(output_dir):
            print(f"- {file}")
        
        return output_dir
        
    except Exception as e:
        print(f"Error during scenario analysis: {e}")
        raise

if __name__ == "__main__":
    output_dir = run_scenario_analysis()
