import pandas as pd
import os
from obamodel import obamodel

# Paths to input data and scenarios
facilities_data_file = "data/facilities_data.csv"
abatement_cost_curve_file = "data/abatement_cost_curve.csv"
scenario_file = "scenarios/scenarios.csv"
output_dir = "scenario_results"

def get_reference_scenario(scenario_file: str) -> dict:
    """Get reference scenario parameters from scenario file."""
    try:
        scenarios = pd.read_csv(scenario_file)
        
        # Look for reference case (either named 'reference' or first row)
        reference_case = scenarios[scenarios['Scenario'].str.lower() == 'reference']
        if reference_case.empty:
            reference_case = scenarios.iloc[0]
            print(f"Using first scenario as reference: {reference_case['Scenario']}")
        else:
            print("Found reference scenario")
            
        # Convert to parameter dictionary
        reference_params = {
            "name": reference_case['Scenario'],
            "floor_price": reference_case['Floor Price'],
            "ceiling_price": reference_case['Ceiling Price'],
            "price_increment": reference_case['Price Increment'],
            "output_growth_rate": reference_case['Output Growth Rate'],
            "emissions_growth_rate": reference_case['Emissions Growth Rate'],
            "benchmark_ratchet_rate": reference_case['Benchmark Ratchet Rate'],
            "max_reduction": reference_case['Max Reduction']
        }
        
        return reference_params
        
    except Exception as e:
        print(f"Error loading reference scenario: {e}")
        raise

# Run scenario analysis
print("Starting scenario analysis...")
try:
    # Load input data
    facilities_data = pd.read_csv(facilities_data_file)
    abatement_cost_curve = pd.read_csv(abatement_cost_curve_file)
    
    # Get reference scenario parameters
    reference_scenario = get_reference_scenario(scenario_file)
    print(f"Reference scenario parameters: {reference_scenario}")
    
    # Initialize model with reference scenario
    model = obamodel(
        facilities_data=facilities_data,
        abatement_cost_curve=abatement_cost_curve,
        start_year=2025,
        end_year=2035,
        scenario_params=reference_scenario
    )
    
    # Run all scenarios
    model.run_all_scenarios(
        scenario_file=scenario_file,
        facilities_data=facilities_data,
        abatement_cost_curve=abatement_cost_curve,
        start_year=2025,
        end_year=2035,
        output_dir=output_dir
    )
    
    print("Scenario analysis completed successfully. Results saved to:", output_dir)
    
except Exception as e:
    print("Error during scenario analysis:", e)
