import pandas as pd
import numpy as np
import os
from pathlib import Path
from obamodel import obamodel

def run_scenario_analysis():
    """Run all scenarios and save results."""
    print("Starting scenario analysis...")
    
    # Get the base directory - adjust this to your setup
    base_dir = Path(r"C:\Users\user\AppData\Local\Programs\Python\Python313\OBA_test")
    print(f"Base directory: {base_dir}")
    
    # Define file paths based on your directory structure
    scenario_file = base_dir / "data" / "input" / "scenarios" / "scenarios.csv"
    facilities_file = base_dir / "data" / "input" / "facilities" / "facilities_data.csv"
    abatement_file = base_dir / "data" / "input" / "facilities" / "abatement_cost_curve.csv"
    
    print("\nAttempting to load files from:")
    print(f"Scenarios: {scenario_file}")
    print(f"Facilities: {facilities_file}")
    print(f"Abatement: {abatement_file}")
    
    # Verify file existence
    for file_path in [scenario_file, facilities_file, abatement_file]:
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            return None
    
    # Create results directory if it doesn't exist
    results_dir = base_dir / "data" / "output" / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Load input data
    try:
        facilities_data = pd.read_csv(facilities_file)
        print(f"\nFacilities data loaded: {len(facilities_data)} rows")
        print(f"Facilities columns: {', '.join(facilities_data.columns)}")
    except Exception as e:
        print(f"Error loading facilities data: {e}")
        return None
        
    try:
        abatement_cost_curve = pd.read_csv(abatement_file)
        print(f"Abatement curves loaded: {len(abatement_cost_curve)} rows")
        print(f"Abatement columns: {', '.join(abatement_cost_curve.columns)}")
    except Exception as e:
        print(f"Error loading abatement curves: {e}")
        return None
    
    # Set time period
    start_year = 2025
    end_year = 2035
    
    # Load scenarios
    try:
        scenarios = obamodel.load_all_scenarios(scenario_file)
        print(f"\nLoaded {len(scenarios)} scenarios")
    except Exception as e:
        print(f"Error loading scenarios: {e}")
        return None
    
    # Run each scenario
    for scenario in scenarios:
        try:
            print(f"\nRunning scenario: {scenario['name']}")
            
            # Create scenario-specific output directory
            scenario_dir = base_dir / "data" / "output" / "scenario_results" / scenario['name'].replace(" ", "_").lower()
            scenario_dir.mkdir(exist_ok=True, parents=True)
            
            # Initialize model
            model = obamodel(
                facilities_data=facilities_data.copy(),  # Added .copy() to prevent modifications
                abatement_cost_curve=abatement_cost_curve.copy(),
                start_year=start_year,
                end_year=end_year,
                scenario_params=scenario
            )
            
            # Run model
            market_summary, facility_results = model.run_model()
            
            # Save results
            market_summary.to_csv(scenario_dir / "market_summary.csv", index=False)
            facility_results.to_csv(scenario_dir / "facility_results.csv", index=False)
            
            print(f"Results saved for scenario: {scenario['name']}")
            
        except Exception as e:
            print(f"Error in scenario {scenario['name']}: {str(e)}")
            continue
    
    return results_dir

if __name__ == "__main__":
    output_dir = run_scenario_analysis()
    if output_dir:
        print(f"\nAnalysis complete. Results saved in: {output_dir}")
    else:
        print("\nAnalysis failed to complete")
