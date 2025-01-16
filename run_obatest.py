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
    
    # Define file paths - FIXED paths to match actual directory structure
    scenario_file = base_dir / "data" / "input" / "scenarios" / "scenarios.csv"  # Fixed path
    facilities_file = base_dir / "data" / "input" / "facilities" / "facilities_data.csv"
    abatement_file = base_dir / "data" / "input" / "facilities" / "abatement_cost_curve.csv"
    
    print("\nAttempting to load files from:")
    print(f"Scenarios: {scenario_file}")  # Updated print statement
    print(f"Facilities: {facilities_file}")
    print(f"Abatement: {abatement_file}")
    
    # Verify file existence with better error messages
    for file_path in [scenario_file, facilities_file, abatement_file]:
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            print(f"Please verify that {file_path.parent} directory exists and contains {file_path.name}")
            return None
        if not os.access(file_path, os.R_OK):
            print(f"ERROR: No permission to read file: {file_path}")
            return None
    
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
    
    # Load scenarios with better error handling
    try:
        print(f"\nAttempting to load scenarios from: {scenario_file}")
        scenarios = obamodel.load_all_scenarios(str(scenario_file))  # Convert Path to string
        print(f"Successfully loaded {len(scenarios)} scenarios")
    except PermissionError as e:
        print(f"Permission error loading scenarios: {e}")
        print("Please check file permissions and ensure you have read access")
        return None
    except FileNotFoundError as e:
        print(f"Scenarios file not found: {e}")
        print(f"Expected file at: {scenario_file}")
        return None
    except Exception as e:
        print(f"Error loading scenarios: {e}")
        return None
        
    # Create results directory if it doesn't exist
    results_dir = base_dir / "data" / "output" / "results"
    try:
        results_dir.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        print(f"Error creating results directory: {e}")
        return None
    
    # Rest of the function remains the same
    # ... (running scenarios and saving results)
    
    return results_dir

if __name__ == "__main__":
    output_dir = run_scenario_analysis()
    if output_dir:
        print(f"\nAnalysis complete. Results saved in: {output_dir}")
    else:
        print("\nAnalysis failed to complete")
