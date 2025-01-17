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
    
    # Define file paths
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
    
    # Load scenarios
    try:
        print(f"\nAttempting to load scenarios from: {scenario_file}")
        scenarios = obamodel.load_all_scenarios(str(scenario_file))
        print(f"Successfully loaded {len(scenarios)} scenarios")
    except Exception as e:
        print(f"Error loading scenarios: {e}")
        return None
        
    # Create results directory
    results_dir = base_dir / "data" / "output" / "results"
    try:
        results_dir.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        print(f"Error creating results directory: {e}")
        return None
    
    # Run each scenario
    scenario_results = []
    start_year = 2025
    end_year = 2030
    
    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario['name']}")
        try:
            # Initialize model for this scenario
            model = obamodel(
                facilities_data=facilities_data,
                abatement_cost_curve=abatement_cost_curve,
                start_year=start_year,
                end_year=end_year,
                scenario_params=scenario
            )
            
            # Run the model
            market_summary, sector_summary, facility_results = model.run_model()
            
            # Save scenario results
            scenario_name = scenario['name'].replace(' ', '_').lower()
            
            market_file = results_dir / f"market_summary_{scenario_name}.csv"
            sector_file = results_dir / f"sector_summary_{scenario_name}.csv"
            facility_file = results_dir / f"facility_results_{scenario_name}.csv"
            
            market_summary.to_csv(market_file, index=False)
            sector_summary.to_csv(sector_file, index=False)
            facility_results.to_csv(facility_file, index=False)
            
            print(f"Results saved for scenario {scenario['name']}:")
            print(f"  Market summary: {market_file}")
            print(f"  Sector summary: {sector_file}")
            print(f"  Facility results: {facility_file}")
            
            # Store results for comparison
            scenario_results.append({
                'name': scenario['name'],
                'market_summary': market_summary,
                'sector_summary': sector_summary,
                'facility_results': facility_results
            })
            
        except Exception as e:
            print(f"Error running scenario {scenario['name']}: {str(e)}")
            continue
    
    # Create comparison analysis if we have results
    if scenario_results:
        try:
            # Create scenario comparison
            comparison_df = pd.DataFrame([
                {
                    'Scenario': result['name'],
                    'Total Emissions': result['market_summary']['Total_Emissions'].sum(),
                    'Total Abatement': result['market_summary']['Total_Abatement'].sum(),
                    'Average Price': result['market_summary']['Market_Price'].mean(),
                    'Total Cost': result['market_summary']['Total_Net_Cost'].sum()
                }
                for result in scenario_results
            ])
            
            comparison_file = results_dir / "scenario_comparison.csv"
            comparison_df.to_csv(comparison_file, index=False)
            print(f"\nScenario comparison saved to: {comparison_file}")
            
        except Exception as e:
            print(f"Error creating scenario comparison: {str(e)}")
    
    return results_dir

if __name__ == "__main__":
    output_dir = run_scenario_analysis()
    if output_dir:
        print(f"\nAnalysis complete. Results saved in: {output_dir}")
    else:
        print("\nAnalysis failed to complete")
