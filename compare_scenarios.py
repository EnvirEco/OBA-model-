import os
import pandas as pd


def combine_scenario_results(output_dir: str = r"C:\Users\user\AppData\Local\Programs\Python\Python313\OBA_test\scenario_results") -> None:
    """
    Combine all market summaries into a single file for scenario comparison.
    """
    print("Combining scenario results into a single summary file...")
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        print(f"Error: Output directory does not exist: {output_dir}")
        return
    
    # Locate all market summary files in the directory
    market_files = [f for f in os.listdir(output_dir) if f.endswith("_market_summary.csv")]
    
    if not market_files:
        print("No market summary files found in the output directory. Exiting.")
        return
    
    combined_results = []
    for file in market_files:
        scenario_name = file.split("_market_summary.csv")[0].replace("_", " ").title()
        file_path = os.path.join(output_dir, file)
        
        try:
            scenario_data = pd.read_csv(file_path)
            scenario_data['Scenario'] = scenario_name
            combined_results.append(scenario_data)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    
    # Combine all scenario data into one DataFrame
    if combined_results:
        combined_summary = pd.concat(combined_results, ignore_index=True)
        combined_file = os.path.join(output_dir, "combined_scenario_summary.csv")
        
        # Save the combined results
        combined_summary.to_csv(combined_file, index=False)
        print(f"Combined scenario results saved to: {combined_file}")
    else:
        print("No valid data to combine. Exiting.")


if __name__ == "__main__":
    # Define the output directory for scenarios
    output_directory = r"C:\Users\user\AppData\Local\Programs\Python\Python313\OBA_test\scenario_results"
    
    # Combine scenario results
    combine_scenario_results(output_directory)
