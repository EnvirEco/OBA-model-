import pandas as pd
import os
from obamodel import obamodel  # Import the updated obamodel class

# Set the working directory
os.chdir(r"C:\Users\user\AppData\Local\Programs\Python\Python313\OBA_test")
print("Current working directory:", os.getcwd())

# Load input data
facilities_data = pd.read_csv("facilities_data.csv")
abatement_cost_curve = pd.read_csv("abatement_cost_curve.csv")

# Initialize the emissions trading model
start_year = 2025
model = obamodel(facilities_data, abatement_cost_curve, start_year)  # Include start_year

# Run the emissions trading model
def run_trading_model(start_year=2025, end_year=2035):
    print("Running the model...")

    # Execute the model's main functionality for multi-year dynamics
    try:
        results = model.run_model(start_year, end_year)
    except KeyError as e:
        print(f"KeyError encountered during model run: {e}")
        return
    except Exception as e:
        print(f"Unexpected error during model run: {e}")
        return

    if not results:  # Handle case where results are None or empty
        print("No results were generated. Please check the model configuration and input data.")
        return

    # Initialize lists to collect yearly summaries for combined files
    combined_market_results = []
    combined_facility_results = []

    # Process and save results for each year
    for result in results:
        year = result['Year']
        market_summary_file = f"market_summary_{year}.csv"
        facility_summary_file = f"facility_summary_{year}.csv"

        # Save market-level summaries to individual files
        pd.DataFrame([result]).to_csv(market_summary_file, index=False)
        print(f"Year {year} Market-level summary saved to {market_summary_file}")

        # Extract year-specific facility data
        facility_summary = model.facilities_data[[
            'Facility ID', f'Emissions_{year}', f'Benchmark_{year}', f'Allocations_{year}',
            f'Allowance Surplus/Deficit_{year}', f'Abatement Cost_{year}',
            f'Trade Cost_{year}', f'Total Cost_{year}', f'Profit_{year}',
            f'Costs to Profits Ratio_{year}', f'Costs to Output Ratio_{year}',
            'Banked Allowances', f'Emission Reductions_{year}'  # Add Emission Reductions to the summary
        ]].copy()

        facility_summary["Year"] = year

        # Save facility-level summaries for the year
        facility_summary.to_csv(facility_summary_file, index=False)
        print(f"Year {year} Facility-level summary saved to {facility_summary_file}")

        # Append results to combined lists
        combined_market_results.append(result)
        combined_facility_results.append(facility_summary)

    # Save combined market-level results to a single file
    combined_market_file = "combined_market_summary.csv"
    pd.DataFrame(combined_market_results).to_csv(combined_market_file, index=False)
    print(f"Combined market-level summary saved to {combined_market_file}")

    # Save combined facility-level results to a single file
    combined_facility_file = "combined_facility_summary.csv"
    pd.concat(combined_facility_results).to_csv(combined_facility_file, index=False)
    print(f"Combined facility-level summary saved to {combined_facility_file}")

# Execute the function
if __name__ == "__main__":
    run_trading_model()
