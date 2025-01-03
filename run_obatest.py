import pandas as pd
import os
from obamodel import obamodel  # Import the updated obamodel class

# Set the working directory
os.chdir(r"C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python313\\OBA_test")
print("Current working directory:", os.getcwd())

# Load input data
facilities_data = pd.read_csv("facilities_data.csv")
abatement_cost_curve = pd.read_csv("abatement_cost_curve.csv")

# Initialize the emissions trading model
start_year = 2025
model = obamodel(facilities_data, abatement_cost_curve, start_year)

# Dynamically calculate annual values

def calculate_annual_values(facilities_data, year, start_year):
    years_since_start = year - start_year
    facilities_data[f'Output_{year}'] = (
        facilities_data['Baseline Output'] * (1 + facilities_data['Output Growth Rate']) ** years_since_start
    )
    facilities_data[f'Emissions_{year}'] = (
        facilities_data['Baseline Emissions'] * (1 + facilities_data['Emissions Growth Rate']) ** years_since_start
    )
    facilities_data[f'Benchmark_{year}'] = (
        facilities_data['Baseline Benchmark'] * (1 + facilities_data['Benchmark Ratchet Rate']) ** years_since_start
    )
    return facilities_data

# Run the emissions trading model
def run_trading_model(start_year=2025, end_year=2035):
    print("Running the model...")

    try:
        # Execute the model
        model.run_model(start_year, end_year)

        # Save reshaped data including trades and carryover credits
        reshaped_output_file = "reshaped_combined_summary.csv"
        model.save_reshaped_facility_summary(start_year, end_year, reshaped_output_file)

        # Load and preview reshaped data to verify inclusion of trades and credits
        if os.path.exists(reshaped_output_file):
            reshaped_data = pd.read_csv(reshaped_output_file)
            print("Preview of reshaped combined summary:")
            print(reshaped_data.head())
        else:
            print(f"Error: {reshaped_output_file} not found!")

        # Save annual market summary
        market_summary_file = "annual_market_summary.csv"
        market_summaries = []
        for year in range(start_year, end_year + 1):
            total_trade_volume = model.facilities_data[f'Trade Volume_{year}'].sum()
            total_trade_cost = model.facilities_data[f'Trade Cost_{year}'].sum()
            total_allocations = model.facilities_data[f'Allocations_{year}'].sum()
            total_emissions = model.facilities_data[f'Emissions_{year}'].sum()
            total_surplus_deficit = model.facilities_data[f'Allowance Surplus/Deficit_{year}'].sum()
            total_credit_carryover = model.facilities_data['Credit Carryover'].sum()

            summary = {
                "Year": year,
                "Total Allocations": total_allocations,
                "Total Emissions": total_emissions,
                "Total Surplus/Deficit": total_surplus_deficit,
                "Total Trade Volume": total_trade_volume,
                "Total Trade Cost": total_trade_cost,
                "Total Credit Carryover": total_credit_carryover
            }
            market_summaries.append(summary)

        pd.DataFrame(market_summaries).to_csv(market_summary_file, index=False)
        print(f"Annual market summary saved to {market_summary_file}")

    except Exception as e:
        print(f"Error during model run: {e}")


# Execute the function
if __name__ == "__main__":
    run_trading_model()
