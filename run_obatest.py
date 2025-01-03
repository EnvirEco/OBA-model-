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
        market_summary_file = "market_summary.csv"
        market_summaries = []
        annual_facility_summaries = []  # Collect annual facility-level data

        for year in range(start_year, end_year + 1):
            print(f"Processing year {year}...")
            total_trade_volume = model.facilities_data[f'Trade Volume_{year}'].sum()
            total_trade_cost = model.facilities_data[f'Trade Cost_{year}'].sum()
            total_allocations = model.facilities_data[f'Allocations_{year}'].sum()
            total_emissions = model.facilities_data[f'Emissions_{year}'].sum()
            total_surplus_deficit = model.facilities_data[f'Allowance Surplus/Deficit_{year}'].sum()
            total_credit_carryover = model.facilities_data['Credit Carryover'].sum()

            # Add additional metrics
            total_supply = model.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
            total_demand = abs(model.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
            net_demand = total_demand - total_supply
            total_output = model.facilities_data[f'Output_{year}'].sum()
            allowance_price = model.facilities_data[f'Allowance Price_{year}'].mean()

            # Prevent allowance price from defaulting to 10 when prices crash
            allowance_price = allowance_price if allowance_price > 0 else 0

            summary = {
                "Year": year,
                "Total Allocations": total_allocations,
                "Total Emissions": total_emissions,
                "Total Surplus/Deficit": total_surplus_deficit,
                "Total Trade Volume": total_trade_volume,
                "Total Trade Cost": total_trade_cost,
                "Total Credit Carryover": total_credit_carryover,
                "Total Supply": total_supply,
                "Total Demand": total_demand,
                "Net Demand": net_demand,
                "Total Output": total_output,
                "Allowance Price ($/MTCO2e)": allowance_price
            }
            market_summaries.append(summary)

            # Facility-level data for the year
            facility_summary = model.facilities_data[[
                "Facility ID", f"Allocations_{year}", f"Emissions_{year}",
                f"Allowance Surplus/Deficit_{year}", f"Trade Volume_{year}",
                f"Trade Cost_{year}", f"Abatement Cost_{year}", "Credit Carryover"
            ]].copy()
            facility_summary["Year"] = year
            facility_summary = facility_summary.rename(columns={
                f"Allocations_{year}": "Allocations",
                f"Emissions_{year}": "Emissions",
                f"Allowance Surplus/Deficit_{year}": "Allowance Surplus/Deficit",
                f"Trade Volume_{year}": "Trade Volume",
                f"Trade Cost_{year}": "Trade Cost",
                f"Abatement Cost_{year}": "Abatement Cost"
            })

            # Calculate cost breakdown
            facility_summary["Compliance Cost"] = facility_summary["Trade Cost"] + facility_summary["Abatement Cost"]
            facility_summary["Cost to Profit Ratio"] = facility_summary["Compliance Cost"] / facility_summary["Allocations"]
            facility_summary["Cost to Output Ratio"] = facility_summary["Compliance Cost"] / model.facilities_data[f"Output_{year}"]

            annual_facility_summaries.append(facility_summary)

        pd.DataFrame(market_summaries).to_csv(market_summary_file, index=False)
        print(f"Annual market summary saved to {market_summary_file}")
        print("Market Summary Debug:")
        print(pd.DataFrame(market_summaries).head())

        # Save combined annual facility summary
        facility_summary_file = "annual_facility_summary.csv"
        pd.concat(annual_facility_summaries).to_csv(facility_summary_file, index=False)
        print(f"Annual facility summary saved to {facility_summary_file}")
        print("Facility Summary Debug:")
        print(pd.concat(annual_facility_summaries).head())

    except Exception as e:
        print(f"Error during model run: {e}")

# Execute the function
if __name__ == "__main__":
    run_trading_model()
