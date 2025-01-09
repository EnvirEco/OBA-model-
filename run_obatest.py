import pandas as pd
import os
from obamodel import obamodel  # Import the updated obamodel class

# Set the working directory
os.chdir(r"C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python313\\OBA_test")
print("Current working directory:", os.getcwd())

# Load input data
try:
    facilities_data = pd.read_csv("facilities_data.csv")
    abatement_cost_curve = pd.read_csv("abatement_cost_curve.csv")
    print("Data loaded successfully:")
    print(f"Number of facilities: {len(facilities_data)}")
    print(f"Number of abatement curves: {len(abatement_cost_curve)}")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit(1)

# Initialize the emissions trading model
start_year = 2025
end_year = 2035
try:
    model = obamodel(facilities_data, abatement_cost_curve, start_year, end_year)
except Exception as e:
    print(f"Error initializing model: {e}")
    exit(1)

# Run the emissions trading model
def run_trading_model():
    print("Running the model...")

    try:
        # Execute the model
        market_summary, facilities_results = model.run_model(output_file="facility_results.csv")

        # Save market summary
        market_summary_file = "market_summary.csv"
        market_summary.to_csv(market_summary_file, index=False)
        print(f"Market summary saved to {market_summary_file}")

        # Save annual facility summary
        facility_summary_file = "annual_facility_summary.csv"
        facilities_results.to_csv(facility_summary_file, index=False)
        print(f"Annual facility summary saved to {facility_summary_file}")

        # Debug outputs
        print("Market Summary Preview:")
        print(market_summary.head())
        print("Facility Summary Preview:")
        print(facilities_results.head())

    except Exception as e:
        print(f"Error during model run: {e}")

# Execute the function
if __name__ == "__main__":
    run_trading_model()
