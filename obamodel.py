import pandas as pd

class obamodel:
    def __init__(self, facilities_data, abatement_cost_curve, start_year):
        # Initialize the class with required data
        self.facilities_data = facilities_data
        self.abatement_cost_curve = abatement_cost_curve
        self.start_year = start_year
        self.market_price = 0.0
        self.government_revenue = 0.0

        # Add default columns to facilities_data
        self.facilities_data['Ceiling Price Payment'] = 0.0
        self.facilities_data['Tonnes Paid at Ceiling'] = 0.0
        self.facilities_data['Allowance Price ($/MTCO2e)'] = 0.0
        self.facilities_data['Trade Volume'] = 0.0
        self.facilities_data['Vintage Year'] = self.start_year
        self.facilities_data['Credit Carryover'] = 0.0

        print("obamodel initialized successfully.")

    def determine_market_price(self, supply, demand):
        print(f"Determining market price with Supply: {supply}, Demand: {demand}")
        if demand <= 0:
            self.market_price = 0
        elif supply == 0:
            self.market_price = max(10, 100)
        else:
            self.market_price = max(10, 100 * (1 / (supply / demand)))
        print(f"Market price determined: {self.market_price}")

    def calculate_dynamic_values(self, year):
        print(f"Calculating dynamic values for year {year}...")
        years_since_start = year - self.start_year
        self.facilities_data[f'Output_{year}'] = (
            self.facilities_data['Baseline Output'] * (1 + self.facilities_data['Output Growth Rate']) ** years_since_start
        )
        self.facilities_data[f'Emissions_{year}'] = (
            self.facilities_data['Baseline Emissions'] * (1 + self.facilities_data['Emissions Growth Rate']) ** years_since_start
        )
        self.facilities_data[f'Benchmark_{year}'] = (
            self.facilities_data['Baseline Benchmark'] * (1 + self.facilities_data['Benchmark Ratchet Rate']) ** years_since_start
        )
        print(f"Dynamic values calculated for year {year}.")

    def calculate_dynamic_allowance_surplus_deficit(self, year):
        print(f"Calculating dynamic allowance surplus/deficit for year {year}...")
        self.calculate_dynamic_values(year)
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * self.facilities_data[f'Benchmark_{year}']
        )
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - self.facilities_data[f'Emissions_{year}']
        )
    
        # Initialize missing columns with default values
        self.facilities_data[f'Abatement Cost_{year}'] = 0.0
        self.facilities_data[f'Total Cost_{year}'] = 0.0
    
        # Update carryover credits for surpluses
        surplus = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0)
        self.facilities_data['Credit Carryover'] += surplus
    
        print(f"Allowance surplus/deficit for year {year} calculated successfully.")


    def trade_allowances(self, year):
        print(f"Trading allowances for year {year}...")
        self.facilities_data[f'Trade Volume_{year}'] = 0.0
        self.facilities_data[f'Trade Cost_{year}'] = 0.0
        buyers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0]
        sellers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0]

        for buyer_idx, buyer in buyers.iterrows():
            deficit = abs(buyer[f'Allowance Surplus/Deficit_{year}'])

            # Apply carryover credits first
            carryover = self.facilities_data.at[buyer_idx, 'Credit Carryover']
            if carryover > 0:
                applied_credits = min(deficit, carryover)
                self.facilities_data.at[buyer_idx, f'Allowance Surplus/Deficit_{year}'] += applied_credits
                self.facilities_data.at[buyer_idx, 'Credit Carryover'] -= applied_credits
                deficit -= applied_credits
                print(f"Buyer {buyer_idx} used {applied_credits} carryover credits to cover deficit.")

            if deficit > 0:
                for seller_idx, seller in sellers.iterrows():
                    surplus = seller[f'Allowance Surplus/Deficit_{year}']
                    if surplus <= 0:
                        continue
                    trade_volume = min(deficit, surplus)
                    trade_cost = trade_volume * self.market_price

                    self.facilities_data.at[buyer_idx, f'Allowance Surplus/Deficit_{year}'] += trade_volume
                    self.facilities_data.at[seller_idx, f'Allowance Surplus/Deficit_{year}'] -= trade_volume
                    self.facilities_data.at[buyer_idx, f'Trade Volume_{year}'] += trade_volume
                    self.facilities_data.at[buyer_idx, f'Trade Cost_{year}'] += trade_cost
                    self.facilities_data.at[seller_idx, f'Trade Volume_{year}'] -= trade_volume
                    self.facilities_data.at[seller_idx, f'Trade Cost_{year}'] -= trade_cost
                    deficit -= trade_volume
                    print(f"Trade executed: Buyer {buyer_idx}, Seller {seller_idx}, Volume: {trade_volume}, Cost: {trade_cost}")

                    if deficit <= 0:
                        break

    def save_reshaped_facility_summary(self, start_year, end_year, output_file):
        reshaped_data = []
    
        for year in range(start_year, end_year + 1):
            required_columns = [
                f"Emissions_{year}", f"Benchmark_{year}", f"Allocations_{year}",
                f"Allowance Surplus/Deficit_{year}", f"Abatement Cost_{year}",
                f"Trade Cost_{year}", f"Total Cost_{year}", f"Trade Volume_{year}"
            ]
            year_data = self.facilities_data[required_columns + ["Facility ID"]].copy()
            year_data = year_data.rename(columns={
                f"Emissions_{year}": "Emissions",
                f"Benchmark_{year}": "Benchmark",
                f"Allocations_{year}": "Allocations",
                f"Allowance Surplus/Deficit_{year}": "Allowance Surplus/Deficit",
                f"Abatement Cost_{year}": "Abatement Cost",
                f"Trade Cost_{year}": "Trade Cost",
                f"Total Cost_{year}": "Total Cost",
                f"Trade Volume_{year}": "Trade Volume"
            })
            year_data["Year"] = year
            reshaped_data.append(pd.melt(
                year_data, id_vars=["Facility ID", "Year"],
                var_name="Variable Name", value_name="Value"
            ))
    
        reshaped_combined_data = pd.concat(reshaped_data, ignore_index=True)
        reshaped_combined_data.to_csv(r"C:\Users\user\AppData\Local\Programs\Python\Python313\OBA_test\reshaped_combined_summary.csv", index=False)
        print(r"Reshaped facility summary saved to C:\Users\user\AppData\Local\Programs\Python\Python313\OBA_test\reshaped_combined_summary.csv")


    def run_model(self, start_year, end_year):
        print("Running emissions trading model...")
        for year in range(start_year, end_year + 1):
            print(f"\nProcessing year {year}...")
            self.calculate_dynamic_allowance_surplus_deficit(year)
            self.trade_allowances(year)
            print(f"Credit Carryover after year {year}:")
            print(self.facilities_data['Credit Carryover'].describe())
        print("Model run complete.")
