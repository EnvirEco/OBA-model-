import pandas as pd

class obamodel:
    def __init__(self, facilities_data, abatement_cost_curve, start_year):
        self.facilities_data = facilities_data
        self.abatement_cost_curve = abatement_cost_curve
        self.start_year = start_year
        self.market_price = 0.0
        self.government_revenue = 0.0
        self.facilities_data['Ceiling Price Payment'] = 0.0
        self.facilities_data['Tonnes Paid at Ceiling'] = 0.0
        self.facilities_data['Allowance Price ($/MTCO2e)'] = 0.0
        self.facilities_data['Trade Volume'] = 0.0
        self.facilities_data['Banked Allowances'] = 0.0
        self.facilities_data['Vintage Year'] = self.start_year
        self.banking_efficiency_rate = 0.8

    def determine_market_price(self, supply, demand):
        print(f"Determining market price with Supply: {supply}, Demand: {demand}")
        if demand <= 0:
            self.market_price = 0
        elif supply == 0:
            self.market_price = max(10, 100)
        else:
            self.market_price = max(10, 100 * (1 / (supply / demand)))
        print(f"Market price determined: {self.market_price}")

    def calculate_dynamic_allowance_surplus_deficit(self, year):
        print(f"Calculating dynamic allowance surplus/deficit for year {year}...")
        required_columns = [f'Output_{year}', f'Benchmark_{year}', f'Emissions_{year}']
        for col in required_columns:
            if col not in self.facilities_data.columns:
                raise KeyError(f"Missing required column for dynamic calculation: {col}")
        
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * self.facilities_data[f'Benchmark_{year}']
        )
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - self.facilities_data[f'Emissions_{year}']
        )
        print(f"Dynamic allowance surplus/deficit for year {year} calculated.")

    def save_reshaped_facility_summary(self, start_year, end_year, output_file):
        reshaped_data = []

        for year in range(start_year, end_year + 1):
            print(f"Processing year {year} for reshaped summary...")
            required_columns = [
                f"Emissions_{year}", f"Benchmark_{year}", f"Allocations_{year}",
                f"Allowance Surplus/Deficit_{year}", f"Abatement Cost_{year}",
                f"Trade Cost_{year}", f"Total Cost_{year}", f"Trade Volume_{year}"
            ]
            missing_columns = [col for col in required_columns if col not in self.facilities_data.columns]
            if missing_columns:
                print(f"Missing columns for year {year}: {missing_columns}")
                continue

            try:
                year_data = self.facilities_data[[
                    "Facility ID", f"Emissions_{year}", f"Benchmark_{year}",
                    f"Allocations_{year}", f"Allowance Surplus/Deficit_{year}",
                    f"Abatement Cost_{year}", f"Trade Cost_{year}", f"Total Cost_{year}", f"Trade Volume_{year}"
                ]].copy()

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

                melted_year_data = pd.melt(
                    year_data,
                    id_vars=["Facility ID", "Year"],
                    var_name="Variable Name",
                    value_name="Value"
                )
                reshaped_data.append(melted_year_data)

                print(f"Year {year}: Reshaped data preview:")
                print(melted_year_data.head())

            except Exception as e:
                print(f"Error processing year {year}: {e}")

        if reshaped_data:
            try:
                reshaped_combined_data = pd.concat(reshaped_data, ignore_index=True)
                reshaped_combined_data.to_csv(output_file, index=False)
                print(f"Reshaped facility summary successfully saved to {output_file}")
            except Exception as e:
                print(f"Error saving reshaped facility summary: {e}")
        else:
            print("No reshaped data available to save.")

    def summarize_market_data(self, start_year, end_year, output_file):
        market_data = []

        for year in range(start_year, end_year + 1):
            print(f"Summarizing market data for year {year}...")
            try:
                summary = {
                    "Year": year,
                    "Total Allocations": self.facilities_data[f"Allocations_{year}"].sum(),
                    "Total Emissions": self.facilities_data[f"Emissions_{year}"].sum(),
                    "Total Allowance Surplus/Deficit": self.facilities_data[f"Allowance Surplus/Deficit_{year}"].sum(),
                    "Total Trade Volume": self.facilities_data[f"Trade Volume_{year}"].sum(),
                    "Total Trade Cost": self.facilities_data[f"Trade Cost_{year}"].sum(),
                    "Total Banked Allowances": self.facilities_data[f"Banked Allowances_{year}"].sum()
                }
                market_data.append(summary)
                print(f"Market summary for year {year}: {summary}")
            except Exception as e:
                print(f"Error summarizing market data for year {year}: {e}")

        if market_data:
            try:
                combined_market_data = pd.DataFrame(market_data)
                combined_market_data.to_csv(output_file, index=False)
                print(f"Market summary successfully saved to {output_file}")
            except Exception as e:
                print(f"Error saving market summary: {e}")
        else:
            print("No market data available to save.")

    def trade_allowances(self, year):
        print(f"Trading allowances for year {year}...")
        self.facilities_data[f'Trade Volume_{year}'] = 0.0
        self.facilities_data[f'Trade Cost_{year}'] = 0.0
        buyers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0]
        sellers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0]
        for buyer_idx, buyer in buyers.iterrows():
            deficit = abs(buyer[f'Allowance Surplus/Deficit_{year}'])
            for seller_idx, seller in sellers.iterrows():
                surplus = seller[f'Allowance Surplus/Deficit_{year}']
                if deficit <= 0 or surplus <= 0:
                    continue
                trade_volume = min(deficit, surplus)
                trade_cost = trade_volume * self.market_price
                self.facilities_data.at[buyer_idx, f'Allowance Surplus/Deficit_{year}'] += trade_volume
                self.facilities_data.at[seller_idx, f'Allowance Surplus/Deficit_{year}'] -= trade_volume
                self.facilities_data.at[buyer_idx, f'Trade Volume_{year}'] += trade_volume
                self.facilities_data.at[buyer_idx, f'Trade Cost_{year}'] += trade_cost
                self.facilities_data.at[seller_idx, f'Trade Volume_{year}'] -= trade_volume
                self.facilities_data.at[seller_idx, f'Trade Cost_{year}'] -= trade_cost
                print(f"Trade executed: Buyer {buyer_idx}, Seller {seller_idx}, Volume: {trade_volume}, Cost: {trade_cost}")

    def bank_allowances(self, year):
        self.facilities_data[f'Banked Allowances_{year}'] = (
            self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0) * self.banking_efficiency_rate
        )
        print(f"Year {year}: Banked allowances updated:")

    def run_model(self, start_year, end_year):
        print("Running emissions trading model...")
        for year in range(start_year, end_year + 1):
            self.calculate_dynamic_allowance_surplus_deficit(year)
            self.trade_allowances(year)
            self.bank_allowances(year)
            print

