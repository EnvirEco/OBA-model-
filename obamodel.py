import pandas as pd

class obamodel:
    def __init__(self, facilities_data, abatement_cost_curve, start_year):
        # Initialize the class with required data
        self.facilities_data = facilities_data.copy()
        self.abatement_cost_curve = abatement_cost_curve
        self.start_year = start_year
        self.market_price = 0.0
        self.government_revenue = 0.0

        # Preinitialize required columns to avoid fragmentation
        self.initialize_columns()
        print("obamodel initialized successfully.")

    def initialize_columns(self):
        required_columns = [
            'Ceiling Price Payment', 'Tonnes Paid at Ceiling',
            'Allowance Price ($/MTCO2e)', 'Trade Volume', 'Credit Carryover'
        ]
        year_columns = []
        for year in range(self.start_year, self.start_year + 20):  # Estimate 20 years
            year_columns.extend([
                f"Output_{year}", f"Emissions_{year}", f"Benchmark_{year}",
                f"Allocations_{year}", f"Allowance Surplus/Deficit_{year}",
                f"Abatement Cost_{year}", f"Trade Cost_{year}",
                f"Total Cost_{year}", f"Trade Volume_{year}",
                f"Allowance Price_{year}"
            ])

        for column in required_columns + year_columns:
            if column not in self.facilities_data.columns:
                self.facilities_data[column] = 0.0

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

        # Debugging supply and demand for surplus/deficit
        total_supply = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
        total_demand = abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
        print(f"Year {year} Debug: Total Supply = {total_supply}, Total Demand = {total_demand}")

    def calculate_marginal_costs(self, year):
        marginal_costs = []
        for index, row in self.facilities_data.iterrows():
            deficit = -row[f'Allowance Surplus/Deficit_{year}']
            if deficit > 0:
                abatement_curve = self.abatement_cost_curve[self.abatement_cost_curve['Facility ID'] == row['Facility ID']]
                if not abatement_curve.empty:
                    slope = abatement_curve.iloc[0]['Slope']
                    marginal_cost = slope  # Assuming linear cost curve (Slope is the marginal cost)
                    marginal_costs.append((marginal_cost, deficit))
        return marginal_costs

    def calculate_abatement(self, year):
        print(f"Calculating abatement for year {year}...")
        for index, row in self.facilities_data.iterrows():
            deficit = -row[f'Allowance Surplus/Deficit_{year}']  # Negative means deficit
            if deficit > 0:
                # Find abatement cost curve for the facility
                abatement_curve = self.abatement_cost_curve[self.abatement_cost_curve['Facility ID'] == row['Facility ID']]
                if not abatement_curve.empty:
                    slope = abatement_curve.iloc[0]['Slope']
                    intercept = abatement_curve.iloc[0]['Intercept']
                    max_reduction = abatement_curve.iloc[0]['Max Reduction (MTCO2e)']
                    
                    # Calculate abatement based on the deficit
                    abatement = min(deficit, max_reduction)
                    abatement_cost = slope * abatement + intercept
                    
                    # Ensure abatement cost is not negative
                    if abatement_cost < 0:
                        abatement_cost = 0
                    
                    # Compare abatement cost with market price
                    purchase_cost = deficit * self.market_price
                    if abatement_cost < purchase_cost:
                        # Abate emissions
                        self.facilities_data.at[index, f'Abatement Cost_{year}'] = abatement_cost
                        self.facilities_data.at[index, f'Allowance Surplus/Deficit_{year}'] += abatement
                        print(f"Facility {row['Facility ID']} abated {abatement} MTCO2e at cost {abatement_cost}")
                    else:
                        print(f"Facility {row['Facility ID']} opts to purchase allowances at market price.")
    
    def determine_market_price(self, supply, demand, year):
        print(f"Determining market price: Supply={supply}, Demand={demand}")
        if demand > 0 and supply > 0:
            marginal_costs = self.calculate_marginal_costs(year)
            marginal_costs.sort()
            accumulated_supply = 0
            for marginal_cost, deficit in marginal_costs:
                accumulated_supply += deficit
                if accumulated_supply >= demand:
                    self.market_price = marginal_cost
                    break
            else:
                self.market_price = marginal_costs[-1][0] if marginal_costs else 0
        elif supply == 0:
            self.market_price = 200  # High price due to lack of supply
        else:
            self.market_price = 0  # No demand or negative demand
        print(f"Market price set to: {self.market_price}")
    
      
    def trade_allowances(self, year):
        print(f"Trading allowances for year {year}...")
        buyers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0]
        sellers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0]
    
        # Calculate total supply and demand
        total_supply = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
        total_demand = abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
    
        print(f"Year {year} Debug: Total Supply = {total_supply}, Total Demand = {total_demand}")
    
        if total_demand == 0 and total_supply == 0:
            print(f"Warning: Both supply and demand are zero for year {year}. No trades executed.")
            self.market_price = 0
            return
    
        # Determine market price dynamically
        self.determine_market_price(total_supply, total_demand, year)
    
        # Store the market price for the year
        self.facilities_data[f'Allowance Price_{year}'] = self.market_price
    
        if buyers.empty or sellers.empty:
            print(f"No buyers or sellers for year {year}. Skipping trades.")
            return
    
        print(f"Buyers: {len(buyers)}, Sellers: {len(sellers)}")
    
        for buyer_idx, buyer in buyers.iterrows():
            deficit = abs(buyer[f'Allowance Surplus/Deficit_{year}'])
            carryover = self.facilities_data.at[buyer_idx, 'Credit Carryover']
    
            if carryover > 0:
                applied_credits = min(deficit, carryover)
                self.facilities_data.at[buyer_idx, f'Allowance Surplus/Deficit_{year}'] += applied_credits
                self.facilities_data.at[buyer_idx, 'Credit Carryover'] -= applied_credits
                deficit -= applied_credits
                print(f"Buyer {buyer_idx} used {applied_credits} credits. Remaining deficit: {deficit}")
    
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
  
    def run_model(self, start_year, end_year):
        print("Running emissions trading model...")
        for year in range(start_year, end_year + 1):
            print(f"\nProcessing year {year}...")
            self.calculate_dynamic_allowance_surplus_deficit(year)
            self.calculate_abatement(year)
            self.trade_allowances(year)
            print(f"Credit Carryover after year {year}:")
            print(self.facilities_data['Credit Carryover'].describe())
        print("Model run complete.")

    def summarize_market_supply_and_demand(self, year):
        print(f"Year {year}: Surplus/Deficit distribution:")
        print(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].describe())
        total_supply = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
        total_demand = abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
        net_demand = total_demand - total_supply
        total_trade_volume = self.facilities_data[f'Trade Volume_{year}'].sum()
        total_banked_allowances = self.facilities_data[f'Banked Allowances_{year}'].sum()
        total_allocations = self.facilities_data[f'Allocations_{year}'].sum()
        total_emissions = self.facilities_data[f'Emissions_{year}'].sum()
        total_output = self.facilities_data[f'Output_{year}'].sum()
        
        print(f"Year {year}: Total Supply: {total_supply}, Total Demand: {total_demand}, Net Demand: {net_demand}, Banked Allowances: {total_banked_allowances}")
        print(f"Year {year}: Total Allocations: {total_allocations}, Total Emissions: {total_emissions}, Total Output: {total_output}")
        
        summary = {
            'Year': year,
            'Total Supply (MTCO2e)': total_supply,
            'Total Demand (MTCO2e)': total_demand,
            'Net Demand (MTCO2e)': net_demand,
            'Total Trade Volume (MTCO2e)': total_trade_volume,
            'Banked Allowances (MTCO2e)': total_banked_allowances,
            'Total Allocations (MTCO2e)': total_allocations,
            'Total Emissions (MTCO2e)': total_emissions,
            'Total Output (MTCO2e)': total_output,
            'Allowance Price ($/MTCO2e)': self.market_price
        }
        return summary
    
    def save_reshaped_facility_summary(self, start_year, end_year, output_file):
        reshaped_data = []
        for year in range(start_year, end_year + 1):
            year_data = self.facilities_data[[
                "Facility ID", f"Emissions_{year}", f"Benchmark_{year}",
                f"Allocations_{year}", f"Allowance Surplus/Deficit_{year}",
                f"Abatement Cost_{year}", f"Trade Cost_{year}",
                f"Total Cost_{year}", f"Trade Volume_{year}",
                f"Allowance Price_{year}"
            ]].copy()
            year_data.rename(columns={
                f"Emissions_{year}": "Emissions",
                f"Benchmark_{year}": "Benchmark",
                f"Allocations_{year}": "Allocations",
                f"Allowance Surplus/Deficit_{year}": "Allowance Surplus/Deficit",
                f"Abatement Cost_{year}": "Abatement Cost",
                f"Trade Cost_{year}": "Trade Cost",
                f"Total Cost_{year}": "Total Cost",
                f"Trade Volume_{year}": "Trade Volume",
                f"Allowance Price_{year}": "Allowance Price"
            }, inplace=True)
            year_data["Year"] = year
            reshaped_data.append(year_data.melt(
                id_vars=["Facility ID", "Year"],
                var_name="Variable Name", value_name="Value"
            ))
    
        reshaped_combined_data = pd.concat(reshaped_data, ignore_index=True)
        reshaped_combined_data.to_csv(output_file, index=False)
        print(f"Reshaped facility summary saved to {output_file}")

