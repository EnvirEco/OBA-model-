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
            'Allowance Price ($/MTCO2e)', 'Trade Volume', 
            'Abatement Cost', 'Allowance Sales Revenue', 'Allowance Purchase Cost'
        ]
        year_columns = []
        for year in range(self.start_year, self.start_year + 20):  # Estimate 20 years
            year_columns.extend([
                f"Output_{year}", f"Emissions_{year}", f"Benchmark_{year}",
                f"Allocations_{year}", f"Allowance Surplus/Deficit_{year}",
                f"Abatement Cost_{year}", f"Trade Cost_{year}",
                f"Total Cost_{year}", f"Trade Volume_{year}",
                f"Allowance Price_{year}", f"Tonnes Abated_{year}",
                f"Allowance Purchase Cost_{year}", f"Allowance Sales Revenue_{year}"
            ])
        
        for column in required_columns + year_columns:
            if column not in self.facilities_data.columns:
                self.facilities_data[column] = 0.0

    def calculate_dynamic_values(self, year):
        print(f"Calculating dynamic values for year {year}...")
        years_since_start = year - self.start_year
        
        # Calculate dynamic outputs, emissions, and benchmarks
        self.facilities_data[f'Output_{year}'] = (
            self.facilities_data['Baseline Output'] * (1 + self.facilities_data['Output Growth Rate']) ** years_since_start
        )
        self.facilities_data[f'Emissions_{year}'] = (
            self.facilities_data['Baseline Emissions'] * (1 + self.facilities_data['Emissions Growth Rate']) ** years_since_start
        )
        self.facilities_data[f'Benchmark_{year}'] = (
            self.facilities_data['Baseline Benchmark'] * (1 + self.facilities_data['Benchmark Ratchet Rate']) ** years_since_start
        )
    
        # Debugging dynamic values
        print(f"Year {year} Dynamic Values Summary:")
        print(self.facilities_data[[f'Output_{year}', f'Emissions_{year}', f'Benchmark_{year}']].describe())

    def calculate_dynamic_allowance_surplus_deficit(self, year):
        print(f"Calculating dynamic allowance surplus/deficit for year {year}...")
        self.calculate_dynamic_values(year)
        
        # Calculate Allocations and Surplus/Deficit
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * self.facilities_data[f'Benchmark_{year}']
        )
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - self.facilities_data[f'Emissions_{year}']
        )
        
        # Debug Allocation and Surplus/Deficit
        print(f"Year {year}: Allocations and Emissions:")
        print(self.facilities_data[[f'Allocations_{year}', f'Emissions_{year}', f'Allowance Surplus/Deficit_{year}']].head())
        
        # Calculate Supply and Demand
        total_supply = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
        total_demand = abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
        
        # Debugging Supply and Demand
        print(f"Year {year}: Total Supply = {total_supply}, Total Demand = {total_demand}")
        print(f"Allowance Surplus/Deficit Details:\n{self.facilities_data[f'Allowance Surplus/Deficit_{year}']}")
        
        if total_demand == 0:
            print(f"Warning: Total demand is zero for year {year}. This may result in no trades or market price changes.")
        
        print(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].describe())

        return total_supply, total_demand


    def calculate_abatement_costs(self, year):
      print(f"Calculating abatement costs for year {year}...")
      for index, facility in self.facilities_data.iterrows():
          surplus_deficit = facility[f'Allowance Surplus/Deficit_{year}']
          if surplus_deficit < 0:  # Facility has a deficit
              facility_curve = self.abatement_cost_curve[
                  self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
              ]
              if not facility_curve.empty:
                  slope = float(facility_curve['Slope'].values[0])
                  intercept = float(facility_curve['Intercept'].values[0])
                  max_reduction = float(facility_curve['Max Reduction (MTCO2e)'].values[0])
  
                  abated = 0
                  abatement_cost = 0
                  while abs(surplus_deficit) > 0 and abated < max_reduction:
                      marginal_abatement_cost = slope * abated + intercept
                      if marginal_abatement_cost > self.market_price:
                          break  # Abatement no longer cost-effective
  
                      increment = min(abs(surplus_deficit), max_reduction - abated)
                      abated += increment
                      abatement_cost += marginal_abatement_cost * increment
                      surplus_deficit += increment
                    
                  # Update facility data
                  self.facilities_data.at[index, f'Tonnes Abated_{year}'] = abated
                  self.facilities_data.at[index, f'Abatement Cost_{year}'] = abatement_cost
                  self.facilities_data.at[index, f'Allowance Surplus/Deficit_{year}'] += abated

    def trade_allowances(self, year):
        print(f"Trading allowances for year {year}...")
        buyers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0].copy()
        sellers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0].copy()
        
        if buyers.empty or sellers.empty:
            print(f"No buyers or sellers for year {year}. Skipping trades.")
            return
    
        if self.market_price <= 0:
            print(f"Warning: Market price is {self.market_price}. No valid trades executed.")
            return
    
        for buyer_idx, buyer_row in buyers.iterrows():
            deficit = abs(buyer_row[f'Allowance Surplus/Deficit_{year}'])
            for seller_idx, seller_row in sellers.iterrows():
                surplus = seller_row[f'Allowance Surplus/Deficit_{year}']
                if deficit <= 0 or surplus <= 0:
                    continue
    
                # Calculate trade volume and cost
                trade_volume = min(deficit, surplus)
                trade_cost = trade_volume * self.market_price
    
                # Update buyer
                self.facilities_data.at[buyer_idx, f'Trade Volume_{year}'] += trade_volume
                self.facilities_data.at[buyer_idx, f'Trade Cost_{year}'] += trade_cost
                self.facilities_data.at[buyer_idx, f'Allowance Surplus/Deficit_{year}'] += trade_volume
                self.facilities_data.at[buyer_idx, f'Allowance Purchase Cost_{year}'] += trade_cost
    
                # Update seller
                self.facilities_data.at[seller_idx, f'Trade Volume_{year}'] -= trade_volume
                self.facilities_data.at[seller_idx, f'Trade Cost_{year}'] -= trade_cost
                self.facilities_data.at[seller_idx, f'Allowance Surplus/Deficit_{year}'] -= trade_volume
                self.facilities_data.at[seller_idx, f'Allowance Sales Revenue_{year}'] += trade_cost
    
                deficit -= trade_volume
                print(f"Trade executed: Buyer {buyer_idx}, Seller {seller_idx}, Volume: {trade_volume}, Cost: {trade_cost}")


    def determine_market_price(self, supply, demand, year):
      print(f"Determining market price for year {year}: Supply={supply}, Demand={demand}")
      
      if supply > 0 and demand > 0:
          sorted_abatement_costs = []
          for index, facility in self.facilities_data.iterrows():
              facility_curve = self.abatement_cost_curve[
                  self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
              ]
              if not facility_curve.empty:
                  slope = float(facility_curve['Slope'].values[0])
                  intercept = float(facility_curve['Intercept'].values[0])
                  max_reduction = int(facility_curve['Max Reduction (MTCO2e)'].values[0])
                  sorted_abatement_costs.extend(
                      [slope * x + intercept for x in range(1, max_reduction + 1)]
                  )
          
          if sorted_abatement_costs:
              sorted_abatement_costs = sorted(sorted_abatement_costs)
              effective_demand = min(int(demand), len(sorted_abatement_costs) - 1)
              self.market_price = sorted_abatement_costs[effective_demand]
          else:
              print(f"Warning: No abatement cost data available. Market price remains zero.")
              self.market_price = 0
      elif supply == 0:
          self.market_price = 200  # Arbitrary high price due to lack of supply
      else:
          self.market_price = 0  # No demand
      
      print(f"Year {year}: Market Price set to {self.market_price}")
      self.facilities_data[f'Allowance Price_{year}'] = self.market_price

    def run_model(self, start_year, end_year, output_file="reshaped_combined_summary.csv"):
        print("Running emissions trading model...")
        market_summary = []
        
        for year in range(start_year, end_year + 1):
            print(f"\nProcessing year {year}...")
            
            # Calculate dynamic allowance surplus/deficit
            total_supply, total_demand = self.calculate_dynamic_allowance_surplus_deficit(year)
            
            # Adjust small discrepancies to avoid negative rounding errors
            self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
                self.facilities_data[f'Allowance Surplus/Deficit_{year}'].round(6)
            )
# Determine market price
            self.determine_market_price(total_supply, total_demand, year)
            
            # Calculate abatement costs
            self.calculate_abatement_costs(year)
            
            # Execute trading logic
            self.trade_allowances(year)
            
            # Ensure total demand is captured
            if total_demand > 0:
                print(f"Year {year} Demand: {total_demand} MTCO2e")
            else:
                print(f"Year {year} Demand is incorrectly reported as zero.")

            # Calculate compliance cost for each facility
            self.facilities_data[f'Compliance Cost_{year}'] = (
                self.facilities_data[f'Abatement Cost_{year}'] +
                self.facilities_data[f'Allowance Purchase Cost_{year}']
            )

            # Calculate total costs for each facility
            self.facilities_data[f'Total Cost_{year}'] = (
                self.facilities_data[f'Abatement Cost_{year}'] +
                self.facilities_data['Allowance Purchase Cost'] -
                self.facilities_data['Allowance Sales Revenue']
            )
            # Collect market-level data
            total_abatement = self.facilities_data[f'Tonnes Abated_{year}'].sum()
            market_summary.append({
                "Year": year,
                "Total Supply (MTCO2e)": total_supply,
                "Total Demand (MTCO2e)": total_demand,
                "Net Demand (MTCO2e)": total_demand - total_supply,
                "Market Price ($/MTCO2e)": self.market_price,
                "Total Abatement (MTCO2e)": total_abatement,
            })
            
            print(f"Year {year} complete.")
        
        # Save reshaped facility summary
        self.save_reshaped_facility_summary(start_year, end_year, output_file)
        
        # Save market summary
        market_summary_df = pd.DataFrame(market_summary)
        market_summary_file = "market_summary.csv"
        market_summary_df.to_csv(market_summary_file, index=False)
        print(f"Market summary saved to {market_summary_file}")
        
        print("Model run complete.")

    def save_reshaped_facility_summary(self, start_year, end_year, output_file):
        reshaped_data = []
        for year in range(start_year, end_year + 1):
            year_data = self.facilities_data[[
               "Facility ID", f"Emissions_{year}", f"Benchmark_{year}",
                f"Allocations_{year}", f"Allowance Surplus/Deficit_{year}",
                f"Abatement Cost_{year}", f"Trade Cost_{year}",
                f"Total Cost_{year}", f"Compliance Cost_{year}",
                f"Trade Volume_{year}", f"Allowance Price_{year}",
                f"Tonnes Abated_{year}", f"Allowance Purchase Cost_{year}",
                f"Allowance Sales Revenue_{year}"
            ]].copy()
            year_data.rename(columns={
                f"Emissions_{year}": "Emissions",
                f"Benchmark_{year}": "Benchmark",
                f"Allocations_{year}": "Allocations",
                f"Allowance Surplus/Deficit_{year}": "Allowance Surplus/Deficit",
                f"Abatement Cost_{year}": "Abatement Cost",
                f"Trade Cost_{year}": "Trade Cost",
                f"Total Cost_{year}": "Total Cost",
                f"Compliance Cost_{year}": "Compliance Cost",
                f"Trade Volume_{year}": "Trade Volume",
                f"Allowance Price_{year}": "Allowance Price",
                f"Tonnes Abated_{year}": "Tonnes Abated",
                f"Allowance Purchase Cost_{year}": "Allowance Purchase Cost",
                f"Allowance Sales Revenue_{year}": "Allowance Sales Revenue"
            }, inplace=True)
            year_data["Year"] = year
            reshaped_data.append(year_data.melt(
                id_vars=["Facility ID", "Year"],
                var_name="Variable Name", value_name="Value"
            ))
        reshaped_combined_data = pd.concat(reshaped_data, ignore_index=True)
        reshaped_combined_data.to_csv(output_file, index=False)
        print(f"Reshaped facility summary saved to {output_file}")

