import pandas as pd

class obamodel:
    def __init__(self, facilities_data, abatement_cost_curve, price_ceiling, start_year):
        """
        Initialize the emissions trading model.

        Parameters:
        facilities_data (pd.DataFrame): Facility-level data with emissions, output, etc.
        abatement_cost_curve (pd.DataFrame): Abatement cost data per facility.
        price_ceiling (float): Maximum allowable price for allowances ($/MTCO2e).
        start_year (int): The starting year of the simulation.
        """
        self.facilities_data = facilities_data
        self.abatement_cost_curve = abatement_cost_curve
        self.price_ceiling = price_ceiling
        self.start_year = start_year  # Store the start_yearself.market_price = None
        self.government_revenue = 0.0
        self.facilities_data['Ceiling Price Payment'] = 0.0
        self.facilities_data['Tonnes Paid at Ceiling'] = 0.0
        
        # Check if 'Tonnes Paid at Ceiling' column exists, if not, create it
        if 'Tonnes Paid at Ceiling' not in self.facilities_data.columns:
            self.facilities_data['Tonnes Paid at Ceiling'] = 0.0
        
        self.facilities_data['Allowance Price ($/MTCO2e)'] = 0.0
        self.facilities_data['Trade Volume'] = 0.0  # Initialize trade volume for safety

    def calculate_dynamic_values(self, year, start_year):
        """
        Dynamically calculate Output, Emissions, and Benchmark for the given year.
        Uses the baseline year values and growth rates to project values for the current year.

        Parameters:
        year (int): The current year of the simulation.
        start_year (int): The starting year of the simulation.
        """
        years_since_start = year - start_year
        dynamic_data = {
            f'Output_{year}': self.facilities_data['Baseline Output'] * (1 + self.facilities_data['Output Growth Rate']) ** years_since_start,
            f'Emissions_{year}': self.facilities_data['Baseline Emissions'] * (1 + self.facilities_data['Emissions Growth Rate']) ** years_since_start,
            f'Benchmark_{year}': self.facilities_data['Baseline Benchmark'] * (1 + self.facilities_data['Benchmark Ratchet Rate']) ** years_since_start
        }
        self.facilities_data = pd.concat([self.facilities_data, pd.DataFrame(dynamic_data)], axis=1)

        # Debug: Print available columns to verify dynamic fields
        print(f"Dynamic values created for {year}: {list(self.facilities_data.columns)}")

    def calculate_allowance_allocation(self, year):
        """
        Calculate tradable output-based allocations for each facility for a specific year.
        """
        # Verify required columns
        required_columns = [f'Output_{year}', f'Benchmark_{year}', f'Emissions_{year}']
        for col in required_columns:
            if col not in self.facilities_data.columns:
                raise KeyError(f"Missing required column for allocation calculation: {col}")

        # Perform allocation calculation
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * self.facilities_data[f'Benchmark_{year}']
        )
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - self.facilities_data[f'Emissions_{year}']
        )

        # Debug: Confirm allocations creation
        print(f"Allocations and Allowance Surplus/Deficit calculated for {year}.")

    def determine_market_price(self, supply, demand):
        """
        Set the market price for allowances based on supply and demand.
        """
        # Debug: Supply and demand inputs
        print(f"Calculating market price...")
        print(f"Total Supply: {supply}, Total Demand: {demand}")
        
        if demand <= 0:
            self.market_price = 0
            print("No demand. Market price set to $0.")
        elif supply == 0:
            self.market_price = self.price_ceiling
            print(f"No supply. Market price set to price ceiling: ${self.price_ceiling}.")
        else:
            supply_demand_ratio = supply / demand
            self.market_price = min(self.price_ceiling, max(10, 100 * (1 / supply_demand_ratio)))
            print(f"Supply-Demand Ratio: {supply_demand_ratio}, Market Price: {self.market_price}")
        
        # Ensure the calculated market price is positive
        if self.market_price < 0:
            raise ValueError(f"Market price calculated as negative: {self.market_price}")
        
        print(f"Final Market Price: {self.market_price}")

    def calculate_abatement_costs(self, year):
        """
        Calculate abatement costs for each facility based on the abatement cost curve.
        """
        self.facilities_data[f'Abatement Cost_{year}'] = 0.0
    
        # Merge the facilities data with the abatement cost curve for vectorized operations
        merged_data = pd.merge(self.facilities_data, self.abatement_cost_curve, on='Facility ID', how='left')
    
        # Only consider rows where surplus_deficit < 0
        negative_deficit_mask = merged_data[f'Allowance Surplus/Deficit_{year}'] < 0
        abatement_needed = merged_data[negative_deficit_mask]
    
        # Calculate abatement required and costs
        abatement_needed[f'Abatement Required_{year}'] = abatement_needed.apply(
            lambda row: min(abs(row[f'Allowance Surplus/Deficit_{year}']), row['Max Reduction (MTCO2e)']), axis=1
        )
        abatement_needed[f'Abatement Cost_{year}'] = abatement_needed.apply(
            lambda row: row['Slope'] * row[f'Abatement Required_{year}'] + row['Intercept'], axis=1
        )
    
        # Update the original facilities data
        self.facilities_data.update(abatement_needed[[f'Abatement Cost_{year}']])
    
        # Debug: Print abatement costs for verification
        print(f"Abatement costs calculated for {year}:")
        print(self.facilities_data[[f'Abatement Cost_{year}']])

    def trade_allowances(self, year):
        """
        Simulate trading of allowances between facilities for the specified year.
        """
        self.facilities_data[f'Trade Cost_{year}'] = 0.0
        self.facilities_data[f'Trade Volume_{year}'] = 0.0
    
        buyers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0]
        sellers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0]
    
        # Debug: Buyers and sellers check
        print(f"Year {year}: Buyers={len(buyers)}, Sellers={len(sellers)}")
        if buyers.empty:
            print(f"Year {year}: No buyers detected. Check surplus/deficit calculations.")
        if sellers.empty:
            print(f"Year {year}: No sellers detected. Check surplus/deficit calculations.")
        if buyers.empty or sellers.empty:
            return
    
        for buyer_idx, buyer_row in buyers.iterrows():
            deficit = abs(buyer_row[f'Allowance Surplus/Deficit_{year}'])
            for seller_idx, seller_row in sellers.iterrows():
                surplus = seller_row[f'Allowance Surplus/Deficit_{year}']
                if deficit <= 0 or surplus <= 0:
                    continue
    
                trade_volume = min(deficit, surplus)
                trade_cost = trade_volume * self.market_price
    
                # Update balances
                self.facilities_data.at[buyer_idx, f'Trade Volume_{year}'] += trade_volume
                self.facilities_data.at[buyer_idx, f'Trade Cost_{year}'] += trade_cost
                self.facilities_data.at[buyer_idx, f'Allowance Surplus/Deficit_{year}'] += trade_volume
    
                self.facilities_data.at[seller_idx, f'Trade Volume_{year}'] -= trade_volume
                self.facilities_data.at[seller_idx, f'Trade Cost_{year}'] -= trade_cost
                self.facilities_data.at[seller_idx, f'Allowance Surplus/Deficit_{year}'] -= trade_volume
    
                # Adjust remaining deficit/surplus
                deficit -= trade_volume
                surplus -= trade_volume
                print(f"Trade executed: Buyer {buyer_idx}, Seller {seller_idx}, Volume={trade_volume}, Cost={trade_cost}")
    
                if deficit <= 0:
                    break
            
    def calculate_dynamic_allowance_surplus_deficit(self, year):
        """
        Dynamically calculate Allowance Surplus/Deficit for each facility for the specified year.
        This method ensures that the calculation is updated dynamically to reflect changes over the years.
    
        Parameters:
        year (int): The current year of the simulation.
        """
        # Calculate dynamic values for Output, Emissions, and Benchmark
        self.calculate_dynamic_values(year, self.start_year)
        
        # Verify required columns
        required_columns = [f'Output_{year}', f'Benchmark_{year}', f'Emissions_{year}']
        for col in required_columns:
            if col not in self.facilities_data.columns:
                raise KeyError(f"Missing required column for dynamic calculation: {col}")
    
        # Perform dynamic allowance allocation calculation
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * self.facilities_data[f'Benchmark_{year}']
        )
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - self.facilities_data[f'Emissions_{year}']
        )
    
        # Debug: Confirm dynamic allowance surplus/deficit calculation
        print(f"Dynamic Allowance Surplus/Deficit calculated for {year}:")
        print(self.facilities_data[[f'Allowance Surplus/Deficit_{year}']])

    # Existing methods...

    def summarize_market_supply_and_demand(self, year):
        """
        Summarize the market supply and demand for the specified year.

        Parameters:
        year (int): The current year of the simulation.

        Returns:
        dict: Summary of market supply and demand.
        """
        supply = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
        demand = abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
        if demand == 0:
            self.market_price = 0
        else:
            self.market_price = min(self.price_ceiling, max(10, 100 * (supply / demand)))
        return {
            'Year': year,
            'Supply': supply,
            'Demand': demand,
            'Market Price': self.market_price
        }
        
    def run_model(self, start_year, end_year):
        yearly_results = []
    
        for year in range(start_year, end_year + 1):
            print(f"Running model for year {year}...")
    
            self.calculate_dynamic_allowance_surplus_deficit(year)  # Use the new method
            
            # Debug: Print allowance surplus/deficit
            print(f"Allowance Surplus/Deficit for {year}:")
            print(self.facilities_data[[f'Allowance Surplus/Deficit_{year}']])
            
            self.calculate_abatement_costs(year)
            self.determine_market_price(
                self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum(),
                abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
            )
            
            # Debug: Print market price
            print(f"Market Price for {year}: {self.market_price}")
            
            self.trade_allowances(year)
            
            # Debug: Print trade volumes
            print(f"Trade Volume for {year}:")
            print(self.facilities_data[[f'Trade Volume_{year}']])
    
            self.facilities_data[f'Total Cost_{year}'] = (
                self.facilities_data[f'Abatement Cost_{year}'] +
                self.facilities_data[f'Trade Cost_{year}'] +
                self.facilities_data['Ceiling Price Payment']
            )
            self.facilities_data[f'Profit_{year}'] = (
                self.facilities_data[f'Output_{year}'] * self.facilities_data['Baseline Profit Rate']
            )
            self.facilities_data[f'Costs to Profits Ratio_{year}'] = (
                self.facilities_data[f'Total Cost_{year}'] / self.facilities_data[f'Profit_{year}']
            )
            self.facilities_data[f'Costs to Output Ratio_{year}'] = (
                self.facilities_data[f'Total Cost_{year}'] / self.facilities_data[f'Output_{year}']
            )
    
            facility_output_file = f"facility_summary_{year}.csv"
            yearly_facility_data = self.facilities_data[[
                'Facility ID', f'Emissions_{year}', f'Benchmark_{year}', f'Allocations_{year}',
                f'Allowance Surplus/Deficit_{year}', f'Abatement Cost_{year}',
                f'Total Cost_{year}', f'Profit_{year}', f'Costs to Profits Ratio_{year}',
                f'Costs to Output Ratio_{year}'
            ]]
            yearly_facility_data.to_csv(facility_output_file, index=False)
            print(f"Facility-level summary saved to {facility_output_file}")
    
            summary = self.summarize_market_supply_and_demand(year)
            market_output_file = f"market_summary_{year}.csv"
            pd.DataFrame([summary]).to_csv(market_output_file, index=False)
            print(f"Market-level summary saved to {market_output_file}")
    
            yearly_results.append(summary)
    
        return yearly_results
