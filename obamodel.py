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
        years_since_start = year - start_year
        self.facilities_data[f'Output_{year}'] = (
            self.facilities_data['Baseline Output'] * (1 + self.facilities_data['Output Growth Rate']) ** years_since_start
        )
        self.facilities_data[f'Emissions_{year}'] = (
            self.facilities_data['Baseline Emissions'] * (1 + self.facilities_data['Emissions Growth Rate']) ** years_since_start
        )
        self.facilities_data[f'Benchmark_{year}'] = (
            self.facilities_data['Baseline Benchmark'] * (1 + self.facilities_data['Benchmark Ratchet Rate']) ** years_since_start
        )
    
        # Debug: Check dynamic values
        print(f"Dynamic values for {year}:")
        print(self.facilities_data[[f'Output_{year}', f'Emissions_{year}', f'Benchmark_{year}']].describe())


   def calculate_allowance_allocation(self, year):
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * self.facilities_data[f'Benchmark_{year}']
        )
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - self.facilities_data[f'Emissions_{year}']
        )
    
        # Debug: Check allocation values
        print(f"Allocations and Surplus/Deficit for {year}:")
        print(self.facilities_data[[f'Allocations_{year}', f'Allowance Surplus/Deficit_{year}']].describe())

    def determine_market_price(self, supply, demand):
        """
        Set the market price for allowances based on supply and demand.
        """
        if demand <= 0:
            self.market_price = 0
        elif supply == 0:
            self.market_price = self.price_ceiling
        else:
            supply_demand_ratio = supply / demand
            self.market_price = min(self.price_ceiling, max(10, 100 * (1 / supply_demand_ratio)))
    
        # Debug: Market price and conditions
        print(f"Year: Market Price: {self.market_price}, Supply: {supply}, Demand: {demand}")

    def calculate_abatement_costs(self, year):
        self.facilities_data[f'Abatement Cost_{year}'] = 0.0
    
        for index, row in self.facilities_data.iterrows():
            facility_curve = self.abatement_cost_curve[self.abatement_cost_curve['Facility ID'] == row['Facility ID']]
            if not facility_curve.empty:
                slope = facility_curve['Slope'].values[0]
                intercept = facility_curve['Intercept'].values[0]
                max_reduction = facility_curve['Max Reduction (MTCO2e)'].values[0]
    
                surplus_deficit = row[f'Allowance Surplus/Deficit_{year}']
                if surplus_deficit < 0:
                    abatement = min(abs(surplus_deficit), max_reduction)
                    cost = slope * abatement + intercept
                    self.facilities_data.at[index, f'Abatement Cost_{year}'] = cost
                    self.facilities_data.at[index, f'Allowance Surplus/Deficit_{year}'] += abatement
    
        # Debug: Check abatement impact
        print(f"Year {year}: Post-Abatement Surplus/Deficit:")
        print(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].describe())

    def trade_allowances(self, year):
        self.facilities_data[f'Trade Cost_{year}'] = 0.0
        self.facilities_data[f'Trade Volume_{year}'] = 0.0
    
        buyers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0]
        sellers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0]
    
        print(f"Year {year}: Buyers Count: {len(buyers)}, Sellers Count: {len(sellers)}")
        if buyers.empty or sellers.empty:
            print(f"Year {year}: No trades executed due to lack of buyers or sellers.")
            return
    
        for buyer_idx, buyer_row in buyers.iterrows():
            deficit = abs(buyer_row[f'Allowance Surplus/Deficit_{year}'])
            for seller_idx, seller_row in sellers.iterrows():
                surplus = seller_row[f'Allowance Surplus/Deficit_{year}']
                if deficit <= 0 or surplus <= 0:
                    continue
    
                trade_volume = min(deficit, surplus)
                trade_cost = trade_volume * self.market_price
    
                # Update buyer and seller balances
                self.facilities_data.at[buyer_idx, f'Trade Volume_{year}'] += trade_volume
                self.facilities_data.at[buyer_idx, f'Trade Cost_{year}'] += trade_cost
                self.facilities_data.at[buyer_idx, f'Allowance Surplus/Deficit_{year}'] += trade_volume
    
                self.facilities_data.at[seller_idx, f'Trade Volume_{year}'] -= trade_volume
                self.facilities_data.at[seller_idx, f'Trade Cost_{year}'] -= trade_cost
                self.facilities_data.at[seller_idx, f'Allowance Surplus/Deficit_{year}'] -= trade_volume
    
                deficit -= trade_volume
                surplus -= trade_volume
    
                print(f"Trade executed: Buyer {buyer_idx}, Seller {seller_idx}, Volume: {trade_volume}, Cost: {trade_cost}")
                if deficit <= 0:
                    break


# Ensure to call the trade_allowances method in the run_model method
           
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
    
            try:
                # Ensure dynamic values are calculated first
                self.calculate_dynamic_values(year, self.start_year)
    
                # Verify the calculated columns exist
                required_columns = [f'Output_{year}', f'Benchmark_{year}', f'Emissions_{year}']
                for col in required_columns:
                    if col not in self.facilities_data.columns:
                        raise KeyError(f"Missing required column: {col}")
    
                # Calculate allocations and surplus/deficit
                self.calculate_allowance_allocation(year)
                
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
            
            except KeyError as e:
                print(f"KeyError during model run for year {year}: {e}")
    
        return yearly_results
