import pandas as pd

class obamodel:
    def __init__(self, facilities_data, abatement_cost_curve, start_year):
        """
        Initialize the emissions trading model.

        Parameters:
        facilities_data (pd.DataFrame): Facility-level data with emissions, output, etc.
        abatement_cost_curve (pd.DataFrame): Abatement cost data per facility.
        start_year (int): The starting year of the simulation.
        """
        self.facilities_data = facilities_data
        self.abatement_cost_curve = abatement_cost_curve
        self.start_year = start_year
        self.market_price = 0.0  # Initialize market_price to 0.0
        self.government_revenue = 0.0
        self.facilities_data['Ceiling Price Payment'] = 0.0
        self.facilities_data['Tonnes Paid at Ceiling'] = 0.0

        # Check if 'Tonnes Paid at Ceiling' column exists, if not, create it
        if 'Tonnes Paid at Ceiling' not in self.facilities_data.columns:
            self.facilities_data['Tonnes Paid at Ceiling'] = 0.0

        self.facilities_data['Allowance Price ($/MTCO2e)'] = 0.0
        self.facilities_data['Trade Volume'] = 0.0  # Initialize trade volume for safety

        # Initialize banked allowances and vintage year
        self.facilities_data['Banked Allowances'] = 0.0
        self.facilities_data['Vintage Year'] = self.start_year

    def determine_market_price(self, supply, demand):
        """
        Set the market price for allowances based on supply and demand.
        """
        print(f"Market Price Calculation: Supply = {supply}, Demand = {demand}")
    
        if demand <= 0:
            self.market_price = 0
        elif supply == 0:
            # Set a default minimal supply value to avoid zero supply
            supply = 1
            self.market_price = max(10, 100 * (1 / supply))
        else:
            supply_demand_ratio = supply / demand
            self.market_price = max(10, 100 * (1 / supply_demand_ratio))
    
        print(f"Determined Market Price: {self.market_price}")

    def calculate_dynamic_allowance_surplus_deficit(self, year):
        """
        Dynamically calculate Allowance Surplus/Deficit for each facility for the specified year.
        This method ensures that the calculation is updated dynamically to reflect changes over the years.
        """
        # Calculate dynamic values for Output, Emissions, and Benchmark
        self.calculate_dynamic_values(year, self.start_year)
        
        # Verify required columns
        required_columns = [f'Output_{year}', f'Benchmark_{year}', f'Emissions_{year}']
        for col in required_columns:
            if col not in self.facilities_data.columns:
                raise KeyError(f"Missing required column for dynamic calculation: {col}")
    
        # Fill NaN values with zero before performing calculations
        self.facilities_data.fillna(0, inplace=True)
    
        # Perform dynamic allowance allocation calculation
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * self.facilities_data[f'Benchmark_{year}']
        )
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - self.facilities_data[f'Emissions_{year}']
        )
    
        # Ensure that the total supply is not zero
        total_supply = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
        if total_supply == 0:
            self.facilities_data[f'Allowance Surplus/Deficit_{year}'] += 1  # Adjust to avoid zero supply
    
        # Bank surplus allowances
        self.bank_allowances(year)
        
        # Debug: Confirm dynamic allowance surplus/deficit calculation
        print(f"Dynamic Allowance Surplus/Deficit calculated for {year}:")
        print(self.facilities_data[[f'Allowance Surplus/Deficit_{year}']])

    def trade_allowances(self, year):
        self.facilities_data[f'Trade Cost_{year}'] = 0.0
        self.facilities_data[f'Trade Volume_{year}'] = 0.0
    
        buyers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0]
        sellers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0]
    
        print(f"Year {year}: Buyers Count: {len(buyers)}, Sellers Count: {len(sellers)}")
        if buyers.empty or sellers.empty:
            print(f"Year {year}: No trades executed due to lack of buyers or sellers.")
            return
    
        if pd.isna(self.market_price):
            print(f"Year {year}: No trades executed due to undefined market price.")
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
        allocation_data = {
            f'Allocations_{year}': self.facilities_data[f'Output_{year}'] * self.facilities_data[f'Benchmark_{year}'],
            f'Allowance Surplus/Deficit_{year}': self.facilities_data[f'Output_{year}'] * self.facilities_data[f'Benchmark_{year}'] - self.facilities_data[f'Emissions_{year}']
        }
        self.facilities_data = pd.concat([self.facilities_data, pd.DataFrame(allocation_data)], axis=1).copy()

    def calculate_abatement_costs(self, year):
        """
        Calculate abatement costs for each facility based on the abatement cost curve.
        """
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
    
        # Debug: Validate post-abatement surplus/deficit
        print(f"Year {year}: Post-Abatement Surplus/Deficit:")
        print(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].describe())

    def bank_allowances(self, year):
        self.facilities_data['Banked Allowances'] += self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0)

    def update_vintages(self, year):
        self.facilities_data['Vintage Year'] = year

    def summarize_market_supply_and_demand(self, year):
        """
        Summarize total market supply, demand, net demand, and other metrics for the year.
        """
        # Debug: Check surplus/deficit distribution
        print(f"Year {year}: Surplus/Deficit distribution:")
        print(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].describe())
        
        # Calculate supply and demand
        total_supply = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
        total_demand = abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
        net_demand = total_demand - total_supply
        total_trade_volume = self.facilities_data[f'Trade Volume_{year}'].sum()
        total_banked_allowances = self.facilities_data['Banked Allowances'].sum()
        
        # Calculate total allocations, emissions, and output
        total_allocations = self.facilities_data[f'Allocations_{year}'].sum()
        total_emissions = self.facilities_data[f'Emissions_{year}'].sum()
        total_output = self.facilities_data[f'Output_{year}'].sum()
        
        # Debug: Validate supply, demand, and net demand
        print(f"Year {year}: Total Supply: {total_supply}, Total Demand: {total_demand}, Net Demand: {net_demand}, Banked Allowances: {total_banked_allowances}")
        print(f"Year {year}: Total Allocations: {total_allocations}, Total Emissions: {total_emissions}, Total Output: {total_output}")
    
        # Return the summary dictionary
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
                    f'Costs to Output Ratio_{year}', 'Banked Allowances'
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
