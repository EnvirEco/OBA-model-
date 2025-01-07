import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

class obamodel:
    def __init__(self, facilities_data: pd.DataFrame, abatement_cost_curve: pd.DataFrame, start_year: int):
        """Initialize OBA model with configuration and market parameters."""
        self.facilities_data = facilities_data.copy()
        self.abatement_cost_curve = abatement_cost_curve
        self.start_year = start_year
        
        # Market parameters
        self.floor_price = 5.0
        self.ceiling_price = 1000.0
        self.price_change_limit = 0.15  # 15% max price change between periods
        self.market_price = 0.0
        
        # Validate input data
        self._validate_input_data()
        self._initialize_columns()
        
    def _validate_input_data(self) -> None:
        """Validate input data structure and relationships."""
        required_facility_cols = {
            'Facility ID', 'Baseline Output', 'Baseline Emissions',
            'Baseline Benchmark', 'Baseline Profit Rate', 'Output Growth Rate',
            'Emissions Growth Rate', 'Benchmark Ratchet Rate'
        }
        required_abatement_cols = {
            'Facility ID', 'Slope', 'Intercept', 'Max Reduction (MTCO2e)'
        }
        
        missing_facility_cols = required_facility_cols - set(self.facilities_data.columns)
        missing_abatement_cols = required_abatement_cols - set(self.abatement_cost_curve.columns)
        
        if missing_facility_cols or missing_abatement_cols:
            raise ValueError(f"Missing required columns: Facilities: {missing_facility_cols}, Abatement: {missing_abatement_cols}")
            
        # Ensure all facilities have abatement curves
        facility_ids = set(self.facilities_data['Facility ID'])
        abatement_ids = set(self.abatement_cost_curve['Facility ID'])
        if facility_ids != abatement_ids:
            raise ValueError("Mismatch between facility IDs in data and abatement curves")

    def _initialize_columns(self) -> None:
        """Initialize all required columns without fragmentation."""
        metrics = [
            "Output", "Emissions", "Benchmark", "Allocations",
            "Allowance Surplus/Deficit", "Abatement Cost", "Trade Cost",
            "Total Cost", "Trade Volume", "Allowance Price",
            "Tonnes Abated", "Allowance Purchase Cost", "Allowance Sales Revenue",
            "Compliance Cost", "Cost to Profit Ratio", "Cost to Output Ratio"
        ]
        
        year_cols = [f"{metric}_{year}" 
                    for year in range(self.start_year, self.start_year + 20)
                    for metric in metrics]
                    
        # Create new columns all at once
        new_cols = pd.DataFrame(0.0, 
                              index=self.facilities_data.index,
                              columns=year_cols)
        
        # Concat with existing data
        self.facilities_data = pd.concat([self.facilities_data, new_cols], axis=1)
        
        # Ensure no fragmentation
        self.facilities_data = self.facilities_data.copy()

    def calculate_dynamic_values(self, year: int) -> None:
        years_elapsed = year - self.start_year
        
        # Calculate output and emissions
        self.facilities_data[f'Output_{year}'] = (
            self.facilities_data['Baseline Output'] * 
            (1 + self.facilities_data['Output Growth Rate']) ** years_elapsed
        )
        
        self.facilities_data[f'Emissions_{year}'] = (
            self.facilities_data['Baseline Emissions'] * 
            (1 + self.facilities_data['Emissions Growth Rate']) ** years_elapsed
        )
        
        self.facilities_data[f'Benchmark_{year}'] = (
            self.facilities_data['Baseline Benchmark'] * 
            (1 + self.facilities_data['Benchmark Ratchet Rate']) ** years_elapsed
        )
        
        # Calculate allocations
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * 
            self.facilities_data[f'Benchmark_{year}']
        )
        
        # Calculate initial surplus/deficit
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - 
            self.facilities_data[f'Emissions_{year}']
        )
        
        print(f"Supply check: {self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()}")
        
        print(f"\nYear {year} Post-calculation:")
        print(f"Average Emissions: {self.facilities_data[f'Emissions_{year}'].mean()}")
        print(f"Average Benchmark: {self.facilities_data[f'Benchmark_{year}'].mean()}")
        print(f"Total Deficit: {self.facilities_data[f'Allowance Surplus/Deficit_{year}'].sum()}")

    def determine_market_price(self, supply: float, demand: float, year: int) -> float:
        MIN_PRICE = 20.0
        MAX_PRICE = 200.0  # More reasonable ceiling
        
        remaining_demand = max(0, demand - supply)
        if remaining_demand <= 0:
            return MIN_PRICE
            
        mac_curve = self._build_mac_curve(year)
        if not mac_curve:
            return MIN_PRICE
            
        price_index = min(int(remaining_demand * 10), len(mac_curve) - 1)  # Scale demand impact
        self.market_price = min(max(mac_curve[price_index], MIN_PRICE), MAX_PRICE)
        
        print(f"Year {year} - Demand: {demand}, Supply: {supply}, Price: {self.market_price}")
        return self.market_price


            
    def _build_mac_curve(self, year: int) -> List[float]:
       mac_points = []
       deficits = self.facilities_data[
           self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0
       ]
       
       for _, facility in deficits.iterrows():
           curve = self.abatement_cost_curve[
               self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
           ].iloc[0]
           
           deficit = abs(facility[f'Allowance Surplus/Deficit_{year}'])
           max_reduction = min(curve['Max Reduction (MTCO2e)'], deficit)
           
           steps = 100
           for i in range(steps):
               qty = max_reduction * (i + 1) / steps
               mac = curve['Slope'] * qty + curve['Intercept'] 
               if mac > 0 and mac <= self.ceiling_price:
                   mac_points.append(mac)
                   
       if not mac_points:
           return [self.floor_price]
           
       return sorted(mac_points)
    
    def _apply_abatement(self, idx: int, amount: float, cost: float, year: int) -> None:
        self.facilities_data.at[idx, f'Tonnes Abated_{year}'] = amount
        self.facilities_data.at[idx, f'Abatement Cost_{year}'] = cost
        self.facilities_data.at[idx, f'Allowance Surplus/Deficit_{year}'] += amount

    def calculate_abatement(self, year: int) -> None:
       for idx, facility in self.facilities_data.iterrows():
           if facility[f'Allowance Surplus/Deficit_{year}'] >= 0:
               continue
               
           deficit = abs(facility[f'Allowance Surplus/Deficit_{year}'])
           curve = self.abatement_cost_curve[
               self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
           ].iloc[0]
           
           max_reduction = float(curve['Max Reduction (MTCO2e)'])
           slope = float(curve['Slope'])
           intercept = float(curve['Intercept'])
           
           steps = 100
           optimal_abatement = 0
           
           for step in range(steps):
               qty = min(max_reduction, deficit) * (step + 1) / steps
               mac = slope * qty + intercept
               
               if mac > self.market_price:
                   optimal_abatement = qty * (step / steps)
                   break
    
           if optimal_abatement > 0:
               # Ensure positive cost
               total_cost = abs((slope * optimal_abatement**2 / 2) + (intercept * optimal_abatement))
               self._apply_abatement(idx, optimal_abatement, total_cost, year)

             
    def execute_trades(self, year: int) -> None:
       if self.market_price <= 0:
           return
           
       while True:
           buyers = self.facilities_data[
               self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0
           ].sort_values(f'Allowance Surplus/Deficit_{year}')
           
           sellers = self.facilities_data[
               self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0
           ].sort_values(f'Allowance Surplus/Deficit_{year}', ascending=False)
           
           if buyers.empty or sellers.empty:
               break
               
           traded = False
           for buyer_idx, buyer in buyers.iterrows():
               deficit = abs(buyer[f'Allowance Surplus/Deficit_{year}'])
               
               for seller_idx, seller in sellers.iterrows():
                   surplus = seller[f'Allowance Surplus/Deficit_{year}']
                   volume = min(deficit, surplus)
                   
                   if volume > 0:
                       cost = volume * self.market_price
                       self._update_trade_positions(buyer_idx, seller_idx, volume, cost, year)
                       traded = True
                       break
                       
               if traded:
                   break
                   
           if not traded:
               break

    def _update_trade_positions(self, buyer_idx: int, seller_idx: int, volume: float, cost: float, year: int) -> None:
       # Update buyer
       self.facilities_data.at[buyer_idx, f'Trade Volume_{year}'] += volume
       self.facilities_data.at[buyer_idx, f'Allowance Surplus/Deficit_{year}'] += volume
       self.facilities_data.at[buyer_idx, f'Allowance Purchase Cost_{year}'] += cost
       
       # Update seller
       self.facilities_data.at[seller_idx, f'Trade Volume_{year}'] -= volume
       self.facilities_data.at[seller_idx, f'Allowance Surplus/Deficit_{year}'] -= volume
       self.facilities_data.at[seller_idx, f'Allowance Sales Revenue_{year}'] += cost                   

    def calculate_costs(self, year: int) -> None:
        # Separate compliance costs from trading revenues
        self.facilities_data[f'Compliance Cost_{year}'] = (
            self.facilities_data[f'Abatement Cost_{year}'].clip(lower=0) +
            self.facilities_data[f'Allowance Purchase Cost_{year}'].clip(lower=0)
        )
        
        self.facilities_data[f'Total Cost_{year}'] = (
            self.facilities_data[f'Compliance Cost_{year}'] -
            self.facilities_data[f'Allowance Sales Revenue_{year}']
        )

    def run_model(self, end_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
       """Run full model simulation and return results."""
       market_summary = []
       
       for year in range(self.start_year, end_year + 1):
           # Calculate year's initial positions
           self.calculate_dynamic_values(year)
           
           # Calculate allocations and initial surplus/deficit
           self.facilities_data[f'Allocations_{year}'] = (
               self.facilities_data[f'Output_{year}'] * 
               self.facilities_data[f'Benchmark_{year}']
           )
           
           self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
               self.facilities_data[f'Allocations_{year}'] - 
               self.facilities_data[f'Emissions_{year}']
           )
           
           # Market operations
           total_supply = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
           total_demand = abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
           
           self.determine_market_price(total_supply, total_demand, year)
           self.calculate_abatement(year)
           self.execute_trades(year)
           self.calculate_costs(year)
           
           # Record market summary
           market_summary.append(self._create_market_summary(year))
           
       return pd.DataFrame(market_summary), self.facilities_data

    def _create_market_summary(self, year: int) -> Dict:
        """Create market summary for given year."""
        return {
            'Year': year,
            'Total Allocations': self.facilities_data[f'Allocations_{year}'].sum(),
            'Total Emissions': self.facilities_data[f'Emissions_{year}'].sum(),
            'Total Abatement': self.facilities_data[f'Tonnes Abated_{year}'].sum(),
            'Market Price': self.market_price,
            'Trade Volume': self.facilities_data[f'Trade Volume_{year}'].abs().sum() / 2,  # Divide by 2 to avoid double counting
            'Total Trade Cost': self.facilities_data[f'Trade Cost_{year}'].abs().sum() / 2,
            'Total Abatement Cost': self.facilities_data[f'Abatement Cost_{year}'].sum(),
            'Total Compliance Cost': self.facilities_data[f'Compliance Cost_{year}'].sum(),
            'Net Market Cost': self.facilities_data[f'Total Cost_{year}'].sum(),
            'Average Cost to Profit Ratio': self.facilities_data[f'Cost to Profit Ratio_{year}'].mean(),
            'Average Cost to Output Ratio': self.facilities_data[f'Cost to Output Ratio_{year}'].mean(),
            'Remaining Surplus': self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum(),
            'Remaining Deficit': abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
        }

    def save_results(self, market_summary: pd.DataFrame, 
                      facilities_data: pd.DataFrame,
                      output_dir: str = '.') -> None:
        """Save model results to CSV files."""
        # Save market summary
        market_file = f"{output_dir}/market_summary.csv"
        market_summary.to_csv(market_file, index=False)
        
        # Save facility-level data in long format
        facility_metrics = [col for col in facilities_data.columns 
                           if any(yr in col for yr in map(str, range(self.start_year, self.start_year + 20)))]
        
        long_data = []
        for year in range(self.start_year, max(market_summary['Year']) + 1):
            year_data = facilities_data[['Facility ID'] + 
                                      [col for col in facility_metrics if str(year) in col]].copy()
            
            # Rename columns to remove year suffix
            year_data.columns = [col.split(f'_{year}')[0] if f'_{year}' in col else col 
                               for col in year_data.columns]
            
            year_data['Year'] = year
            long_data.append(year_data)
        
        facility_file = f"{output_dir}/facility_results.csv"
        pd.concat(long_data).to_csv(facility_file, index=False)
    
    def get_compliance_report(self, year: int) -> pd.DataFrame:
        """Generate compliance report for specified year."""
        metrics = [
            'Output', 'Emissions', 'Allocations', 'Allowance Surplus/Deficit',
            'Tonnes Abated', 'Trade Volume', 'Compliance Cost', 'Total Cost',
            'Cost to Profit Ratio', 'Cost to Output Ratio'
        ]
        
        columns = ['Facility ID'] + [f'{metric}_{year}' for metric in metrics]
        report = self.facilities_data[columns].copy()
        
        # Rename columns to remove year suffix
        report.columns = ['Facility ID'] + metrics
        return report
