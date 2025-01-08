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
        
        # Initialize model
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
        
        # Calculate initial profit if not provided
        if 'Profit' not in self.facilities_data.columns:
            self.facilities_data['Profit'] = (
                self.facilities_data['Baseline Output'] * 
                self.facilities_data['Baseline Profit Rate']
            )

    def calculate_dynamic_values(self, year: int) -> None:
        """Calculate dynamic values for output, emissions, and allocations with detailed diagnostics."""
        years_elapsed = year - self.start_year
        
        print(f"\n=== Dynamic Value Analysis for Year {year} ===")
        print(f"Years elapsed: {years_elapsed}")
        
        # Calculate and store initial values for comparison
        baseline_allocations = (
            self.facilities_data['Baseline Output'] * 
            self.facilities_data['Baseline Benchmark']
        )
        
        # Calculate benchmark with ratchet
        self.facilities_data[f'Benchmark_{year}'] = (
            self.facilities_data['Baseline Benchmark'] * 
            (1 + self.facilities_data['Benchmark Ratchet Rate']) ** years_elapsed
        )
        
        # Calculate output with growth
        self.facilities_data[f'Output_{year}'] = (
            self.facilities_data['Baseline Output'] * 
            (1 + self.facilities_data['Output Growth Rate']) ** years_elapsed
        )
        
        # Calculate emissions with growth
        self.facilities_data[f'Emissions_{year}'] = (
            self.facilities_data['Baseline Emissions'] * 
            (1 + self.facilities_data['Emissions Growth Rate']) ** years_elapsed
        )
        
        # Calculate allocations with minimum provision
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * 
            self.facilities_data[f'Benchmark_{year}']
        )
        
        # Calculate initial surplus/deficit
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - 
            self.facilities_data[f'Emissions_{year}']
        )
        
        # Print diagnostic information
        print("\nBenchmark Analysis:")
        print(f"Average Starting Benchmark: {self.facilities_data['Baseline Benchmark'].mean():.4f}")
        print(f"Average Current Benchmark: {self.facilities_data[f'Benchmark_{year}'].mean():.4f}")
        print(f"Benchmark Reduction: {((1 - self.facilities_data[f'Benchmark_{year}'].mean() / self.facilities_data['Baseline Benchmark'].mean()) * 100):.1f}%")
        
        print("\nOutput Analysis:")
        print(f"Total Baseline Output: {self.facilities_data['Baseline Output'].sum():,.0f}")
        print(f"Total Current Output: {self.facilities_data[f'Output_{year}'].sum():,.0f}")
        print(f"Output Growth: {((self.facilities_data[f'Output_{year}'].sum() / self.facilities_data['Baseline Output'].sum() - 1) * 100):.1f}%")
        
        print("\nEmissions Analysis:")
        print(f"Total Baseline Emissions: {self.facilities_data['Baseline Emissions'].sum():,.0f}")
        print(f"Total Current Emissions: {self.facilities_data[f'Emissions_{year}'].sum():,.0f}")
        print(f"Emissions Growth: {((self.facilities_data[f'Emissions_{year}'].sum() / self.facilities_data['Baseline Emissions'].sum() - 1) * 100):.1f}%")
        
        print("\nAllocation Analysis:")
        print(f"Total Baseline Allocations: {baseline_allocations.sum():,.0f}")
        print(f"Total Current Allocations: {self.facilities_data[f'Allocations_{year}'].sum():,.0f}")
        print(f"Allocation Change: {((self.facilities_data[f'Allocations_{year}'].sum() / baseline_allocations.sum() - 1) * 100):.1f}%")
        
        # Market balance analysis
        total_surplus = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
        total_deficit = abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
        
        print("\nMarket Balance Analysis:")
        print(f"Total Surplus: {total_surplus:,.2f}")
        print(f"Total Deficit: {total_deficit:,.2f}")
        print(f"Net Position: {(total_surplus - total_deficit):,.2f}")
        
        # Check for severe imbalance
        if total_surplus == 0 or total_deficit == 0:
            print("\nWARNING: Market imbalance detected!")
            print("Consider adjusting benchmark ratchet rate or implementing minimum allocation provisions.")

    def calculate_dynamic_allowance_surplus_deficit(self, year: int) -> Tuple[float, float]:
        """Calculate supply and demand for a given year."""
        self.calculate_dynamic_values(year)
        
        # Calculate total supply (positive positions)
        total_supply = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
        
        # Calculate total demand (negative positions)
        total_demand = abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
        
        print(f"\nMarket Balance for Year {year}:")
        print(f"Total Supply: {total_supply:,.2f}")
        print(f"Total Demand: {total_demand:,.2f}")
        
        return total_supply, total_demand

    def determine_market_price(self, supply: float, demand: float, year: int) -> float:
        """Determine market price based on supply, demand and MAC curve."""
        MIN_PRICE = 20.0
        MAX_PRICE = 200.0
        
        remaining_demand = max(0, demand - supply)
        if remaining_demand <= 0:
            return MIN_PRICE
            
        mac_curve = self._build_mac_curve(year)
        if not mac_curve:
            return MIN_PRICE
            
        price_index = min(int(remaining_demand * 10), len(mac_curve) - 1)
        self.market_price = min(max(mac_curve[price_index], MIN_PRICE), MAX_PRICE)
        
        print(f"\nMarket Price for Year {year}:")
        print(f"Price: ${self.market_price:,.2f}")
        return self.market_price

    def _build_mac_curve(self, year: int) -> List[float]:
        """Build marginal abatement cost curve."""
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
                    
        return sorted(mac_points) if mac_points else [self.floor_price]

    def calculate_abatement(self, year: int) -> None:
        """Calculate and apply optimal abatement for facilities with deficits."""
        print(f"\n=== Abatement Analysis for Year {year} ===")
        
        abatement_summary = []
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
                total_cost = abs((slope * optimal_abatement**2 / 2) + (intercept * optimal_abatement))
                self._apply_abatement(idx, optimal_abatement, total_cost, year)
                
                abatement_summary.append({
                    'Facility ID': facility['Facility ID'],
                    'Abatement': optimal_abatement,
                    'Cost': total_cost,
                    'Final MAC': slope * optimal_abatement + intercept
                })
        
        if abatement_summary:
            summary_df = pd.DataFrame(abatement_summary)
            print("\nAbatement Summary:")
            print(summary_df.to_string())
            print(f"\nTotal Abatement: {summary_df['Abatement'].sum():,.2f}")
            print(f"Total Cost: ${summary_df['Cost'].sum():,.2f}")

    def _apply_abatement(self, idx: int, amount: float, cost: float, year: int) -> None:
        """Apply abatement results to facility data."""
        self.facilities_data.at[idx, f'Tonnes Abated_{year}'] = amount
        self.facilities_data.at[idx, f'Abatement Cost_{year}'] = cost
        self.facilities_data.at[idx, f'Allowance Surplus/Deficit_{year}'] += amount

    def trade_allowances(self, year: int) -> None:
        """Execute allowance trades between facilities."""
        print(f"\n=== Trading Analysis for Year {year} ===")
        
        # Pre-trade analysis
        buyers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0]
        sellers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0]
        
        print("\nPre-trade Positions:")
        print(f"Buyers found: {len(buyers)} with total deficit: {abs(buyers[f'Allowance Surplus/Deficit_{year}'].sum()):,.2f}")
        print(f"Sellers found: {len(sellers)} with total surplus: {sellers[f'Allowance Surplus/Deficit_{year}'].sum():,.2f}")
        
        if buyers.empty or sellers.empty:
            print("\nMarket imbalance detected!")
            print("\nEmissions vs Allocations:")
            print(self.facilities_data[[
                'Facility ID',
                f'Emissions_{year}',
                f'Allocations_{year}',
                f'Allowance Surplus/Deficit_{year}'
            ]].to_string())
            return  # Exit if no trading pairs found
        
        # Execute trades
        trades_executed = []
        for buyer_idx, buyer in buyers.iterrows():
            deficit = abs(buyer[f'Allowance Surplus/Deficit_{year}'])
            
            for seller_idx, seller in sellers.iterrows():
                surplus = seller[f'Allowance Surplus/Deficit_{year}']
                trade_volume = min(deficit, surplus)
                trade_cost = trade_volume * self.market_price
                
                if trade_volume > 0:
                    self._update_trade_positions(buyer_idx, seller_idx, trade_volume, trade_cost, year)
                    trades_executed.append({
                        'Buyer': buyer['Facility ID'],
                        'Seller': seller['Facility ID'],
                        'Volume': trade_volume,
                        'Price': self.market_price,
                        'Total Cost': trade_cost
                    })
                    deficit -= trade_volume
                    
                    if deficit <= 0:
                        break
        
        # Report results
        if trades_executed:
            trades_df = pd.DataFrame(trades_executed)
            print("\nTrades Executed:")
            print(trades_df.to_string())
            print(f"\nTotal Trade Volume: {trades_df['Volume'].sum():,.2f}")
        else:
            print("No trades were executed")

    def _update_trade_positions(self, buyer_idx: int, seller_idx: int, 
                              trade_volume: float, trade_cost: float, year: int) -> None:
        """Update positions after a trade."""
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

    def analyze_market_positions(self, year: int) -> pd.DataFrame:
        """Add detailed diagnostic logging for market positions."""
        print(f"\n=== Market Position Analysis for Year {year} ===")
        
        positions = self.facilities_data[[
            'Facility ID',
            f'Allocations_{year}',
            f'Emissions_{year}',
            f'Allowance Surplus/Deficit_{year}'
        ]].copy()
        
        print("\nAllocation Statistics:")
        print(f"Total Allocations: {positions[f'Allocations_{year}'].sum():,.2f}")
        print(f"Total Emissions: {positions[f'Emissions_{year}'].sum():,.2f}")
        print(f"Net Position: {positions[f'Allowance Surplus/Deficit_{year}'].sum():,.2f}")
        
        # Count buyers and sellers
        buyers = positions[positions[f'Allowance Surplus/Deficit_{year}'] < 0]
        sellers = positions[positions[f'Allowance Surplus/Deficit_{year}'] > 0]
        
        print(f"\nNumber of Buyers: {len(buyers)}")
        print(f"Total Deficit: {abs(buyers[f'Allowance Surplus/Deficit_{year}'].sum()):,.2f}")
        print(f"Number of Sellers: {len(sellers)}")
        print(f"Total Surplus: {sellers[f'Allowance Surplus/Deficit_{year}'].sum():,.2f}")
        
        return positions

    def calculate_costs(self, year: int) -> None:
        """Calculate various cost metrics for facilities."""
        # Compliance costs
        self.facilities_data[f'Compliance Cost_{year}'] = (
            self.facilities_data[f'Abatement Cost_{year}'].clip(lower=0) +
            self.facilities_data[f'Allowance Purchase Cost_{year}'].clip(lower=0)
        )
        
        # Total costs including trading revenues
        self.facilities_data[f'Total Cost_{year}'] = (
            self.facilities_data[f'Compliance Cost_{year}'] -
            self.facilities_data[f'Allowance Sales Revenue_{year}']
        )
        
    def calculate_cost_ratios(self, year: int) -> None:
        """Calculate cost ratios relative to profit and output."""
        # Cost to Profit ratio
        self.facilities_data[f'Cost to Profit Ratio_{year}'] = (
            self.facilities_data[f'Total Cost_{year}'] / self.facilities_data['Profit']
        ).replace([float('inf'), -float('inf')], 0).fillna(0)
        
        # Cost to Output ratio
        self.facilities_data[f'Cost to Output Ratio_{year}'] = (
            self.facilities_data[f'Total Cost_{year}'] / self.facilities_data[f'Output_{year}']
        ).replace([float('inf'), -float('inf')], 0).fillna(0)

    def run_model(self, start_year: int, end_year: int, output_file: str = "reshaped_combined_summary.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the complete model simulation for specified time period."""
        print("Running emissions trading model...")
        market_summary = []
        
        for year in range(start_year, end_year + 1):
            print(f"\nProcessing year {year}...")
            
            # Market operations
            total_supply, total_demand = self.calculate_dynamic_allowance_surplus_deficit(year)
            self.determine_market_price(total_supply, total_demand, year)
            self.calculate_abatement(year)
            self.trade_allowances(year)
            
            # Cost calculations
            self.calculate_costs(year)
            self.calculate_cost_ratios(year)
            
            # Collect market summary
            market_summary.append(self._create_market_summary(year))
            
            print(f"Year {year} complete.")
        
        # Create summary DataFrames
        market_summary_df = pd.DataFrame(market_summary)
        facility_results = self._prepare_facility_results(start_year, end_year)
        
        # Save results
        self.save_results(market_summary_df, facility_results, output_file)
        
        return market_summary_df, facility_results

    def _prepare_facility_results(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Prepare facility results in long format."""
        results = []
        
        metrics = [
            "Output", "Emissions", "Benchmark", "Allocations",
            "Allowance Surplus/Deficit", "Tonnes Abated", "Abatement Cost",
            "Trade Volume", "Trade Cost", "Allowance Purchase Cost",
            "Allowance Sales Revenue", "Compliance Cost", "Total Cost",
            "Cost to Profit Ratio", "Cost to Output Ratio"
        ]
        
        for year in range(start_year, end_year + 1):
            year_data = self.facilities_data[['Facility ID'] + 
                [f'{metric}_{year}' for metric in metrics]].copy()
            
            # Remove year suffix from column names
            year_data.columns = ['Facility ID'] + metrics
            year_data['Year'] = year
            results.append(year_data)
        
        return pd.concat(results, ignore_index=True)

    def save_results(self, market_summary: pd.DataFrame, facility_results: pd.DataFrame, 
                    output_file: str) -> None:
        """Save model results to CSV files."""
        # Save market summary
        market_summary.to_csv("market_summary.csv", index=False)
        print("Market summary saved to market_summary.csv")
        
        # Save facility results
        facility_results.to_csv(output_file, index=False)
        print(f"Facility results saved to {output_file}")

    def get_compliance_report(self, year: int) -> pd.DataFrame:
        """Generate a compliance report for a specific year."""
        metrics = [
            'Output', 'Emissions', 'Allocations', 'Allowance Surplus/Deficit',
            'Tonnes Abated', 'Trade Volume', 'Compliance Cost', 'Total Cost',
            'Cost to Profit Ratio', 'Cost to Output Ratio'
        ]
        
        report = self.facilities_data[['Facility ID'] + 
            [f'{metric}_{year}' for metric in metrics]].copy()
        
        # Clean up column names
        report.columns = ['Facility ID'] + metrics
        
        return report

    def _create_market_summary(self, year: int) -> Dict:
        """Create market summary dictionary for a specific year."""
        return {
            'Year': year,
            'Total Allocations': self.facilities_data[f'Allocations_{year}'].sum(),
            'Total Emissions': self.facilities_data[f'Emissions_{year}'].sum(),
            'Total Abatement': self.facilities_data[f'Tonnes Abated_{year}'].sum(),
            'Market Price': self.market_price,
            'Trade Volume': self.facilities_data[f'Trade Volume_{year}'].abs().sum() / 2,
            'Total Trade Cost': self.facilities_data[f'Trade Cost_{year}'].abs().sum() / 2,
            'Total Abatement Cost': self.facilities_data[f'Abatement Cost_{year}'].sum(),
            'Total Compliance Cost': self.facilities_data[f'Compliance Cost_{year}'].sum(),
            'Net Market Cost': self.facilities_data[f'Total Cost_{year}'].sum(),
            'Average Cost to Profit Ratio': self.facilities_data[f'Cost to Profit Ratio_{year}'].mean(),
            'Average Cost to Output Ratio': self.facilities_data[f'Cost to Output Ratio_{year}'].mean(),
            'Remaining Surplus': self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum(),
            'Remaining Deficit': abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
        }
