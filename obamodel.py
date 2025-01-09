import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

class obamodel:
    def __init__(self, facilities_data: pd.DataFrame, abatement_cost_curve: pd.DataFrame, start_year: int, end_year: int):
        """Initialize OBA model with configuration and market parameters."""
        self.facilities_data = facilities_data.copy()
        self.abatement_cost_curve = abatement_cost_curve
        self.start_year = start_year
        self.end_year = end_year if end_year is not None else start_year + 10

       # Initialize baseline allocations as a column in facilities_data
        self.facilities_data['Baseline Allocations'] = (
            self.facilities_data['Baseline Output'] * self.facilities_data['Baseline Benchmark']
        )

        print(f"Initialized Baseline Allocations: {self.facilities_data['Baseline Allocations'].sum():,.2f}")
        
        # Market parameters
        self.floor_price = 5
        self.ceiling_price = 1000.0
        self.price_change_limit = 0.15  # 15% max price change between periods
        self.market_price = 0.0
        
        # Initialize price schedule
        self.price_schedule = {
            year: 50 + 5 * (year - start_year) for year in range(start_year, end_year + 1)
        }

        print("Initialized price schedule:")
        for year, price in self.price_schedule.items():
            print(f"  Year {year}: ${price:.2f}")
        
        # Initialize model
        self._validate_input_data()
        self._initialize_columns()

    def run_scenario_analysis(self, scenario_file: str = "scenarios.csv") -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Run model for multiple scenarios defined in CSV file."""
        print("Starting scenario analysis...")
        
        # Load scenarios
        scenarios = pd.read_csv(scenario_file)
        results = {}
        
        for _, scenario in scenarios.iterrows():
            print(f"\nRunning scenario: {scenario['scenario_name']}")
            print(f"Description: {scenario['description']}")
            
            # Store original parameters
            original_params = {
                'benchmark_ratchet': self.facilities_data['Benchmark Ratchet Rate'].copy(),
                'floor_price': self.floor_price,
                'ceiling_price': self.ceiling_price
            }
            
            try:
                # Apply scenario parameters
                self.facilities_data['Benchmark Ratchet Rate'] = scenario['benchmark_ratchet_rate']
                self.floor_price = scenario['market_price_floor']
                self.ceiling_price = scenario['market_price_ceiling']
                
                # Run model
                market_summary, facility_results = self.run_model(
                    scenario['start_year'],
                    scenario['end_year']
                )
                
                # Store results
                results[scenario['scenario_name']] = {
                    'market_summary': market_summary,
                    'facility_results': facility_results,
                    'parameters': scenario.to_dict()
                }
                
            finally:
                # Restore original parameters
                self.facilities_data['Benchmark Ratchet Rate'] = original_params['benchmark_ratchet']
                self.floor_price = original_params['floor_price']
                self.ceiling_price = original_params['ceiling_price']
        
        # Save comparative analysis
        self._save_scenario_comparison(results)
        
        return results
        
    def _save_scenario_comparison(self, scenario_results: Dict) -> None:
        """Create and save comparative analysis of scenarios."""
        # Market price comparison
        price_comparison = pd.DataFrame({
            scenario: results['market_summary'].set_index('Year')['Market Price']
            for scenario, results in scenario_results.items()
        })
        price_comparison.to_csv('scenario_price_comparison.csv')
        
        # Emissions comparison
        emissions_comparison = pd.DataFrame({
            scenario: results['market_summary'].set_index('Year')['Total Emissions']
            for scenario, results in scenario_results.items()
        })
        emissions_comparison.to_csv('scenario_emissions_comparison.csv')
        
        # Abatement comparison
        abatement_comparison = pd.DataFrame({
            scenario: results['market_summary'].set_index('Year')['Total Abatement']
            for scenario, results in scenario_results.items()
        })
        abatement_comparison.to_csv('scenario_abatement_comparison.csv')
        
        # Create summary report
        with open('scenario_analysis_report.txt', 'w') as f:
            f.write("Scenario Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for scenario, results in scenario_results.items():
                f.write(f"\nScenario: {scenario}\n")
                f.write(f"Parameters: {results['parameters']}\n")
                f.write("\nKey Metrics (Average over period):\n")
                summary = results['market_summary'].mean()
                f.write(f"Average Price: ${summary['Market Price']:.2f}\n")
                f.write(f"Average Emissions: {summary['Total Emissions']:.2f}\n")
                f.write(f"Average Abatement: {summary['Total Abatement']:.2f}\n")
                f.write("-" * 50 + "\n")
    
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
        """Calculate dynamic values for output, emissions, and allocations with robust diagnostics."""
        years_elapsed = year - self.start_year
    
        print(f"\n=== Dynamic Value Analysis for Year {year} ===")
        print(f"Years elapsed: {years_elapsed}")
    
        # Adjust emissions intensity based on prior abatement
        if year > self.start_year:
            prior_emissions = self.facilities_data[f'Emissions_{year - 1}']
            prior_abatement = self.facilities_data[f'Tonnes Abated_{year - 1}']
            prior_output = self.facilities_data[f'Output_{year - 1}']
            emissions_intensity = (prior_emissions - prior_abatement) / prior_output
            emissions_intensity = emissions_intensity.clip(lower=0)  # Ensure non-negative intensity
        else:
            emissions_intensity = (
                self.facilities_data['Baseline Emissions'] /
                self.facilities_data['Baseline Output']
            )
    
        # Calculate current and target surplus
        total_allocations = self.facilities_data['Baseline Allocations'].sum()
        target_surplus = 0.05 * total_allocations  # Target surplus is 5% of total allocations
        if year > self.start_year:
            current_surplus = self.facilities_data[f'Allowance Surplus/Deficit_{year - 1}'].clip(lower=0).sum()
        else:
            current_surplus = target_surplus  # Assume balanced in the first year
    
        # Compute required ratchet adjustment to maintain target surplus ratio
        allocation_decline_rate = (total_allocations - target_surplus) / total_allocations
        required_ratchet_rate = allocation_decline_rate / (1 + years_elapsed)  # Spread over time
    
        # Apply bounds to the computed ratchet rate
        self.facilities_data['Benchmark Ratchet Rate'] = np.clip(
            required_ratchet_rate, 0.01, 0.20  # Limit annual decline to 1-20%
        )
    
        # Calculate benchmark with adjusted ratchet rate
        self.facilities_data[f'Benchmark_{year}'] = (
            self.facilities_data['Baseline Benchmark'] *
            (1 - self.facilities_data['Benchmark Ratchet Rate']) ** years_elapsed
        )
    
        # Calculate output with growth
        self.facilities_data[f'Output_{year}'] = (
            self.facilities_data['Baseline Output'] *
            (1 + self.facilities_data['Output Growth Rate']) ** years_elapsed
        )
    
        # Calculate emissions based on adjusted intensity and output
        self.facilities_data[f'Emissions_{year}'] = (
            self.facilities_data[f'Output_{year}'] * emissions_intensity
        )
        self.facilities_data[f'Emissions_{year}'] = np.clip(
            self.facilities_data[f'Emissions_{year}'], 0, None
        )
    
        # Calculate allocations
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] *
            self.facilities_data[f'Benchmark_{year}']
        )
        self.facilities_data[f'Allocations_{year}'] = np.clip(
            self.facilities_data[f'Allocations_{year}'], 0, None
        )
    
        # Calculate initial surplus/deficit
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] -
            self.facilities_data[f'Emissions_{year}']
        )
    
        # Diagnostics
        print(f"  Total Allocations: {self.facilities_data[f'Allocations_{year}'].sum():,.2f}")
        print(f"  Total Emissions: {self.facilities_data[f'Emissions_{year}'].sum():,.2f}")
        print(f"  Total Surplus/Deficit: {self.facilities_data[f'Allowance Surplus/Deficit_{year}'].sum():,.2f}")
        print(f"  Adjusted Benchmark Ratchet Rate: {self.facilities_data['Benchmark Ratchet Rate'].mean():.4f}")
        print(f"  Adjusted Allocations Diagnostic: {self.facilities_data[f'Allocations_{year}'].describe()}")


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
        """Determine market price based on supply, demand, and exogenous price schedule."""
        target_price = self.price_schedule.get(year, self.floor_price)
        remaining_demand = max(0, demand - supply)
    
        if remaining_demand <= 0:
            self.market_price = target_price  # Default to exogenous price
            return self.market_price
    
        mac_curve = self._build_mac_curve(year)
        if not mac_curve:
            self.market_price = target_price
            return self.market_price
    
        # Adjust price dynamically based on remaining demand
        price_index = min(int(remaining_demand * 10), len(mac_curve) - 1)
        self.market_price = max(target_price, mac_curve[price_index])
    
        print(f"Year {year} Market Price (Target: ${target_price:.2f}): ${self.market_price:.2f}")
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
            slope = float(curve['Slope'])
            intercept = float(curve['Intercept'])
    
            # Ensure intercept is reasonable
            if intercept < 0:
                print(f"Warning: Negative intercept ({intercept}) for Facility ID {facility['Facility ID']}. Adjusting to 0.")
                intercept = 0
            
            steps = 100
            for i in range(steps):
                qty = max_reduction * (i + 1) / steps
                mac = curve['Slope'] * qty + curve['Intercept']
                if mac > 0 and mac <= self.ceiling_price:
                    mac_points.append(mac)
        if not mac_points:
            print("Warning: Empty MAC curve. Using floor price.")
            mac_points = [self.floor_price]
            
        print(f"MAC Curve: Min={min(mac_points):.2f}, Max={max(mac_points):.2f}, Points={len(mac_points)}")                  
                    
        return sorted(mac_points) if mac_points else [self.floor_price]

    def calculate_abatement(self, year: int) -> None:
        """Calculate and apply optimal abatement for facilities with deficits."""
        print(f"\n=== Abatement Analysis for Year {year} ===")
        
        total_abatement = 0.0
        for idx, facility in self.facilities_data.iterrows():
            if facility[f'Allowance Surplus/Deficit_{year}'] >= 0:
                continue  # Skip facilities with no deficit
    
            deficit = abs(facility[f'Allowance Surplus/Deficit_{year}'])
            curve = self.abatement_cost_curve[
                self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
            ]
            
            if curve.empty:
                print(f"Warning: Missing abatement curve for Facility ID {facility['Facility ID']}")
                continue
            
            curve = curve.iloc[0]
            max_reduction = float(curve['Max Reduction (MTCO2e)'])
            slope = float(curve['Slope'])
            intercept = max(0, float(curve['Intercept']))
            
            # Incrementally abate to reduce deficit
            abated = 0.0
            while deficit > 0 and abated < max_reduction:
                marginal_cost = slope * abated + intercept
                if marginal_cost > self.market_price:
                    break  # Stop if MAC exceeds market price
                
                step_reduction = min(max_reduction - abated, deficit, 0.01)  # Abate in small steps
                deficit -= step_reduction
                abated += step_reduction
    
            if abated > 0:
                total_cost = (slope * abated**2 / 2) + (intercept * abated)
                self._apply_abatement(idx, abated, total_cost, year)
                total_abatement += abated
    
        # Log total abatement
        print(f"Year {year}: Total Abatement: {total_abatement:.2f}")

    def _apply_abatement(self, idx: int, abated: float, cost: float, year: int) -> None:
        """Apply abatement results to the facility's data."""
        self.facilities_data.at[idx, f'Tonnes Abated_{year}'] += abated
        self.facilities_data.at[idx, f'Abatement Cost_{year}'] += cost
        self.facilities_data.at[idx, f'Allowance Surplus/Deficit_{year}'] += abated
    
        # Log updates for debugging
        print(f"Facility {self.facilities_data.at[idx, 'Facility ID']} - Year {year}:")
        print(f"  Abated: {abated:.2f}, Cost: ${cost:.2f}")
        print(f"  Updated Surplus/Deficit: {self.facilities_data.at[idx, f'Allowance Surplus/Deficit_{year}']:.2f}")


    def trade_allowances(self, year: int) -> None:
        """Execute allowance trades between facilities."""
        print(f"\n=== Trading Analysis for Year {year} ===")
        
        # Pre-trade analysis
        pre_trade = self.analyze_market_positions(year)
        
        if self.market_price <= 0:
            print(f"Warning: Invalid market price (${self.market_price:,.2f})")
            return
    
        buyers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0].copy()
        sellers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0].copy()
        
        if buyers.empty or sellers.empty:
            print("No valid trading pairs found")
            return
    
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

    def run_model(self, output_file: str = "reshaped_combined_summary.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the complete model simulation for the initialized time period."""
        print("Running emissions trading model...")
        market_summary = []
        
        for year in range(self.start_year, self.end_year + 1):
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
            
            # Diagnostics
            print(f"Year {year} Results:")
            print(f"  Target Market Price: ${self.price_schedule.get(year, self.floor_price):.2f}")
            print(f"  Achieved Market Price: ${self.market_price:.2f}")
            print(f"  Total Supply: {total_supply:.2f}")
            print(f"  Total Demand: {total_demand:.2f}")
            print(f"  Remaining Surplus: {self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum():,.2f}")
            print(f"  Remaining Deficit: {abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum()):,.2f}")
            print(f"  Average Benchmark Ratchet Rate: {self.facilities_data['Benchmark Ratchet Rate'].mean():.4f}")
            
            print(f"Year {year} complete.")
        
        # Create summary DataFrames
        market_summary_df = pd.DataFrame(market_summary)
        facility_results = self._prepare_facility_results(self.start_year, self.end_year)
        
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
        # Calculate total abatement from facilities
        facility_total_abatement = self.facilities_data[f'Tonnes Abated_{year}'].sum()
        
        # Ensure alignment between facility and market-level totals
        if 'Total Abatement' in self.facilities_data.columns:
            market_total_abatement = self.facilities_data['Total Abatement'].sum()
        else:
            market_total_abatement = facility_total_abatement  # Fallback if not pre-calculated
    
        # Compare totals for diagnostics
        if not np.isclose(facility_total_abatement, market_total_abatement, atol=1e-5):
            print(f"Warning: Abatement mismatch for Year {year}.")
            print(f"Facility-level Abatement: {facility_total_abatement}")
            print(f"Market-level Abatement: {market_total_abatement}")
    
        # Calculate the emission-weighted average ratchet rate
        weighted_ratchet_rate = (
            (self.facilities_data['Baseline Emissions'] *
             self.facilities_data['Benchmark Ratchet Rate']).sum() /
            self.facilities_data['Baseline Emissions'].sum()
        )
    
        #validate sum of allcoatiosn against expecations 
        total_allocations = self.facilities_data[f'Allocations_{year}'].sum()
        if total_allocations < 0 or np.isnan(total_allocations):
            print(f"Warning: Total Allocations for Year {year} is invalid: {total_allocations}")

        #bound ratchet rate 
        self.facilities_data['Benchmark Ratchet Rate'] = np.clip(
            self.facilities_data['Benchmark Ratchet Rate'], 0, 1
)
        
        # Return the market summary
        return {
            'Year': year,
            'Total Allocations': self.facilities_data[f'Allocations_{year}'].sum(),
            'Total Emissions': self.facilities_data[f'Emissions_{year}'].sum(),
            'Total Abatement': facility_total_abatement,  # Use facility-level data for accuracy
            'Market Price': self.market_price,
            'Trade Volume': self.facilities_data[f'Trade Volume_{year}'].abs().sum() / 2,
            'Total Trade Cost': self.facilities_data[f'Trade Cost_{year}'].abs().sum() / 2,
            'Total Abatement Cost': self.facilities_data[f'Abatement Cost_{year}'].sum(),
            'Total Compliance Cost': self.facilities_data[f'Compliance Cost_{year}'].sum(),
            'Net Market Cost': self.facilities_data[f'Total Cost_{year}'].sum(),
            'Average Cost to Profit Ratio': self.facilities_data[f'Cost to Profit Ratio_{year}'].mean(),
            'Average Cost to Output Ratio': self.facilities_data[f'Cost to Output Ratio_{year}'].mean(),
            'Remaining Surplus': self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum(),
            'Remaining Deficit': abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum()),
            'Emission-Weighted Ratchet Rate': weighted_ratchet_rate
        }

