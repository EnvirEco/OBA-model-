```python
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from scipy import optimize

class obamodel:
    def __init__(self, facilities_data: pd.DataFrame, abatement_cost_curve: pd.DataFrame, start_year: int):
        """Initialize OBA model with configuration and market parameters."""
        self.facilities_data = facilities_data.copy()
        self.abatement_cost_curve = abatement_cost_curve
        self.start_year = start_year
        
        # Market parameters
        self.floor_price = 100
        self.ceiling_price = 1000.0
        self.price_change_limit = 0.15
        self.market_price = 0.0
        
        # Production function parameters
        self.sigma_ek = 0.5  # Energy-capital substitution elasticity
        self.sigma_l = 0.8   # Labor substitution elasticity
        self.alpha_e = 0.3   # Energy share
        self.alpha_k = 0.7   # Capital share
        self.alpha_l = 0.6   # Labor share
        self.alpha_ek = 0.4  # Energy-capital bundle share
        self.tech_progress = 0.01  # Annual technical progress
        
        # Initialize model
        self._validate_input_data()
        self._initialize_columns()
        
        # Store baseline allocations for reference
        self.baseline_allocations = (
            self.facilities_data['Baseline Output'] * 
            self.facilities_data['Baseline Benchmark']
        )

    def _validate_input_data(self) -> None:
        """Validate input data structure and relationships."""
        required_facility_cols = {
            'Facility ID', 'Baseline Output', 'Baseline Emissions',
            'Baseline Benchmark', 'Baseline Profit Rate', 'Output Growth Rate',
            'Emissions Growth Rate', 'Benchmark Ratchet Rate',
            'Baseline Capital', 'Baseline Energy', 'Baseline Labor',
            'Capital Growth Rate', 'Energy Growth Rate', 'Labor Growth Rate',
            'Energy Efficiency Improvement Rate'
        }
        
        required_abatement_cols = {
            'Facility ID', 'Slope', 'Intercept', 'Max Reduction (MTCO2e)'
        }
        
        missing_facility_cols = required_facility_cols - set(self.facilities_data.columns)
        missing_abatement_cols = required_abatement_cols - set(self.abatement_cost_curve.columns)
        
        if missing_facility_cols or missing_abatement_cols:
            raise ValueError(f"Missing required columns: Facilities: {missing_facility_cols}, Abatement: {missing_abatement_cols}")
        
        # Validate production parameters
        min_values = {
            'Baseline Capital': 0,
            'Baseline Energy': 0,
            'Baseline Labor': 0,
            'Capital Growth Rate': -0.1,
            'Energy Growth Rate': -0.1,
            'Labor Growth Rate': -0.1,
            'Energy Efficiency Improvement Rate': 0
        }
        
        for col, min_val in min_values.items():
            if (self.facilities_data[col] < min_val).any():
                raise ValueError(f"{col} contains invalid values below {min_val}")

    def _initialize_columns(self) -> None:
        """Initialize all required columns without fragmentation."""
        metrics = [
            "Output", "Emissions", "Benchmark", "Allocations",
            "Allowance Surplus/Deficit", "Abatement Cost", "Trade Cost",
            "Total Cost", "Trade Volume", "Allowance Price",
            "Tonnes Abated", "Allowance Purchase Cost", "Allowance Sales Revenue",
            "Compliance Cost", "Cost to Profit Ratio", "Cost to Output Ratio",
            "Energy Capital Bundle", "Energy Use", "Capital Use", "Labor Use"
        ]
        
        year_cols = [f"{metric}_{year}" 
                    for year in range(self.start_year, self.start_year + 20)
                    for metric in metrics]
        
        new_cols = pd.DataFrame(0.0, 
                              index=self.facilities_data.index,
                              columns=year_cols)
        
        self.facilities_data = pd.concat([self.facilities_data, new_cols], axis=1)
        
        if 'Profit' not in self.facilities_data.columns:
            self.facilities_data['Profit'] = (
                self.facilities_data['Baseline Output'] * 
                self.facilities_data['Baseline Profit Rate']
            )

    def _calculate_energy_capital_bundle(self, facility: pd.Series, year: int, 
                                       years_elapsed: int) -> float:
        """Calculate energy-capital bundle using CES function."""
        capital = facility['Baseline Capital'] * (1 + facility['Capital Growth Rate']) ** years_elapsed
        energy = facility['Baseline Energy'] * (1 + facility['Energy Growth Rate']) ** years_elapsed
        
        tech_factor = (1 + self.tech_progress) ** years_elapsed
        
        bundle = tech_factor * (
            self.alpha_k * capital ** ((self.sigma_ek - 1)/self.sigma_ek) +
            self.alpha_e * energy ** ((self.sigma_ek - 1)/self.sigma_ek)
        ) ** (self.sigma_ek/(self.sigma_ek - 1))
        
        return bundle, energy, capital

    def _calculate_output(self, facility: pd.Series, energy_capital_bundle: float,
                         year: int, years_elapsed: int) -> float:
        """Calculate final output using CES production function."""
        labor = facility['Baseline Labor'] * (1 + facility['Labor Growth Rate']) ** years_elapsed
        
        output = (
            self.alpha_l * labor ** ((self.sigma_l - 1)/self.sigma_l) +
            self.alpha_ek * energy_capital_bundle ** ((self.sigma_l - 1)/self.sigma_l)
        ) ** (self.sigma_l/(self.sigma_l - 1))
        
        return output, labor

    def calculate_dynamic_values(self, year: int) -> None:
        """Calculate dynamic values with nested CES production structure."""
        years_elapsed = year - self.start_year
        
        print(f"\n=== Production and Emissions Analysis for Year {year} ===")
        
        for idx, facility in self.facilities_data.iterrows():
            # Calculate production components
            bundle, energy, capital = self._calculate_energy_capital_bundle(
                facility, year, years_elapsed
            )
            
            output, labor = self._calculate_output(
                facility, bundle, year, years_elapsed
            )
            
            # Store production values
            self.facilities_data.at[idx, f'Energy Capital Bundle_{year}'] = bundle
            self.facilities_data.at[idx, f'Energy Use_{year}'] = energy
            self.facilities_data.at[idx, f'Capital Use_{year}'] = capital
            self.facilities_data.at[idx, f'Labor Use_{year}'] = labor
            self.facilities_data.at[idx, f'Output_{year}'] = output
            
            # Calculate emissions with efficiency improvements
            base_intensity = facility['Baseline Emissions'] / facility['Baseline Output']
            efficiency = 1 - facility['Energy Efficiency Improvement Rate'] * years_elapsed
            
            # Consider previous abatement
            if year > self.start_year and f'Tonnes Abated_{year-1}' in self.facilities_data.columns:
                prev_abatement = facility[f'Tonnes Abated_{year-1}']
                prev_emissions = facility[f'Emissions_{year-1}']
                abatement_effect = max(0, 1 - prev_abatement / prev_emissions) if prev_emissions > 0 else 1
            else:
                abatement_effect = 1.0
            
            emissions = output * base_intensity * efficiency * abatement_effect
            self.facilities_data.at[idx, f'Emissions_{year}'] = emissions
        
        # Calculate benchmarks and allocations
        self.facilities_data[f'Benchmark_{year}'] = (
            self.facilities_data['Baseline Benchmark'] * 
            (1 + self.facilities_data['Benchmark Ratchet Rate']) ** years_elapsed
        )
        
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * 
            self.facilities_data[f'Benchmark_{year}']
        )
        
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - 
            self.facilities_data[f'Emissions_{year}']
        )
```python
    def calculate_dynamic_allowance_surplus_deficit(self, year: int) -> Tuple[float, float]:
        """Calculate supply and demand based on production outcomes."""
        print(f"\n=== Market Balance Analysis for Year {year} ===")
        
        # Update production and emissions first
        self.calculate_dynamic_values(year)
        
        # Calculate market positions
        total_supply = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
        total_demand = abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
        
        print("\nMarket Balance:")
        print(f"Total Supply: {total_supply:,.2f}")
        print(f"Total Demand: {total_demand:,.2f}")
        print(f"Net Position: {total_supply - total_demand:,.2f}")
        
        return total_supply, total_demand

    def determine_market_price(self, supply: float, demand: float, year: int) -> float:
        """Determine market price with production considerations."""
        print(f"\n=== Price Determination for Year {year} ===")
        
        # Build MAC curve considering production structure
        mac_curve = self._build_mac_curve(year)
        
        if supply >= demand:
            self.market_price = self.floor_price
            print(f"Market oversupplied, price set to floor: ${self.floor_price:,.2f}")
            return self.market_price
        
        remaining_demand = demand - supply
        
        # Find price that balances remaining demand with abatement supply
        try:
            def excess_demand(price):
                abatement_supply = self._calculate_abatement_supply(price, year)
                return remaining_demand - abatement_supply
            
            # Use numerical optimization to find market clearing price
            result = optimize.root_scalar(
                excess_demand,
                bracket=[self.floor_price, self.ceiling_price],
                method='brentq'
            )
            
            self.market_price = result.root
            
        except ValueError:
            # If no solution found, use MAC curve based price
            if mac_curve:
                price_index = min(int(remaining_demand * 10), len(mac_curve) - 1)
                self.market_price = min(max(mac_curve[price_index], self.floor_price), self.ceiling_price)
            else:
                self.market_price = self.floor_price
        
        print(f"Market price determined: ${self.market_price:,.2f}")
        return self.market_price

    def _calculate_abatement_supply(self, price: float, year: int) -> float:
        """Calculate total abatement supply at given price."""
        total_abatement = 0
        
        for _, facility in self.facilities_data.iterrows():
            if facility[f'Allowance Surplus/Deficit_{year}'] >= 0:
                continue
            
            curve = self.abatement_cost_curve[
                self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
            ].iloc[0]
            
            deficit = abs(facility[f'Allowance Surplus/Deficit_{year}'])
            max_reduction = min(curve['Max Reduction (MTCO2e)'], deficit)
            
            # Calculate economic abatement at this price
            if curve['Slope'] > 0:
                economic_abatement = max(0, min(
                    max_reduction,
                    (price - curve['Intercept']) / curve['Slope']
                ))
                total_abatement += economic_abatement
        
        return total_abatement

    def calculate_abatement(self, year: int) -> None:
        """Calculate optimal abatement with production considerations."""
        print(f"\n=== Abatement Analysis for Year {year} ===")
        
        total_abatement = 0
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
            intercept = max(0, float(curve['Intercept']))
            
            # Calculate optimal abatement considering production impacts
            optimal_abatement = 0
            if slope > 0:
                optimal_abatement = min(
                    max_reduction,
                    deficit,
                    max(0, (self.market_price - intercept) / slope)
                )
            
            if optimal_abatement > 0:
                # Calculate total cost using quadratic cost function
                total_cost = (slope * optimal_abatement**2 / 2) + (intercept * optimal_abatement)
                
                # Apply abatement
                self._apply_abatement(idx, optimal_abatement, total_cost, year)
                total_abatement += optimal_abatement
                
                # Record for summary
                abatement_summary.append({
                    'Facility ID': facility['Facility ID'],
                    'Deficit': deficit,
                    'Abatement': optimal_abatement,
                    'Cost': total_cost,
                    'Final MAC': slope * optimal_abatement + intercept
                })
        
        if abatement_summary:
            summary_df = pd.DataFrame(abatement_summary)
            print("\nAbatement Summary:")
            print(summary_df.to_string())
            print(f"\nTotal Abatement: {total_abatement:,.2f}")
            print(f"Average Cost per Tonne: ${summary_df['Cost'].sum() / total_abatement:,.2f}")

    def trade_allowances(self, year: int) -> None:
        """Execute allowance trades considering production costs."""
        print(f"\n=== Trading Analysis for Year {year} ===")
        
        # Get buyers and sellers after abatement
        buyers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0]
        sellers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0]
        
        print(f"Potential Trades - Buyers: {len(buyers)}, Sellers: {len(sellers)}")
        
        if buyers.empty or sellers.empty:
            print("No valid trading pairs found")
            return
        
        trades_executed = []
        for buyer_idx, buyer in buyers.iterrows():
            remaining_deficit = abs(buyer[f'Allowance Surplus/Deficit_{year}'])
            
            for seller_idx, seller in sellers.iterrows():
                available_surplus = seller[f'Allowance Surplus/Deficit_{year}']
                
                # Calculate optimal trade volume
                trade_volume = min(remaining_deficit, available_surplus)
                trade_cost = trade_volume * self.market_price
                
                if trade_volume > 0:
                    # Execute trade
                    self._update_trade_positions(buyer_idx, seller_idx, trade_volume, trade_cost, year)
                    
                    trades_executed.append({
                        'Buyer': buyer['Facility ID'],
                        'Seller': seller['Facility ID'],
                        'Volume': trade_volume,
                        'Price': self.market_price,
                        'Total Cost': trade_cost
                    })
                    
                    remaining_deficit -= trade_volume
                    if remaining_deficit <= 0:
                        break
        
        if trades_executed:
            trades_df = pd.DataFrame(trades_executed)
            print("\nTrades Executed:")
            print(trades_df.to_string())
            print(f"\nTotal Trade Volume: {trades_df['Volume'].sum():,.2f}")
            print(f"Total Trade Value: ${trades_df['Total Cost'].sum():,.2f}")

```python
    def calculate_costs(self, year: int) -> None:
        """Calculate comprehensive cost metrics including production costs."""
        print(f"\n=== Cost Analysis for Year {year} ===")
        
        # Production-related costs
        self.facilities_data[f'Production Cost_{year}'] = (
            self.facilities_data[f'Energy Use_{year}'] * self.energy_price +
            self.facilities_data[f'Capital Use_{year}'] * self.capital_price +
            self.facilities_data[f'Labor Use_{year}'] * self.labor_price
        )
        
        # Compliance costs
        self.facilities_data[f'Compliance Cost_{year}'] = (
            self.facilities_data[f'Abatement Cost_{year}'].clip(lower=0) +
            self.facilities_data[f'Allowance Purchase Cost_{year}'].clip(lower=0)
        )
        
        # Total costs including trading revenues
        self.facilities_data[f'Total Cost_{year}'] = (
            self.facilities_data[f'Production Cost_{year}'] +
            self.facilities_data[f'Compliance Cost_{year}'] -
            self.facilities_data[f'Allowance Sales Revenue_{year}']
        )
        
        # Calculate economic metrics
        self.facilities_data[f'Value Added_{year}'] = (
            self.facilities_data[f'Output_{year}'] * self.output_price -
            self.facilities_data[f'Production Cost_{year}']
        )
        
        print("\nCost Summary:")
        print("Average Production Cost: "
              f"${self.facilities_data[f'Production Cost_{year}'].mean():,.2f}")
        print("Average Compliance Cost: "
              f"${self.facilities_data[f'Compliance Cost_{year}'].mean():,.2f}")
        print("Average Total Cost: "
              f"${self.facilities_data[f'Total Cost_{year}'].mean():,.2f}")

    def calculate_performance_metrics(self, year: int) -> None:
        """Calculate economic and environmental performance metrics."""
        # Cost intensity metrics
        self.facilities_data[f'Cost to Output Ratio_{year}'] = (
            self.facilities_data[f'Total Cost_{year}'] / 
            self.facilities_data[f'Output_{year}']
        ).replace([float('inf'), -float('inf')], 0).fillna(0)
        
        self.facilities_data[f'Cost to Value Added Ratio_{year}'] = (
            self.facilities_data[f'Total Cost_{year}'] / 
            self.facilities_data[f'Value Added_{year}']
        ).replace([float('inf'), -float('inf')], 0).fillna(0)
        
        # Environmental performance metrics
        self.facilities_data[f'Emissions Intensity_{year}'] = (
            self.facilities_data[f'Emissions_{year}'] /
            self.facilities_data[f'Output_{year}']
        )
        
        self.facilities_data[f'Abatement Rate_{year}'] = (
            self.facilities_data[f'Tonnes Abated_{year}'] /
            (self.facilities_data[f'Emissions_{year}'] + 
             self.facilities_data[f'Tonnes Abated_{year}'])
        )

    def run_model(self, start_year: int, end_year: int, 
                 output_file: str = "model_results.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run complete model simulation with enhanced production structure."""
        print("\nStarting OBA Model Simulation...")
        market_summary = []
        
        for year in range(start_year, end_year + 1):
            print(f"\nProcessing Year {year}")
            
            # Core market operations
            total_supply, total_demand = self.calculate_dynamic_allowance_surplus_deficit(year)
            self.determine_market_price(total_supply, total_demand, year)
            self.calculate_abatement(year)
            self.trade_allowances(year)
            
            # Economic calculations
            self.calculate_costs(year)
            self.calculate_performance_metrics(year)
            
            # Collect market summary
            market_summary.append(self._create_market_summary(year))
        
        # Prepare and save results
        market_summary_df = pd.DataFrame(market_summary)
        facility_results = self._prepare_facility_results(start_year, end_year)
        
        self.save_results(market_summary_df, facility_results, output_file)
        print("\nSimulation Complete")
        
        return market_summary_df, facility_results

    def _create_market_summary(self, year: int) -> Dict:
        """Create comprehensive market summary including production metrics."""
        summary = {
            'Year': year,
            'Market Price': self.market_price,
            'Total Output': self.facilities_data[f'Output_{year}'].sum(),
            'Total Emissions': self.facilities_data[f'Emissions_{year}'].sum(),
            'Total Abatement': self.facilities_data[f'Tonnes Abated_{year}'].sum(),
            'Average Emissions Intensity': self.facilities_data[f'Emissions Intensity_{year}'].mean(),
            'Total Trade Volume': self.facilities_data[f'Trade Volume_{year}'].abs().sum() / 2,
            'Total Production Cost': self.facilities_data[f'Production Cost_{year}'].sum(),
            'Total Compliance Cost': self.facilities_data[f'Compliance Cost_{year}'].sum(),
            'Total Value Added': self.facilities_data[f'Value Added_{year}'].sum(),
            'Average Cost to Output Ratio': self.facilities_data[f'Cost to Output Ratio_{year}'].mean(),
            'Average Cost to Value Added Ratio': self.facilities_data[f'Cost to Value Added Ratio_{year}'].mean(),
            'Average Abatement Rate': self.facilities_data[f'Abatement Rate_{year}'].mean(),
        }
        
        # Add market balance metrics
        summary.update({
            'Remaining Surplus': self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum(),
            'Remaining Deficit': abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
        })
        
        return summary

    def _prepare_facility_results(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Prepare detailed facility-level results."""
        metrics = [
            "Output", "Emissions", "Emissions Intensity", "Energy Use", 
            "Capital Use", "Labor Use", "Production Cost", "Value Added",
            "Allowance Surplus/Deficit", "Tonnes Abated", "Abatement Rate",
            "Abatement Cost", "Trade Volume", "Trade Cost",
            "Compliance Cost", "Total Cost", "Cost to Output Ratio",
            "Cost to Value Added Ratio"
        ]
        
        results = []
        for year in range(start_year, end_year + 1):
            year_data = self.facilities_data[['Facility ID'] + 
                [f'{metric}_{year}' for metric in metrics]].copy()
            
            year_data.columns = ['Facility ID'] + metrics
            year_data['Year'] = year
            results.append(year_data)
        
        return pd.concat(results, ignore_index=True)

    def save_results(self, market_summary: pd.DataFrame, 
                    facility_results: pd.DataFrame, 
                    output_file: str) -> None:
        """Save comprehensive model results."""
        # Save market summary
        market_summary.to_csv("market_summary.csv", index=False)
        print("Market summary saved to market_summary.csv")
        
        # Save detailed facility results
        facility_results.to_csv(output_file, index=False)
        print(f"Facility results saved to {output_file}")
        
        # Generate and save additional analysis
        self._save_additional_analysis(market_summary, facility_results)

    def _save_additional_analysis(self, market_summary: pd.DataFrame, 
                                facility_results: pd.DataFrame) -> None:
        """Generate and save additional analysis of results."""
        # Market trends analysis
        market_trends = market_summary.set_index('Year').rolling(window=3).mean()
        market_trends.to_csv("market_trends.csv")
        
        # Facility performance analysis
        facility_performance = facility_results.groupby('Facility ID').agg({
            'Output': 'mean',
            'Emissions': 'mean',
            'Emissions Intensity': 'mean',
            'Abatement Rate': 'mean',
            'Cost to Output Ratio': 'mean',
            'Cost to Value Added Ratio': 'mean'
        })
        facility_performance.to_csv("facility_performance.csv")
