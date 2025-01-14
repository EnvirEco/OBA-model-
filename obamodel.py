import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os

class obamodel:
    # 1. Initialization and Setup
    @staticmethod
    def load_all_scenarios(scenario_file: str) -> List[Dict]:
        """Load and validate scenarios from CSV file."""
        print(f"Loading scenarios from file: {scenario_file}")
        try:
            scenarios = pd.read_csv(scenario_file)
            
            # Clean column names
            scenarios.columns = scenarios.columns.str.strip()
            
            # Define required parameters and their bounds
            param_bounds = {
                'Floor Price': (0, None),
                'Ceiling Price': (0, None),
                'Price Increment': (0, None),
                'Output Growth Rate': (-0.5, 0.5),
                'Emissions Growth Rate': (-0.5, 0.5),
                'Benchmark Ratchet Rate': (0, 1),
                'MSR Active': (0, 1),
                'MSR Upper Threshold': (0, 1),
                'MSR Lower Threshold': (-1, 0),
                'MSR Adjustment Rate': (0, 1)
            }
            
            # Validate parameters
            for param, (min_val, max_val) in param_bounds.items():
                if param not in scenarios.columns:
                    raise ValueError(f"Missing required parameter: {param}")
                if min_val is not None and (scenarios[param] < min_val).any():
                    raise ValueError(f"{param} contains values below {min_val}")
                if max_val is not None and (scenarios[param] > max_val).any():
                    raise ValueError(f"{param} contains values above {max_val}")
            
            # Convert to list of dictionaries with standardized parameter names
            scenario_list = []
            for _, row in scenarios.iterrows():
                scenario_list.append({
                    "name": row["Scenario"],
                    "floor_price": row["Floor Price"],
                    "ceiling_price": row["Ceiling Price"],
                    "price_increment": row["Price Increment"],
                    "output_growth_rate": row["Output Growth Rate"],
                    "emissions_growth_rate": row["Emissions Growth Rate"],
                    "benchmark_ratchet_rate": row["Benchmark Ratchet Rate"],
                    "msr_active": bool(row["MSR Active"]),
                    "msr_upper_threshold": row["MSR Upper Threshold"],
                    "msr_lower_threshold": row["MSR Lower Threshold"],
                    "msr_adjustment_rate": row["MSR Adjustment Rate"]
                })
            
            print(f"Successfully loaded {len(scenario_list)} scenarios")
            return scenario_list
            
        except Exception as e:
            print(f"Error loading scenarios: {e}")
            raise
  
    def __init__(self, facilities_data: pd.DataFrame, abatement_cost_curve: pd.DataFrame, 
                 start_year: int, end_year: int, scenario_params: Dict):
        """Initialize OBA model with facility-specific benchmarks."""
        # Validate required columns
        required_cols = [
            'Facility ID', 'Sector', 'Baseline Output', 'Baseline Emissions',
            'Baseline Benchmark', 'Benchmark Ratchet Rate'
        ]
        missing_cols = [col for col in required_cols if col not in facilities_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Store base data
        self.facilities_data = facilities_data.copy()
        self.abatement_cost_curve = abatement_cost_curve.copy()
        self.start_year = start_year
        self.end_year = end_year
        
        # Extract scenario parameters
        self.floor_price = scenario_params.get("floor_price", 20)
        self.ceiling_price = scenario_params.get("ceiling_price", 200)
        self.price_increment = scenario_params.get("price_increment", 5)
        
        # Initialize price schedule
        self.price_schedule = {
            year: min(self.floor_price + self.price_increment * (year - start_year), 
                     self.ceiling_price)
            for year in range(start_year, end_year + 1)
        }
        
        # Growth rates
        self.output_growth_rate = scenario_params.get("output_growth_rate", 0.02)
        self.emissions_growth_rate = scenario_params.get("emissions_growth_rate", 0.01)
        
        # MSR parameters
        self.msr_active = scenario_params.get("msr_active", False)
        self.msr_upper_threshold = scenario_params.get("msr_upper_threshold", 0.15)
        self.msr_lower_threshold = scenario_params.get("msr_lower_threshold", -0.05)
        self.msr_adjustment_rate = scenario_params.get("msr_adjustment_rate", 0.03)
        
        # Initialize model columns
        self._initialize_columns()
        
        print("\nModel Initialized:")
        print(f"Time period: {start_year} - {end_year}")
        print(f"Price range: ${self.floor_price} - ${self.ceiling_price}")
        print(f"Number of facilities: {len(facilities_data)}")
        print("\nSector Summary:")
        sector_summary = facilities_data.groupby('Sector').agg({
            'Baseline Benchmark': 'mean',
            'Benchmark Ratchet Rate': 'mean'
        })
        print(sector_summary.to_string())
    
  
    def analyze_benchmark_coverage(self, sector: str, benchmark: float) -> None:
        """Analyze how many facilities meet the benchmark."""
        sector_data = self.facilities_data[self.facilities_data['Sector'] == sector]
        intensities = sector_data['Baseline Emissions'] / sector_data['Baseline Output']
        
        meets_benchmark = (intensities <= benchmark).sum()
        total_facilities = len(sector_data)
        
        print(f"\nBenchmark Coverage Analysis for {sector}:")
        print(f"  Benchmark: {benchmark:.6f}")
        print(f"  Facilities Meeting Benchmark: {meets_benchmark} of {total_facilities}")
        print(f"  Coverage Percentage: {(meets_benchmark/total_facilities)*100:.1f}%")

    def _initialize_model_parameters(self, scenario_params: Dict) -> None:
        """Initialize model parameters including sector-specific settings."""
        # Base parameters
        self.output_growth_rate = scenario_params.get("output_growth_rate", 0.02)
        self.emissions_growth_rate = scenario_params.get("emissions_growth_rate", 0.01)
             
        # MSR parameters
        self.msr_active = scenario_params.get("msr_active", False)
        self.msr_upper_threshold = scenario_params.get("msr_upper_threshold", 0.15)
        self.msr_lower_threshold = scenario_params.get("msr_lower_threshold", -0.05)
        self.msr_adjustment_rate = scenario_params.get("msr_adjustment_rate", 0.03)
                     
    def _validate_input_data(self) -> None:
        """Validate input data structure and relationships."""
        required_facility_cols = {
            'Facility ID', 'Baseline Output', 'Baseline Emissions',
            'Baseline Benchmark', 'Baseline Profit Rate', 'Output Growth Rate',
            'Emissions Growth Rate'
        }
        required_abatement_cols = {
            'Facility ID', 'Slope', 'Intercept', 'Max Reduction (MTCO2e)'
        }
        
        # Check for missing columns
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
        """Initialize all required columns with explicit creation and verification."""
        # Core metrics that need to be tracked annually
        metrics = [
            "Output", "Emissions", "Benchmark", "Allocations",
            "Allowance Surplus/Deficit", "Abatement Cost", "Trade Cost",
            "Total Cost", "Trade Volume", "Allowance Price",
            "Tonnes Abated", "Allowance Purchase Cost", "Allowance Sales Revenue",
            "Compliance Cost", "Cost to Profit Ratio", "Cost to Output Ratio"
        ]
        
        # Add MSR-specific metrics if MSR is active
        if self.msr_active:
            metrics.extend([
                "MSR_Adjustment",
                "MSR_Active"
            ])
        
        # Create and verify year-specific columns
        year_cols = []
        for year in range(self.start_year, self.end_year + 1):
            for metric in metrics:
                col_name = f"{metric}_{year}"
                year_cols.append(col_name)
                
        # Create new columns with explicit zeros
        new_cols = pd.DataFrame(
            data=0.0,
            index=self.facilities_data.index,
            columns=year_cols
        )
        
        # Verify all required columns exist before concatenating
        missing_cols = set(year_cols) - set(new_cols.columns)
        if missing_cols:
            raise ValueError(f"Failed to create columns: {missing_cols}")
        
        # Concat with existing data
        self.facilities_data = pd.concat([self.facilities_data, new_cols], axis=1)
        
        # Verify critical columns after concatenation
        for year in range(self.start_year, self.end_year + 1):
            critical_cols = [
                f"Output_{year}",
                f"Emissions_{year}",
                f"Tonnes Abated_{year}",
                f"Allocations_{year}"
            ]
            missing = [col for col in critical_cols if col not in self.facilities_data.columns]
            if missing:
                raise ValueError(f"Critical columns missing after initialization: {missing}")
        
        # Calculate Baseline Allocations if needed
        if 'Baseline Allocations' not in self.facilities_data.columns:
            self.facilities_data['Baseline Allocations'] = (
                self.facilities_data['Baseline Output'] *
                self.facilities_data['Baseline Benchmark']
            )
    
        # Calculate initial profit if not provided
        if 'Profit' not in self.facilities_data.columns:
            self.facilities_data['Profit'] = (
                self.facilities_data['Baseline Output'] * 
                self.facilities_data['Baseline Profit Rate']
            )
        
        # Print verification of critical columns
        print("\nInitialized columns verification:")
        print(f"Total columns created: {len(year_cols)}")
        print(f"First year columns present: {all(f'{m}_{self.start_year}' in self.facilities_data.columns for m in metrics)}")
        print(f"Last year columns present: {all(f'{m}_{self.end_year}' in self.facilities_data.columns for m in metrics)}")
        
    
  
    # 2. Core Market Mechanisms
    def run_model(self, output_file: str = "results.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the complete model simulation with proper market sequence."""
        print("Running emissions trading model...")
        market_summary = []
        
        for year in range(self.start_year, self.end_year + 1):
            print(f"\nProcessing year {year}...")
            
            # 1. Calculate initial positions and benchmarks
            self.calculate_dynamic_values(year)
            
            # 2. Initial market analysis
            total_supply, total_demand = self.calculate_market_positions(year)
            
            # 3. First round price determination
            initial_price = self.determine_market_price(total_supply, total_demand, year)
            
            # 4. Calculate abatement based on price signal
            self.calculate_abatement(year)
            
            # 5. Recalculate market positions after abatement
            post_abatement_supply, post_abatement_demand = self.calculate_market_positions(year)
            
            # 6. Update price based on post-abatement positions
            final_price = self.determine_market_price(post_abatement_supply, post_abatement_demand, year)
            
            # 7. Execute trades at final price
            self.trade_allowances(year)
            
            # 8. Calculate final costs and metrics
            self.calculate_costs(year)
            self.calculate_cost_ratios(year)
            
            # 9. Analyze market stability
            self.analyze_market_stability(year)
            
            # 10. Collect market summary
            market_summary.append(self._create_market_summary(year))
        
        # Save results
        market_summary_df = pd.DataFrame(market_summary)
        facility_results = self._prepare_facility_results(self.start_year, self.end_year)
        self.save_results(market_summary_df, facility_results, output_file)
        
        return market_summary_df, facility_results
        
    def calculate_sector_benchmarks(self, year: int) -> pd.Series:
        """
        Calculate sector-specific benchmarks for a given year.
        
        Args:
            year: The year to calculate benchmarks for
            
        Returns:
            Series with benchmark values by sector
        """
        print(f"\nCalculating sector benchmarks for year {year}")
        
        # Get benchmark method and parameters from scenario
        method = self.sector_params.get('benchmark_method', 'percentile')
        
        sector_benchmarks = {}
        
        # Calculate benchmark for each sector
        for sector in self.facilities_data['Sector'].unique():
            # Get sector data
            sector_data = self.facilities_data[self.facilities_data['Sector'] == sector]
            
            # Get sector-specific parameters
            sector_config = self.sector_params.get(sector, {})
            
            print(f"\nProcessing sector: {sector}")
            print(f"Method: {method}")
            print(f"Configuration: {sector_config}")
            
            # Calculate initial benchmark
            initial_benchmark = self.get_sector_benchmark(sector_data, method, sector_config)
            
            # Apply ratchet rate if not first year
            if year > self.start_year:
                years_elapsed = year - self.start_year
                ratchet_rate = sector_config.get('ratchet_rate', self.benchmark_ratchet_rate)
                final_benchmark = initial_benchmark * (1 - ratchet_rate) ** years_elapsed
                print(f"Applied ratchet rate {ratchet_rate} for {years_elapsed} years")
                print(f"Final benchmark: {final_benchmark:.6f}")
            else:
                final_benchmark = initial_benchmark
                
            sector_benchmarks[sector] = final_benchmark
        
        # Convert to pandas Series for easy mapping
        result = pd.Series(sector_benchmarks)
        
        print("\nFinal sector benchmarks:")
        for sector, benchmark in result.items():
            print(f"{sector}: {benchmark:.6f}")
            
        return result
        
    def calculate_dynamic_values(self, year: int) -> None:
        """Calculate dynamic values using facility-specific baseline benchmarks."""
        years_elapsed = year - self.start_year
        print(f"\n=== Dynamic Value Analysis for Year {year} ===")
        print(f"MSR Status: {'Active' if self.msr_active else 'Inactive'}")
        
        # Calculate emissions intensity
        if year > self.start_year:
            prior_emissions = self.facilities_data[f'Emissions_{year - 1}']
            prior_abatement = self.facilities_data.get(f'Tonnes Abated_{year - 1}', 0)
            prior_output = self.facilities_data[f'Output_{year - 1}']
            emissions_intensity = ((prior_emissions - prior_abatement) / prior_output).clip(lower=0)
        else:
            emissions_intensity = (
                self.facilities_data['Baseline Emissions'] /
                self.facilities_data['Baseline Output']
            )
        
        # Calculate output and emissions
        self.facilities_data[f'Output_{year}'] = (
            self.facilities_data['Baseline Output'] *
            (1 + self.output_growth_rate) ** years_elapsed
        )
        
        self.facilities_data[f'Emissions_{year}'] = (
            self.facilities_data[f'Output_{year}'] * 
            emissions_intensity
        ).clip(lower=0)
        
        # Calculate benchmarks with ratchet rate
        self.facilities_data[f'Benchmark_{year}'] = (
            self.facilities_data['Baseline Benchmark'] * 
            (1 - self.facilities_data['Benchmark Ratchet Rate']) ** years_elapsed
        )
        
        # Calculate allocations
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * 
            self.facilities_data[f'Benchmark_{year}']
        )
        
        # Calculate market position
        total_emissions = self.facilities_data[f'Emissions_{year}'].sum()
        total_allocations = self.facilities_data[f'Allocations_{year}'].sum()
        current_surplus_ratio = (total_allocations - total_emissions) / total_allocations if total_allocations > 0 else 0
        
        # Apply MSR adjustments if active
        adjustment_factor = 1.0
        if self.msr_active:
            if current_surplus_ratio > self.msr_upper_threshold:
                # Too much surplus - tighten benchmarks
                adjustment_factor = (1 + self.msr_upper_threshold) / (1 + current_surplus_ratio)
                print(f"MSR: Surplus ({current_surplus_ratio:.4f}) above threshold - tightening")
            elif current_surplus_ratio < self.msr_lower_threshold:
                # Too much deficit - loosen benchmarks
                adjustment_factor = (1 + self.msr_lower_threshold) / (1 + current_surplus_ratio)
                print(f"MSR: Surplus ({current_surplus_ratio:.4f}) below threshold - loosening")
            
            if adjustment_factor != 1.0:
                self.facilities_data[f'Benchmark_{year}'] *= adjustment_factor
                self.facilities_data[f'Allocations_{year}'] *= adjustment_factor
                
                # Recalculate market position
                total_allocations = self.facilities_data[f'Allocations_{year}'].sum()
                current_surplus_ratio = (total_allocations - total_emissions) / total_allocations
        
        # Calculate and store surplus/deficit
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - 
            self.facilities_data[f'Emissions_{year}']
        )
        
        # Store MSR metrics if active
        if self.msr_active:
            self.facilities_data[f'MSR_Adjustment_{year}'] = adjustment_factor
        
        # Report results by sector
        print("\nSector Performance:")
        sector_results = self.facilities_data.groupby('Sector').agg({
            f'Benchmark_{year}': 'mean',
            f'Allocations_{year}': 'sum',
            f'Emissions_{year}': 'sum',
            f'Allowance Surplus/Deficit_{year}': 'sum'
        })
        
        for sector in sector_results.index:
            print(f"\n{sector}:")
            print(f"  Average Benchmark: {sector_results.loc[sector, f'Benchmark_{year}']:.6f}")
            print(f"  Total Allocations: {sector_results.loc[sector, f'Allocations_{year}']:,.2f}")
            print(f"  Total Emissions: {sector_results.loc[sector, f'Emissions_{year}']:,.2f}")
            print(f"  Net Position: {sector_results.loc[sector, f'Allowance Surplus/Deficit_{year}']:,.2f}")
        
        # Overall market status
        print(f"\nOverall Market Status Year {year}:")
        print(f"Total Emissions: {total_emissions:,.2f}")
        print(f"Total Allocations: {total_allocations:,.2f}")
        print(f"Current Surplus Ratio: {current_surplus_ratio:.4f}")
        if self.msr_active:
            print(f"MSR Adjustment Factor: {adjustment_factor:.4f}")
            
    def calculate_market_positions(self, year: int) -> Tuple[float, float]:
        """Calculate current market positions."""
        # Calculate total supply (positive positions)
        total_supply = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
        
        # Calculate total demand (negative positions)
        total_demand = abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
        
        print(f"\nMarket Balance for Year {year}:")
        print(f"Total Supply: {total_supply:,.2f}")
        print(f"Total Demand: {total_demand:,.2f}")
        
        return total_supply, total_demand

    def determine_market_price(self, supply: float, demand: float, year: int) -> float:
        """Determine market price based on supply, demand, and MAC curves."""
        # Get scheduled price as starting point
        scheduled_price = self.price_schedule.get(year, self.floor_price)
        
        # If market is balanced, use scheduled price
        if abs(supply - demand) < 0.001:
            self.market_price = scheduled_price
            return self.market_price
        
        # Build MAC curve for price determination
        mac_curve = self._build_mac_curve(year)
        
        # Calculate market clearing price
        if demand > supply:
            # Find price that would incentivize enough abatement
            needed_abatement = demand - supply
            price_index = min(int(needed_abatement * 10), len(mac_curve) - 1)
            clearing_price = mac_curve[price_index] if mac_curve else scheduled_price
            
            # Ensure price is within bounds
            self.market_price = min(
                max(clearing_price, scheduled_price),
                self.ceiling_price
            )
        else:
            # Market has excess supply, use floor price
            self.market_price = max(self.floor_price, scheduled_price)
        
        print(f"\nMarket Price Determination for Year {year}:")
        print(f"Supply: {supply:.2f}, Demand: {demand:.2f}")
        print(f"Determined Price: ${self.market_price:.2f}")
        return self.market_price
        
    def _build_mac_curve(self, year: int) -> List[float]:
        """Build marginal abatement cost curve."""
        mac_points = []
        
        # Get facilities with deficits
        deficits = self.facilities_data[
            self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0
        ]
        
        for _, facility in deficits.iterrows():
            # Get facility's abatement curve
            curve = self.abatement_cost_curve[
                self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
            ].iloc[0]
            
            # Calculate maximum abatement needed
            deficit = abs(facility[f'Allowance Surplus/Deficit_{year}'])
            max_reduction = min(curve['Max Reduction (MTCO2e)'], deficit)
            
            # Get curve parameters
            slope = float(curve['Slope'])
            intercept = float(curve['Intercept'])
    
            # Ensure intercept is reasonable
            if intercept < 0:
                print(f"Warning: Negative intercept ({intercept}) for Facility ID {facility['Facility ID']}. Adjusting to 0.")
                intercept = 0
            
            # Build curve points
            steps = 100  # Number of points to generate
            for i in range(steps):
                qty = max_reduction * (i + 1) / steps
                mac = slope * qty + intercept
                if mac > 0 and mac <= self.ceiling_price:
                    mac_points.append(mac)
        
        # Handle empty MAC curve case
        if not mac_points:
            print("Warning: Empty MAC curve. Using floor price.")
            mac_points = [self.floor_price]
        else:
            # Sort points for proper price determination
            mac_points.sort()
            
        print(f"MAC Curve: Min=${min(mac_points):.2f}, Max=${max(mac_points):.2f}, Points={len(mac_points)}")
        
        return mac_points
        
    def calculate_abatement(self, year: int) -> None:
        """Calculate and apply optimal abatement based on price signal."""
        print(f"\n=== Abatement Analysis for Year {year} ===")
        
        total_abatement = 0.0
        for idx, facility in self.facilities_data.iterrows():
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
            
            # Calculate profitable abatement at current price
            if slope > 0:
                profitable_abatement = min(
                    max_reduction,
                    (self.market_price - intercept) / slope
                )
                
                # Calculate costs and expected revenue
                if profitable_abatement > 0:
                    abatement_cost = (slope * profitable_abatement**2 / 2) + (intercept * profitable_abatement)
                    expected_revenue = profitable_abatement * self.market_price
                    
                    # Execute abatement if profitable
                    if expected_revenue > abatement_cost:
                        self._apply_abatement(idx, profitable_abatement, abatement_cost, year)
                        total_abatement += profitable_abatement
                        
                        print(f"\nFacility {facility['Facility ID']} Abatement:")
                        print(f"  Amount: {profitable_abatement:.2f}")
                        print(f"  Cost: ${abatement_cost:.2f}")
                        print(f"  Revenue: ${expected_revenue:.2f}")
                        print(f"  Profit: ${expected_revenue - abatement_cost:.2f}")
        
        print(f"\nTotal Abatement Summary:")
        print(f"  Volume: {total_abatement:.2f}")
        print(f"  Price: ${self.market_price:.2f}")

    def _apply_abatement(self, idx: int, abated: float, cost: float, year: int) -> None:
        """
        Apply abatement results to the facility's data.
        
        Args:
            idx: Index of facility in facilities_data
            abated: Amount of emissions abated
            cost: Total cost of abatement
            year: Year being processed
        """
        # Update abatement amount
        self.facilities_data.at[idx, f'Tonnes Abated_{year}'] += abated
        
        # Update abatement cost
        self.facilities_data.at[idx, f'Abatement Cost_{year}'] += cost
        
        # Update allowance position
        self.facilities_data.at[idx, f'Allowance Surplus/Deficit_{year}'] += abated
    
        # Log updates for debugging
        print(f"\nFacility {self.facilities_data.at[idx, 'Facility ID']} - Year {year}:")
        print(f"  Abated: {abated:.2f}")
        print(f"  Cost: ${cost:.2f}")
        print(f"  Updated Surplus/Deficit: {self.facilities_data.at[idx, f'Allowance Surplus/Deficit_{year}']:.2f}")
    
    def trade_allowances(self, year: int) -> None:
        """Execute trades at market clearing price."""
        print(f"\n=== Trading Analysis for Year {year} ===")
        
        # Identify buyers and sellers after abatement
        buyers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0].copy()
        sellers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0].copy()
        
        if buyers.empty or sellers.empty:
            print("No trading needed - market is balanced")
            return
            
        # Calculate total positions
        total_demand = abs(buyers[f'Allowance Surplus/Deficit_{year}'].sum())
        total_supply = sellers[f'Allowance Surplus/Deficit_{year}'].sum()
        
        print(f"Pre-trade positions:")
        print(f"Buyers: {len(buyers)}, Total Demand: {total_demand:.2f}")
        print(f"Sellers: {len(sellers)}, Total Supply: {total_supply:.2f}")
        
        # Sort buyers by willingness to pay
        buyers = buyers.sort_values('Baseline Profit Rate', ascending=False)
        
        # Sort sellers by marginal cost
        def get_mac(row):
            curve = self.abatement_cost_curve[
                self.abatement_cost_curve['Facility ID'] == row['Facility ID']
            ].iloc[0]
            return curve['Intercept']
        
        sellers['mac'] = sellers.apply(get_mac, axis=1)
        sellers = sellers.sort_values('mac')
        
        # Execute trades
        trades_executed = []
        for _, buyer in buyers.iterrows():
            buyer_demand = abs(buyer[f'Allowance Surplus/Deficit_{year}'])
            
            for seller_idx, seller in sellers.iterrows():
                seller_supply = seller[f'Allowance Surplus/Deficit_{year}']
                
                # Calculate trade volume
                trade_volume = min(buyer_demand, seller_supply)
                if trade_volume <= 0:
                    continue
                    
                # Execute trade at market price
                trade_cost = trade_volume * self.market_price
                self._update_trade_positions(
                    buyer.name, seller_idx,
                    trade_volume, trade_cost, year
                )
                
                trades_executed.append({
                    'Buyer': buyer['Facility ID'],
                    'Seller': seller['Facility ID'],
                    'Volume': trade_volume,
                    'Price': self.market_price,
                    'Total Cost': trade_cost
                })
                
                # Update remaining positions
                buyer_demand -= trade_volume
                sellers.at[seller_idx, f'Allowance Surplus/Deficit_{year}'] -= trade_volume
                
                if buyer_demand <= 0:
                    break
        
        if trades_executed:
            trades_df = pd.DataFrame(trades_executed)
            print("\nTrades Executed:")
            print(trades_df.to_string())
            print(f"\nTotal Volume: {trades_df['Volume'].sum():,.2f}")
            print(f"Average Price: ${trades_df['Price'].mean():,.2f}")                

    def _update_trade_positions(self, buyer_idx, seller_idx, trade_volume, trade_cost, year):
        """Update the trade positions for the buyer and seller after a trade."""
        # Update buyer's positions
        self.facilities_data.at[buyer_idx, f'Allowance Surplus/Deficit_{year}'] += trade_volume
        self.facilities_data.at[buyer_idx, f'Trade Cost_{year}'] += trade_cost
        self.facilities_data.at[buyer_idx, f'Trade Volume_{year}'] += trade_volume
    
        # Update seller's positions
        self.facilities_data.at[seller_idx, f'Allowance Surplus/Deficit_{year}'] -= trade_volume
        self.facilities_data.at[seller_idx, f'Trade Cost_{year}'] -= trade_cost
        self.facilities_data.at[seller_idx, f'Trade Volume_{year}'] -= trade_volume
        
   

    def analyze_market_stability(self, year: int) -> None:
        """Analyze market stability with sector-specific reporting."""
        print(f"\n=== Market Stability Analysis Year {year} ===")
        
        # Overall market stability
        total_emissions = self.facilities_data[f'Emissions_{year}'].sum()
        total_allocations = self.facilities_data[f'Allocations_{year}'].sum()
        overall_surplus_ratio = (total_allocations - total_emissions) / total_allocations if total_allocations > 0 else 0
        
        print("\nOverall Market Stability:")
        print(f"Market-wide Surplus Ratio: {overall_surplus_ratio:.4f}")
        
        # Sector-specific stability analysis
        print("\nSector Stability Analysis:")
        for sector in self.facilities_data['Sector'].unique():
            sector_data = self.facilities_data[self.facilities_data['Sector'] == sector]
            
            # Calculate sector metrics
            sector_emissions = sector_data[f'Emissions_{year}'].sum()
            sector_allocations = sector_data[f'Allocations_{year}'].sum()
            sector_surplus_ratio = (sector_allocations - sector_emissions) / sector_allocations if sector_allocations > 0 else 0
            
            print(f"\n{sector}:")
            print(f"  Surplus Ratio: {sector_surplus_ratio:.4f}")
            
            # Calculate year-over-year changes if not first year
            if year > self.start_year:
                try:
                    # Emissions change
                    prev_emissions = sector_data[f'Emissions_{year-1}'].sum()
                    emissions_change = (sector_emissions - prev_emissions) / prev_emissions if prev_emissions > 0 else 0
                    
                    # Benchmark change
                    current_benchmark = sector_data[f'Benchmark_{year}'].mean()
                    prev_benchmark = sector_data[f'Benchmark_{year-1}'].mean()
                    benchmark_change = (current_benchmark - prev_benchmark) / prev_benchmark if prev_benchmark > 0 else 0
                    
                    print(f"  Emissions Change: {emissions_change:.4f}")
                    print(f"  Benchmark Change: {benchmark_change:.4f}")
                    
                except Exception as e:
                    print(f"  Error calculating changes: {str(e)}")
    
    # 3. Cost and Performance Calculations
    def calculate_costs(self, year: int) -> None:
        """Calculate various cost metrics for facilities."""
        # Calculate compliance costs (abatement costs + allowance purchases)
        self.facilities_data[f'Compliance Cost_{year}'] = (
            self.facilities_data[f'Abatement Cost_{year}'].clip(lower=0) +
            self.facilities_data[f'Allowance Purchase Cost_{year}'].clip(lower=0)
        )
        
        # Calculate total costs (compliance costs - trading revenues)
        self.facilities_data[f'Total Cost_{year}'] = (
            self.facilities_data[f'Compliance Cost_{year}'] -
            self.facilities_data[f'Allowance Sales Revenue_{year}']
        )
        
        # Calculate cost ratios
        self.calculate_cost_ratios(year)
        
        print(f"\nCost calculations for year {year}:")
        print(f"Total Compliance Cost: {self.facilities_data[f'Compliance Cost_{year}'].sum():,.2f}")
        print(f"Net Market Cost: {self.facilities_data[f'Total Cost_{year}'].sum():,.2f}")
    
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
    
    # 4. Model Execution and Results
    def prepare_facility_results(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Prepare facility-level results in long format with all metrics.
        
        Args:
            start_year: First year of simulation period
            end_year: Last year of simulation period
            
        Returns:
            DataFrame with facility results across all years
        """
        # Define core metrics to track
        metrics = [
            "Output", "Emissions", "Benchmark", "Allocations",
            "Allowance Surplus/Deficit", "Tonnes Abated", "Abatement Cost",
            "Trade Volume", "Trade Cost", "Allowance Purchase Cost",
            "Allowance Sales Revenue", "Compliance Cost", "Total Cost",
            "Cost to Profit Ratio", "Cost to Output Ratio"
        ]
        
        results = []
        for year in range(start_year, end_year + 1):
            # Extract year-specific data
            year_data = self.facilities_data[
                ['Facility ID'] + [f'{metric}_{year}' for metric in metrics]
            ].copy()
            
            # Clean column names and add year
            year_data.columns = ['Facility ID'] + metrics
            year_data['Year'] = year
            results.append(year_data)
        
        # Combine all years into single DataFrame
        combined_results = pd.concat(results, ignore_index=True)
        
        # Add additional identifier columns if available
        if 'Sector' in self.facilities_data.columns:
            sector_map = self.facilities_data[['Facility ID', 'Sector']].set_index('Facility ID')
            combined_results = combined_results.merge(
                sector_map, on='Facility ID', how='left'
            )
        
        print(f"\nPrepared facility results:")
        print(f"Years: {start_year}-{end_year}")
        print(f"Facilities: {len(self.facilities_data)}")
        print(f"Total records: {len(combined_results)}")
        
        return combined_results

    def _prepare_facility_results(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Prepare facility results in long format.
        
        Args:
            start_year: First year of simulation
            end_year: Last year of simulation
            
        Returns:
            DataFrame with facility results across all years
        """
        # Core metrics to track
        metrics = [
            "Output", "Emissions", "Benchmark", "Allocations",
            "Allowance Surplus/Deficit", "Tonnes Abated", "Abatement Cost",
            "Trade Volume", "Trade Cost", "Allowance Purchase Cost",
            "Allowance Sales Revenue", "Compliance Cost", "Total Cost",
            "Cost to Profit Ratio", "Cost to Output Ratio"
        ]
        
        # Store results for each year
        results = []
        for year in range(start_year, end_year + 1):
            # Get year-specific data
            year_data = self.facilities_data[
                ['Facility ID'] + [f'{metric}_{year}' for metric in metrics]
            ].copy()
            
            # Remove year suffix from column names
            year_data.columns = ['Facility ID'] + metrics
            
            # Add year column
            year_data['Year'] = year
            
            results.append(year_data)
        
        # Combine all years
        combined_results = pd.concat(results, ignore_index=True)
        
        # Add facility identifiers if available
        if 'Sector' in self.facilities_data.columns:
            sector_map = self.facilities_data[['Facility ID', 'Sector']].set_index('Facility ID')
            combined_results = combined_results.merge(
                sector_map, on='Facility ID', how='left'
            )
        
        print(f"\nPrepared facility results:")
        print(f"Years: {start_year}-{end_year}")
        print(f"Facilities: {len(self.facilities_data)}")
        print(f"Total records: {len(combined_results)}")
        
        return combined_results
    
    def save_results(self, market_summary: pd.DataFrame, facility_results: pd.DataFrame, 
                    output_file: str, output_dir: str = ".") -> None:
        """
        Save model results to CSV files with proper organization.
        
        Args:
            market_summary: DataFrame with market-level metrics
            facility_results: DataFrame with facility-level results
            output_file: Base name for output files
            output_dir: Directory to save results (default: current directory)
        """
        import os
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Save market summary
        market_file = os.path.join(output_dir, "market_summary.csv")
        market_summary.to_csv(market_file, index=False)
        print(f"Market summary saved to {market_file}")
        
        # Save facility results
        facility_file = os.path.join(output_dir, output_file)
        facility_results.to_csv(facility_file, index=False)
        print(f"Facility results saved to {facility_file}")
        
        # Generate and save summary statistics
        summary_stats = self._generate_summary_statistics(facility_results)
        stats_file = os.path.join(output_dir, "summary_statistics.csv")
        summary_stats.to_csv(stats_file, index=True)
        print(f"Summary statistics saved to {stats_file}")

    def _generate_summary_statistics(self, facility_results: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics from facility results."""
        stats = []
        
        # Calculate annual averages
        annual_stats = facility_results.groupby('Year').agg({
            'Emissions': 'sum',
            'Allocations': 'sum',
            'Tonnes Abated': 'sum',
            'Total Cost': 'sum',
            'Cost to Profit Ratio': 'mean',
            'Cost to Output Ratio': 'mean'
        })
        
        # Add to summary
        stats.append(annual_stats)
        
        # If sector information is available, add sector-level statistics
        if 'Sector' in facility_results.columns:
            sector_stats = facility_results.groupby(['Year', 'Sector']).agg({
                'Emissions': 'sum',
                'Allocations': 'sum',
                'Tonnes Abated': 'sum',
                'Total Cost': 'sum'
            }).reset_index()
            
            # Calculate sector shares
            total_emissions = sector_stats.groupby('Year')['Emissions'].transform('sum')
            sector_stats['Emissions Share'] = sector_stats['Emissions'] / total_emissions
            
        return pd.concat(stats, keys=['Annual', 'Sector']) if 'Sector' in facility_results.columns else annual_stats

    def get_compliance_report(self, year: int) -> pd.DataFrame:
        """Generate compliance report with sector information."""
        # Core metrics
        metrics = [
            'Sector', 'Output', 'Emissions', 'Benchmark', 'Allocations',
            'Allowance Surplus/Deficit', 'Tonnes Abated', 'Trade Volume',
            'Compliance Cost', 'Total Cost', 'Cost to Profit Ratio'
        ]
        
        # Create report with facility and sector information
        report = self.facilities_data[['Facility ID', 'Sector'] + 
            [col for col in self.facilities_data.columns if f'_{year}' in col]].copy()
        
        # Clean up column names
        report.columns = [col.replace(f'_{year}', '') if f'_{year}' in col else col 
                         for col in report.columns]
        
        # Add compliance status
        report['Compliance Status'] = report['Allowance Surplus/Deficit'].apply(
            lambda x: 'Compliant' if x >= 0 else 'Non-Compliant'
        )
        
        # Add sector summary
        sector_summary = report.groupby('Sector').agg({
            'Emissions': 'sum',
            'Allocations': 'sum',
            'Tonnes Abated': 'sum',
            'Compliance Cost': 'sum',
            'Compliance Status': lambda x: (x == 'Compliant').mean()
        }).round(4)
        
        sector_summary = sector_summary.rename(
            columns={'Compliance Status': 'Compliance Rate'}
        )
        
        print(f"\nCompliance Report Summary for Year {year}:")
        print("\nSector Performance:")
        print(sector_summary.to_string())
        
        return report, sector_summary

    #sensitivity testing 
    def run_all_scenarios(self, scenario_file: str, facilities_data: pd.DataFrame, 
                         abatement_cost_curve: pd.DataFrame, start_year: int, 
                         end_year: int, output_dir: str = "scenario_results") -> pd.DataFrame:
        """Run model for multiple scenarios focusing on key parameters."""
        import os
        
        # Load and validate scenarios
        scenarios = self.load_all_scenarios(scenario_file)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Track scenario results
        scenario_results = []
        
        # Process each scenario
        for scenario in scenarios:
            scenario_name = scenario["name"].replace(" ", "_").lower()
            print(f"\nProcessing Scenario: {scenario['name']}")
            print("Key Parameters:")
            print(f"  Price Range: ${scenario['floor_price']} - ${scenario['ceiling_price']}")
            print(f"  Ratchet Rate: {scenario['benchmark_ratchet_rate']:.3f}")
            print(f"  Growth Rate: {scenario['output_growth_rate']:.3f}")
            print(f"  MSR Active: {scenario['msr_active']}")
            
            try:
                # Run scenario
                result = self._run_single_scenario(
                    scenario, facilities_data.copy(), abatement_cost_curve.copy(),
                    start_year, end_year, output_dir
                )
                scenario_results.append(result)
                print(f"Results saved for scenario: {scenario['name']}")
                
            except Exception as e:
                print(f"Error in scenario {scenario['name']}: {str(e)}")
                continue
        
        # Create and save comparison
        comparison = self._create_scenario_comparison(scenario_results)
        comparison.to_csv(os.path.join(output_dir, 'scenario_comparison.csv'), index=False)
        
        return comparison

    def _run_single_scenario(self, scenario: Dict, facilities_data: pd.DataFrame,
                            abatement_cost_curve: pd.DataFrame, start_year: int,
                            end_year: int, output_dir: str) -> Dict:
        """Run a single scenario with simplified parameters."""
        scenario_name = scenario["name"].replace(" ", "_").lower()
        
        # Initialize model
        model = obamodel(
            facilities_data=facilities_data,
            abatement_cost_curve=abatement_cost_curve,
            start_year=start_year,
            end_year=end_year,
            scenario_params={
                'floor_price': scenario['floor_price'],
                'ceiling_price': scenario['ceiling_price'],
                'price_increment': scenario['price_increment'],
                'output_growth_rate': scenario['output_growth_rate'],
                'emissions_growth_rate': scenario['emissions_growth_rate'],
                'benchmark_ratchet_rate': scenario['benchmark_ratchet_rate'],
                'msr_active': scenario['msr_active'],
                'msr_upper_threshold': scenario.get('msr_upper_threshold', 0.15),
                'msr_lower_threshold': scenario.get('msr_lower_threshold', -0.05),
                'msr_adjustment_rate': scenario.get('msr_adjustment_rate', 0.03)
            }
        )
        
        # Run model
        market_summary, facility_results = model.run_model()
        
        # Add scenario identifier
        market_summary['Scenario'] = scenario['name']
        facility_results['Scenario'] = scenario['name']
        
        # Save results
        os.makedirs(os.path.join(output_dir, scenario_name), exist_ok=True)
        market_summary.to_csv(os.path.join(output_dir, scenario_name, 'market_summary.csv'), index=False)
        facility_results.to_csv(os.path.join(output_dir, scenario_name, 'facility_results.csv'), index=False)
        
        return {
            'name': scenario['name'],
            'market_summary': market_summary,
            'facility_results': facility_results,
            'parameters': {
                'floor_price': scenario['floor_price'],
                'ceiling_price': scenario['ceiling_price'],
                'benchmark_ratchet_rate': scenario['benchmark_ratchet_rate'],
                'output_growth_rate': scenario['output_growth_rate'],
                'msr_active': scenario['msr_active']
            }
        }

    def _create_scenario_comparison(self, scenario_results: List[Dict]) -> pd.DataFrame:
        """Create comparative analysis of scenario results."""
        comparisons = []
        
        for result in scenario_results:
            summary = result['market_summary']
            
            # Calculate key metrics
            comparisons.append({
                'Scenario': result['name'],
                'Average Price': summary['Market Price'].mean(),
                'Final Price': summary['Market Price'].iloc[-1],
                'Total Abatement': summary['Total Abatement'].sum(),
                'Average Annual Abatement': summary['Total Abatement'].mean(),
                'Cumulative Emissions': summary['Total Emissions'].sum(),
                'Final Year Emissions': summary['Total Emissions'].iloc[-1],
                'Total Compliance Cost': summary['Total Compliance Cost'].sum(),
                'Average Annual Cost': summary['Total Compliance Cost'].mean(),
                'Cost Effectiveness': (
                    summary['Total Compliance Cost'].sum() / 
                    summary['Total Abatement'].sum()
                ) if summary['Total Abatement'].sum() > 0 else float('inf'),
                'Average Market Balance': summary['Surplus Ratio'].mean()
            })
        
        return pd.DataFrame(comparisons)

    def _save_scenario_files(self, scenario_name: str, market_summary: pd.DataFrame,
                            facility_results: pd.DataFrame, output_dir: str) -> None:
        """Save individual scenario result files."""
        market_summary.to_csv(
            os.path.join(output_dir, f"{scenario_name}_market_summary.csv"),
            index=False
        )
        facility_results.to_csv(
            os.path.join(output_dir, f"{scenario_name}_facility_results.csv"),
            index=False
        )

    def _save_scenario_results(self, comparison: pd.DataFrame, 
                              scenario_results: List[Dict], output_dir: str) -> None:
        """Save scenario comparison and detailed results."""
        # Save main comparison
        comparison.to_csv(
            os.path.join(output_dir, 'scenario_comparison.csv'),
            index=False
        )
        
        # Create detailed analysis
        detailed_results = self._create_detailed_analysis(scenario_results)
        detailed_results.to_csv(
            os.path.join(output_dir, 'detailed_scenario_analysis.csv'),
            index=False
        )

    def _create_detailed_analysis(self, scenario_results: List[Dict]) -> pd.DataFrame:
        """Create detailed analysis of scenario results."""
        detailed_records = []
        
        for result in scenario_results:
            scenario_name = result['name']
            market_data = result['market_summary']
            
            for _, row in market_data.iterrows():
                record = {
                    'Scenario': scenario_name,
                    'Year': row['Year'],
                    'Market Price': row['Market Price'],
                    'Total Emissions': row['Total Emissions'],
                    'Total Abatement': row['Total Abatement'],
                    'Market Balance': row['Surplus Ratio'],
                    'Compliance Cost': row['Total Compliance Cost']
                }
                detailed_records.append(record)
        
        return pd.DataFrame(detailed_records)
    
    def _create_market_summary(self, year: int) -> Dict:
        """Create market summary with sector-specific reporting."""
        if f'Allowance Surplus/Deficit_{year}' not in self.facilities_data.columns:
            raise KeyError(f"Required data missing for year {year}")
        
        # Overall market metrics
        total_metrics = {
            'Year': year,
            'Total Allocations': self.facilities_data[f'Allocations_{year}'].sum(),
            'Total Emissions': self.facilities_data[f'Emissions_{year}'].sum(),
            'Total Abatement': self.facilities_data[f'Tonnes Abated_{year}'].sum(),
            'Market Price': self.market_price,
        }
        
        # Print overall market status
        print(f"\n=== Market Summary for Year {year} ===")
        print("\nOverall Market Status:")
        print(f"Total Allocations: {total_metrics['Total Allocations']:,.2f}")
        print(f"Total Emissions: {total_metrics['Total Emissions']:,.2f}")
        print(f"Total Abatement: {total_metrics['Total Abatement']:,.2f}")
        print(f"Market Price: ${self.market_price:.2f}")
        
        # Sector-specific reporting
        print("\nSector-Specific Performance:")
        sector_metrics = {}
        for sector in self.facilities_data['Sector'].unique():
            sector_data = self.facilities_data[self.facilities_data['Sector'] == sector]
            
            # Calculate sector metrics
            allocations = sector_data[f'Allocations_{year}'].sum()
            emissions = sector_data[f'Emissions_{year}'].sum()
            abatement = sector_data[f'Tonnes Abated_{year}'].sum()
            benchmark = sector_data[f'Benchmark_{year}'].mean()
            
            # Store sector metrics
            sector_metrics[sector] = {
                'Allocations': allocations,
                'Emissions': emissions,
                'Abatement': abatement,
                'Benchmark': benchmark,
                'Surplus/Deficit': allocations - emissions
            }
            
            # Print sector details
            print(f"\n{sector}:")
            print(f"  Benchmark: {benchmark:.6f}")
            print(f"  Allocations: {allocations:,.2f}")
            print(f"  Emissions: {emissions:,.2f}")
            print(f"  Abatement: {abatement:,.2f}")
            print(f"  Surplus/Deficit: {(allocations - emissions):,.2f}")
        
        # Calculate overall market balance
        remaining_surplus = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
        remaining_deficit = abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
        surplus_ratio = remaining_surplus / total_metrics['Total Allocations'] if total_metrics['Total Allocations'] > 0 else 0.0
        
        # Add market balance metrics
        total_metrics.update({
            'Remaining Surplus': remaining_surplus,
            'Remaining Deficit': remaining_deficit,
            'Surplus Ratio': surplus_ratio,
            'Sector_Metrics': sector_metrics  # Include sector data in return value
        })
        
        return total_metrics
