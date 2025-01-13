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
        """Initialize OBA model with configuration and scenario parameters."""
        self.facilities_data = facilities_data.copy()
        self.abatement_cost_curve = abatement_cost_curve.copy()
        self.start_year = start_year
        self.end_year = end_year
        
        # Extract scenario parameters
        self.floor_price = scenario_params.get("floor_price", 20)
        self.ceiling_price = scenario_params.get("ceiling_price", 200)
        self.price_increment = scenario_params.get("price_increment", 5)
        self.output_growth_rate = scenario_params.get("output_growth_rate", 0.02)
        self.emissions_growth_rate = scenario_params.get("emissions_growth_rate", 0.01)
        self.benchmark_ratchet_rate = scenario_params.get("benchmark_ratchet_rate", 0.03)
        
        # MSR parameters
        self.msr_active = scenario_params.get("msr_active", False)
        self.msr_upper_threshold = scenario_params.get("msr_upper_threshold", 0.15)
        self.msr_lower_threshold = scenario_params.get("msr_lower_threshold", -0.05)
        self.msr_adjustment_rate = scenario_params.get("msr_adjustment_rate", 0.03)
        
        # Initialize price schedule
        self.price_schedule = {
            year: self.floor_price + self.price_increment * (year - start_year)
            for year in range(start_year, end_year + 1)
        }
        
        # Print initialization parameters
        print("\nInitializing OBA Model with parameters:")
        print(f"Time period: {start_year} - {end_year}")
        print(f"Price range: {self.floor_price} - {self.ceiling_price}")
        print(f"Growth rates: Output {self.output_growth_rate}, Emissions {self.emissions_growth_rate}")
        print(f"Benchmark ratchet rate: {self.benchmark_ratchet_rate}")
        print(f"MSR active: {self.msr_active}")
        
        # Initialize model columns and validate data
        self._initialize_columns()
        self._validate_input_data()
                     
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
            
    # 3. Cost and Performance Calculations
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
        """
        Generate a detailed compliance report for a specific year.
        
        Args:
            year: Year to generate report for
            
        Returns:
            DataFrame with compliance metrics for each facility
        """
        # Core compliance metrics
        metrics = [
            'Output', 'Emissions', 'Allocations', 'Allowance Surplus/Deficit',
            'Tonnes Abated', 'Trade Volume', 'Compliance Cost', 'Total Cost',
            'Cost to Profit Ratio', 'Cost to Output Ratio'
        ]
        
        # Create report with facility information
        report = self.facilities_data[['Facility ID'] + 
            [f'{metric}_{year}' for metric in metrics]].copy()
        
        # Clean column names
        report.columns = ['Facility ID'] + metrics
        
        # Add compliance status
        report['Compliance Status'] = report['Allowance Surplus/Deficit'].apply(
            lambda x: 'Compliant' if x >= 0 else 'Non-Compliant'
        )
        
        # Calculate cost effectiveness metrics
        report['Abatement Rate'] = report['Tonnes Abated'] / report['Emissions']
        report['Cost per Tonne Abated'] = (
            report['Total Cost'] / report['Tonnes Abated']
        ).replace([np.inf, -np.inf], np.nan)
        
        print(f"\nCompliance Report for Year {year}:")
        print(f"Total Facilities: {len(report)}")
        print(f"Compliant Facilities: {(report['Compliance Status'] == 'Compliant').sum()}")
        
        return report

    #sensitivity testing 
     def run_all_scenarios(self, scenario_file: str, facilities_data: pd.DataFrame, 
                         abatement_cost_curve: pd.DataFrame, start_year: int, 
                         end_year: int, output_dir: str = "scenario_results") -> pd.DataFrame:
        """
        Run model for multiple scenarios and generate comparative analysis.
        
        Args:
            scenario_file: Path to scenario configuration file
            facilities_data: Base facility data
            abatement_cost_curve: Abatement cost curves
            start_year: Simulation start year
            end_year: Simulation end year
            output_dir: Directory for results
            
        Returns:
            DataFrame with scenario comparison results
        """
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
            print(f"Parameters:")
            for key, value in scenario.items():
                if key != 'name':
                    print(f"  {key}: {value}")
            
            try:
                # Run scenario
                result = self._run_single_scenario(
                    scenario, facilities_data, abatement_cost_curve,
                    start_year, end_year, output_dir
                )
                scenario_results.append(result)
                
            except Exception as e:
                print(f"Error in scenario {scenario['name']}: {str(e)}")
                continue
        
        # Generate comparison
        comparison = self._create_scenario_comparison(scenario_results)
        
        # Save results
        self._save_scenario_results(comparison, scenario_results, output_dir)
        
        return comparison

    def _run_single_scenario(self, scenario: Dict, facilities_data: pd.DataFrame,
                            abatement_cost_curve: pd.DataFrame, start_year: int,
                            end_year: int, output_dir: str) -> Dict:
        """Run a single scenario and save results."""
        scenario_name = scenario["name"].replace(" ", "_").lower()
        
        # Initialize model for scenario
        model = obamodel(
            facilities_data=facilities_data.copy(),
            abatement_cost_curve=abatement_cost_curve.copy(),
            start_year=start_year,
            end_year=end_year,
            scenario_params=scenario
        )
        
        # Run model
        market_summary, facility_results = model.run_model()
        
        # Add scenario identifier
        market_summary['Scenario'] = scenario['name']
        facility_results['Scenario'] = scenario['name']
        
        # Save individual scenario results
        self._save_scenario_files(
            scenario_name, market_summary, facility_results, output_dir
        )
        
        return {
            'name': scenario['name'],
            'market_summary': market_summary,
            'facility_results': facility_results
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
    
    def create_market_summary(self, year: int) -> Dict:
        """Create summary of market conditions for a given year."""
        # Validate required data
        if f'Allowance Surplus/Deficit_{year}' not in self.facilities_data.columns:
            raise KeyError(f"Required data missing for year {year}")
        
        # Calculate core metrics
        metrics = {
            'Year': year,
            'Total Allocations': self.facilities_data[f'Allocations_{year}'].sum(),
            'Total Emissions': self.facilities_data[f'Emissions_{year}'].sum(),
            'Total Abatement': self.facilities_data[f'Tonnes Abated_{year}'].sum(),
            'Market Price': self.market_price,
            'Trade Volume': self.facilities_data[f'Trade Volume_{year}'].abs().sum() / 2,
            'Total Trade Cost': self.facilities_data[f'Trade Cost_{year}'].abs().sum() / 2,
            'Total Abatement Cost': self.facilities_data[f'Abatement Cost_{year}'].sum(),
            'Total Compliance Cost': self.facilities_data[f'Compliance Cost_{year}'].sum(),
            'Net Market Cost': self.facilities_data[f'Total Cost_{year}'].sum()
        }
        
        # Calculate market balance metrics
        surplus = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
        deficit = abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
        
        metrics.update({
            'Remaining Surplus': surplus,
            'Remaining Deficit': deficit,
            'Surplus Ratio': surplus / metrics['Total Allocations'] if metrics['Total Allocations'] > 0 else 0.0
        })
        
        # Calculate efficiency metrics
        metrics.update({
            'Average Cost to Profit Ratio': self.facilities_data[f'Cost to Profit Ratio_{year}'].mean(),
            'Average Cost to Output Ratio': self.facilities_data[f'Cost to Output Ratio_{year}'].mean()
        })
        
        # Print summary
        print(f"\n=== Market Summary for Year {year} ===")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:,.2f}")
            else:
                print(f"{key}: {value}")
        
        return metrics
