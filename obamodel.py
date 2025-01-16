import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
from pathlib import Path

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
        
   
  
# 3. Core Model Execution Methods
    def run_model(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Main entry point that orchestrates the entire model execution.
        Returns tuple of (market_summary_df, sector_summary_df, facility_results)
        """
        print("Running emissions trading model...")
        market_summary = []
        sector_summaries = []
        
        for year in range(self.start_year, self.end_year + 1):
            print(f"\nProcessing year {year}...")
            
            # 1. Calculate initial positions 
            self.calculate_dynamic_values(year)
            
            try:
                # 2. Initial market positions - Add error handling
                market_positions = self.calculate_market_positions(year)
                if not isinstance(market_positions, tuple) or len(market_positions) != 2:
                    raise ValueError(f"Invalid market positions return value: {market_positions}")
                total_supply, total_demand = market_positions
                
                # 3. Determine initial market price
                self.determine_market_price(total_supply, total_demand, year)
                
                # 4. Calculate abatement based on price signal
                self.calculate_abatement(year)
                
                # 5. Recalculate market positions after abatement - Add error handling
                market_positions = self.calculate_market_positions(year)
                if not isinstance(market_positions, tuple) or len(market_positions) != 2:
                    raise ValueError(f"Invalid post-abatement market positions: {market_positions}")
                post_abatement_supply, post_abatement_demand = market_positions
                
                # 6. Update price based on post-abatement positions
                self.determine_market_price(post_abatement_supply, post_abatement_demand, year)
                
                # Continue with rest of processing...
                self.trade_allowances(year)
                self.calculate_costs(year)
                self.calculate_cost_ratios(year)
                
                # Create summaries
                market_summary.append(self._create_market_summary(year))
                sector_summary = self.create_sector_summary(year)
                sector_summaries.append(sector_summary)
                
                # Analyze stability
                self.analyze_market_stability(year)
                
            except Exception as e:
                print(f"Error processing year {year}: {str(e)}")
                continue
        
        # Convert summaries to DataFrames
        market_summary_df = pd.DataFrame(market_summary) if market_summary else pd.DataFrame()
        sector_summary_df = pd.concat(sector_summaries, ignore_index=True) if sector_summaries else pd.DataFrame()
        facility_results = self._prepare_facility_results(self.start_year, self.end_year)
        
        return market_summary_df, sector_summary_df, facility_results
    
    def validate_scenario_parameters(self, scenario_type: str, params: Dict) -> bool:
        """Validate parameters for any scenario type"""
        # Base parameter validation
        base_params = {
            'floor_price': (0, None),  # No upper limit
            'ceiling_price': (0, None),
            'price_increment': (0, None),
            'output_growth_rate': (-0.5, 0.5),
            'emissions_growth_rate': (-0.5, 0.5),
            'benchmark_ratchet_rate': (0, 1)
        }
        
        # Validate base parameters first
        for param, (min_val, max_val) in base_params.items():
            value = params.get(param)
            if value is None:
                print(f"Missing required parameter: {param}")
                return False
            if not isinstance(value, (int, float)):
                print(f"Parameter {param} must be numeric")
                return False
            if min_val is not None and value < min_val:
                print(f"Parameter {param} must be >= {min_val}")
                return False
            if max_val is not None and value > max_val:
                print(f"Parameter {param} must be <= {max_val}")
                return False

        # MSR-specific validation
        if scenario_type == 'msr':
            msr_params = {
                'msr_upper_threshold': (0, 1),
                'msr_lower_threshold': (-1, 0),
                'msr_adjustment_rate': (0, 1)
            }
            
            for param, (min_val, max_val) in msr_params.items():
                value = params.get(param)
                if value is None:
                    print(f"Missing required MSR parameter: {param}")
                    return False
                if not isinstance(value, (int, float)):
                    print(f"MSR parameter {param} must be numeric")
                    return False
                if value < min_val or value > max_val:
                    print(f"MSR parameter {param} must be between {min_val} and {max_val}")
                    return False
            
            if params['msr_upper_threshold'] <= params['msr_lower_threshold']:
                print("MSR upper threshold must be greater than lower threshold")
                return False
                
        return True
    
    def run_msr_scenario(self) -> Tuple[float, float]:
        """
        Run MSR scenario and return stability metrics.
        """
        try:
            # Run through main scenario handler
            result = self.run_scenario('msr')
            
            # Just extract the metrics we want
            if isinstance(result, dict) and 'metrics' in result:
                return (
                    result['metrics'].get('stability', 0.0),
                    result['metrics'].get('balance', 0.0)
                )
            return 0.0, 0.0
            
        except Exception as e:
            print(f"Error in MSR scenario: {str(e)}")
            return 0.0, 0.0

    def run_scenario(self, scenario_type: str, params: Dict = None) -> Dict:
        """
        Single entry point for running any scenario type.
        """
        print("Starting scenario analysis...")
    
        # Define fixed paths
        base_dir = Path(r"C:\Users\user\AppData\Local\Programs\Python\Python313\OBA_test")
        scenario_file = base_dir / "scenarios" / "scenarios.csv"
        facilities_file = base_dir / "data" / "input" / "facilities" / "facilities_data.csv"
        abatement_file = base_dir / "data" / "input" / "facilities" / "abatement_cost_curve.csv"
        results_dir = base_dir / "results"
    
        # Verify paths
        print("\nFile paths:")
        print(f"Scenarios file: {scenario_file}")
        print(f"Facilities file: {facilities_file}")
        print(f"Abatement file: {abatement_file}")
        print(f"Results directory: {results_dir}")
    
        # Ensure results directory exists
        try:
            results_dir.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            print(f"Error creating results directory: {e}")
            return None
    
        if params is None:
            params = {}
    
        # Set up parameters
        scenario_params = {
            'floor_price': params.get('floor_price', self.floor_price),
            'ceiling_price': params.get('ceiling_price', self.ceiling_price),
            'price_increment': params.get('price_increment', self.price_increment),
            'output_growth_rate': params.get('output_growth_rate', self.output_growth_rate),
            'emissions_growth_rate': params.get('emissions_growth_rate', self.emissions_growth_rate),
            'benchmark_ratchet_rate': params.get('benchmark_ratchet_rate', 0.02)
        }
    
        if scenario_type == 'msr':
            scenario_params['msr_active'] = True
            scenario_params['msr_upper_threshold'] = params.get('msr_upper_threshold', self.msr_upper_threshold)
            scenario_params['msr_lower_threshold'] = params.get('msr_lower_threshold', self.msr_lower_threshold)
            scenario_params['msr_adjustment_rate'] = params.get('msr_adjustment_rate', self.msr_adjustment_rate)
        else:
            scenario_params['msr_active'] = False
    
        try:
            print(f"\nExecuting {scenario_type} scenario...")
            # Run core model
            market_summary, sector_summary, facility_results = self.run_model()
    
            # Calculate metrics (these need to be defined elsewhere in the class)
            metrics = {
                'stability': 1.0 - (market_summary['Market_Price'].std() / market_summary['Market_Price'].mean() if market_summary['Market_Price'].mean() > 0 else 0),
                'balance': market_summary['Market_Balance_Ratio'].mean() if 'Market_Balance_Ratio' in market_summary.columns else 0
            }
    
            return {
                'type': scenario_type,
                'parameters': scenario_params,
                'market_summary': market_summary,
                'sector_summary': sector_summary,
                'facility_results': facility_results,
                'metrics': metrics
            }
    
        except Exception as e:
            print(f"Error in scenario execution: {str(e)}")
            return self._empty_scenario_result(scenario_type, scenario_params)
        
    def _empty_scenario_result(self, scenario_type: str, params: Dict) -> Dict:
        """Return empty result structure for failed scenarios"""
        return {
            'type': scenario_type,
            'parameters': params,
            'market_summary': pd.DataFrame(),
            'sector_summary': pd.DataFrame(),
            'facility_results': pd.DataFrame(),
            'metrics': {
                'stability': 0.0,
                'balance': 0.0
            }
        }

    def integrate_periodic_adjustment(self, year: int) -> Tuple[float, float]:
        """Integration point for periodic benchmark adjustment in the main model execution."""
        # Calculate initial values
        self.calculate_dynamic_values(year)
        
        # Apply periodic benchmark adjustment
        self.adjust_benchmarks_with_rolling_average(year)
        
        # Return updated market positions
        return self.calculate_market_positions(year)

# 2. Dynamic Value Calculations
    def calculate_dynamic_values(self, year: int) -> None:
        """Calculate dynamic values using facility-specific baseline benchmarks."""
        years_elapsed = year - self.start_year
        print(f"\n=== Dynamic Value Analysis for Year {year} ===")
    
        # 1. Calculate output with growth rate
        self.facilities_data[f'Output_{year}'] = (
            self.facilities_data['Baseline Output'] *
            (1 + self.output_growth_rate) ** years_elapsed
        )
    
        # 2. Calculate emissions using baseline emissions and growth rate
        self.facilities_data[f'Emissions_{year}'] = (
            self.facilities_data['Baseline Emissions'] *
            (1 + self.emissions_growth_rate) ** years_elapsed
        )
    
        # 3. Calculate current total emissions for ratchet calculation
        total_emissions = self.facilities_data[f'Emissions_{year}'].sum()
        
        # 4. Define target and calculate required ratchet
        target_surplus_ratio = 0.05  # 5% oversupply target
        target_surplus = total_emissions * (1 + target_surplus_ratio)
        
        # 5. Calculate initial allocations using baseline benchmark
        initial_allocations = (
            self.facilities_data[f'Output_{year}'] * 
            self.facilities_data['Baseline Benchmark']
        ).sum()
        
        # 6. Compute required ratchet adjustment
        if initial_allocations > 0:
            allocation_decline_rate = (initial_allocations - target_surplus) / initial_allocations
            required_ratchet_rate = allocation_decline_rate / max(1, years_elapsed)
            bounded_ratchet_rate = np.clip(required_ratchet_rate, 0.01, 0.20)
        else:
            bounded_ratchet_rate = 0.01
            
        print(f"\nRatchet Rate Calculation:")
        print(f"Initial Allocations: {initial_allocations:,.2f}")
        print(f"Total Emissions: {total_emissions:,.2f}")
        print(f"Target Surplus: {target_surplus:,.2f}")
        print(f"Applied Ratchet Rate: {bounded_ratchet_rate:.4f}")
        
        # 7. Calculate final benchmark with ratchet
        self.facilities_data[f'Benchmark_{year}'] = (
            self.facilities_data['Baseline Benchmark'] *
            (1 - bounded_ratchet_rate) ** years_elapsed
        )
        
        # 8. Calculate final allocations
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * 
            self.facilities_data[f'Benchmark_{year}']
        )
        
        # 9. Calculate and store surplus/deficit
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - 
            self.facilities_data[f'Emissions_{year}']
        )

    def adjust_benchmarks_with_rolling_average(self, year: int) -> None:
        """Adjust benchmarks every two years based on rolling averages of historical data."""
        # Check if this is an adjustment year (every two years)
        is_adjustment_year = (year - self.start_year) % 2 == 0
        if not is_adjustment_year and year != self.start_year:
            print(f"\nYear {year} is not an adjustment year. Keeping current benchmarks.")
            return
    
        print(f"\n=== Periodic Benchmark Adjustment Analysis for Year {year} ===")
        
        # Calculate rolling averages from previous periods
        look_back_years = 2
        historical_data = {}
        
        for prev_year in range(max(self.start_year, year - look_back_years), year):
            historical_data[prev_year] = {
                'emissions': self.facilities_data[f'Emissions_{prev_year}'].sum(),
                'allocations': self.facilities_data[f'Allocations_{prev_year}'].sum(),
                'output': self.facilities_data[f'Output_{prev_year}'].sum()
            }
            
        if historical_data:
            avg_emissions = sum(data['emissions'] for data in historical_data.values()) / len(historical_data)
            avg_allocations = sum(data['allocations'] for data in historical_data.values()) / len(historical_data)
            avg_output = sum(data['output'] for data in historical_data.values()) / len(historical_data)
        else:
            avg_emissions = self.facilities_data[f'Emissions_{year}'].sum()
            avg_allocations = self.facilities_data[f'Allocations_{year}'].sum()
            avg_output = self.facilities_data[f'Output_{year}'].sum()
            
        # Calculate target metrics
        target_surplus_ratio = 0.05
        target_allocations = avg_emissions * (1 + target_surplus_ratio)
        
        if avg_allocations > 0:
            required_adjustment = (target_allocations - avg_allocations) / avg_allocations
            required_adjustment = max(min(required_adjustment, 0.10), -0.10)
        else:
            required_adjustment = 0
            
        print(f"\nRequired Adjustment: {required_adjustment:.4f}")
        
        # Apply sector-specific adjustments
        for sector in self.facilities_data['Sector'].unique():
            sector_mask = self.facilities_data['Sector'] == sector
            sector_data = self.facilities_data[sector_mask]
            
            sector_historical = {}
            for prev_year in historical_data.keys():
                sector_historical[prev_year] = {
                    'emissions': sector_data[f'Emissions_{prev_year}'].sum(),
                    'output': sector_data[f'Output_{prev_year}'].sum()
                }
                
            if sector_historical:
                sector_avg_emissions = sum(data['emissions'] for data in sector_historical.values()) / len(sector_historical)
                sector_avg_output = sum(data['output'] for data in sector_historical.values()) / len(sector_historical)
            else:
                sector_avg_emissions = sector_data[f'Emissions_{year}'].sum()
                sector_avg_output = sector_data[f'Output_{year}'].sum()
            
            sector_intensity = sector_avg_emissions / sector_avg_output if sector_avg_output > 0 else 0
            average_intensity = avg_emissions / avg_output if avg_output > 0 else 0
            
            sector_adjustment = required_adjustment
            
            if average_intensity > 0:
                intensity_factor = sector_intensity / average_intensity
                sector_adjustment *= 1.2 if intensity_factor > 1 else 0.8
                
            current_benchmarks = self.facilities_data.loc[sector_mask, f'Benchmark_{year}']
            new_benchmarks = current_benchmarks * (1 + sector_adjustment)
            
            self.facilities_data.loc[sector_mask, f'Benchmark_{year}'] = new_benchmarks
            
            if year < self.end_year:
                self.facilities_data.loc[sector_mask, f'Benchmark_{year+1}'] = new_benchmarks
                
        # Recalculate allocations and positions
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * 
            self.facilities_data[f'Benchmark_{year}']
        )
        
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - 
            self.facilities_data[f'Emissions_{year}']
        )

# 3. Market Position Analysis

    def calculate_market_positions(self, year: int) -> Tuple[float, float]:
        """Calculate current market positions with improved balance calculation."""
        positions = self.facilities_data[f'Allowance Surplus/Deficit_{year}']
        
        # Calculate total supply (positive positions)
        total_supply = positions[positions > 0].sum()
        
        # Calculate total demand (negative positions)
        total_demand = abs(positions[positions < 0].sum())
        
        return total_supply, total_demand

   
    def analyze_market_stability(self, year: int) -> None:
        """Analyze market stability with sector-specific reporting."""
        print(f"\n=== Market Stability Analysis Year {year} ===")
        
        total_emissions = self.facilities_data[f'Emissions_{year}'].sum()
        total_allocations = self.facilities_data[f'Allocations_{year}'].sum()
        overall_surplus_ratio = (total_allocations - total_emissions) / total_allocations if total_allocations > 0 else 0
        
        print("\nOverall Market Stability:")
        print(f"Market-wide Surplus Ratio: {overall_surplus_ratio:.4f}")
        
        print("\nSector Stability Analysis:")
        for sector in self.facilities_data['Sector'].unique():
            sector_data = self.facilities_data[self.facilities_data['Sector'] == sector]
            
            sector_emissions = sector_data[f'Emissions_{year}'].sum()
            sector_allocations = sector_data[f'Allocations_{year}'].sum()
            sector_surplus_ratio = (sector_allocations - sector_emissions) / sector_allocations if sector_allocations > 0 else 0
            
            print(f"\n{sector}:")
            print(f"  Surplus Ratio: {sector_surplus_ratio:.4f}")
            
            if year > self.start_year:
                try:
                    prev_emissions = sector_data[f'Emissions_{year-1}'].sum()
                    emissions_change = (sector_emissions - prev_emissions) / prev_emissions if prev_emissions > 0 else 0
                    
                    current_benchmark = sector_data[f'Benchmark_{year}'].mean()
                    prev_benchmark = sector_data[f'Benchmark_{year-1}'].mean()
                    benchmark_change = (current_benchmark - prev_benchmark) / prev_benchmark if prev_benchmark > 0 else 0
                    
                    print(f"  Emissions Change: {emissions_change:.4f}")
                    print(f"  Benchmark Change: {benchmark_change:.4f}")
                    
                except Exception as e:
                    print(f"  Error calculating changes: {str(e)}")
                    
    def analyze_trading_conditions(self, year: int) -> None:
        """Analyze market conditions affecting trade volumes and costs."""
        print(f"\n=== Detailed Market Analysis for Year {year} ===")
        
        positions = self.facilities_data[f'Allowance Surplus/Deficit_{year}']
        buyers = self.facilities_data[positions < 0]
        sellers = self.facilities_data[positions > 0]
        
        total_surplus = positions[positions > 0].sum()
        total_deficit = abs(positions[positions < 0].sum())
        
        print("\nPosition Analysis:")
        print(f"Total facilities: {len(self.facilities_data)}")
        print(f"Sellers: {len(sellers)} facilities")
        print(f"Total surplus available: {total_surplus:.4f}")
        print(f"Average surplus per seller: {(total_surplus/len(sellers) if len(sellers)>0 else 0):.4f}")
        
        print(f"\nBuyers: {len(buyers)} facilities")
        print(f"Total deficit needing coverage: {total_deficit:.4f}")
        print(f"Average deficit per buyer: {(total_deficit/len(buyers) if len(buyers)>0 else 0):.4f}")
        
        print(f"\nPrice Analysis:")
        print(f"Current market price: ${self.market_price:.2f}")
        
        # Sample abatement costs
        sample_size = min(5, len(self.abatement_cost_curve))

# 4. Price Determination and Abatement
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

# 5. Trading Execution
    def trade_allowances(self, year: int) -> None:
        """Execute trades at market clearing price with improved balance handling."""
        print(f"\n=== Trading Analysis for Year {year} ===")
        
        # Initialize trading metrics
        self.facilities_data[f'Trade Volume_{year}'] = 0.0
        self.facilities_data[f'Allowance Purchase Cost_{year}'] = 0.0
        self.facilities_data[f'Allowance Sales Revenue_{year}'] = 0.0
        
        # Calculate initial market balance
        total_supply, total_demand = self.calculate_market_positions(year)
        print(f"\nInitial market positions:")
        print(f"Total Supply: {total_supply:.4f}")
        print(f"Total Demand: {total_demand:.4f}")
        
        # Set minimum trade threshold
        MIN_TRADE_VOLUME = 0.0001
        
        # Only proceed if there's meaningful imbalance
        if abs(total_demand - total_supply) < MIN_TRADE_VOLUME:
            print("Market is balanced within tolerance - no trading needed")
            return
        
        trading_rounds = 0
        max_rounds = 10  # Prevent infinite loops
        total_volume_traded = 0
        
        while trading_rounds < max_rounds:
            # Get current buyers and sellers
            buyers = self.facilities_data[
                self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < -MIN_TRADE_VOLUME
            ].copy()
            
            sellers = self.facilities_data[
                self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > MIN_TRADE_VOLUME
            ].copy()
            
            if buyers.empty or sellers.empty:
                print(f"No more valid trading pairs after {trading_rounds} rounds")
                break
                
            # Sort buyers by willingness to pay (higher profit rate first)
            buyers['trade_priority'] = buyers['Baseline Profit Rate'] / buyers[f'Allowance Surplus/Deficit_{year}'].abs()
            buyers = buyers.sort_values('trade_priority', ascending=False)
            
            # Sort sellers by cost (lower MAC first)
            sellers['mac'] = sellers.apply(
                lambda row: self._get_facility_mac(row['Facility ID'], year),
                axis=1
            )
            sellers = sellers.sort_values('mac')
            
            round_volume = 0
            trades_this_round = []
            
            # Execute trades for this round
            for _, buyer in buyers.iterrows():
                buyer_demand = abs(buyer[f'Allowance Surplus/Deficit_{year}'])
                if buyer_demand < MIN_TRADE_VOLUME:
                    continue
                    
                for seller_idx, seller in sellers.iterrows():
                    seller_supply = seller[f'Allowance Surplus/Deficit_{year}']
                    if seller_supply < MIN_TRADE_VOLUME:
                        continue
                    
                    # Calculate optimal trade volume
                    trade_volume = min(buyer_demand, seller_supply)
                    if trade_volume < MIN_TRADE_VOLUME:
                        continue
                    
                    # Calculate trade cost using current market price
                    trade_cost = trade_volume * self.market_price
                    
                    # Execute trade
                    # Update buyer
                    self.facilities_data.loc[buyer.name, f'Allowance Surplus/Deficit_{year}'] += trade_volume
                    self.facilities_data.loc[buyer.name, f'Trade Volume_{year}'] += trade_volume
                    self.facilities_data.loc[buyer.name, f'Allowance Purchase Cost_{year}'] += trade_cost
                    
                    # Update seller
                    self.facilities_data.loc[seller_idx, f'Allowance Surplus/Deficit_{year}'] -= trade_volume
                    self.facilities_data.loc[seller_idx, f'Trade Volume_{year}'] += trade_volume
                    self.facilities_data.loc[seller_idx, f'Allowance Sales Revenue_{year}'] += trade_cost
                    
                    # Record trade
                    trades_this_round.append({
                        'Buyer': buyer['Facility ID'],
                        'Seller': seller['Facility ID'],
                        'Volume': trade_volume,
                        'Price': self.market_price,
                        'Cost': trade_cost
                    })
                    
                    round_volume += trade_volume
                    total_volume_traded += trade_volume
                    
                    # Update remaining needs
                    buyer_demand -= trade_volume
                    sellers.loc[seller_idx, f'Allowance Surplus/Deficit_{year}'] -= trade_volume
                    
                    if buyer_demand < MIN_TRADE_VOLUME:
                        break
                        
            # Check if meaningful trading occurred this round
            if round_volume < MIN_TRADE_VOLUME:
                print(f"No significant trades in round {trading_rounds + 1}")
                break
                
            print(f"\nRound {trading_rounds + 1} Summary:")
            print(f"Volume traded: {round_volume:.4f}")
            print(f"Trades executed: {len(trades_this_round)}")
            
            # Recalculate market balance
            new_supply, new_demand = self.calculate_market_positions(year)
            print(f"Updated market balance - Supply: {new_supply:.4f}, Demand: {new_demand:.4f}")
            
            trading_rounds += 1
        
        # Final summary
        print(f"\nFinal Trading Summary:")
        print(f"Total rounds: {trading_rounds}")
        print(f"Total volume traded: {total_volume_traded:.4f}")
        
        final_supply, final_demand = self.calculate_market_positions(year)
        print(f"Final market balance:")
        print(f"Supply: {final_supply:.4f}")
        print(f"Demand: {final_demand:.4f}")
        print(f"Imbalance: {abs(final_supply - final_demand):.4f}")

    def debug_trade_conditions(self, year: int) -> None:
        """Debug why trades are not occurring by checking key conditions."""
        print(f"\n=== Trade Debug Analysis for Year {year} ===")
        
        # Check market positions
        positions = self.facilities_data[f'Allowance Surplus/Deficit_{year}']
        buyers = self.facilities_data[positions < 0]
        sellers = self.facilities_data[positions > 0]
        
        print("\nMarket Position Analysis:")
        print(f"Total facilities: {len(self.facilities_data)}")
        print(f"Number of buyers: {len(buyers)}")
        print(f"Number of sellers: {len(sellers)}")
        print(f"Total demand: {abs(positions[positions < 0].sum()):.2f}")
        print(f"Total supply: {positions[positions > 0].sum():.2f}")
        
        if len(buyers) == 0 or len(sellers) == 0:
            print("ERROR: Missing buyers or sellers!")
            return
            
        # Check price analysis
        print("\nPrice Analysis:")
        print(f"Current market price: ${self.market_price:.2f}")
        
        # Sample MAC curves and analyze trade potential
        sample_size = min(5, len(self.abatement_cost_curve))
        print(f"\nSample of {sample_size} MAC curves:")
        sample_curves = self.abatement_cost_curve.head(sample_size)
        for _, curve in sample_curves.iterrows():
            print(f"Facility {curve['Facility ID']}:")
            print(f"  Intercept: ${curve['Intercept']:.2f}")
            print(f"  Slope: {curve['Slope']:.4f}")        
      

    def _get_facility_mac(self, facility_id: str, year: int) -> float:
        """Calculate facility's marginal abatement cost at current position."""
        try:
            curve = self.abatement_cost_curve[
                self.abatement_cost_curve['Facility ID'] == facility_id
            ].iloc[0]
            
            current_abatement = self.facilities_data.loc[
                self.facilities_data['Facility ID'] == facility_id,
                f'Tonnes Abated_{year}'
            ].iloc[0]
            
            # Calculate MAC at current abatement level
            mac = float(curve['Slope']) * current_abatement + float(curve['Intercept'])
            return max(0, mac)  # Ensure non-negative MAC
            
        except Exception as e:
            print(f"Warning: Error calculating MAC for facility {facility_id}: {str(e)}")
            return float('inf')  # Return high cost to discourage trading


# 6. Cost Calculations
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



# 8. Data Analysis & Summary Creation Methods

    def _prepare_facility_results(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Prepare facility-level results for the simulation period.
        
        Args:
            start_year: First year of simulation
            end_year: Last year of simulation
            
        Returns:
            DataFrame containing facility results for all years
        """
        results_data = []
        
        for year in range(start_year, end_year + 1):
            for _, facility in self.facilities_data.iterrows():
                # Prepare base record
                record = {
                    'Year': year,
                    'Facility ID': facility['Facility ID'],
                    'Sector': facility['Sector'],
                    'Output': facility[f'Output_{year}'],
                    'Emissions': facility[f'Emissions_{year}'],
                    'Benchmark': facility[f'Benchmark_{year}'],
                    'Allocations': facility[f'Allocations_{year}'],
                    'Allowance Surplus/Deficit': facility[f'Allowance Surplus/Deficit_{year}']
                }
                
                # Add optional metrics with safe get()
                optional_metrics = [
                    'Tonnes Abated',
                    'Abatement Cost',
                    'Trade Volume',
                    'Trade Cost',
                    'Allowance Purchase Cost',
                    'Allowance Sales Revenue',
                    'Compliance Cost',
                    'Total Cost',
                    'Cost to Profit Ratio',
                    'Cost to Output Ratio'
                ]
                
                for metric in optional_metrics:
                    metric_year = f'{metric}_{year}'
                    record[metric] = float(facility.get(metric_year, 0))
                
                results_data.append(record)
        
        # Create DataFrame and enforce float type for numeric columns
        results_df = pd.DataFrame(results_data)
        
        # Ensure non-string columns are float
        for col in results_df.columns:
            if col not in ['Year', 'Facility ID', 'Sector']:
                results_df[col] = results_df[col].astype(float)
        
        return results_df
    
    def create_sector_summary(self, year: int) -> pd.DataFrame:
        """
        Create summary of metrics by sector for a given year.
        
        Args:
            year: The year to summarize
                
        Returns:
            DataFrame with sector-level metrics
        """
        sector_data = []
            
        for sector in self.facilities_data['Sector'].unique():
            sector_slice = self.facilities_data[self.facilities_data['Sector'] == sector]
                
            # Calculate sector metrics
            record = {
                'Sector': sector,
                'Year': year,
                'Output': float(sector_slice[f'Output_{year}'].sum()),
                'Emissions': float(sector_slice[f'Emissions_{year}'].sum()),
                'Benchmark': float(sector_slice[f'Benchmark_{year}'].mean()),
                'Allocations': float(sector_slice[f'Allocations_{year}'].sum()),
                'Allowance Surplus/Deficit': float(sector_slice[f'Allowance Surplus/Deficit_{year}'].sum()),
                'Tonnes Abated': float(sector_slice[f'Tonnes Abated_{year}'].sum()),
                'Abatement Cost': float(sector_slice[f'Abatement Cost_{year}'].sum()),
                'Trade Volume': float(sector_slice[f'Trade Volume_{year}'].sum()),
                'Trade Cost': float(sector_slice[f'Trade Cost_{year}'].sum()),
                'Allowance Purchase Cost': float(sector_slice[f'Allowance Purchase Cost_{year}'].sum()),
                'Allowance Sales Revenue': float(sector_slice[f'Allowance Sales Revenue_{year}'].sum()),
                'Compliance Cost': float(sector_slice[f'Compliance Cost_{year}'].sum()),
                'Total Cost': float(sector_slice[f'Total Cost_{year}'].sum())
            }
                
            # Calculate ratios
            total_profit = sector_slice['Profit'].sum()
            total_output = record['Output']
                
            record['Cost to Profit Ratio'] = record['Total Cost'] / total_profit if total_profit > 0 else 0
            record['Cost to Output Ratio'] = record['Total Cost'] / total_output if total_output > 0 else 0
                
            sector_data.append(record)
            
        # Convert to DataFrame
        sector_summary = pd.DataFrame(sector_data)
            
        # Reorder columns
        column_order = [
            'Sector', 'Year', 'Output', 'Emissions', 'Benchmark', 'Allocations',
            'Allowance Surplus/Deficit', 'Tonnes Abated', 'Abatement Cost',
            'Trade Volume', 'Trade Cost', 'Allowance Purchase Cost',
            'Allowance Sales Revenue', 'Compliance Cost', 'Total Cost',
            'Cost to Profit Ratio', 'Cost to Output Ratio'
        ]
            
        return sector_summary[column_order]   

    def _calculate_stability_metric(self, market_summary: pd.DataFrame) -> float:
        """Calculate price stability metric."""
        try:
            if market_summary.empty or 'Market_Price' not in market_summary.columns:
                return 0.0
                
            prices = market_summary['Market_Price']
            if len(prices) <= 1:
                return 1.0
                
            # Calculate price stability as 1 - coefficient of variation
            mean_price = prices.mean()
            if mean_price > 0:
                return 1.0 - (prices.std() / mean_price)
            return 0.0
            
        except Exception as e:
            print(f"Error calculating stability metric: {str(e)}")
            return 0.0

    def _calculate_balance_metric(self, market_summary: pd.DataFrame) -> float:
        """Calculate market balance metric."""
        try:
            if market_summary.empty or 'Market_Balance_Ratio' not in market_summary.columns:
                return 0.0
                
            # Use average absolute deviation from target balance
            target_balance = 0.05  # 5% surplus target
            balance_ratios = market_summary['Market_Balance_Ratio']
            
            return 1.0 - abs(balance_ratios.mean() - target_balance)
            
        except Exception as e:
            print(f"Error calculating balance metric: {str(e)}")
            return 0.0
    
    def _create_market_summary(self, year: int) -> Dict:
        """Create system-wide market summary for a given year."""
        if f'Allowance Surplus/Deficit_{year}' not in self.facilities_data.columns:
            raise KeyError(f"Required data missing for year {year}")
        
        # Calculate market-wide metrics
        summary = {
            'Year': year,
            'Market_Price': float(self.market_price),
            'Total_Allocations': float(self.facilities_data[f'Allocations_{year}'].sum()),
            'Total_Emissions': float(self.facilities_data[f'Emissions_{year}'].sum()),
            'Total_Abatement': float(self.facilities_data[f'Tonnes Abated_{year}'].sum()),
            'Total_Trade_Volume': float(abs(self.facilities_data[f'Trade Volume_{year}'].sum())/2),
            'Total_Trade_Cost': float(self.facilities_data.get(f'Trade Cost_{year}', 0).abs().sum()/2),
            'Total_Abatement_Cost': float(self.facilities_data[f'Abatement Cost_{year}'].sum()),
            'Total_Compliance_Cost': float(self.facilities_data[f'Compliance Cost_{year}'].sum()),
            'Total_Net_Cost': float(self.facilities_data[f'Total Cost_{year}'].sum())
        }
        
        # Add market balance metrics
        summary['Market_Balance'] = summary['Total_Allocations'] - summary['Total_Emissions']
        summary['Market_Balance_Ratio'] = (summary['Market_Balance'] / summary['Total_Allocations'] 
                                         if summary['Total_Allocations'] > 0 else 0.0)
        
        # Calculate average costs
        if summary['Total_Abatement'] > 0:
            summary['Average_Abatement_Cost'] = summary['Total_Abatement_Cost'] / summary['Total_Abatement']
        else:
            summary['Average_Abatement_Cost'] = 0
            
        if summary['Total_Trade_Volume'] > 0:
            summary['Average_Trade_Price'] = summary['Total_Trade_Cost'] / summary['Total_Trade_Volume']
        else:
            summary['Average_Trade_Price'] = 0
        
        return summary

    def _save_results_to_files(self, market_summary: pd.DataFrame, 
                                  sector_summary: pd.DataFrame,
                                  facility_results: pd.DataFrame, 
                                  output_file: str) -> None:
            """
            Internal method to save results to files.
            
            Args:
                market_summary: DataFrame with market-level metrics
                sector_summary: DataFrame with sector-level metrics
                facility_results: DataFrame with facility-level results
                output_file: Base name for output files
            """
            try:
                # Save market summary
                market_file = f"market_summary_{output_file}"
                market_summary.to_csv(market_file, index=False)
                print(f"Market summary saved to {market_file}")
                
                # Save sector summary if not empty
                if not sector_summary.empty:
                    sector_file = f"sector_summary_{output_file}"
                    sector_summary.to_csv(sector_file, index=False)
                    print(f"Sector summary saved to {sector_file}")
                
                # Save facility results
                facility_results.to_csv(output_file, index=False)
                print(f"Facility results saved to {output_file}")
                
            except Exception as e:
                print(f"Error saving results: {str(e)}")

    def _create_scenario_comparison(self, scenario_results: List[Dict]) -> pd.DataFrame:
        """
        Create comparative analysis of different scenario results.
        
        Args:
            scenario_results: List of dictionaries containing scenario results
            
        Returns:
            DataFrame with scenario comparisons
        """
        comparisons = []
        
        for result in scenario_results:
            summary = result['market_summary']
            
            comparison = {
                'Scenario': result['type'],
                'Average Price': summary['Market_Price'].mean(),
                'Final Price': summary['Market_Price'].iloc[-1],
                'Total Abatement': summary['Total_Abatement'].sum(),
                'Average Annual Abatement': summary['Total_Abatement'].mean(),
                'Cumulative Emissions': summary['Total_Emissions'].sum(),
                'Final Year Emissions': summary['Total_Emissions'].iloc[-1],
                'Total Compliance Cost': summary['Total_Compliance_Cost'].sum(),
                'Average Annual Cost': summary['Total_Compliance_Cost'].mean(),
                'Cost Effectiveness': (
                    summary['Total_Compliance_Cost'].sum() / 
                    summary['Total_Abatement'].sum()
                ) if summary['Total_Abatement'].sum() > 0 else float('inf'),
                'Average Market Balance': summary['Market_Balance_Ratio'].mean(),
                'Price Stability': result['metrics']['stability'],
                'Market Balance': result['metrics']['balance']
            }
            comparisons.append(comparison)
        
        return pd.DataFrame(comparisons)

    def _create_detailed_analysis(self, scenario_results: List[Dict]) -> pd.DataFrame:
        """
        Create detailed year-by-year analysis of scenario results.
        
        Args:
            scenario_results: List of dictionaries containing scenario results
            
        Returns:
            DataFrame with detailed scenario analysis
        """
        detailed_records = []
        
        for result in scenario_results:
            market_data = result['market_summary']
            
            for _, row in market_data.iterrows():
                record = {
                    'Scenario': result['type'],
                    'Year': row['Year'],
                    'Market_Price': row['Market_Price'],
                    'Total_Emissions': row['Total_Emissions'],
                    'Total_Abatement': row['Total_Abatement'],
                    'Market_Balance_Ratio': row['Market_Balance_Ratio'],
                    'Compliance_Cost': row['Total_Compliance_Cost'],
                    'Trade_Volume': row.get('Total_Trade_Volume', 0),
                    'Stability_Metric': result['metrics']['stability'],
                    'Balance_Metric': result['metrics']['balance']
                }
                detailed_records.append(record)
        
        return pd.DataFrame(detailed_records)

    
