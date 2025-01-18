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
        
        # Growth rates and benchmark parameters
        self.output_growth_rate = scenario_params.get("output_growth_rate", 0.02)
        self.emissions_growth_rate = scenario_params.get("emissions_growth_rate", 0.01)
        self.benchmark_ratchet_rate = scenario_params.get("benchmark_ratchet_rate", 0.02)  # Added this line
        
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
        print(f"Benchmark ratchet rate: {self.benchmark_ratchet_rate:.4f}")  # Added this line
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
    def calculate_dynamic_values(self, year: int) -> None:
        """Calculate base values with corrected surplus/deficit tracking."""
        print(f"\n=== Calculating Dynamic Values for Year {year} ===")
        years_elapsed = year - self.start_year
        
        # 1. Calculate output and emissions growth
        self.facilities_data[f'Output_{year}'] = (
            self.facilities_data['Baseline Output'] *
            (1 + self.output_growth_rate) ** years_elapsed
        )
        
        self.facilities_data[f'Emissions_{year}'] = (
            self.facilities_data['Baseline Emissions'] *
            (1 + self.emissions_growth_rate) ** years_elapsed
        )
        
        # 2. Apply benchmark ratchet to baseline benchmarks
        self.facilities_data[f'Benchmark_{year}'] = (
            self.facilities_data['Baseline Benchmark'] * 
            (1 - self.benchmark_ratchet_rate) ** years_elapsed
        )
        
        # 3. Calculate allocations
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * 
            self.facilities_data[f'Benchmark_{year}']
        )
        
        # 4. Calculate initial surplus/deficit positions 
        # Key fix: Ensure this calculation happens and verify the results
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - 
            self.facilities_data[f'Emissions_{year}']
        )
        
        # Add detailed verification
        print("\nAllowance Position Verification:")
        sample_size = min(5, len(self.facilities_data))
        sample = self.facilities_data.head(sample_size)
        
        for _, facility in sample.iterrows():
            fid = facility['Facility ID']
            allocations = facility[f'Allocations_{year}']
            emissions = facility[f'Emissions_{year}']
            surplus_deficit = facility[f'Allowance Surplus/Deficit_{year}']
            
            print(f"\nFacility {fid}:")
            print(f"  Allocations:      {allocations:,.2f}")
            print(f"  Emissions:        {emissions:,.2f}")
            print(f"  Surplus/Deficit:  {surplus_deficit:,.2f}")
            print(f"  Verification:     {(allocations - emissions):,.2f}")
        
        # Market-wide verification
        total_allocations = self.facilities_data[f'Allocations_{year}'].sum()
        total_emissions = self.facilities_data[f'Emissions_{year}'].sum()
        total_surplus_deficit = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].sum()
        
        print("\nMarket-wide Position:")
        print(f"Total Allocations:     {total_allocations:,.2f}")
        print(f"Total Emissions:       {total_emissions:,.2f}")
        print(f"Net Surplus/Deficit:   {total_surplus_deficit:,.2f}")
        print(f"Verification:          {(total_allocations - total_emissions):,.2f}")
        
        # Distribution analysis
        surplus_count = (self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0).sum()
        deficit_count = (self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0).sum()
        neutral_count = (self.facilities_data[f'Allowance Surplus/Deficit_{year}'] == 0).sum()
        
        print("\nPosition Distribution:")
        print(f"Facilities in Surplus: {surplus_count}")
        print(f"Facilities in Deficit: {deficit_count}")
        print(f"Facilities Neutral:    {neutral_count}")
        
        # Add check for zeros
        zero_positions = self.facilities_data[
            self.facilities_data[f'Allowance Surplus/Deficit_{year}'] == 0
        ]
        if not zero_positions.empty:
            print("\nWARNING: Found facilities with zero surplus/deficit!")
            print("Sample of facilities with zero positions:")
            print(zero_positions[['Facility ID', f'Allocations_{year}', f'Emissions_{year}']].head())
        
    def run_model(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute model with surplus/deficit verification at each step."""
        print("\nExecuting emission trading model...")
        market_summary = []
        sector_summaries = []
        
        for year in range(self.start_year, self.end_year + 1):
            print(f"\nProcessing year {year}")
            try:
                # Verify surplus/deficit at each major step
                def verify_positions(step_name: str):
                    total_surplus_deficit = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].sum()
                    zero_count = (self.facilities_data[f'Allowance Surplus/Deficit_{year}'] == 0).sum()
                    print(f"\nPosition check after {step_name}:")
                    print(f"Net Surplus/Deficit: {total_surplus_deficit:,.2f}")
                    print(f"Facilities with zero position: {zero_count}")
                
                # 1. Calculate base values
                self.calculate_dynamic_values(year)
                verify_positions("initial calculation")
                
                # 2. First market clearing
                supply, demand = self.calculate_market_positions(year)
                self.determine_market_price(supply, demand, year)
                verify_positions("market clearing")
                
                # 3. Abatement response
                self.calculate_abatement(year)
                verify_positions("abatement")
                
                # 4. Final market clearing
                final_supply, final_demand = self.calculate_market_positions(year)
                self.determine_market_price(final_supply, final_demand, year)
                verify_positions("final market clearing")
                
                # 5. Trading and costs
                self.trade_allowances(year)
                verify_positions("trading")
                
                self.calculate_costs(year)
                verify_positions("cost calculation")
                
                # 6. Record results
                market_summary.append(self._create_market_summary(year))
                sector_summaries.append(self.create_sector_summary(year))
                
                print(f"Year {year} completed")
                verify_positions("year end")
                
            except Exception as e:
                print(f"Error in year {year}: {str(e)}")
                break
        
        # Convert to DataFrames
        market_summary_df = pd.DataFrame(market_summary)
        sector_summary_df = pd.concat(sector_summaries, ignore_index=True)
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
        """Calculate base values with corrected surplus/deficit tracking."""
        print(f"\n=== Calculating Dynamic Values for Year {year} ===")
        years_elapsed = year - self.start_year
        
        # 1. Calculate output and emissions growth
        self.facilities_data[f'Output_{year}'] = (
            self.facilities_data['Baseline Output'] *
            (1 + self.output_growth_rate) ** years_elapsed
        )
        
        self.facilities_data[f'Emissions_{year}'] = (
            self.facilities_data['Baseline Emissions'] *
            (1 + self.emissions_growth_rate) ** years_elapsed
        )
        
        # 2. Apply benchmark ratchet to baseline benchmarks
        self.facilities_data[f'Benchmark_{year}'] = (
            self.facilities_data['Baseline Benchmark'] * 
            (1 - self.benchmark_ratchet_rate) ** years_elapsed
        )
        
        # 3. Calculate allocations
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * 
            self.facilities_data[f'Benchmark_{year}']
        )
        
        # 4. Calculate initial surplus/deficit positions 
        # Key fix: Ensure this calculation happens and verify the results
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - 
            self.facilities_data[f'Emissions_{year}']
        )
        
        # Add detailed verification
        print("\nAllowance Position Verification:")
        sample_size = min(5, len(self.facilities_data))
        sample = self.facilities_data.head(sample_size)
        
        for _, facility in sample.iterrows():
            fid = facility['Facility ID']
            allocations = facility[f'Allocations_{year}']
            emissions = facility[f'Emissions_{year}']
            surplus_deficit = facility[f'Allowance Surplus/Deficit_{year}']
            
            print(f"\nFacility {fid}:")
            print(f"  Allocations:      {allocations:,.2f}")
            print(f"  Emissions:        {emissions:,.2f}")
            print(f"  Surplus/Deficit:  {surplus_deficit:,.2f}")
            print(f"  Verification:     {(allocations - emissions):,.2f}")
        
        # Market-wide verification
        total_allocations = self.facilities_data[f'Allocations_{year}'].sum()
        total_emissions = self.facilities_data[f'Emissions_{year}'].sum()
        total_surplus_deficit = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].sum()
        
        print("\nMarket-wide Position:")
        print(f"Total Allocations:     {total_allocations:,.2f}")
        print(f"Total Emissions:       {total_emissions:,.2f}")
        print(f"Net Surplus/Deficit:   {total_surplus_deficit:,.2f}")
        print(f"Verification:          {(total_allocations - total_emissions):,.2f}")
        
        # Distribution analysis
        surplus_count = (self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0).sum()
        deficit_count = (self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0).sum()
        neutral_count = (self.facilities_data[f'Allowance Surplus/Deficit_{year}'] == 0).sum()
        
        print("\nPosition Distribution:")
        print(f"Facilities in Surplus: {surplus_count}")
        print(f"Facilities in Deficit: {deficit_count}")
        print(f"Facilities Neutral:    {neutral_count}")
        
        # Add check for zeros
        zero_positions = self.facilities_data[
            self.facilities_data[f'Allowance Surplus/Deficit_{year}'] == 0
        ]
        if not zero_positions.empty:
            print("\nWARNING: Found facilities with zero surplus/deficit!")
            print("Sample of facilities with zero positions:")
            print(zero_positions[['Facility ID', f'Allocations_{year}', f'Emissions_{year}']].head())

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
        """Calculate market positions with better logging."""
        print("\nCalculating market positions")
        positions = self.facilities_data[f'Allowance Surplus/Deficit_{year}']
        
        total_supply = positions[positions > 0].sum()
        total_demand = abs(positions[positions < 0].sum())
        
        print(f"Total supply: {total_supply:.2f}")
        print(f"Total demand: {total_demand:.2f}")
        
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
        """Determine market price from marginal abatement cost curves and market balance."""
        print(f"\n=== Market Price Determination for Year {year} ===")
        print(f"Supply: {supply:.2f}")
        print(f"Demand: {demand:.2f}")
        print(f"Net Shortage: {(demand - supply):.2f}")
        
        # If market is balanced/long, start from floor but consider selling costs
        if supply >= demand:
            # Find minimum profitable selling price
            min_sell_price = float('inf')
            for _, facility in self.facilities_data.iterrows():
                if facility[f'Allowance Surplus/Deficit_{year}'] > 0:
                    curve = self.abatement_cost_curve[
                        self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
                    ]
                    if not curve.empty:
                        min_sell_price = min(min_sell_price, float(curve.iloc[0]['Intercept']))
            
            clearing_price = max(self.floor_price, min_sell_price / 2)
            self.market_price = clearing_price
            print(f"Market is long - price set to: ${clearing_price:.2f}")
            return clearing_price
        
        # Market is short - find clearing price from MAC curves
        needed_abatement = demand - supply
        print(f"Market is short - need {needed_abatement:.2f} abatement")
        
        # Build array of possible clearing prices
        clearing_prices = [self.floor_price]  # Start with floor
        
        # Add MAC-based prices
        for _, facility in self.facilities_data.iterrows():
            # Skip if no emissions to abate
            if facility[f'Emissions_{year}'] <= 0:
                continue
                
            curve = self.abatement_cost_curve[
                self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
            ]
            if curve.empty:
                continue
                
            curve = curve.iloc[0]
            intercept = float(curve['Intercept'])
            slope = float(curve['Slope'])
            max_reduction = min(
                float(curve['Max Reduction (MTCO2e)']),
                facility[f'Emissions_{year}']
            )
            
            # Add facility's MAC curve prices
            if slope > 0:
                # Price at start of abatement
                if intercept > 0:
                    clearing_prices.append(intercept)
                
                # Price at max abatement
                max_price = intercept + (slope * max_reduction)
                if max_price <= self.ceiling_price:
                    clearing_prices.append(max_price)
                
                # Add intermediate points
                for pct in [0.25, 0.5, 0.75]:
                    price = intercept + (slope * max_reduction * pct)
                    if price <= self.ceiling_price:
                        clearing_prices.append(price)
        
        # Add ceiling price and some points below
        clearing_prices.append(self.ceiling_price)
        
        # Sort and deduplicate
        clearing_prices = sorted(set(clearing_prices))
        print(f"\nTesting {len(clearing_prices)} price points...")
        
        # Find clearing price
        best_price = self.ceiling_price
        best_gap = float('inf')
        
        for price in clearing_prices:
            potential_abatement = 0
            abatement_costs = []
            
            # Calculate total abatement at this price
            for _, facility in self.facilities_data.iterrows():
                # Skip if no emissions
                if facility[f'Emissions_{year}'] <= 0:
                    continue
                    
                curve = self.abatement_cost_curve[
                    self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
                ]
                if curve.empty:
                    continue
                    
                curve = curve.iloc[0]
                slope = float(curve['Slope'])
                intercept = float(curve['Intercept'])
                max_reduction = min(
                    float(curve['Max Reduction (MTCO2e)']),
                    facility[f'Emissions_{year}']
                )
                
                if slope > 0 and price > intercept:
                    # Calculate economic abatement
                    econ_abatement = min(
                        max_reduction,
                        (price - intercept) / slope
                    )
                    
                    if econ_abatement > 0:
                        # Calculate average cost
                        total_cost = (slope * econ_abatement * econ_abatement / 2) + (intercept * econ_abatement)
                        avg_cost = total_cost / econ_abatement
                        
                        # Only count if profitable
                        if avg_cost < price:
                            potential_abatement += econ_abatement
                            abatement_costs.append(avg_cost)
            
            # Check how well this price balances the market
            gap = abs(potential_abatement - needed_abatement)
            print(f"Price ${price:.2f} -> Abatement: {potential_abatement:.2f} (gap: {gap:.2f})")
            
            if gap < best_gap:
                best_gap = gap
                best_price = price
                
            # Stop if we've found sufficient abatement
            if potential_abatement >= needed_abatement:
                break
        
        # Set final price
        self.market_price = best_price
        print(f"\nFinal Price Analysis:")
        print(f"Clearing Price: ${best_price:.2f}")
        print(f"Target Abatement: {needed_abatement:.2f}")
        print(f"Price Gap: {best_gap:.2f}")
        
        return best_price
    
    def _build_mac_curve(self, year: int) -> List[float]:
        """Build MAC curve with facility-specific characteristics."""
        mac_points = []
        
        print("\nBuilding MAC Curve...")
        for _, facility in self.facilities_data.iterrows():
            # Skip if no emissions
            if facility[f'Emissions_{year}'] <= 0:
                continue
                
            curve = self.abatement_cost_curve[
                self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
            ]
            if curve.empty:
                continue
                
            curve = curve.iloc[0]
            slope = float(curve['Slope'])
            intercept = float(curve['Intercept'])
            max_reduction = min(
                float(curve['Max Reduction (MTCO2e)']),
                facility[f'Emissions_{year}']
            )
            
            if slope > 0:
                # Generate more granular points
                for pct in range(5, 101, 5):  # Every 5%
                    abatement = max_reduction * (pct/100)
                    mac = intercept + (slope * abatement)
                    
                    if mac > 0 and mac <= self.ceiling_price:
                        mac_points.append(mac)
                        
                print(f"Facility {facility['Facility ID']}:")
                print(f"  Slope: {slope:.4f}")
                print(f"  Intercept: {intercept:.2f}")
                print(f"  Max Reduction: {max_reduction:.2f}")
        
        mac_points.sort()
        if mac_points:
            print(f"\nMAC Points: {len(mac_points)}")
            print(f"Range: ${min(mac_points):.2f} - ${max(mac_points):.2f}")
        
        return mac_points  

    def calculate_abatement(self, year: int) -> None:
        """Calculate profit-maximizing abatement for each facility and update emissions."""
        print(f"\n=== Abatement Analysis for Year {year} ===")
        print(f"Market Price: ${self.market_price:.2f}")
        
        total_abatement = 0.0
        total_cost = 0.0
        
        for idx, facility in self.facilities_data.iterrows():
            print(f"\nAnalyzing Facility {facility['Facility ID']}")
            
            # Get facility's MAC curve
            curve = self.abatement_cost_curve[
                self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
            ]
            if curve.empty:
                print("  No MAC curve found")
                continue
                
            curve = curve.iloc[0]
            emissions = facility[f'Emissions_{year}']
            if emissions <= 0:
                print("  No emissions to abate")
                continue
                
            # Get MAC curve parameters
            slope = float(curve['Slope'])
            intercept = float(curve['Intercept'])
            max_reduction = float(curve['Max Reduction (MTCO2e)'])
            
            print(f"  MAC Parameters:")
            print(f"  - Slope: {slope:.4f}")
            print(f"  - Intercept: ${intercept:.2f}")
            print(f"  - Max Reduction: {max_reduction:.2f}")
            print(f"  - Current Emissions: {emissions:.2f}")
            
            if slope <= 0:
                print("  Invalid MAC curve (slope <= 0)")
                continue
                
            # Calculate optimal abatement
            optimal_quantity = (self.market_price - intercept) / slope
            
            # Bound by constraints
            bounded_quantity = min(
                optimal_quantity,
                max_reduction,
                emissions
            )
            
            if bounded_quantity <= 0:
                print("  No profitable abatement")
                continue
                
            # Calculate total and average costs
            total_abatement_cost = (
                (slope * bounded_quantity * bounded_quantity / 2) + 
                (intercept * bounded_quantity)
            )
            avg_cost = total_abatement_cost / bounded_quantity
            
            # Only abate if profitable vs. market price
            if avg_cost < self.market_price:
                print(f"  Profitable Abatement Found:")
                print(f"  - Quantity: {bounded_quantity:.2f}")
                print(f"  - Total Cost: ${total_abatement_cost:.2f}")
                print(f"  - Average Cost: ${avg_cost:.2f}/unit")
                print(f"  - Market Price: ${self.market_price:.2f}/unit")
                print(f"  - Profit/unit: ${(self.market_price - avg_cost):.2f}")
                
                # Execute abatement and update emissions
                self.facilities_data.at[idx, f'Tonnes Abated_{year}'] = bounded_quantity
                self.facilities_data.at[idx, f'Abatement Cost_{year}'] = total_abatement_cost
                self.facilities_data.at[idx, f'Allowance Surplus/Deficit_{year}'] += bounded_quantity
                
                # Critical fix: Update emissions to reflect abatement
                self.facilities_data.at[idx, f'Emissions_{year}'] -= bounded_quantity
                
                total_abatement += bounded_quantity
                total_cost += total_abatement_cost
            else:
                print("  Abatement not profitable vs. market price")
                
        # Print summary
        if total_abatement > 0:
            print(f"\nMarket Summary:")
            print(f"Total Abatement: {total_abatement:.2f} tonnes")
            print(f"Total Cost: ${total_cost:.2f}")
            print(f"Average Cost: ${(total_cost/total_abatement):.2f}/tonne")
            print(f"Market Price: ${self.market_price:.2f}/tonne")
            
            # Print emissions impact
            total_emissions = self.facilities_data[f'Emissions_{year}'].sum()
            print(f"Total Emissions After Abatement: {total_emissions:.2f} tonnes")
        else:
            print("\nNo profitable abatement at current market price")
            
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
        """Execute trades with detailed cost tracking."""
        print(f"\n=== TRADE EXECUTION DEBUG - Year {year} ===")
        print(f"Current Market Price: ${self.market_price:.2f}")
        
        # 1. Setup tracking
        class TradeTracker:
            def __init__(self):
                self.total_volume = 0.0
                self.total_cost = 0.0
                self.trades = []
                
            def add_trade(self, buyer_id, seller_id, volume, price):
                cost = volume * price
                self.trades.append({
                    'buyer': buyer_id,
                    'seller': seller_id,
                    'volume': volume,
                    'price': price,
                    'cost': cost
                })
                self.total_volume += volume
                self.total_cost += cost
                
            def print_summary(self):
                print("\nDetailed Trade Log:")
                for t in self.trades:
                    print(f"Trade: {t['volume']:.2f} units @ ${t['price']:.2f} = ${t['cost']:.2f}")
                    print(f"  Buyer: {t['buyer']}, Seller: {t['seller']}")
                print(f"\nTotal Volume: {self.total_volume:.2f}")
                print(f"Total Cost: ${self.total_cost:.2f}")
                print(f"Average Price: ${(self.total_cost/self.total_volume if self.total_volume > 0 else 0):.2f}")
        
        tracker = TradeTracker()
        
        # 2. Initialize columns if needed
        trade_columns = [
            f'Trade Volume_{year}',
            f'Allowance Purchase Cost_{year}',
            f'Allowance Sales Revenue_{year}',
            f'Trade Cost_{year}'
        ]
        for col in trade_columns:
            if col not in self.facilities_data.columns:
                print(f"Creating column: {col}")
                self.facilities_data[col] = 0.0
            else:
                print(f"Resetting column: {col}")
                self.facilities_data[col] = 0.0
                
        # 3. Get market positions
        positions = self.facilities_data[f'Allowance Surplus/Deficit_{year}']
        print("\nInitial Positions:")
        print(f"Total Short: {abs(positions[positions < 0].sum()):.2f}")
        print(f"Total Long: {positions[positions > 0].sum():.2f}")
        
        # 4. Execute trades
        MIN_TRADE = 0.0001
        buyers = self.facilities_data[positions < -MIN_TRADE].copy()
        sellers = self.facilities_data[positions > MIN_TRADE].copy()
        
        for _, buyer in buyers.iterrows():
            demand = abs(buyer[f'Allowance Surplus/Deficit_{year}'])
            remaining_demand = demand
            
            for seller_idx, seller in sellers.iterrows():
                supply = seller[f'Allowance Surplus/Deficit_{year}']
                if supply <= MIN_TRADE:
                    continue
                    
                # Calculate trade
                volume = min(remaining_demand, supply)
                if volume < MIN_TRADE:
                    continue
                    
                # Execute trade and track
                cost = volume * self.market_price
                print(f"\nExecuting Trade:")
                print(f"Volume: {volume:.4f}")
                print(f"Price: ${self.market_price:.2f}")
                print(f"Cost: ${cost:.2f}")
                
                # Update buyer
                print("\nUpdating Buyer:")
                print(f"Before - Volume: {self.facilities_data.at[buyer.name, f'Trade Volume_{year}']:.4f}")
                print(f"Before - Cost: {self.facilities_data.at[buyer.name, f'Allowance Purchase Cost_{year}']:.2f}")
                
                self.facilities_data.at[buyer.name, f'Allowance Surplus/Deficit_{year}'] += volume
                self.facilities_data.at[buyer.name, f'Trade Volume_{year}'] += volume
                self.facilities_data.at[buyer.name, f'Allowance Purchase Cost_{year}'] += cost
                self.facilities_data.at[buyer.name, f'Trade Cost_{year}'] += cost
                
                print(f"After - Volume: {self.facilities_data.at[buyer.name, f'Trade Volume_{year}']:.4f}")
                print(f"After - Cost: {self.facilities_data.at[buyer.name, f'Allowance Purchase Cost_{year}']:.2f}")
                
                # Update seller
                print("\nUpdating Seller:")
                print(f"Before - Volume: {self.facilities_data.at[seller_idx, f'Trade Volume_{year}']:.4f}")
                print(f"Before - Revenue: {self.facilities_data.at[seller_idx, f'Allowance Sales Revenue_{year}']:.2f}")
                
                self.facilities_data.at[seller_idx, f'Allowance Surplus/Deficit_{year}'] -= volume
                self.facilities_data.at[seller_idx, f'Trade Volume_{year}'] += volume
                self.facilities_data.at[seller_idx, f'Allowance Sales Revenue_{year}'] += cost
                
                print(f"After - Volume: {self.facilities_data.at[seller_idx, f'Trade Volume_{year}']:.4f}")
                print(f"After - Revenue: {self.facilities_data.at[seller_idx, f'Allowance Sales Revenue_{year}']:.2f}")
                
                # Track trade
                tracker.add_trade(buyer['Facility ID'], seller['Facility ID'], volume, self.market_price)
                
                # Update remaining demand
                remaining_demand -= volume
                if remaining_demand <= MIN_TRADE:
                    break
                    
                # Update seller's remaining supply
                sellers.loc[seller_idx, f'Allowance Surplus/Deficit_{year}'] -= volume
                if sellers.loc[seller_idx, f'Allowance Surplus/Deficit_{year}'] <= MIN_TRADE:
                    continue
        
        # 5. Print final verification
        tracker.print_summary()
        
        # Verify trade costs in data
        total_purchase_costs = self.facilities_data[f'Allowance Purchase Cost_{year}'].sum()
        total_sales_revenue = self.facilities_data[f'Allowance Sales Revenue_{year}'].sum()
        total_trade_volume = self.facilities_data[f'Trade Volume_{year}'].sum() / 2  # Divide by 2 as both sides count
        
        print("\nFinal Data Verification:")
        print(f"Total Purchase Costs: ${total_purchase_costs:.2f}")
        print(f"Total Sales Revenue: ${total_sales_revenue:.2f}")
        print(f"Total Trade Volume: {total_trade_volume:.2f}")
        if total_trade_volume > 0:
            print(f"Implied Average Price: ${total_purchase_costs/total_trade_volume:.2f}")
        
        # 6. Check final positions
        final_positions = self.facilities_data[f'Allowance Surplus/Deficit_{year}']
        final_short = abs(final_positions[final_positions < 0].sum())
        final_long = final_positions[final_positions > 0].sum()
        
        print("\nFinal Positions:")
        print(f"Final Short: {final_short:.2f}")
        print(f"Final Long: {final_long:.2f}")
        print(f"Final Net: {(final_long - final_short):.2f}")

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

    
