
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
from typing import List, Dict, Tuple
from pathlib import Path

class obamodel:
    # 1. Initialization and Setup
    @staticmethod
    def load_all_scenarios(scenario_file: str) -> List[Dict]:
        """Load and validate scenarios from CSV file."""
        print(f"Loading scenarios from file: {scenario_file}")
        try:
            # Read scenario file
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
                'MSR Adjustment Rate': (0, 1),
                'Historical Bank Share': (0, 1),  
                'Min Bank Price Ratio': (0, 1),
                'Vintage Limit': (0, 10)
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
                scenario = {
                    "name": row["Scenario"],
                    "floor_price": float(row["Floor Price"]),
                    "ceiling_price": float(row["Ceiling Price"]),
                    "price_increment": float(row["Price Increment"]), 
                    "output_growth_rate": float(row["Output Growth Rate"]),
                    "emissions_growth_rate": float(row["Emissions Growth Rate"]),
                    "benchmark_ratchet_rate": float(row["Benchmark Ratchet Rate"]),
                    "msr_active": bool(row["MSR Active"]),
                    "msr_upper_threshold": float(row["MSR Upper Threshold"]),
                    "msr_lower_threshold": float(row["MSR Lower Threshold"]),
                    "msr_adjustment_rate": float(row["MSR Adjustment Rate"]),
                    "historical_bank_share": float(row["Historical Bank Share"]),
                    "min_bank_price_ratio": float(row["Min Bank Price Ratio"]), 
                    "vintage_limit": int(row["Vintage Limit"])
                }
                
                scenario_list.append(scenario)
            
            print(f"Successfully loaded {len(scenario_list)} scenarios")
            return scenario_list
            
        except pd.errors.EmptyDataError:
            print("Error: Scenario file is empty")
            raise
        except FileNotFoundError:
            print(f"Error: Could not find scenario file: {scenario_file}")
            raise
        except Exception as e:
            print(f"Error loading scenarios: {e}")
            raise

    def __init__(self, facilities_data: pd.DataFrame, abatement_cost_curve: pd.DataFrame, 
                 start_year: int, end_year: int, scenario_params: Dict):
        """Initialize OBA model with market-driven pricing."""
    
          # Ensure scenario_params is a dictionary
        if not isinstance(scenario_params, dict):
            raise TypeError(f"Expected scenario_params to be dict, got {type(scenario_params)}")
    
        # Store scenario parameters as an instance attribute
        self.scenario_params = scenario_params
    
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
    
        self.floor_price = scenario_params.get("floor_price", 20)
        self.ceiling_price = scenario_params.get("ceiling_price", 200)
        self.price_increment = scenario_params.get("price_increment", 10)  # Get from scenario
        self.market_price = self.floor_price  # Initialize market price
        self.historical_bank_share = scenario_params.get('historical_bank_share', 0)
        self.min_bank_price_ratio = scenario_params.get('min_bank_price_ratio', 0)
        self.vintage_limit = scenario_params.get("vintage_limit", 0)
    
        print("\nPrice Control Parameters:")
        print(f"Floor Price: ${self.floor_price:.2f}")
        print(f"Base Ceiling Price: ${self.ceiling_price:.2f}")
        print(f"Price Increment: ${self.price_increment:.2f}/year")
        print(f"Initial Market Price: ${self.market_price:.2f}")
    
        # Verify facility data
        print("\nFacility Data Verification:")
        print(f"Number of facilities: {len(self.facilities_data)}")
        print("\nTotal Baseline Values:")
        print(f"Total Baseline Emissions: {self.facilities_data['Baseline Emissions'].sum():.4f}")
        print(f"Total Baseline Output: {self.facilities_data['Baseline Output'].sum():.4f}")
    
        # Sample verification
        print("\nSample Facility Data:")
        sample = self.facilities_data.head()
        print(sample[['Facility ID', 'Baseline Emissions', 'Baseline Output', 'Baseline Benchmark']])
    
        # Store scenario parameters
        self.output_growth_rate = scenario_params.get("output_growth_rate", 0.00)
        self.emissions_growth_rate = scenario_params.get("emissions_growth_rate", 0.00)
        self.benchmark_ratchet_rate = scenario_params.get("benchmark_ratchet_rate", 0.00)
    
        # MSR parameters
        self.msr_active = scenario_params.get("msr_active", False)
        self.msr_upper_threshold = scenario_params.get("msr_upper_threshold", 0.15)
        self.msr_lower_threshold = scenario_params.get("msr_lower_threshold", -0.05)
        self.msr_adjustment_rate = scenario_params.get("msr_adjustment_rate", 0.03)
    
        # Initialize model columns
        self._initialize_columns()
    
        # Initialize bank balances
        self._initialize_bank_balances()
    
        # Initialize historical bank
        self.initialize_historical_bank()
    
        print("\nModel Parameters:")
        print(f"Start Year: {start_year}")
        print(f"End Year: {end_year}")
        print(f"Price Range: ${self.floor_price} - ${self.ceiling_price}")
        print(f"Benchmark Ratchet Rate: {self.benchmark_ratchet_rate:.4f}")
                 
    def calculate_initial_period(self, year: int) -> None:
        """Calculate first period values with direct allocation calculation."""
        print(f"\n=== First Period Calculation (Year {year}) ===")
        
        # Set initial output, emissions and benchmark
        self.facilities_data[f'Output_{year}'] = self.facilities_data['Baseline Output']
        self.facilities_data[f'Emissions_{year}'] = self.facilities_data['Baseline Emissions']
        self.facilities_data[f'Benchmark_{year}'] = self.facilities_data['Baseline Benchmark']
        
        # Calculate allocations directly
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * 
            self.facilities_data[f'Benchmark_{year}']
        )
        
        # Calculate initial positions
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - 
            self.facilities_data[f'Emissions_{year}']
        )
        
        # Verify calculations
        print("\nFirst Period Verification:")
        print("\nSample Calculations:")
        sample_size = min(5, len(self.facilities_data))
        for _, facility in self.facilities_data.head(sample_size).iterrows():
            print(f"\nFacility: {facility['Facility ID']}")
            print(f"Output: {facility[f'Output_{year}']:.6f}")
            print(f"Benchmark: {facility[f'Benchmark_{year}']:.6f}")
            print(f"Calculated Allocation: {facility[f'Allocations_{year}']:.6f}")
            manual_calc = facility[f'Output_{year}'] * facility[f'Benchmark_{year}']
            print(f"Manual Check: {manual_calc:.6f}")
            print(f"Difference: {abs(facility[f'Allocations_{year}'] - manual_calc):.10f}")
            
        # Print summary statistics
        print("\nMarket Summary:")
        summary = {
            'Total Output': self.facilities_data[f'Output_{year}'].sum(),
            'Total Emissions': self.facilities_data[f'Emissions_{year}'].sum(),
            'Total Allocations': self.facilities_data[f'Allocations_{year}'].sum(),
            'Net Position': self.facilities_data[f'Allowance Surplus/Deficit_{year}'].sum()
        }
        
        for metric, value in summary.items():
            print(f"{metric}: {value:.2f}")
            
        # Verify all allocations are Output * Benchmark
        self.facilities_data['Verification'] = (
            self.facilities_data[f'Output_{year}'] * 
            self.facilities_data[f'Benchmark_{year}']
        )
        
        discrepancies = self.facilities_data[
            abs(self.facilities_data[f'Allocations_{year}'] - 
                self.facilities_data['Verification']) > 1e-10
        ]
        
        if not discrepancies.empty:
            print("\nWARNING: Found allocation discrepancies:")
            for _, row in discrepancies.iterrows():
                print(f"\nFacility: {row['Facility ID']}")
                print(f"Actual Allocation: {row[f'Allocations_{year}']:.6f}")
                print(f"Expected Allocation: {row['Verification']:.6f}")
                
        self.facilities_data.drop('Verification', axis=1, inplace=True)
        
        # Print sector-level summary
        print("\nSector Summary:")
        sector_summary = self.facilities_data.groupby('Sector').agg({
            f'Output_{year}': 'sum',
            f'Emissions_{year}': 'sum',
            f'Benchmark_{year}': 'mean',
            f'Allocations_{year}': 'sum',
            f'Allowance Surplus/Deficit_{year}': 'sum'
        }).round(4)
        
        print(sector_summary)

    def analyze_initial_performance(self) -> None:
        """Analyze initial facility performance vs benchmarks."""
        print("\n=== Initial Performance Analysis ===")
        
        # Calculate initial emission intensities
        self.facilities_data['Initial_Intensity'] = (
            self.facilities_data['Baseline Emissions'] / 
            self.facilities_data['Baseline Output']
        )
        
        # Compare against benchmarks
        results = []
        for sector in self.facilities_data['Sector'].unique():
            sector_data = self.facilities_data[self.facilities_data['Sector'] == sector]
            
            baseline_intensity = sector_data['Initial_Intensity'].mean()
            facility_benchmark = sector_data['Baseline Benchmark'].mean()
            scenario_benchmark = self.scenario_benchmarks[sector]
            
            facilities_total = len(sector_data)
            beating_facility = (sector_data['Initial_Intensity'] < sector_data['Baseline Benchmark']).sum()
            beating_scenario = (sector_data['Initial_Intensity'] < self.scenario_benchmarks[sector]).sum()
            
            results.append({
                'Sector': sector,
                'Baseline_Intensity': baseline_intensity,
                'Facility_Benchmark': facility_benchmark,
                'Scenario_Benchmark': scenario_benchmark,
                'Facilities_Total': facilities_total,
                'Beating_Facility_Benchmark': beating_facility,
                'Beating_Scenario_Benchmark': beating_scenario
            })
        
        results_df = pd.DataFrame(results)
        print("\nSector Performance Analysis:")
        print(results_df.round(4))
        
        # Identify potential issues
        print("\nPotential Issues:")
        for _, row in results_df.iterrows():
            if row['Scenario_Benchmark'] > row['Baseline_Intensity']:
                print(f"\nWARNING - {row['Sector']}:")
                print(f"  Scenario benchmark ({row['Scenario_Benchmark']:.4f}) is higher than")
                print(f"  baseline intensity ({row['Baseline_Intensity']:.4f})")
                print("  This may result in excess allocations")
            
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

    # Define the _initialize_bank_balances method
    def _initialize_bank_balances(self):
        """Initialize bank balances for all facilities."""
        for year in range(self.start_year, self.end_year + 1):
            self.facilities_data[f'Historical_Bank_Balance_{year}'] = 0.0
            self.facilities_data[f'New_Bank_Balance_{year}'] = 0.0
            self.facilities_data[f'Historical_Bank_Use_{year}'] = 0.0
            self.facilities_data[f'New_Bank_Use_{year}'] = 0.0

    def initialize_historical_bank(self):
        """Set up historical permit bank at start."""
        total_allocations = self.facilities_data['Baseline Allocations'].sum()

        # Debugging print
        print(f"DEBUG: scenario_params = {self.scenario_params}")

        historical_share = self.scenario_params.get('historical_bank_share', 0)
        min_price_ratio = self.scenario_params.get('min_bank_price_ratio', 0)
        
        self.historical_bank = {
            'volume': total_allocations * self.scenario_params.get('historical_bank_share', 0),
            'vintage': self.start_year - 1,  # Pre-existing credits
            'min_price': self.ceiling_price * min_price_ratio
        }
        
        # Initialize first year balances
        self.facilities_data[f'Historical_Bank_Balance_{self.start_year}'] = (
            self.facilities_data['Baseline Allocations'] / total_allocations * 
            self.historical_bank['volume']
        )

        
    def run_model(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute model with banking capabilities."""
        print("\nExecuting emission trading model with banking...")
    
        # Initialize banking columns if needed
        if hasattr(self, '_initialize_banking_columns'):
            self._initialize_banking_columns()
    
        market_summary = []
        sector_summaries = []
        market_reports = []
        compliance_reports = []
    
        for year in range(self.start_year, self.end_year + 1):
            print(f"\nProcessing year {year}")
            try:
                # 1. Calculate initial positions
                self.calculate_dynamic_values(year)
                
                # 2. Track and enforce bank limits
                self.track_vintage_limits(year)
                
                # 3. Try bank use for compliance
                self.use_banked_permits(year)
                
                # 4. Calculate remaining market needs
                supply, demand = self.calculate_market_positions(year)
                
                # 5. Set market price
                self.market_price = self.determine_market_price(supply, demand, year)
                
                # 6. Calculate economic abatement
                self.calculate_abatement(year)
                
                # 7. Execute trading
                if self.validate_market_price():
                    trade_volume, trade_cost = self.trade_allowances(year)
                
                # 8. Bank remaining surplus
                self.make_banking_decision(year)
                
                # 9. Last resort: ceiling price
                self.apply_ceiling_price_compliance(year)
                
                # 10. Calculate costs
                self.calculate_costs(year)
                
                # 11. Generate reports
                if hasattr(self, 'generate_market_report'):
                    market_report = self.generate_market_report(year)
                    market_reports.append(market_report)
                
                if hasattr(self, 'generate_compliance_report'):
                    compliance_report = self.generate_compliance_report(year)
                    compliance_reports.append(compliance_report)
                
                # 12. Store results
                market_summary.append(self._create_market_summary(year))
                sector_summaries.append(self.create_sector_summary(year))
                
            except Exception as e:
                print(f"Error in year {year}: {str(e)}")
                print("Current DataFrame columns:")
                print(self.facilities_data.columns)                
                raise
    
        # Convert results to DataFrames    
        try:
            market_summary_df = pd.DataFrame(market_summary)
            sector_summary_df = pd.concat(sector_summaries, ignore_index=True) if sector_summaries else pd.DataFrame()
            facility_results = self._prepare_facility_results(self.start_year, self.end_year)
            market_reports_df = pd.concat(market_reports, ignore_index=True) if market_reports else pd.DataFrame()
            compliance_reports_df = pd.concat(compliance_reports, ignore_index=True) if compliance_reports else pd.DataFrame()
            
            return (market_summary_df, sector_summary_df, facility_results, compliance_reports_df, market_reports_df)
        except Exception as e:
            print(f"Error converting results to DataFrames: {str(e)}")
            return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        
    def validate_scenario_parameters(self, scenario_type: str, params: Dict) -> bool:
        """Validate parameters for any scenario type"""
        # Validate base parameters
        base_params = {
            'floor_price': (0, None),
            'ceiling_price': (0, None),
            'price_increment': (0, None),  # Add explicit validation for price_increment
            'output_growth_rate': (-0.5, 0.5),
            'emissions_growth_rate': (-0.5, 0.5),
            'benchmark_ratchet_rate': (0, 1),
            'historical_bank_share': (0, 1),
            'min_bank_price_ratio': (0, 1),
            'vintage_limit': (0, 10)
        }
        
        # Validate each parameter
        for param, (min_val, max_val) in base_params.items():
            value = params.get(param)
            if value is not None:  # Only validate if parameter is provided
                if not isinstance(value, (int, float)):
                    print(f"Parameter {param} must be numeric")
                    return False
                if min_val is not None and value < min_val:
                    print(f"Parameter {param} must be >= {min_val}")
                    return False
                if max_val is not None and value > max_val:
                    print(f"Parameter {param} must be <= {max_val}")
                    return False
        
        # Additional validation for ceiling price relationship
        if 'ceiling_price' in params and 'floor_price' in params:
            if params['ceiling_price'] <= params['floor_price']:
                print("Ceiling price must be greater than floor price")
                return False
        
        return True    

    def run_scenario(self, scenario_type: str, params: Dict = None) -> Dict:
        """Execute any scenario type with specified parameters."""
        print(f"\nExecuting {scenario_type} scenario...")
        print(f"DEBUG: Scenario parameters received: {params}")
        
        
        # Set up base parameters
        if params is None:
            params = {}

        

        # Ensure params is a dictionary
        if not isinstance(params, dict):
            raise TypeError(f"Expected params to be dict, got {type(params)}")
             
        # Store original values to restore later
        original_values = {
            'ceiling_price': self.ceiling_price,
            'price_increment': self.price_increment,
            'output_growth_rate': self.output_growth_rate,
            'emissions_growth_rate': self.emissions_growth_rate,
            'benchmark_ratchet_rate': self.benchmark_ratchet_rate,
            'msr_active': self.msr_active if hasattr(self, 'msr_active') else False,
            'historical_bank_share': self.historical_bank_share if hasattr(self, 'historical_bank_share') else 0.0,
        }
        
        try:
            # Create scenario parameters dictionary
            scenario_params = {
                'floor_price': params.get('floor_price', self.floor_price),
                'ceiling_price': params.get('ceiling_price', self.ceiling_price),
                'price_increment': params.get('price_increment', self.price_increment),
                'output_growth_rate': params.get('output_growth_rate', self.output_growth_rate),
                'emissions_growth_rate': params.get('emissions_growth_rate', self.emissions_growth_rate),
                'benchmark_ratchet_rate': params.get('benchmark_ratchet_rate', 0.02),
                'msr_active': params.get('msr_active', self.msr_active),
                'msr_upper_threshold': params.get('msr_upper_threshold', self.msr_upper_threshold),
                'msr_lower_threshold': params.get('msr_lower_threshold', self.msr_lower_threshold),
                'msr_adjustment_rate': params.get('msr_adjustment_rate', self.msr_adjustment_rate),
                'historical_bank_share': params.get('historical_bank_share', self.historical_bank_share),
                'min_bank_price_ratio': params.get('min_bank_price_ratio', self.min_bank_price_ratio),
                'vintage_limit': params.get('vintage_limit', self.vintage_limit)
            }
    
            # Update class attributes with scenario parameters
            self.ceiling_price = scenario_params['ceiling_price']
            self.price_increment = scenario_params['price_increment']  # Ensure this is set
            self.output_growth_rate = scenario_params['output_growth_rate']
            self.emissions_growth_rate = scenario_params['emissions_growth_rate']
            self.benchmark_ratchet_rate = scenario_params['benchmark_ratchet_rate']
            self.msr_active = scenario_params['msr_active']
            
            # Print scenario settings for verification
            print("\nScenario Parameter Verification:")
            print(f"Ceiling Price: ${self.ceiling_price:.2f}")
            print(f"Price Increment: ${self.price_increment:.2f}/year")
            print(f"Output Growth Rate: {self.output_growth_rate:.3f}")
            print(f"Emissions Growth Rate: {self.emissions_growth_rate:.3f}")
            
            # Run model
            (market_summary, sector_summary, facility_results, 
             compliance_reports, market_reports) = self.run_model()
            
            return {
                'type': scenario_type,
                'parameters': scenario_params,  # Use the full scenario_params
                'market_summary': market_summary,
                'sector_summary': sector_summary,
                'facility_results': facility_results,
                'compliance_reports': compliance_reports,
                'market_reports': market_reports,
                'metrics': {
                    'stability': self._calculate_stability_metric(market_summary),
                    'balance': self._calculate_balance_metric(market_summary)
                }
            }
            
        finally:
            # Restore original values
            for key, value in original_values.items():
                setattr(self, key, value)    
     
       
    def _empty_scenario_result(self, scenario_type: str, params: Dict) -> Dict:
        """Return empty result structure for failed scenarios"""
        return {
            'type': scenario_type,
            'parameters': params,
            'market_summary': pd.DataFrame(),
            'sector_summary': pd.DataFrame(),
            'facility_results': pd.DataFrame(),
            'compliance_reports': pd.DataFrame(),
            'market_reports': pd.DataFrame(),
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
       """Calculate base values with growth rates and abatement."""
       print(f"\n=== Calculating Values for Year {year} ===")
       
       if year == self.start_year:
           # First period - use baseline values
           self.facilities_data[f'Output_{year}'] = self.facilities_data['Baseline Output']
           self.facilities_data[f'Emissions_{year}'] = self.facilities_data['Baseline Emissions']
           
           print("\nFirst Period Values:")
           print(f"Total Output: {self.facilities_data[f'Output_{year}'].sum():.4f}")
           print(f"Total Emissions: {self.facilities_data[f'Emissions_{year}'].sum():.4f}")
           
       else:
           # Subsequent years
           years_elapsed = year - self.start_year
           
           # Only apply growth if rates are non-zero
           output_factor = (1 + self.output_growth_rate) ** years_elapsed if self.output_growth_rate != 0 else 1
           emissions_factor = (1 + self.emissions_growth_rate) ** years_elapsed if self.emissions_growth_rate != 0 else 1
           
           print(f"\nYear {year} Growth Factors:")
           print(f"Output Growth: {output_factor:.4f}")
           print(f"Emissions Growth: {emissions_factor:.4f}")
           
           # Apply growth factors
           self.facilities_data[f'Output_{year}'] = (
               self.facilities_data['Baseline Output'] * output_factor
           )
           
           # Calculate emissions with growth and subtract previous year's abatement
           base_emissions = self.facilities_data['Baseline Emissions'] * emissions_factor
           prev_abatement = self.facilities_data[f'Tonnes Abated_{year-1}'].fillna(0)
           self.facilities_data[f'Emissions_{year}'] = base_emissions - prev_abatement
       
       # Calculate benchmarks with ratchet
       years_of_ratchet = year - (self.start_year - 1)
       ratchet_factor = (1 - self.benchmark_ratchet_rate) ** years_of_ratchet if self.benchmark_ratchet_rate != 0 else 1
       
       self.facilities_data[f'Benchmark_{year}'] = (
           self.facilities_data['Baseline Benchmark'] * ratchet_factor
       )
       
       # Calculate allocations and positions
       self.facilities_data[f'Allocations_{year}'] = (
           self.facilities_data[f'Output_{year}'] * 
           self.facilities_data[f'Benchmark_{year}']
       )
       
       self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
           self.facilities_data[f'Allocations_{year}'] - 
           self.facilities_data[f'Emissions_{year}']
       )
       
       print("\nFinal Values:")
       print(f"Total Output: {self.facilities_data[f'Output_{year}'].sum():.4f}")
       print(f"Total Emissions: {self.facilities_data[f'Emissions_{year}'].sum():.4f}")
       print(f"Total Allocations: {self.facilities_data[f'Allocations_{year}'].sum():.4f}")
       print(f"Net Position: {self.facilities_data[f'Allowance Surplus/Deficit_{year}'].sum():.4f}")
    
    def update_surplus_deficit(self, year: int) -> None:
        """Update allowance surplus/deficit positions after trading."""
        print(f"\n=== Updating Surplus/Deficit Positions for Year {year} ===")
        
        # Recalculate net positions
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - 
            self.facilities_data[f'Emissions_{year}']
        )
        
        # Calculate market statistics
        total_surplus = self.facilities_data[
            self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0
        ][f'Allowance Surplus/Deficit_{year}'].sum()
        
        total_deficit = abs(self.facilities_data[
            self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0
        ][f'Allowance Surplus/Deficit_{year}'].sum())
        
        net_position = total_surplus - total_deficit
        
        # Print verification
        print("\nMarket Position Summary:")
        print(f"Total Surplus: {total_surplus:.4f}")
        print(f"Total Deficit: {total_deficit:.4f}")
        print(f"Net Position: {net_position:.4f}")
        
        # Distribution analysis
        n_surplus = (self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0).sum()
        n_deficit = (self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0).sum()
        n_neutral = (self.facilities_data[f'Allowance Surplus/Deficit_{year}'] == 0).sum()
        
        print("\nPosition Distribution:")
        print(f"Facilities in Surplus: {n_surplus}")
        print(f"Facilities in Deficit: {n_deficit}")
        print(f"Facilities in Balance: {n_neutral}")
        
        # Verify total allowances preserved
        total_allowances = self.facilities_data[f'Allocations_{year}'].sum()
        total_emissions = self.facilities_data[f'Emissions_{year}'].sum()
        expected_position = total_allowances - total_emissions
        
        if abs(net_position - expected_position) > 0.0001:
            print("\nWARNING: Position reconciliation error")
            print(f"Net Position: {net_position:.4f}")
            print(f"Expected Position: {expected_position:.4f}")
            print(f"Difference: {abs(net_position - expected_position):.4f}")
            
        # Sample verification
        print("\nSample Position Verification:")
        sample_size = min(3, len(self.facilities_data))
        for _, facility in self.facilities_data.head(sample_size).iterrows():
            print(f"\nFacility: {facility['Facility ID']}")
            print(f"Allocations: {facility[f'Allocations_{year}']:.4f}")
            print(f"Emissions: {facility[f'Emissions_{year}']:.4f}")
            print(f"Position: {facility[f'Allowance Surplus/Deficit_{year}']:.4f}")

           
    def calculate_banking_incentive(self, current_year: int, facility_id: str) -> float:
        current_price = self.market_price
        current_ceiling = self.calculate_price_ceiling(current_year)
        next_year_ceiling = self.calculate_price_ceiling(current_year + 1)
        
        # Price closer to ceiling increases incentive
        price_ratio = current_price / current_ceiling
        # Higher ceiling growth increases incentive  
        ceiling_growth = (next_year_ceiling - current_ceiling) / current_ceiling
        
        banking_incentive = 0.7 * price_ratio + 0.3 * ceiling_growth
        
        return max(0.0, min(1.0, banking_incentive))

    def track_vintage_limits(self, year: int) -> None:
        """Track and enforce vintage limits on banked permits."""
        for idx, facility in self.facilities_data.iterrows():
            # Track vintages for new bank
            new_bank_vintages = {}
            for past_year in range(max(self.start_year, year - self.vintage_limit), year + 1):
                banked = facility[f'New_Bank_Balance_{past_year}']
                if banked > 0:
                    new_bank_vintages[past_year] = banked

            # Expire old vintages
            valid_balance = sum(new_bank_vintages.values())
            self.facilities_data.at[idx, f'New_Bank_Balance_{year}'] = valid_balance

            # Historical bank doesn't expire but track usage
            if year - self.start_year > self.vintage_limit:
                self.facilities_data.at[idx, f'Historical_Bank_Balance_{year}']
    
    def _initialize_banking_columns(self) -> None:
            """Initialize banking columns for all facilities."""
            for year in range(self.start_year, self.end_year + 1):
                self.facilities_data[f'Historical_Bank_Balance_{year}'] = 0.0
                self.facilities_data[f'New_Bank_Balance_{year}'] = 0.0
                self.facilities_data[f'Historical_Bank_Use_{year}'] = 0.0
                self.facilities_data[f'New_Bank_Use_{year}'] = 0.0
                        
    def make_banking_decision(self, year: int) -> None:
        """Execute banking decisions for facilities."""
        if year >= self.end_year:
            return
        
        # Ensure the Banking_Decision column exists for this year
        if f'Banking_Decision_{year}' not in self.facilities_data.columns:
            self.facilities_data[f'Banking_Decision_{year}'] = 0.0
        
        for idx, facility in self.facilities_data.iterrows():
            surplus = facility[f'Allowance Surplus/Deficit_{year}']
            if surplus <= 0:
                continue
            
            # More aggressive banking when prices are rising
            price_ceiling_ratio = self.market_price / self.calculate_price_ceiling(year)
            banking_incentive = min(0.8, price_ceiling_ratio + 0.2)  # More aggressive banking
            
            bank_volume = surplus * banking_incentive
            
            # Update banking decision and positions
            self.facilities_data.at[idx, f'Banking_Decision_{year}'] = bank_volume
            self.facilities_data.at[idx, f'New_Bank_Balance_{year}'] = (
                self.facilities_data.at[idx, f'New_Bank_Balance_{year}'] + bank_volume
            )
            self.facilities_data.at[idx, f'Allowance Surplus/Deficit_{year}'] -= bank_volume
  
    def use_banked_permits(self, year: int) -> None:
        for idx, facility in self.facilities_data.iterrows():
            deficit = abs(min(0, facility[f'Allowance Surplus/Deficit_{year}']))
            if deficit <= 0:
                continue
    
            # Use more bank as price rises
            use_rate = min(0.9, self.market_price / self.calculate_price_ceiling(year))
            target_use = deficit * use_rate
    
            bank_balance = facility[f'New_Bank_Balance_{year}']
            use_amount = min(bank_balance, target_use)
            
            if use_amount > 0:
                self.facilities_data.at[idx, f'New_Bank_Use_{year}'] = use_amount
                self.facilities_data.at[idx, f'New_Bank_Balance_{year}'] -= use_amount
                self.facilities_data.at[idx, f'Allowance Surplus/Deficit_{year}'] += use_amount 

  
# 3. Market Position Analysis
    def calculate_market_positions(self, year: int) -> Tuple[float, float]:
        """Calculate market supply and demand positions."""
        print("\nCalculating market positions")
        
        # Get current positions
        positions = self.facilities_data[f'Allowance Surplus/Deficit_{year}']
        
        # Calculate total supply (sum of surpluses)
        total_supply = positions[positions > 0].sum()
        
        # Calculate total demand (absolute sum of deficits)
        total_demand = abs(positions[positions < 0].sum())
        
        # Print verification
        print(f"Total supply: {total_supply:.2f}")
        print(f"Total demand: {total_demand:.2f}")
        
        # Print detailed analysis
        if total_demand > 0:
            coverage_ratio = total_supply / total_demand
            print(f"Supply/Demand ratio: {coverage_ratio:.2f}")
        
        buyers = (positions < 0).sum()
        sellers = (positions > 0).sum()
        neutral = (positions == 0).sum()
        
        print("\nPosition Distribution:")
        print(f"Buyers (deficit): {buyers}")
        print(f"Sellers (surplus): {sellers}")
        print(f"Neutral: {neutral}")
        
        # Calculate average position sizes
        if buyers > 0:
            avg_deficit = total_demand / buyers
            print(f"Average deficit size: {avg_deficit:.2f}")
        if sellers > 0:
            avg_surplus = total_supply / sellers
            print(f"Average surplus size: {avg_surplus:.2f}")
        
        # Identify largest positions
        if not positions.empty:
            largest_deficit = abs(positions[positions < 0].min()) if any(positions < 0) else 0
            largest_surplus = positions[positions > 0].max() if any(positions > 0) else 0
            print(f"\nLargest deficit: {largest_deficit:.2f}")
            print(f"Largest surplus: {largest_surplus:.2f}")
            
            # Show facilities with large positions (>10% of total)
            threshold = max(total_supply, total_demand) * 0.1
            large_positions = self.facilities_data[abs(positions) > threshold]
            if not large_positions.empty:
                print("\nFacilities with large positions (>10% of market):")
                for _, facility in large_positions.iterrows():
                    position = facility[f'Allowance Surplus/Deficit_{year}']
                    print(f"Facility {facility['Facility ID']}: {position:.2f}")
        
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
    def calculate_price_ceiling(self, year: int) -> float:
        """Calculate the price ceiling for a given year using scenario parameters."""
        years_elapsed = year - self.start_year
        # Use scenario-specific price increment
        ceiling = self.ceiling_price + (self.price_increment * years_elapsed)
        
        print(f"Ceiling calculation for year {year}:")
        print(f"Base ceiling: ${self.ceiling_price:.2f}")
        print(f"Years elapsed: {years_elapsed}")
        print(f"Price increment: ${self.price_increment:.2f}")
        print(f"Calculated ceiling: ${ceiling:.2f}")
        
        return ceiling
        
    def determine_market_price(self, supply: float, demand: float, year: int) -> float:
        """Determine market price based on supply/demand and MAC curves."""
        print(f"\n=== Market Price Determination for Year {year} ===")
        current_ceiling = self.calculate_price_ceiling(year)
        STABILITY_THRESHOLD = 0.1
        
        net_position = supply - demand
        position_ratio = abs(net_position) / max(supply, demand)
    
        if position_ratio < STABILITY_THRESHOLD:
            return min(self.market_price, current_ceiling)
            
        if supply >= demand:
            min_sell_price = float('inf')
            for _, facility in self.facilities_data.iterrows():
                if facility[f'Allowance Surplus/Deficit_{year}'] > 0:
                    curve = self.abatement_cost_curve[
                        self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
                    ].iloc[0]
                    min_sell_price = min(min_sell_price, float(curve['Intercept']))
                    
            price_drop = (position_ratio - STABILITY_THRESHOLD) * min_sell_price
            new_price = max(self.floor_price, self.market_price - price_drop)
            return min(new_price, current_ceiling)
        
        needed_abatement = demand - supply
        clearing_prices = []
        
        for _, facility in self.facilities_data.iterrows():
            curve = self.abatement_cost_curve[
                self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
            ].iloc[0]
            slope = float(curve['Slope'])
            intercept = float(curve['Intercept'])
            max_reduction = float(curve['Max Reduction (MTCO2e)'])
            
            if slope > 0:
                clearing_prices.append(intercept)
                max_price = min(intercept + (slope * max_reduction), current_ceiling)
                clearing_prices.append(max_price)
        
        clearing_prices.append(current_ceiling)
        clearing_prices = sorted(set(clearing_prices))
        
        best_price = current_ceiling
        min_excess_demand = float('inf')
        
        for price in clearing_prices:
            potential_abatement = 0
            for _, facility in self.facilities_data.iterrows():
                curve = self.abatement_cost_curve[
                    self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
                ].iloc[0]
                slope = float(curve['Slope'])
                intercept = float(curve['Intercept'])
                max_reduction = float(curve['Max Reduction (MTCO2e)'])
                
                if slope > 0 and price > intercept:
                    econ_abatement = min(max_reduction, (price - intercept) / slope)
                    potential_abatement += econ_abatement
            
            excess_demand = needed_abatement - potential_abatement
            if abs(excess_demand) < min_excess_demand:
                min_excess_demand = abs(excess_demand)
                best_price = price
                if excess_demand <= 0:
                    break
        
        price_increase = (position_ratio - STABILITY_THRESHOLD) * (best_price - self.market_price) 
        new_price = min(self.market_price + price_increase, current_ceiling)
        return max(self.floor_price, new_price) 
    
    def validate_market_price(self) -> bool:
        """Validate that the market price is within bounds."""
        if not isinstance(self.market_price, (int, float)):
            print(f"ERROR: Invalid market price type: {type(self.market_price)}")
            return False
        if self.market_price < self.floor_price:
            print(f"ERROR: Market price ${self.market_price:.2f} below floor ${self.floor_price:.2f}")
            return False
        return True
      
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
        print(f"\n=== Calculating Abatement for Year {year} ===")
        print(f"Market Price: ${self.market_price:.2f}")
        
        total_abatement = 0.0
        total_cost = 0.0
        
        # First pass: Calculate market balance
        total_allocations = self.facilities_data[f'Allocations_{year}'].sum()
        total_emissions = self.facilities_data[f'Emissions_{year}'].sum()
        market_shortage = max(0, total_emissions - total_allocations)
        
        # Calculate per-facility share of needed market balance
        n_facilities = len(self.facilities_data)
        balance_share = market_shortage / n_facilities if n_facilities > 0 else 0
        
        for idx, facility in self.facilities_data.iterrows():
            current_emissions = facility[f'Emissions_{year}']
            allocation = facility[f'Allocations_{year}']
            initial_gap = current_emissions - allocation
            
            curve = self.abatement_cost_curve[
                self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
            ]
            if curve.empty:
                continue
                
            curve = curve.iloc[0]
            slope = float(curve['Slope'])
            intercept = float(curve['Intercept'])
            max_reduction = float(curve['Max Reduction (MTCO2e)'])
            
            # Calculate cumulative abatement and adjust max reduction
            cumulative_abatement = sum(
                facility.get(f'Tonnes Abated_{prev_year}', 0) 
                for prev_year in range(self.start_year, year)
            )
            adjusted_max_reduction = max(0, max_reduction - cumulative_abatement)
            
            if self.market_price > intercept and slope > 0:
                # Base economic abatement
                economic_quantity = (self.market_price - intercept) / slope
                
                # Additional abatement if profitable to trade
                trading_potential = 0
                if initial_gap < 0:  # Already long
                    # Consider additional abatement for trading
                    if self.market_price > (intercept + slope * economic_quantity):
                        trading_potential = min(
                            adjusted_max_reduction - economic_quantity,
                            -initial_gap * 0.5  # Use up to 50% of surplus for additional abatement
                        )
                
                # Calculate target abatement including trading consideration
                target_abatement = min(
                    economic_quantity + trading_potential,
                    adjusted_max_reduction,
                    current_emissions,
                    # Ensure some facilities generate surplus for trading
                    current_emissions + max(-initial_gap, balance_share * 1.2)  # Allow 20% extra
                )
                
                if target_abatement > 0:
                    abatement_cost = (
                        (slope * target_abatement * target_abatement / 2) +
                        (intercept * target_abatement)
                    )
                    
                    # Update facility data
                    self.facilities_data.at[idx, f'Tonnes Abated_{year}'] = target_abatement
                    self.facilities_data.at[idx, f'Abatement Cost_{year}'] = abatement_cost
                    self.facilities_data.at[idx, f'Emissions_{year}'] -= target_abatement
                    
                    total_abatement += target_abatement
                    total_cost += abatement_cost
        
        # Verify post-abatement positions
        positions = self.facilities_data[f'Allowance Surplus/Deficit_{year}']
        total_short = abs(positions[positions < 0].sum())
        total_long = positions[positions > 0].sum()
        
        print("\n=== Abatement Summary ===")
        print(f"Total Abatement: {total_abatement:.2f}")
        print(f"Average Cost per Tonne: ${(total_cost/total_abatement if total_abatement > 0 else 0):.2f}")
        print(f"\nPost-Abatement Positions:")
        print(f"Total Short: {total_short:.2f}")
        print(f"Total Long: {total_long:.2f}")
            
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
    def evaluate_trade_profitability(self, buyer_id: str, seller_id: str, 
                               volume: float, price: float, year: int) -> Tuple[float, float]:
        """
        Evaluate the profitability of a potential trade for both parties.
        
        Returns:
            Tuple of (buyer_profit_impact, seller_profit_impact)
        """
        buyer = self.facilities_data[self.facilities_data['Facility ID'] == buyer_id].iloc[0]
        seller = self.facilities_data[self.facilities_data['Facility ID'] == seller_id].iloc[0]
        
        # Get MAC curves
        buyer_curve = self.abatement_cost_curve[
            self.abatement_cost_curve['Facility ID'] == buyer_id
        ].iloc[0]
        seller_curve = self.abatement_cost_curve[
            self.abatement_cost_curve['Facility ID'] == seller_id
        ].iloc[0]
        
        # Calculate buyer's alternative cost (MAC)
        buyer_mac = float(buyer_curve['Intercept']) + (
            float(buyer_curve['Slope']) * buyer[f'Tonnes Abated_{year}']
        )
        
        # Calculate seller's opportunity cost
        seller_mac = float(seller_curve['Intercept']) + (
            float(seller_curve['Slope']) * seller[f'Tonnes Abated_{year}']
        )
        
        # Calculate profit impacts
        buyer_profit_impact = (buyer_mac - price) * volume  # Savings vs abating
        seller_profit_impact = (price - seller_mac) * volume  # Revenue vs abating
        
        return buyer_profit_impact, seller_profit_impact
        
    def optimize_facility_response(self, facility_id: str, market_price: float, year: int) -> Dict:
        """
        Optimize a facility's response to current market conditions.
        Returns optimal mix of abatement and trading.
        """
        facility = self.facilities_data[
            self.facilities_data['Facility ID'] == facility_id
        ].iloc[0]
        
        curve = self.abatement_cost_curve[
            self.abatement_cost_curve['Facility ID'] == facility_id
        ].iloc[0]
        
        current_emissions = facility[f'Emissions_{year}']
        allocation = facility[f'Allocations_{year}']
        position = allocation - current_emissions
        
        slope = float(curve['Slope'])
        intercept = float(curve['Intercept'])
        max_reduction = float(curve['Max Reduction (MTCO2e)'])
        
        # Calculate optimal abatement (where MAC = market price)
        if slope > 0:
            optimal_abatement = min(
                max_reduction,
                max(0, (market_price - intercept) / slope)
            )
        else:
            optimal_abatement = 0
            
        # Calculate trading needs after optimal abatement
        post_abatement_position = position + optimal_abatement
        
        return {
            'optimal_abatement': optimal_abatement,
            'trading_volume': abs(post_abatement_position),
            'is_buyer': post_abatement_position < 0,
            'abatement_cost': (
                (slope * optimal_abatement * optimal_abatement / 2) +
                (intercept * optimal_abatement)
            ) if optimal_abatement > 0 else 0
        }
    
    def trade_allowances(self, year: int) -> Tuple[float, float]:
        """Execute trades and return trading metrics."""
        print(f"\n=== TRADE EXECUTION - Year {year} ===")
        print(f"Market Price: ${self.market_price:.2f}")
        
        if self.market_price > self.calculate_price_ceiling(year):
            self.market_price = self.calculate_price_ceiling(year)
    
        if not self.validate_market_price():
            print("ERROR: Invalid market price, cannot execute trades")
            return 0.0, 0.0
    
        # Initialize trading columns
        trade_columns = [
            f'Trade Volume_{year}',
            f'Allowance Purchase Cost_{year}',
            f'Allowance Sales Revenue_{year}',
            f'Trade Cost_{year}'
        ]
        for col in trade_columns:
            if col not in self.facilities_data.columns:
                self.facilities_data[col] = 0.0
    
        # Get positions AFTER abatement
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - 
            self.facilities_data[f'Emissions_{year}']
        )
    
        positions = self.facilities_data[f'Allowance Surplus/Deficit_{year}']
    
        # Identify buyers (short positions) and sellers (long positions)
        buyers = self.facilities_data[positions < 0].copy()
        sellers = self.facilities_data[positions > 0].copy()
    
        print("\nMarket Participants:")
        print(f"Buyers (Short): {len(buyers)}")
        print(f"Sellers (Long): {len(sellers)}")
    
        if buyers.empty or sellers.empty:
            print("No valid trading pairs found.")
            return 0.0, 0.0
    
        total_volume = 0.0
        total_cost = 0.0
        MIN_TRADE = 0.0001
    
        current_ceiling = self.calculate_price_ceiling(year)
        print(f"Current ceiling price: ${current_ceiling:.2f}")
    
        # Allow trades at or below the ceiling price
        print("\nExecuting trades at market price (at or below ceiling)")
        buyers = buyers.sort_values(f'Allowance Surplus/Deficit_{year}', ascending=True)
        sellers = sellers.sort_values(f'Allowance Surplus/Deficit_{year}', ascending=False)
    
        for buyer_idx, buyer in buyers.iterrows():
            buyer_demand = abs(buyer[f'Allowance Surplus/Deficit_{year}'])
            if buyer_demand < MIN_TRADE:
                continue
    
            print(f"\nBuyer {buyer['Facility ID']} demand: {buyer_demand:.4f}")
    
            for seller_idx, seller in sellers.iterrows():
                seller_supply = seller[f'Allowance Surplus/Deficit_{year}']
                if seller_supply < MIN_TRADE:
                    continue
    
                print(f"Seller {seller['Facility ID']} supply: {seller_supply:.4f}")
    
                # Calculate trade volume
                volume = min(buyer_demand, seller_supply)
                if volume < MIN_TRADE:
                    continue
    
                cost = volume * self.market_price
    
                print(f"\nExecuting Trade:")
                print(f"Volume: {volume:.4f}")
                print(f"Price: ${self.market_price:.2f}")
                print(f"Cost: ${cost:.2f}")
    
                # Update buyer
                self.facilities_data.at[buyer_idx, f'Allowance Surplus/Deficit_{year}'] += volume
                self.facilities_data.at[buyer_idx, f'Trade Volume_{year}'] += volume
                self.facilities_data.at[buyer_idx, f'Allowance Purchase Cost_{year}'] += cost
                self.facilities_data.at[buyer_idx, f'Trade Cost_{year}'] += cost
    
                # Update seller
                self.facilities_data.at[seller_idx, f'Allowance Surplus/Deficit_{year}'] -= volume
                self.facilities_data.at[seller_idx, f'Trade Volume_{year}'] += volume
                self.facilities_data.at[seller_idx, f'Allowance Sales Revenue_{year}'] += cost
    
                total_volume += volume
                total_cost += cost
    
                # Update remaining demand and supply
                buyer_demand -= volume
                sellers.loc[seller_idx, f'Allowance Surplus/Deficit_{year}'] -= volume
    
                if buyer_demand < MIN_TRADE:
                    break
    
        # Verify final positions
        final_positions = self.facilities_data[f'Allowance Surplus/Deficit_{year}']
        final_short = abs(final_positions[final_positions < 0].sum())
        final_long = final_positions[final_positions > 0].sum()
    
        print("\nTrading Results:")
        print(f"Total Volume: {total_volume:.2f}")
        print(f"Total Cost: ${total_cost:.2f}")
        print(f"Final Short: {final_short:.2f}")
        print(f"Final Long: {final_long:.2f}")
    
        return total_volume, total_cost
    
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

    def initialize_historical_bank(self):
        """Set up historical permit bank at start."""
        total_allocations = self.facilities_data['Baseline Allocations'].sum()
        
        # Use self.historical_bank_share, falling back to scenario_params if needed
        historical_share = getattr(self, 'historical_bank_share', self.scenario_params.get('historical_bank_share', 0))
        min_price_ratio = getattr(self, 'min_bank_price_ratio', self.scenario_params.get('min_bank_price_ratio', 0))
        
        self.historical_bank = {
            'volume': total_allocations * historical_share,
            'vintage': self.start_year - 1,
            'min_price': self.ceiling_price * min_price_ratio
        }
        
        # Initialize first year balances
        self.facilities_data[f'Historical_Bank_Balance_{self.start_year}'] = (
            self.facilities_data['Baseline Allocations'] / total_allocations * 
            self.historical_bank['volume']
        )

    def use_historical_bank(self, year: int):
        """Use banked permits for compliance when profitable."""
        if year - self.historical_bank['vintage'] > self.scenario_params.get('vintage_limit', 0):
            return
    
           
        # Calculate deficits eligible for bank use
        deficits = self.facilities_data[f'Allowance Surplus/Deficit_{year}']
        total_deficit = abs(deficits[deficits < 0].sum())
        bank_use = min(total_deficit, self.historical_bank['volume'])
    
        # Allocate bank credits proportionally to facilities in deficit
        if bank_use > 0:
            for idx, facility in self.facilities_data[deficits < 0].iterrows():
                share = abs(facility[f'Allowance Surplus/Deficit_{year}']) / total_deficit
                allocation = bank_use * share
                self.facilities_data.at[idx, f'Allowance Surplus/Deficit_{year}'] += allocation
                self.facilities_data.at[idx, f'Historical Bank Use_{year}'] = allocation
    
            self.historical_bank['volume'] -= bank_use

    def apply_ceiling_price_compliance(self, year: int) -> None:
        """Apply ceiling price compliance as last resort."""
        ceiling_price = self.calculate_price_ceiling(year)
        
        if self.market_price >= ceiling_price:
            for idx, facility in self.facilities_data.iterrows():
                deficit = facility[f'Allowance Surplus/Deficit_{year}']
                if deficit >= 0:
                    continue
                    
                # Try banking first
                banked_balance = (
                    facility[f'Historical_Bank_Balance_{year}'] +
                    facility[f'New_Bank_Balance_{year}']
                )
                
                bank_use = min(abs(deficit), banked_balance)
                if bank_use > 0:
                    deficit += bank_use
                    # Update bank balances...
                    
                # Only use ceiling price for remaining deficit
                if deficit < 0:
                    payment = abs(deficit) * ceiling_price
                    self.facilities_data.at[idx, f'Ceiling Price Payment_{year}'] = payment
                    self.facilities_data.at[idx, f'Allowance Surplus/Deficit_{year}'] = 0    

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
        
        # Calculate profits after costs and trading
        self.calculate_facility_profits(year)
        
        print(f"\nCost calculations for year {year}:")
        print(f"Total Compliance Cost: {self.facilities_data[f'Compliance Cost_{year}'].sum():,.2f}")
        print(f"Net Market Cost: {self.facilities_data[f'Total Cost_{year}'].sum():,.2f}")
    
    def calculate_facility_profits(self, year: int) -> None:
        """Calculate facility profits considering production, abatement, and trading."""
        for idx, facility in self.facilities_data.iterrows():
            # Base production profit
            output = facility[f'Output_{year}']
            profit_rate = facility['Baseline Profit Rate']
            base_profit = output * profit_rate
            
            # Compliance costs
            abatement_cost = facility[f'Abatement Cost_{year}']
            allowance_purchases = facility[f'Allowance Purchase Cost_{year}']
            allowance_sales = facility[f'Allowance Sales Revenue_{year}']
            
            # Net trading position impact on profits
            trading_profit = allowance_sales - allowance_purchases
            
            # Calculate total profit
            total_profit = base_profit - abatement_cost + trading_profit
            
            # Store results
            self.facilities_data.at[idx, f'Base Profit_{year}'] = base_profit
            self.facilities_data.at[idx, f'Trading Profit_{year}'] = trading_profit
            self.facilities_data.at[idx, f'Total Profit_{year}'] = total_profit
            
            # Calculate profit metrics
            self.facilities_data.at[idx, f'Profit Margin_{year}'] = (
                total_profit / (output * profit_rate) if output * profit_rate > 0 else 0
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


# 8. Data Analysis & Summary Creation Methods

    def _prepare_facility_results(self, start_year: int, end_year: int) -> pd.DataFrame:
        results_data = []
        
        for year in range(start_year, end_year + 1):
            for _, facility in self.facilities_data.iterrows():
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
                
                # Add bank metrics
                bank_metrics = [
                    'Historical_Bank_Balance',
                    'New_Bank_Balance', 
                    'Historical_Bank_Use',
                    'New_Bank_Use'
                ]
                for metric in bank_metrics:
                    record[metric] = facility.get(f'{metric}_{year}', 0)
                
                # Add optional metrics
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
                    record[metric] = float(facility.get(f'{metric}_{year}', 0))
                
                results_data.append(record)
    
        results_df = pd.DataFrame(results_data)
        
        # Convert numeric columns to float
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
        """Create system-wide market summary with enhanced abatement metrics."""
        summary = {
            'Year': year,
            'Market_Price': float(self.market_price),
            'Total_Allocations': float(self.facilities_data[f'Allocations_{year}'].sum()),
            'Total_Emissions': float(self.facilities_data[f'Emissions_{year}'].sum()),
            'Total_Abatement': float(self.facilities_data[f'Tonnes Abated_{year}'].sum()),
            'Total_Trade_Volume': float(self.facilities_data[f'Trade Volume_{year}'].sum()/2),
            'Total_Trade_Cost': float(self.facilities_data[f'Trade Cost_{year}'].sum()/2),
            'Total_Abatement_Cost': float(self.facilities_data[f'Abatement Cost_{year}'].sum()),
            'Total_Compliance_Cost': float(self.facilities_data[f'Compliance Cost_{year}'].sum()),
            'Total_Net_Cost': float(self.facilities_data[f'Total Cost_{year}'].sum())
        }
        
        # Calculate market balance metrics
        summary['Market_Balance'] = summary['Total_Allocations'] - summary['Total_Emissions']
        summary['Market_Balance_Ratio'] = (summary['Market_Balance'] / summary['Total_Allocations'] 
                                         if summary['Total_Allocations'] > 0 else 0.0)
        
        if summary['Total_Trade_Volume'] > 0:
            summary['Average_Trade_Price'] = summary['Total_Trade_Cost'] / summary['Total_Trade_Volume']
        else:
            summary['Average_Trade_Price'] = 0
        
        # Calculate highest marginal cost of abatement
        highest_mac = 0
        if summary['Total_Abatement'] > 0:
            summary['Average_Abatement_Cost'] = summary['Total_Abatement_Cost'] / summary['Total_Abatement']
            
            for _, facility in self.facilities_data.iterrows():
                abatement = facility[f'Tonnes Abated_{year}']
                if abatement > 0:
                    curve = self.abatement_cost_curve[
                        self.abatement_cost_curve['Facility ID'] == facility['Facility ID']
                    ]
                    if not curve.empty:
                        curve = curve.iloc[0]
                        mac = float(curve['Intercept']) + (float(curve['Slope']) * abatement)
                        highest_mac = max(highest_mac, mac)
        else:
            summary['Average_Abatement_Cost'] = 0
            
        summary['Highest_Marginal_Cost'] = highest_mac
        
        return summary

    def calculate_compliance_value(self, shortfall: float, year: int) -> Tuple[float, float]:
        """Calculate compliance volume and value using appropriate price."""
        if shortfall >= 0:  # No shortfall
            return 0.0, 0.0
            
        # Use market price if available and lower than ceiling
        price = min(self.market_price, self.calculate_price_ceiling(year))
        volume = abs(shortfall)
        value = volume * price
        
        return volume, value
    
    def generate_compliance_report(self, year: int) -> pd.DataFrame:
        """
        Generate a compliance report showing how obligations were met and banking status.
        """
        print(f"\n=== Compliance Report for Year {year} ===")
        
        # Create report dataframe
        report_data = []
        
        total_ceiling_volume = 0
        total_ceiling_value = 0
        
        for _, facility in self.facilities_data.iterrows():
            # Get basic facility info
            facility_id = facility['Facility ID']
            sector = facility['Sector']
            
            # Get compliance requirements
            emissions = facility[f'Emissions_{year}']
            allocations = facility[f'Allocations_{year}']
            
            # Get compliance actions
            abatement = facility[f'Tonnes Abated_{year}']
            trade_volume = facility[f'Trade Volume_{year}']
            purchased = facility[f'Allowance Purchase Cost_{year}'] > 0
            sold = facility[f'Allowance Sales Revenue_{year}'] > 0
            
            # Get banking information
            banking_decision_col = f'Banking_Decision_{year}'
            banked_this_year = facility[banking_decision_col] if banking_decision_col in facility.index else 0
            used_banked = facility[f'Used_Banked_Allowances_{year}'] if f'Used_Banked_Allowances_{year}' in facility.index else 0
            total_banked = facility[f'Historical_Bank_Balance_{year}'] + facility[f'New_Bank_Balance_{year}']

            
            
            # Calculate compliance status
            final_position = facility[f'Allowance Surplus/Deficit_{year}']
            compliant = final_position >= 0
            
            # Calculate compliance values using new method
            compliance_volume, compliance_value = self.calculate_compliance_value(final_position, year)
            
            total_ceiling_volume += compliance_volume
            total_ceiling_value += compliance_value
            
            # Create compliance method string
            methods = []
            if allocations > 0:
                methods.append("Initial Allocation")
            if abatement > 0:
                methods.append("Abatement")
            if purchased:
                methods.append("Purchases")
            if used_banked > 0:
                methods.append("Used Banked")
            
            report_data.append({
                'Facility ID': facility_id,
                'Sector': sector,
                'Compliance Status': 'Compliant' if compliant else 'Non-Compliant',
                'Emissions (tCO2e)': emissions,
                'Allocations (tCO2e)': allocations,
                'Abatement (tCO2e)': abatement,
                'Compliance Methods': ", ".join(methods),
                'Trading Activity': 'Purchased' if purchased else ('Sold' if sold else 'None'),
                'Final Position (tCO2e)': final_position,
                'Banked This Year (tCO2e)': banked_this_year,
                'Used Banked This Year (tCO2e)': used_banked,
                'Total Currently Banked (tCO2e)': total_banked,
                'Compliance Volume (tCO2e)': compliance_volume,
                'Compliance Value ($)': compliance_value
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Print summary statistics
        print("\nCompliance Summary:")
        print(f"Total Facilities: {len(report_df)}")
        print(f"Compliant Facilities: {(report_df['Compliance Status'] == 'Compliant').sum()}")
        print(f"Total Emissions: {report_df['Emissions (tCO2e)'].sum():,.2f} tCO2e")
        print(f"Total Abatement: {report_df['Abatement (tCO2e)'].sum():,.2f} tCO2e")
        print(f"Total Banked: {report_df['Total Currently Banked (tCO2e)'].sum():,.2f} tCO2e")
        print(f"Total Compliance Volume: {total_ceiling_volume:,.2f} tCO2e")
        print(f"Total Compliance Value: ${total_ceiling_value:,.2f}")
        
        # Print sector summary
        print("\nSector Compliance Summary:")
        sector_summary = report_df.groupby('Sector').agg({
            'Facility ID': 'count',
            'Compliance Status': lambda x: (x == 'Compliant').sum(),
            'Emissions (tCO2e)': 'sum',
            'Abatement (tCO2e)': 'sum',
            'Total Currently Banked (tCO2e)': 'sum',
            'Compliance Volume (tCO2e)': 'sum',
            'Compliance Value ($)': 'sum'
        }).round(2)
        
        sector_summary.columns = [
            'Total Facilities', 'Compliant Facilities', 
            'Total Emissions', 'Total Abatement', 'Total Banked',
            'Compliance Volume', 'Compliance Value'
        ]
        print(sector_summary)
        
        return report_df
    
    def generate_market_report(self, year: int) -> pd.DataFrame:
        """Generate market report with banking metrics."""
        positions = (
            self.facilities_data[f'Allocations_{year}'] - 
            self.facilities_data[f'Emissions_{year}']
        )
        net_position = positions.sum()
        total_short = abs(net_position) if net_position < 0 else 0
        total_long = net_position if net_position > 0 else 0
    
        total_emissions = self.facilities_data[f'Emissions_{year}'].sum()
        total_allocations = self.facilities_data[f'Allocations_{year}'].sum()
        total_abatement = self.facilities_data[f'Tonnes Abated_{year}'].sum()
        total_trade_volume = self.facilities_data[f'Trade Volume_{year}'].sum() / 2
        
        # Banking metrics
        total_hist_bank = self.facilities_data[f'Historical_Bank_Balance_{year}'].sum()
        total_new_bank = self.facilities_data[f'New_Bank_Balance_{year}'].sum()
        hist_bank_used = self.facilities_data[f'Historical_Bank_Use_{year}'].sum()
        new_bank_used = self.facilities_data[f'New_Bank_Use_{year}'].sum()
        
        current_ceiling = self.calculate_price_ceiling(year)
        price_to_ceiling_ratio = self.market_price / current_ceiling if current_ceiling > 0 else 0
        
        if self.market_price < current_ceiling:
            ceiling_volume = ceiling_value = 0
        else:
            ceiling_volume = abs(positions[positions < 0].sum())
            ceiling_value = ceiling_volume * current_ceiling
    
        market_data = {
            'Year': year,
            'Market_Price': self.market_price,
            'Price_Ceiling': current_ceiling,
            'Price_Ceiling_Ratio': price_to_ceiling_ratio,
            'Total_Allocations': total_allocations,
            'Total_Emissions': total_emissions,
            'Net_Position': net_position,
            'Total_Short_Position': total_short,
            'Total_Long_Position': total_long,
            'Total_Trade_Volume': total_trade_volume,
            'Trading_Turnover_Rate': total_trade_volume / total_allocations if total_allocations > 0 else 0,
            'Total_Abatement': total_abatement,
            'Abatement_Rate': total_abatement / total_emissions if total_emissions > 0 else 0,
            'Historical_Bank_Balance': total_hist_bank,
            'New_Bank_Balance': total_new_bank,
            'Historical_Bank_Use': hist_bank_used,
            'New_Bank_Use': new_bank_used,
            'Total_Bank_Balance': total_hist_bank + total_new_bank,
            'Total_Bank_Use': hist_bank_used + new_bank_used,
            'Banking_Rate': (total_hist_bank + total_new_bank) / total_allocations if total_allocations > 0 else 0,
            'Ceiling_Price_Volume': ceiling_volume,
            'Ceiling_Price_Value': ceiling_value
        }
        
        report_df = pd.DataFrame([market_data])
        
        print("\nMARKET REPORT")
        print("=============")
        print(f"Market Price: ${market_data['Market_Price']:.2f}")
        print(f"Price Ceiling: ${market_data['Price_Ceiling']:.2f}")
        print(f"Price/Ceiling Ratio: {market_data['Price_Ceiling_Ratio']:.2%}")
        print(f"Total Trade Volume: {market_data['Total_Trade_Volume']:,.0f}")
        print(f"Total Bank Balance: {market_data['Total_Bank_Balance']:,.0f}")
        print(f"Total Bank Use: {market_data['Total_Bank_Use']:,.0f}")
        print(f"Ceiling Price Volume: {market_data['Ceiling_Price_Volume']:,.0f}")
        print(f"Ceiling Price Value: ${market_data['Ceiling_Price_Value']:,.2f}")
        
        return report_df
    
    def _save_results_to_files(self, market_summary: pd.DataFrame, 
                              sector_summary: pd.DataFrame,
                              facility_results: pd.DataFrame,
                              compliance_reports: pd.DataFrame,
                              market_reports: pd.DataFrame, 
                              output_file: str) -> None:
        """Save all model results to files."""
        try:
            base_path = Path(output_file).parent
            base_name = Path(output_file).stem
            
            # Save market summary
            market_file = base_path / f"{base_name}_market_summary.csv"
            market_summary.to_csv(market_file, index=False)
            print(f"Market summary saved to {market_file}")
            
            # Save sector summary
            if not sector_summary.empty:
                sector_file = base_path / f"{base_name}_sector_summary.csv"
                sector_summary.to_csv(sector_file, index=False)
                print(f"Sector summary saved to {sector_file}")
            
            # Save facility results
            facility_file = base_path / f"{base_name}_facility_results.csv"
            facility_results.to_csv(facility_file, index=False)
            print(f"Facility results saved to {facility_file}")
            
            # Save compliance reports
            if not compliance_reports.empty:
                compliance_file = base_path / f"{base_name}_compliance_reports.csv"
                compliance_reports.to_csv(compliance_file, index=False)
                print(f"Compliance reports saved to {compliance_file}")
            
            # Save market reports
            if not market_reports.empty:
                market_report_file = base_path / f"{base_name}_market_reports.csv"
                market_reports.to_csv(market_report_file, index=False)
                print(f"Market reports saved to {market_report_file}")
                
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            raise
     

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

    
