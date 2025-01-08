# run_obatest.py
import pandas as pd
import os
from obamodel import obamodel

def run_trading_model(start_year=2025, end_year=2035):
    """Run the emissions trading model with proper initialization and error handling."""
    print("\nInitializing OBA model simulation...")
    
    try:
        # Set working directory
        os.chdir(r"C:\Users\user\AppData\Local\Programs\Python\Python313\OBA_test")
        print("Working directory:", os.getcwd())
        
        # Load input data
        facilities_data = pd.read_csv("facilities_data.csv")
        abatement_cost_curve = pd.read_csv("abatement_cost_curve.csv")
        
        print("\nData loaded successfully:")
        print(f"Number of facilities: {len(facilities_data)}")
        print(f"Number of abatement curves: {len(abatement_cost_curve)}")
        
        # Initialize model
        model = obamodel(facilities_data, abatement_cost_curve, start_year)
        
        # Run model
        print("\nExecuting model simulation...")
        market_summary, facility_results = model.run_model(start_year, end_year)
        
        # Save and analyze results
        print("\nSaving results...")
        
        # Market summary analysis
        print("\nMarket Summary Statistics:")
        print(market_summary[['Year', 'Market Price', 'Trade Volume', 'Total Abatement']].describe())
        market_summary.to_csv("market_summary.csv", index=False)
        
        # Facility results analysis
        print("\nFacility Results Summary:")
        summary_stats = facility_results.groupby('Year').agg({
            'Emissions': 'sum',
            'Allocations': 'sum',
            'Allowance Surplus/Deficit': 'sum',
            'Trade Volume': 'sum',
            'Abatement Cost': 'sum',
            'Total Cost': 'sum',
            'Cost to Profit Ratio': 'mean',
            'Cost to Output Ratio': 'mean'
        })
        print(summary_stats)
        facility_results.to_csv("facility_results.csv", index=False)
        
        return market_summary, facility_results
        
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
        raise
    except Exception as e:
        print(f"Error during model execution: {e}")
        raise

def analyze_results(market_summary, facility_results):
    """Analyze model results in detail."""
    print("\n=== Detailed Results Analysis ===")
    
    # Market analysis
    print("\nMarket Price Analysis:")
    price_analysis = market_summary.groupby('Year')['Market Price'].agg(['mean', 'min', 'max'])
    print(price_analysis)
    
    print("\nTrading Activity:")
    trading_analysis = market_summary.groupby('Year').agg({
        'Trade Volume': 'sum',
        'Total Trade Cost': 'sum'
    })
    print(trading_analysis)
    
    # Abatement analysis
    print("\nAbatement Analysis:")
    abatement_analysis = market_summary.groupby('Year').agg({
        'Total Abatement': 'sum',
        'Total Abatement Cost': 'sum'
    })
    print(abatement_analysis)
    
    # Facility performance
    print("\nFacility Performance Metrics:")
    facility_metrics = facility_results.groupby('Year').agg({
        'Compliance Cost': ['mean', 'min', 'max'],
        'Cost to Profit Ratio': ['mean', 'min', 'max'],
        'Cost to Output Ratio': ['mean', 'min', 'max']
    })
    print(facility_metrics)
    
    # Save detailed analysis
    with open("analysis_report.txt", "w") as f:
        f.write("=== OBA Model Analysis Report ===\n\n")
        f.write("Market Price Analysis:\n")
        f.write(price_analysis.to_string())
        f.write("\n\nTrading Activity:\n")
        f.write(trading_analysis.to_string())
        f.write("\n\nAbatement Analysis:\n")
        f.write(abatement_analysis.to_string())
        f.write("\n\nFacility Performance Metrics:\n")
        f.write(facility_metrics.to_string())

if __name__ == "__main__":
    try:
        print("Starting OBA model simulation...")
        market_summary, facility_results = run_trading_model()
        analyze_results(market_summary, facility_results)
        print("\nSimulation completed successfully.")
        
    except Exception as e:
        print(f"\nSimulation failed: {e}")
