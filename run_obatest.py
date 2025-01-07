import pandas as pd
import os
from obamodel import obamodel

def run_trading_model(start_year=2025, end_year=2035):
    """Run the OBA trading model simulation."""
    try:
        facilities_data = pd.read_csv("facilities_data.csv")
        abatement_cost_curve = pd.read_csv("abatement_cost_curve.csv")
        model = obamodel(facilities_data, abatement_cost_curve, start_year)
        market_summary, facilities_results = model.run_model(end_year)
        model.save_results(market_summary, facilities_results)
        
        print("\nMarket Summary:")
        print(market_summary[['Year', 'Market Price', 'Trade Volume', 'Total Abatement']].head())
        
        print("\nCompliance Report for Final Year:")
        final_compliance = model.get_compliance_report(end_year)
        print(final_compliance.head())

        return market_summary, facilities_results

    except FileNotFoundError as e:
        print(f"Error: Required input file not found - {e}")
    except Exception as e:
        print(f"Error during model run: {e}")
        raise

def analyze_results(market_summary, facilities_results):
    """Analyze and print key model results."""
    try:
        print("\nMarket Price Trends:")
        price_stats = market_summary.groupby('Year')['Market Price'].agg(['mean', 'min', 'max'])
        print(price_stats)

        print("\nTrading Activity:")
        trade_stats = market_summary.groupby('Year')['Trade Volume'].sum()
        print(trade_stats)

        print("\nCompliance Cost Summary:")
        cost_stats = market_summary.groupby('Year')['Total Compliance Cost'].agg(['mean', 'min', 'max', 'sum'])
        print(cost_stats)

    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    working_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(working_dir)
    print(f"Working directory: {working_dir}")
    
    market_summary, facilities_results = run_trading_model()
    
    if market_summary is not None and facilities_results is not None:
        analyze_results(market_summary, facilities_results)
