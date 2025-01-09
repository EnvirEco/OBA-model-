import pandas as pd

def create_scenario_template():
    """Create template for scenario CSV."""
    scenarios_df = pd.DataFrame({
        'scenario_name': ['base_case', 'tight_market', 'loose_market'],
        'benchmark_ratchet_rate': [-0.02, -0.03, -0.01],
        'market_price_floor': [100, 150, 50],
        'market_price_ceiling': [1000, 1000, 800],
        'start_year': [2025, 2025, 2025],
        'end_year': [2035, 2035, 2035],
        'description': [
            'Base case scenario',
            'Tighter market conditions',
            'Looser market conditions'
        ]
    })
    
    scenarios_df.to_csv('scenarios.csv', index=False)
    return scenarios_df

if __name__ == "__main__":
    create_scenario_template()
