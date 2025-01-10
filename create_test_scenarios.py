# Save this as C:\Users\user\AppData\Local\Programs\Python\Python313\OBA_test\create_test_scenarios.py

import pandas as pd
import os

def create_scenario_file():
    """Create test scenarios CSV with reference case."""
    # Set working directory to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Ensure scenarios directory exists
    if not os.path.exists('scenarios'):
        os.makedirs('scenarios')

    # Create scenarios with exact column names
    scenarios = pd.DataFrame({
        'Scenario': ['reference', 'high_price', 'low_price', 'aggressive_reduction'],
        'Floor Price': [20.0, 50.0, 10.0, 30.0],
        'Ceiling Price': [200.0, 300.0, 150.0, 250.0],
        'Price Increment': [5.0, 10.0, 2.5, 7.5],
        'Output Growth Rate': [0.02, 0.02, 0.02, 0.015],
        'Emissions Growth Rate': [0.01, 0.01, 0.01, 0.005],
        'Benchmark Ratchet Rate': [0.03, 0.03, 0.02, 0.04],
        'Max Reduction': [100, 100, 75, 150],
        'Description': [
            'Reference case - baseline parameters',
            'High price scenario with steeper increments',
            'Low price scenario with gentler increases',
            'Aggressive emissions reduction pathway'
        ]
    })

    # Save to CSV with explicit encoding
    output_path = os.path.join('scenarios', 'scenarios.csv')
    scenarios.to_csv(output_path, index=False, encoding='utf-8')
    
    print("\nCreated scenarios.csv with test scenarios:")
    print(scenarios[['Scenario', 'Floor Price', 'Ceiling Price', 'Benchmark Ratchet Rate']].to_string())
    print(f"\nFile saved to: {output_path}")

if __name__ == "__main__":
    create_scenario_file()
