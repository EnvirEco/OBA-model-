# create_test_scenarios.py
import pandas as pd

def create_scenario_file():
    """Create test scenarios CSV with reference case."""
    scenarios = pd.DataFrame({
        'Scenario': ['reference', 'high_price', 'low_price', 'aggressive_reduction'],
        'Floor Price': [20.0, 50.0, 10.0, 30.0],
        'Ceiling Price': [200.0, 300.0, 150.0, 250.0],
        'Price Increment': [5.0, 10.0, 2.5, 7.5],
        'Output Growth Rate': [0.02, 0.02, 0.02, 0.015],
        'Emissions Growth Rate': [0.01, 0.01, 0.01, 0.005],
        'Benchmark Ratchet Rate': [0.03, 0.03, 0.02, 0.04],
        'Max Reduction': [100, 100, 75, 150]
    })

    # Save to CSV
    scenarios.to_csv('scenarios/scenarios.csv', index=False)
    print("Created scenarios.csv with test scenarios:")
    print(scenarios.to_string())

if __name__ == "__main__":
    create_scenario_file()
