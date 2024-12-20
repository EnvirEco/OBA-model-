import pandas as pd

class OBAModel:
    def __init__(self, facilities_data, abatement_cost_curve, start_year):
        self.facilities_data = facilities_data
        self.abatement_cost_curve = abatement_cost_curve
        self.start_year = start_year
        self.market_price = 0.0
        self.government_revenue = 0.0
        self.facilities_data['Ceiling Price Payment'] = 0.0
        self.facilities_data['Tonnes Paid at Ceiling'] = 0.0

        if 'Tonnes Paid at Ceiling' not in self.facilities_data.columns:
            self.facilities_data['Tonnes Paid at Ceiling'] = 0.0

        self.facilities_data['Allowance Price ($/MTCO2e)'] = 0.0
        self.facilities_data['Trade Volume'] = 0.0
        self.facilities_data['Banked Allowances'] = 0.0
        self.facilities_data['Vintage Year'] = self.start_year

    def determine_market_price(self, supply, demand):
        if demand <= 0:
            self.market_price = 0
        elif supply == 0:
            supply = 1
            self.market_price = (100 / supply)
        else:
            supply_demand_ratio = supply / demand
            self.market_price = (100 / supply_demand_ratio)
        print(f"Determined Market Price: {self.market_price}")

    def execute_trades(self, year):
        buyers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] < 0]
        sellers = self.facilities_data[self.facilities_data[f'Allowance Surplus/Deficit_{year}'] > 0]
        total_trade_volume = 0

        for _, buyer in buyers.iterrows():
            for _, seller in sellers.iterrows():
                if buyer[f'Allowance Surplus/Deficit_{year}'] >= 0 or seller[f'Allowance Surplus/Deficit_{year}'] <= 0:
                    continue

                trade_volume = min(abs(buyer[f'Allowance Surplus/Deficit_{year}']), seller[f'Allowance Surplus/Deficit_{year}'])
                trade_cost = trade_volume * self.market_price

                buyer[f'Allowance Surplus/Deficit_{year}'] += trade_volume
                seller[f'Allowance Surplus/Deficit_{year}'] -= trade_volume

                buyer[f'Trade Volume_{year}'] += trade_volume
                seller[f'Trade Volume_{year}'] -= trade_volume

                buyer[f'Trade Cost_{year}'] += trade_cost
                seller[f'Trade Cost_{year}'] -= trade_cost

                total_trade_volume += trade_volume

                if buyer[f'Allowance Surplus/Deficit_{year}'] >= 0:
                    break
        print(f"Total trade volume for year {year}: {total_trade_volume}")

    def calculate_dynamic_values(self, year, start_year):
        years_since_start = year - start_year
        self.facilities_data[f'Output_{year}'] = (
            self.facilities_data['Baseline Output'] * (1 + self.facilities_data['Output Growth Rate']) ** years_since_start
        )
        self.facilities_data[f'Emissions_{year}'] = (
            self.facilities_data['Baseline Emissions'] * (1 + self.facilities_data['Emissions Growth Rate']) ** years_since_start
        )
        self.facilities_data[f'Benchmark_{year}'] = (
            self.facilities_data['Baseline Benchmark'] * (1 + self.facilities_data['Benchmark Ratchet Rate']) ** years_since_start
        )
        print(f"Dynamic values for {year}:")
        print(self.facilities_data[[f'Output_{year}', f'Emissions_{year}', f'Benchmark_{year}']].describe())

    def calculate_allowance_allocation(self, year):
        self.facilities_data[f'Allocations_{year}'] = (
            self.facilities_data[f'Output_{year}'] * self.facilities_data[f'Benchmark_{year}']
        ).fillna(0)
        self.facilities_data[f'Allowance Surplus/Deficit_{year}'] = (
            self.facilities_data[f'Allocations_{year}'] - self.facilities_data[f'Emissions_{year}']
        ).fillna(0)
        total_supply = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
        if total_supply == 0:
            print("Warning: No surplus allowances available. Adjusting for minimal supply.")
            self.facilities_data[f'Allowance Surplus/Deficit_{year}'] += 0.1
        print("Debug: Allowance Allocation and Surplus/Deficit:")
        print(self.facilities_data[['Facility ID', f'Allocations_{year}', f'Allowance Surplus/Deficit_{year}']])

    def calculate_abatement_costs(self, year):
        self.facilities_data[f'Abatement Cost_{year}'] = 0.0
        for index, row in self.facilities_data.iterrows():
            facility_curve = self.abatement_cost_curve[self.abatement_cost_curve['Facility ID'] == row['Facility ID']]
            if not facility_curve.empty:
                slope = facility_curve['Slope'].values[0]
                intercept = facility_curve['Intercept'].values[0]
                max_reduction = facility_curve['Max Reduction (MTCO2e)'].values[0]
                surplus_deficit = row[f'Allowance Surplus/Deficit_{year}']
                if surplus_deficit < 0:
                    abatement = min(abs(surplus_deficit), max_reduction)
                    cost = slope * abatement + intercept
                    self.facilities_data.at[index, f'Abatement Cost_{year}'] = cost
                    self.facilities_data.at[index, f'Allowance Surplus/Deficit_{year}'] += abatement
        print(f"Year {year}: Post-Abatement Surplus/Deficit:")
        print(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].describe())

    def bank_allowances(self, year):
        self.facilities_data['Banked Allowances'] += self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0)
        print(f"Year {year}: Banked Allowances:")
        print(self.facilities_data[['Facility ID', 'Banked Allowances']])

    def update_vintages(self, year):
        self.facilities_data['Vintage Year'] = year

    def market_clearing(self, year):
        max_iterations = 100
        tolerance = 1e-3
        iteration = 0
        while iteration < max_iterations:
            self.determine_market_price(
                self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum(),
                abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
            )
            self.execute_trades(year)
            total_supply = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
            total_demand = abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
            if abs(total_supply - total_demand) < tolerance:
                break
            iteration += 1
        print(f"Market cleared after {iteration} iterations with price: {self.market_price}")

    def summarize_market_supply_and_demand(self, year):
        print(f"Year {year}: Surplus/Deficit distribution:")
        print(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].describe())
        total_supply = self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum()
        total_demand = abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
        net_demand = total_demand - total_supply
        total_trade_volume = self.facilities_data[f'Trade Volume_{year}'].sum()
        total_banked_allowances = self.facilities_data['Banked Allowances'].sum()
        total_allocations = self.facilities_data[f'Allocations_{year}'].sum()
        total_emissions = self.facilities_data[f'Emissions_{year}'].sum()
        total_output = self.facilities_data[f'Output_{year}'].sum()
        print(f"Year {year}: Total Supply: {total_supply}, Total Demand: {total_demand}, Net Demand: {net_demand}, Banked Allowances: {total_banked_allowances}")
        print(f"Year {year}: Total Allocations: {total_allocations}, Total Emissions: {total_emissions}, Total Output: {total_output}")
        summary = {
            'Year': year,
            'Total Supply (MTCO2e)': total_supply,
            'Total Demand (MTCO2e)': total_demand,
            'Net Demand (MTCO2e)': net_demand,
            'Total Trade Volume (MTCO2e)': total_trade_volume,
            'Banked Allowances (MTCO2e)': total_banked_allowances,
            'Total Allocations (MTCO2e)': total_allocations,
            'Total Emissions (MTCO2e)': total_emissions,
            'Total Output (MTCO2e)': total_output,
            'Allowance Price ($/MTCO2e)': self.market_price
        }
        return summary

    def run_model(self, start_year, end_year):
        yearly_results = []
        for year in range(start_year, end_year + 1):
            print(f"Running model for year {year}...")
            try:
                self.calculate_dynamic_values(year, self.start_year)
                self.calculate_allowance_allocation(year)
                self.calculate_abatement_costs(year)
                self.determine_market_price(
                    self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(lower=0).sum(),
                    abs(self.facilities_data[f'Allowance Surplus/Deficit_{year}'].clip(upper=0).sum())
                )
                self.execute_trades(year)
                self.bank_allowances(year)
                summary = self.summarize_market_supply_and_demand(year)
                yearly_results.append(summary)
            except KeyError as e:
                print(f"KeyError during model run for year {year}: {e}")
        return yearly_results
