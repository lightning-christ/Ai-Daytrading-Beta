import pandas as pd
import numpy as np
import datetime

# Set seed for reproducibility
np.random.seed(42)

# Generate a date range
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2024, 1, 1)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate random closing prices
prices = np.random.uniform(low=100, high=500, size=len(date_range))

# Generate random features
features = {
    'Open': np.random.uniform(low=100, high=500, size=len(date_range)),
    'High': np.random.uniform(low=100, high=500, size=len(date_range)),
    'Low': np.random.uniform(low=100, high=500, size=len(date_range)),
    'Volume': np.random.randint(1000, 10000, size=len(date_range)),
    'SMA50': np.random.uniform(low=100, high=500, size=len(date_range)),
    'SMA200': np.random.uniform(low=100, high=500, size=len(date_range)),
    'MACrossover': np.random.choice([0, 1], size=len(date_range)),
    'RSI': np.random.uniform(low=0, high=100, size=len(date_range)),
    'RSI_Signal': np.random.uniform(low=0, high=100, size=len(date_range)),
    'BB_Mid': np.random.uniform(low=100, high=500, size=len(date_range)),
    'BB_Low': np.random.uniform(low=100, high=500, size=len(date_range)),
    'BB_High': np.random.uniform(low=100, high=500, size=len(date_range)),
    'Hour': np.random.randint(0, 24, size=len(date_range)),
    'DayOfWeek': np.random.randint(0, 7, size=len(date_range)),
    'Month': np.random.randint(1, 13, size=len(date_range))
}

# Create DataFrame
data = pd.DataFrame(features)
data['Date'] = date_range
data['Close'] = prices

# Save to CSV
data.to_csv('sample.csv', index=False)
print("Data generated and saved to '5min_intraday_data_with_indicators.csv'.")
