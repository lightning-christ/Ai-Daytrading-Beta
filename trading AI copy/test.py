import pandas as pd

# Load the data
data = pd.read_csv('5min_intraday_data_with_indicators.csv')

# Print the column names
print("Column names:", data.columns)

# Print the first few rows
print(data.head())