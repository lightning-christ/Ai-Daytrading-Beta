import yfinance as yf
import pandas as pd

# Define stock symbols and date range
tickers = ['AAPL', 'MSFT', 'GOOGL']
start_date = '2024-07-01'
end_date = '2024-07-27'

# Create an empty list to store dataframes
data_frames = []

for ticker in tickers:
    # Download 5-minute intraday data
    data = yf.download(ticker, start=start_date, end=end_date, interval='5m')
    
    # Add a column for the ticker symbol
    data['Ticker'] = ticker
    
    # Calculate SMA50 and SMA200
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    
    # Calculate MACrossover (Boolean indicating if SMA50 is above SMA200)
    data['MACrossover'] = data['SMA50'] > data['SMA200']
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # RSI Signal (Boolean indicating if RSI is below 30)
    data['RSI_Signal'] = data['RSI'] < 30
    
    # Calculate Bollinger Bands
    rolling_mean = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['BB_Mid'] = rolling_mean
    data['BB_High'] = rolling_mean + (rolling_std * 2)
    data['BB_Low'] = rolling_mean - (rolling_std * 2)
    
    # Fill NaN values with the previous value for simplicity
    data.fillna(method='bfill', inplace=True)
    
    # Append dataframe to the list
    data_frames.append(data)

# Concatenate all dataframes into one
all_data = pd.concat(data_frames)

# Save to CSV
all_data.to_csv('5min_intraday_data_with_indicators.csv', index=True)
