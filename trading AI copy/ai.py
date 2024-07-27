import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Define the model
class StockPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(StockPricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load the trained model
input_size = len([
    'Open', 'High', 'Low', 'Volume', 'SMA50', 'SMA200', 'MACrossover', 'RSI', 'RSI_Signal', 'BB_Mid', 'BB_High', 'BB_Low',
    'Hour', 'DayOfWeek', 'Month'
])

model = StockPricePredictor(input_size)
try:
    model.load_state_dict(torch.load('stock_price_predictor.pth', map_location=torch.device('cpu')))
except RuntimeError as e:
    print(f"Error loading model state_dict: {e}")
    # You may need to retrain the model or check the feature count

model.eval()

# Load and prepare new data for prediction
new_data = pd.read_csv('5min_intraday_data_with_indicators.csv')

# Drop unnecessary columns if needed
new_data = new_data.drop(columns=['Date', 'Ticker'], errors='ignore')

# Update the features list based on available columns
features = [
    'Open', 'High', 'Low', 'Volume', 'SMA50', 'SMA200', 'MACrossover', 'RSI', 'RSI_Signal', 'BB_Mid', 'BB_High', 'BB_Low',
    'Hour', 'DayOfWeek', 'Month'
]

# Check which features are in the DataFrame
available_features = [feature for feature in features if feature in new_data.columns]
print("Available features in new_data:", available_features)

# Prepare features if all required features are available
if len(available_features) == len(features):
    X_new = new_data[available_features].values
    scaler_X = StandardScaler()  # Make sure to use the same scaler as used for training
    X_new_scaled = scaler_X.fit_transform(X_new)

    # Convert to PyTorch tensor
    X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

    # Make predictions
    with torch.no_grad():
        predictions = model(X_new_tensor).squeeze().numpy()

    # Simple strategy for Buy, Sell, Hold
    def decision(predicted_close, actual_close):
        if predicted_close > actual_close:
            return 'Buy'
        elif predicted_close < actual_close:
            return 'Sell'
        else:
            return 'Hold'

    # Add predictions to DataFrame if 'Close' exists
    if 'Close' in new_data.columns:
        new_data['Predicted_Close'] = predictions
        new_data['Decision'] = new_data.apply(lambda row: decision(row['Predicted_Close'], row['Close']), axis=1)
        
        # Print only decisions
        for index, row in new_data.iterrows():
            print(f"Date: {row.get('Date', 'N/A')}, Ticker: {row.get('Ticker', 'N/A')}, Decision: {row['Decision']}")
    else:
        print("Warning: 'Close' column is missing from the data.")
else:
    print("Not all required features are available in the new data.")
