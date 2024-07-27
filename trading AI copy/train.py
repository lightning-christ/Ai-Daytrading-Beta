import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# Load and prepare the data
data = pd.read_csv('5min_intraday_data_with_indicators.csv')

# Drop unnecessary columns
data = data.drop(columns=['Date', 'Ticker'], errors='ignore')

# Define features and target
features = ['Open', 'High', 'Low', 'Volume', 'SMA50', 'SMA200', 'MACrossover', 'RSI', 'RSI_Signal', 'BB_Mid', 'BB_High', 'BB_Low']
target = 'Close'

# Ensure target column exists
if target not in data.columns:
    raise ValueError(f"Target column '{target}' not found in the data.")

X = data[features].values
y = data[target].values

# Scale features and target
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader objects
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model
input_size = len(features)
model = StockPricePredictor(input_size)

# Load existing model parameters if they exist
try:
    model.load_state_dict(torch.load('stock_price_predictor.pth', map_location=torch.device('cpu')))
except FileNotFoundError:
    print("No existing model found. Starting training from scratch.")

# Set up the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, data_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(data_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=20)

# Evaluate the model on the test set
def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    
    avg_loss = total_loss / len(data_loader.dataset)
    print(f'Test Loss: {avg_loss:.4f}')

# Evaluate the model
evaluate_model(model, test_loader)

# Save the updated model parameters
torch.save(model.state_dict(), 'stock_price_predictor.pth')
