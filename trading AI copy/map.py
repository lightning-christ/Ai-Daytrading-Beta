import torch

# Load the model checkpoint
checkpoint = torch.load('stock_price_predictor.pth', map_location=torch.device('cpu'))

# Print the checkpoint content
print(checkpoint)
