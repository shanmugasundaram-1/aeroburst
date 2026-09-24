import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

# 1. Neural Network Architecture for Tabular Time-Series Bust Prediction
class TabularBustLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TabularBustLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the output from the final time-step of the lookback period
        out = self.fc(out[:, -1, :]) 
        out = self.sigmoid(out) * 100 # Output 0-100% Probability
        return out

# 2. PyTorch Dataset preparation
class WeatherSequenceDataset(Dataset):
    def __init__(self, data, seq_length=7):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        # Sequence of previous `seq_length` days (X)
        x = self.data[index:index+self.seq_length, :-1]
        # Target bust probability of the current day (Y)
        y = self.data[index+self.seq_length-1, -1] 
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)

def train_aerobust_model():
    data_path = "../data/processed_aerobust.csv"
    if not os.path.exists(data_path):
        print("Processed dataset not found! Please run 'python prepare_data.py' first.")
        return

    print("Loading processed dataset...")
    df = pd.read_csv(data_path)
    
    # Focusing on a single City (Delhi) for high-speed prototyping
    df_city = df[df['city'] == 'Delhi'].copy()
    
    # Features vector mapped from NWP / Observational data
    features = ['temperature_2m_max', 'wind_speed_10m_max', 'forecast_rain_sum']
    target = 'bust_probability' # Ground Truth Error Margin
    
    X = df_city[features].values
    Y = df_city[target].values.reshape(-1, 1)
    
    # Normalizing data to optimize Neural Network convergence gradients
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Stack processed values
    data_combined = np.hstack((X_scaled, Y))
    
    # Creates batches with a 7-day retrospective
    dataset = WeatherSequenceDataset(data_combined, seq_length=7) 
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model Setup
    model = TabularBustLSTM(input_size=len(features), hidden_size=32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 5
    print(f"Starting Training for {epochs} Epochs on {len(dataset)} chronological sequences...")
    print(f"--------------------------------------------------")
    
    for epoch in range(epochs):
        total_loss = 0
        for seq, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {total_loss/len(dataloader):.4f}")
        
    # Save the trained weights to load into FastAPI Backend later
    model_path = "aerobust_lstm.pth"
    torch.save(model.state_dict(), model_path)
    print(f"--------------------------------------------------")
    print(f"✅ Training Complete. Model weights saved to: ml_pipeline/{model_path}")

if __name__ == "__main__":
    train_aerobust_model()
