from data_processing import *

from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna

import numpy as np
from tqdm import tqdm
import copy

##########################################################
# Data loading
##########################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "../01_Datenaufbereitung/Output/Calculated/"
all_data = load_data(data_dir)

train_df, val_df, test_df = split_data(all_data, train=13, val=1, test=1,parts = 5)
train_scaled, val_scaled, test_scaled = scale_data(train_df, val_df, test_df)

class SequenceDataset(Dataset):
    """
    (X[t], y[t]) mit seq_len Zeitschritten.
    - X[t] = [Voltage, Current, SOC] für t..t+seq_len-1
    - y[t] = SOC an (t + seq_len)
    """
    def __init__(self, df, seq_len=60):
        # Hier NICHT SOC skalieren, da er bereits in 0..1 liegt
        self.seq_len = seq_len
        data_array = df[["Voltage[V]", "Current[A]", "SOC_ZHU"]].values
        self.data = data_array

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.data[idx : idx + self.seq_len]     # shape (seq_len, 3)
        y_val = self.data[idx + self.seq_len, 2]        # SOC in Spalte 2
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

# Erstellen der Datasets
seq_length = 60
train_dataset = SequenceDataset(df_train_scaled, seq_len=seq_length)
val_dataset   = SequenceDataset(df_val_scaled,   seq_len=seq_length)
test_dataset  = SequenceDataset(df_test_scaled,  seq_len=seq_length)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=50000, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=50000, shuffle=False, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=50000, shuffle=False, drop_last=True)

###############################################################################
# 4) LSTM-Modell (pytorch_forecasting)
###############################################################################
class LSTMSOCModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=1, batch_first=True):
        super().__init__()
        self.lstm = ForecastingLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, 3)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        soc_pred = self.fc(last_out)
        return soc_pred.squeeze(-1)    # (batch_size,)

model = LSTMSOCModel(input_size=3, hidden_size=32, num_layers=2, batch_first=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)