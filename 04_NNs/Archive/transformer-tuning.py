import argparse
import os
import json
import copy
import random
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import optuna
import matplotlib.pyplot as plt

# Import custom data processing functions
from data_processing import load_data, split_data, scale_data

# ---------------------------
# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# ---------------------------

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_folder", type=str, default=None,
                    help="Folder path to save models and logs.")
args = parser.parse_args()

# Create a run folder if not provided
if args.run_folder is None:
    args.run_folder = os.path.join("models", datetime.datetime.now().strftime("run_transformer_%Y%m%d_%H%M%S"))
os.makedirs(args.run_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directly define training parameters
EPOCHS = 100
PATIENCE = 10

# Data loading and preprocessing
data_dir = "../01_Datenaufbereitung/Output/Calculated/"
all_data = load_data(data_dir)
train_df, val_df, test_df = split_data(all_data, train=13, val=1, test=1, parts=1)
train_scaled, val_scaled, test_scaled = scale_data(train_df, val_df, test_df)

# Custom Dataset for cell data
class CellDataset(Dataset):
    def __init__(self, df, sequence_length=60, pred_len=1, stride=1):
        """
        Args:
            df (DataFrame): input data
            sequence_length (int): input sequence length
            pred_len (int): prediction length
            stride (int): sliding window stride
        """
        self.sequence_length = sequence_length
        self.pred_len = pred_len

        # Define feature and label columns
        features_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        label_col = 'SOH_ZHU'
        cell_id_col = 'cell_id'

        self.features = []
        self.labels = []

        # Process data for each cell separately
        for cell_id in df[cell_id_col].unique():
            cell_data = df[df[cell_id_col] == cell_id].sort_index()
            cell_features = torch.tensor(cell_data[features_cols].values, dtype=torch.float32)
            cell_labels = torch.tensor(cell_data[label_col].values, dtype=torch.float32)
            n_samples = len(cell_data) - sequence_length - pred_len + 1

            if n_samples > 0:
                for i in range(0, n_samples, stride):
                    feature_window = cell_features[i:i + sequence_length]
                    label_window = cell_labels[i + sequence_length:i + sequence_length + pred_len]
                    self.features.append(feature_window)
                    self.labels.append(label_window)

        self.features = torch.stack(self.features)  # (n_samples, sequence_length, n_features)
        self.labels = torch.stack(self.labels)       # (n_samples, pred_len)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Generate square subsequent mask for transformer's causal attention
def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Positional encoding for transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Transformer model definition
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout, output_length=1):
        """
        Args:
            input_dim (int): number of input features
            hidden_dim (int): transformer hidden dimension
            num_layers (int): number of transformer layers
            num_heads (int): number of attention heads
            feedforward_dim (int): dimension of the feedforward network
            dropout (float): dropout probability
            output_length (int): number of prediction steps
        """
        super(TransformerModel, self).__init__()
        
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4*hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_length)
        )
    
    def forward(self, x):
        # Embed the input features
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer encoder (without mask for simplicity)
        output = self.transformer_encoder(x)
        
        # Use the last time step for prediction
        output = output[:, -1, :]
        
        # Apply fully connected layers
        output = self.fc(output)
        
        return output

# Import math module for positional encoding
import math

# Objective function for hyperparameter tuning using Optuna
def objective(trial):
    # Suggest hyperparameters for tuning
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.2, 0.5, step = 0.1)
    sequence_length = trial.suggest_int('sequence_length', 60, 1440, step=60)
    pred_len = trial.suggest_categorical('pred_len', [1, 60])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    # Create temporary datasets and dataloaders based on the suggested sequence and prediction lengths
    train_dataset_t = CellDataset(train_scaled, sequence_length=sequence_length, pred_len=pred_len)
    val_dataset_t = CellDataset(val_scaled, sequence_length=sequence_length, pred_len=pred_len)
    train_loader_t = DataLoader(train_dataset_t, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader_t = DataLoader(val_dataset_t, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    
    # Ensure num_heads divides hidden_dim evenly
    if hidden_dim % num_heads != 0:
        return float('inf')  # Skip this trial if dimensions don't align
    
    # Instantiate the model with the suggested hyperparameters
    model_t = TransformerModel(
        input_dim=3,  # Voltage, Current, Temperature
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        output_length=pred_len
    )
    model_t.to(device)
    
    optimizer_t = optim.Adam(model_t.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(EPOCHS):
        model_t.train()
        train_loss_epoch = 0.0
        for features, labels in train_loader_t:
            features, labels = features.to(device), labels.to(device)
            optimizer_t.zero_grad()
            outputs = model_t(features)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model_t.parameters(), max_norm=1)
            optimizer_t.step()
            train_loss_epoch += loss.item()
        train_loss_epoch /= len(train_loader_t)
        history["train_loss"].append(train_loss_epoch)
        
        # Validation phase
        model_t.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for features, labels in val_loader_t:
                features, labels = features.to(device), labels.to(device)
                outputs = model_t(features)
                loss = criterion(outputs, labels)
                val_loss_epoch += loss.item()
        val_loss_epoch /= len(val_loader_t)
        history["val_loss"].append(val_loss_epoch)
        
        # Print progress every 20 epochs
        if epoch % 20 == 0:
            print(f"Trial {trial.number}, Epoch {epoch}: Val Loss = {val_loss_epoch:.6f}")
        
        # Early stopping check
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    return best_val_loss

# Main function: perform hyperparameter search and train the final model
def main():
    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Val Loss): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_params = study.best_trial.params

    # Update hyperparameters for final training using the best trial
    final_hyperparams = {
        "SEQUENCE_LENGTH": best_params['sequence_length'],
        "PREDICT_LENGTH": best_params['pred_len'],
        "HIDDEN_DIM": best_params['hidden_dim'],
        "NUM_LAYERS": best_params['num_layers'],
        "NUM_HEADS": best_params['num_heads'],
        "DROPOUT": best_params['dropout'],
        "BATCH_SIZE": best_params['batch_size'],
        "LEARNING_RATE": best_params['learning_rate'],
        "WEIGHT_DECAY": 1e-4,
        "EPOCHS": 200,
        "PATIENCE": 20
    }
    
    # Save the hyperparameters to the run folder
    with open(os.path.join(args.run_folder, 'hyperparameters.json'), 'w') as f:
        json.dump(final_hyperparams, f, indent=4)
