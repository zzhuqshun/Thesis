import argparse
import os
import json
import copy
import random
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error

# Import custom data processing functions
from data_processing import load_data, split_data, scale_data

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_folder", type=str, default=None,
                    help="Folder path to save models and logs.")
args = parser.parse_args()

# If no run_folder passed, create a default (optional)
if args.run_folder is None:
    # fallback: use a time-based folder or something else
    args.run_folder = os.path.join("models", datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S"))

os.makedirs(args.run_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters dictionary
hyperparams = {
    "MODEL" : "LSTM 10min liqun",
    "SEQUENCE_LENGTH": 144,
    "PREDICT_LENGTH": 1,
    "HIDDEN_SIZE": 64,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.2,
    "BATCH_SIZE": 64,
    "LEARNING_RATE": 1e-3,
    "EPOCHS": 200,
    "PATIENCE": 20,
    "WEIGHT_DECAY": 0.0
}

# Save hyperparameters in the run folder
with open(os.path.join(args.run_folder, "hyperparameters.json"), "w") as f:
    json.dump(hyperparams, f, indent=4)

# Data loading and preprocessing
data_dir = "../01_Datenaufbereitung/Output/Calculated/"
all_data = load_data(data_dir)

# Split data into training, validation, and test sets
train_df, val_df, test_df = split_data(all_data, train=13, val=1, test=1, parts=1)
train_scaled, val_scaled, test_scaled = scale_data(train_df, val_df, test_df)

# Custom Dataset for cell data
class CellDataset(Dataset):
    def __init__(self, df, sequence_length=60, pred_len=1):
        """
        Args:
            df (DataFrame): input data
            sequence_length (int): input sequence length
            pred_len (int): prediction length
            stride (int): sliding window stride
        """
        self.sequence_length = sequence_length
        
        # Get the features and labels
        features_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        label_col = 'SOH_ZHU'
        features = torch.tensor(df[features_cols].values, dtype=torch.float32)
        labels = torch.tensor(df[label_col].values, dtype=torch.float32)
        
        # Create the sequence data using efficient preallocated memory method
        n_samples = len(df) - sequence_length - pred_len + 1
        if pred_len == 1:
            self.labels = torch.zeros(n_samples, dtype=torch.float32)
        else:
            self.labels = torch.zeros((n_samples, pred_len), dtype=torch.float32)

        self.features = torch.zeros((n_samples, sequence_length, len(features_cols)), dtype=torch.float32)
        
        # Create the sequence window data for each sample using tensor slicing
        for i in range(n_samples):
            self.features[i] = features[i:i+sequence_length]
            if pred_len == 1:
                self.labels[i] = labels[i+sequence_length]
            else:
                self.labels[i] = labels[i+sequence_length:i+sequence_length+pred_len]


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create datasets and dataloaders using hyperparameters
train_dataset = CellDataset(train_scaled, sequence_length=hyperparams["SEQUENCE_LENGTH"],
                            pred_len=hyperparams["PREDICT_LENGTH"])
val_dataset = CellDataset(val_scaled, sequence_length=hyperparams["SEQUENCE_LENGTH"],
                          pred_len=hyperparams["PREDICT_LENGTH"])
test_dataset = CellDataset(test_scaled, sequence_length=hyperparams["SEQUENCE_LENGTH"],
                           pred_len=hyperparams["PREDICT_LENGTH"])

train_loader = DataLoader(train_dataset, batch_size=hyperparams["BATCH_SIZE"],
                          shuffle=True, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=hyperparams["BATCH_SIZE"],
                        shuffle=False, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_dataset, batch_size=hyperparams["BATCH_SIZE"],
                         shuffle=False, pin_memory=torch.cuda.is_available())

# LSTM model definition
class LSTMmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_length=1):
        """
        Args:
            input_dim (int): number of input features
            hidden_dim (int): LSTM hidden layer size
            num_layers (int): number of LSTM layers
            dropout (float): dropout probability
            output_length (int): number of prediction steps
        """
        super(LSTMmodel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_length)
        )
    
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim,
                         device=x.device, dtype=x.dtype)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim,
                         device=x.device, dtype=x.dtype)
        lstm_out, _ = self.lstm(x, (h0, c0))  # (batch_size, seq_len, hidden_dim)
        final_hidden = lstm_out[:, -1, :]      # Use the output from the last time step
        output = self.fc(final_hidden)         # (batch_size, output_length)
        return output, _

# Training and validation function
def train_and_validation(model, train_loader, val_loader, hyperparams):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hyperparams["LEARNING_RATE"],
                                 weight_decay=hyperparams["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=10,
                                                           min_lr=1e-6)

    epochs_no_improve = 0
    best_val_loss = float('inf')

    history = {
        "train_loss": [],
        "val_loss": [],
        "epoch": []
    }

    print("\nStart training...")
    for epoch in range(hyperparams["EPOCHS"]):
        # Training phase
        model.train()
        train_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{hyperparams["EPOCHS"]}', leave=False) as pbar:
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs, _ = model(features)
                if hyperparams["PREDICT_LENGTH"] > 1:
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs.squeeze(-1), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                train_loss += loss.item()
                pbar.update(1)
        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs, _ = model(features)
                if hyperparams["PREDICT_LENGTH"] > 1:
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs.squeeze(-1), labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)
        history["epoch"].append(epoch + 1)

        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}/{hyperparams["EPOCHS"]} | Train Loss: {train_loss:.3e} | Val Loss: {val_loss:.3e} | LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # Early stopping mechanism
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.run_folder, 'best.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= hyperparams["PATIENCE"]:
                print(f'Early stopping triggered after {epoch+1} epochs!')
                break

    # Save all files in the unique run folder
    history_df = pd.DataFrame(history)
    history_df.to_parquet(os.path.join(args.run_folder, 'history.parquet'), index=False)
    torch.save(model.state_dict(), os.path.join(args.run_folder, 'last.pth'))

    return history_df

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs, _ = model(features)
            if hyperparams["PREDICT_LENGTH"] > 1:
                predictions.append(outputs.cpu().numpy())
            else:
                predictions.append(outputs.squeeze(-1).cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    predictions = np.concatenate(predictions)
    true_labels = np.concatenate(true_labels)

    mae = mean_absolute_error(true_labels, predictions)
    mape =  mean_absolute_percentage_error(true_labels, predictions)
    rmse = np.sqrt(mean_squared_error(true_labels, predictions))
    r2 = r2_score(true_labels, predictions)

    print("Evaluation Metrics:")
    print("MAE: {:.4e}".format(mae))
    print("MAPE: {:.4e}".format(mape))
    print("RMSE: {:.4e}".format(rmse))
    print("R2: {:.4f}".format(r2))

    return predictions, true_labels, mae, rmse, r2

# Main function
def main():
    # Initialize the model
    model = LSTMmodel(input_dim=3,
                      hidden_dim=hyperparams["HIDDEN_SIZE"],
                      num_layers=hyperparams["NUM_LAYERS"],
                      dropout=hyperparams["DROPOUT"],
                      output_length=hyperparams["PREDICT_LENGTH"])
    model.to(device)
    start_time = time.time()
    # Train and validate
    train_and_validation(model, train_loader, val_loader, hyperparams)
    
    end_time = time.time()
    elapsed = end_time - start_time
    print("Total training time: {:.2f} seconds".format(elapsed))
    model.load_state_dict(torch.load(os.path.join(args.run_folder, 'best.pth'), weights_only=True))
    # Evaluate the model on the test set
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
