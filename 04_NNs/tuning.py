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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
parser.add_argument("--learning_rate", type=float, default=1e-4,
                    help="Initial learning rate for the optimizer.")
args = parser.parse_args()

# Create a run folder if not provided
if args.run_folder is None:
    args.run_folder = os.path.join("models", datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S"))
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

# LSTM model definition (returns only predictions)
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
        return output

# Objective function for hyperparameter tuning using Optuna
def objective(trial):
    # Suggest hyperparameters for tuning
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128])
    num_layers = trial.suggest_int('num_layers', 2, 5)
    dropout = trial.suggest_float('dropout', 0.2, 0.5, step = 0.1)
    
    # Use discrete weight_decay values as requested
    weight_decay = trial.suggest_categorical('weight_decay', [1e-5, 1e-4])
    
    sequence_length = trial.suggest_int('sequence_length', 72, 432, step=72) # 0.5 day - 3day
    pred_len = trial.suggest_categorical('pred_len', [1, 6])
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    
    # Fixed learning rate instead of searching for it
    learning_rate = args.learning_rate
    
    # Create temporary datasets and dataloaders based on the suggested sequence and prediction lengths
    train_dataset_t = CellDataset(train_scaled, sequence_length=sequence_length, pred_len=pred_len)
    val_dataset_t = CellDataset(val_scaled, sequence_length=sequence_length, pred_len=pred_len)
    train_loader_t = DataLoader(train_dataset_t, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader_t = DataLoader(val_dataset_t, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    
    # Instantiate the model with the suggested hyperparameters
    model_t = LSTMmodel(input_dim=3, hidden_dim=hidden_size, num_layers=num_layers,
                        dropout=dropout, output_length=pred_len)
    model_t.to(device)
    
    optimizer_t = optim.Adam(model_t.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer_t, mode='min', factor=0.1, patience=5, min_lr=1e-6)
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
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss_epoch)
        
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
    study.optimize(objective, n_trials=50)
    print("Best trial:")
    print(study.best_trial.params)

    best_params = study.best_trial.params

    # Update hyperparameters for final training using the best trial
    final_hyperparams = {
        "SEQUENCE_LENGTH": best_params['sequence_length'],
        "PREDICT_LENGTH": best_params['pred_len'],
        "HIDDEN_SIZE": best_params['hidden_size'],
        "NUM_LAYERS": best_params['num_layers'],
        "DROPOUT": best_params['dropout'],
        "BATCH_SIZE": best_params['batch_size'],
        "LEARNING_RATE": args.learning_rate,
        "WEIGHT_DECAY": best_params['weight_decay'],
        "EPOCHS": 200,
        "PATIENCE": 20
    }
    
    # Save hyperparameters to the run folder
    with open(os.path.join(args.run_folder, 'hyperparams.json'), 'w') as f:
        json.dump(final_hyperparams, f, indent=4)
    
    # Create final train, validation, and test datasets using the final hyperparameters
    train_dataset_final = CellDataset(train_scaled, sequence_length=final_hyperparams["SEQUENCE_LENGTH"],
                                      pred_len=final_hyperparams["PREDICT_LENGTH"])
    val_dataset_final = CellDataset(val_scaled, sequence_length=final_hyperparams["SEQUENCE_LENGTH"],
                                    pred_len=final_hyperparams["PREDICT_LENGTH"])
    test_dataset_final = CellDataset(test_scaled, sequence_length=final_hyperparams["SEQUENCE_LENGTH"],
                                     pred_len=final_hyperparams["PREDICT_LENGTH"])
    
    train_loader_final = DataLoader(train_dataset_final, batch_size=final_hyperparams["BATCH_SIZE"],
                                    shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader_final = DataLoader(val_dataset_final, batch_size=final_hyperparams["BATCH_SIZE"],
                                  shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader_final = DataLoader(test_dataset_final, batch_size=final_hyperparams["BATCH_SIZE"],
                                   shuffle=False, pin_memory=torch.cuda.is_available())
    
    # Instantiate and train the final model
    final_model = LSTMmodel(input_dim=3, hidden_dim=final_hyperparams["HIDDEN_SIZE"],
                            num_layers=final_hyperparams["NUM_LAYERS"],
                            dropout=final_hyperparams["DROPOUT"],
                            output_length=final_hyperparams["PREDICT_LENGTH"])
    final_model.to(device)
    
    optimizer_final = optim.Adam(final_model.parameters(), 
                                 lr=final_hyperparams["LEARNING_RATE"],
                                 weight_decay=final_hyperparams["WEIGHT_DECAY"])
    
    # Add learning rate scheduler for final training
    scheduler_final = ReduceLROnPlateau(optimizer_final, mode='min', factor=0.5, 
                                        patience=7, min_lr=1e-6)
    
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "epoch": [], "learning_rate": []}
    best_model = None
    
    for epoch in range(final_hyperparams["EPOCHS"]):
        final_model.train()
        train_loss_epoch = 0.0
        for features, labels in train_loader_final:
            features, labels = features.to(device), labels.to(device)
            optimizer_final.zero_grad()
            outputs = final_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1)
            optimizer_final.step()
            train_loss_epoch += loss.item()
        train_loss_epoch /= len(train_loader_final)
        
        final_model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for features, labels in val_loader_final:
                features, labels = features.to(device), labels.to(device)
                outputs = final_model(features)
                loss = criterion(outputs, labels)
                val_loss_epoch += loss.item()
        val_loss_epoch /= len(val_loader_final)
        
        # Get current learning rate
        current_lr = optimizer_final.param_groups[0]['lr']
        
        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss_epoch)
        history["epoch"].append(epoch + 1)
        history["learning_rate"].append(current_lr)
        
        # Update the learning rate scheduler
        scheduler_final.step(val_loss_epoch)
        
        print(f'Epoch {epoch+1}/{final_hyperparams["EPOCHS"]} | Train Loss: {train_loss_epoch:.3e} | Val Loss: {val_loss_epoch:.3e} | LR: {current_lr}')
        
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            epochs_no_improve = 0
            best_model = copy.deepcopy(final_model)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= final_hyperparams["PATIENCE"]:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Save training history and models in the run folder
    history_df = pd.DataFrame(history)
    history_df.to_parquet(os.path.join(args.run_folder, 'final_history.parquet'), index=False)
    torch.save(best_model, os.path.join(args.run_folder, 'final_best.pth'))
    torch.save(final_model, os.path.join(args.run_folder, 'final_last.pth'))
    
    # Evaluate the best model on the test set
    best_model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for features, labels in test_loader_final:
            features, labels = features.to(device), labels.to(device)
            outputs = best_model(features)
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    mae = mean_absolute_error(true_labels, predictions)
    rmse = np.sqrt(mean_squared_error(true_labels, predictions))
    r2 = r2_score(true_labels, predictions)
    
    print("Final Evaluation Metrics:")
    print("MAE: {:.4e}".format(mae))
    print("RMSE: {:.4e}".format(rmse))
    print("R2: {:.4f}".format(r2))
    
    # Plot the comparison between predictions and ground truth, then save the plot
    true_labels_flat = true_labels.reshape(-1)
    predictions_flat = predictions.reshape(-1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(true_labels_flat, label='Ground Truth')
    plt.plot(predictions_flat, label='Predicted')
    plt.title(f"SOH Prediction - Test Set\nR2: {r2:.4f} | MAE: {mae:.4e} | RMSE: {rmse:.4e}")
    plt.xlabel('Time Steps')
    plt.ylabel('SOH')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot to the run folder
    plot_path = os.path.join(args.run_folder, "evaluation_plot.png")
    plt.savefig(plot_path)
    
if __name__ == "__main__":
    main()
