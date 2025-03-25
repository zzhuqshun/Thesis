import argparse
import os
import json
import copy
import random
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

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
    "MODEL": "Transformer",
    "SEQUENCE_LENGTH": 360,
    "PREDICT_LENGTH": 1,
    "HIDDEN_SIZE": 64,
    "NUM_LAYERS": 3,
    "DROPOUT": 0.3,
    "BATCH_SIZE": 64,
    "LEARNING_RATE": 0.001,
    "EPOCHS": 200,
    "PATIENCE": 20,
    "NUM_HEADS": 4,  # Added for transformer
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

# Function to generate causal mask (added)
def generate_square_subsequent_mask(sz, device):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
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
            dim_feedforward= 4*hidden_dim,
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
        # Generate causal mask to ensure the model only looks at past time steps
        seq_len = x.size(1)
        mask = generate_square_subsequent_mask(seq_len, x.device)
        
        # Embed the input features
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer encoder with causal mask
        output = self.transformer_encoder(x, mask=mask)
        
        # Use the last time step for prediction
        output = output[:, -1, :]
        
        # Apply fully connected layers
        output = self.fc(output)
        
        # For compatibility with the LSTM model's return type
        hidden_state = None
        
        return output, hidden_state

# Training and validation function
def train_and_validation(model, train_loader, val_loader, hyperparams):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hyperparams["LEARNING_RATE"],
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=10)

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
                loss = criterion(outputs, labels)
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
                loss = criterion(outputs, labels)
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
            best_model = copy.deepcopy(model)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= hyperparams["PATIENCE"]:
                print(f'Early stopping triggered after {epoch+1} epochs!')
                break

    last_model = copy.deepcopy(model)
    history_df = pd.DataFrame(history)
    # Save all files in the unique run folder
    history_df.to_parquet(os.path.join(args.run_folder, 'history.parquet'), index=False)
    torch.save(best_model, os.path.join(args.run_folder, 'best.pth'))
    torch.save(last_model, os.path.join(args.run_folder, 'last.pth'))

    return history_df, best_model, last_model

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs, _ = model(features)
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    mae = mean_absolute_error(true_labels, predictions)
    rmse = np.sqrt(mean_squared_error(true_labels, predictions))
    r2 = r2_score(true_labels, predictions)

    print("Evaluation Metrics:")
    print("MAE: {:.4e}".format(mae))
    print("RMSE: {:.4e}".format(rmse))
    print("R2: {:.4f}".format(r2))

    return predictions, true_labels, mae, rmse, r2

# Main function
def main():
    # Initialize the model
    model = TransformerModel(
        input_dim=3,
        hidden_dim=hyperparams["HIDDEN_SIZE"],
        num_layers=hyperparams["NUM_LAYERS"],
        num_heads=hyperparams["NUM_HEADS"],
        dropout=hyperparams["DROPOUT"],
        output_length=hyperparams["PREDICT_LENGTH"]
    )
    model.to(device)

    # Train and validate
    history, best_model, last_model = train_and_validation(model, train_loader, val_loader, hyperparams)

    # Evaluate the model on the test set
    evaluate_model(best_model, test_loader)

if __name__ == "__main__":
    main()
