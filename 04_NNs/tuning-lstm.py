import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import random
import json
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from tqdm import tqdm
import optuna
import copy

# Parse command line arguments
parser = argparse.ArgumentParser(description="Hyperparameter tuning for SOH LSTM model")
parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results")
args = parser.parse_args()

# Set configurations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create output directory if not specified
if args.output_dir is None:
    save_dir = Path(__file__).parent / f"tuning_results_{timestamp}"
else:
    save_dir = Path(args.output_dir)
save_dir.mkdir(exist_ok=True, parents=True)

# Base hyperparameters - some will be tuned by Optuna
base_hyperparams = {
    "MODEL": "SOH LSTM Tuning",
    "SEQUENCE_LENGTH": 144,  # will be tuned
    "HIDDEN_SIZE": 64,       # will be tuned
    "NUM_LAYERS": 3,         # will be tuned
    "DROPOUT": 0.2,          # will be tuned
    "BATCH_SIZE": 64,        # will be tuned
    "LEARNING_RATE": 1e-3,   # will be tuned
    "EPOCHS": 50,
    "PATIENCE": 10,
    "WEIGHT_DECAY": 0.0,     # will be tuned
    "device": str(device)
}

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Load and process data function (similar to soh_lstm.py)
def load_data(data_dir: Path):
    """
    Load training, validation and test data from a single directory.
    Assumes filenames in the format: df_01.parquet, df_03.parquet, ...
    """
    # Get all parquet files and sort them by the numeric part of the filename
    parquet_files = sorted(
        [f for f in data_dir.glob('*.parquet') if f.is_file()],
        key=lambda x: int(x.stem.split('_')[-1])  # Assuming the number is after the last underscore
    )

    # Randomly select one file as the test set
    test_file = random.choice(parquet_files)

    # Remaining files for training and validation
    train_val_files = [f for f in parquet_files if f != test_file]

    # Randomly select 1/5 of the files for validation
    val_files = random.sample(train_val_files, len(train_val_files) // 5)

    # The remaining files are for training
    train_files = [f for f in train_val_files if f not in val_files]

    def process_file(file_path: Path):
        """Internal function to read and process each parquet file."""
        df = pd.read_parquet(file_path)
        
        # Keep only needed columns to reduce memory usage
        columns_to_keep = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 
                           'Temperature[°C]', 'SOC_ZHU', 'SOH_ZHU']

        df_processed = df[columns_to_keep].copy()
        df_processed.dropna(inplace=True)
        # Process time column into integers and generate corresponding Datetime column
        df_processed['Testtime[s]'] = df_processed['Testtime[s]'].round().astype(int)
        start_date = pd.Timestamp("2023-02-02")
        df_processed['Datetime'] = pd.date_range(
            start=start_date,
            periods=len(df_processed),
            freq='s'
        )
        
        # Sample data every 10 minutes to reduce data size
        df_sampled = df_processed.resample('10min', on='Datetime').first().reset_index(drop=False)
        
        return df_sampled, file_path.name

    # Process training, validation, and test files
    test_data = [process_file(test_file)]
    val_data = [process_file(f) for f in val_files]
    train_data = [process_file(f) for f in train_files]

    # Print filenames for each dataset
    print(f"Training files: {[t[1] for t in train_data]}")
    print(f"Validation files: {[v[1] for v in val_data]}")
    print(f"Testing file: {test_data[0][1]}")

    # Combine data
    df_train = pd.concat([t[0] for t in train_data], ignore_index=True)
    df_val = pd.concat([v[0] for v in val_data], ignore_index=True)
    df_test = test_data[0][0]

    print(f"\nTraining dataframe shape: {df_train.shape}")
    print(f"Validation dataframe shape: {df_val.shape}")
    print(f"Testing dataframe shape: {df_test.shape}\n")

    return df_train, df_val, df_test

def scale_data(df_train, df_val, df_test):
    """
    Scaling the features using StandardScaler.
    """
    features_to_scale = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']

    # Create a copy of the scaled dataframes
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()

    # Use StandardScaler as in the original file
    scaler = StandardScaler()

    # Fit the features only on the training data
    scaler.fit(df_train[features_to_scale])

    # Transform the features on the training, validation and testing dataframes
    df_train_scaled[features_to_scale] = scaler.transform(df_train[features_to_scale])
    df_val_scaled[features_to_scale] = scaler.transform(df_val[features_to_scale])
    df_test_scaled[features_to_scale] = scaler.transform(df_test[features_to_scale])

    print('Features scaled using StandardScaler\n')
    
    return df_train_scaled, df_val_scaled, df_test_scaled

class BatteryDataset(Dataset):
    def __init__(self, df, sequence_length=60):
        self.sequence_length = sequence_length
        
        # Get the features and labels
        features_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        label_col = 'SOH_ZHU'
        features = torch.tensor(df[features_cols].values, dtype=torch.float32)
        labels = torch.tensor(df[label_col].values, dtype=torch.float32)
        
        # Create the sequence data using efficient preallocated memory method
        n_samples = len(df) - sequence_length
        self.features = torch.zeros((n_samples, sequence_length, len(features_cols)), dtype=torch.float32)
        self.labels = torch.zeros(n_samples, dtype=torch.float32)
        
        # Create the sequence window data for each sample using tensor slicing
        for i in range(n_samples):
            self.features[i] = features[i:i+sequence_length]
            self.labels[i] = labels[i+sequence_length]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# LSTM model for SOH prediction
class SOHLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(SOHLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layer
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Only take the output from the last time
        out = lstm_out[:, -1, :]  # The shape: [batch_size, hidden_size]

        # Through the fully connected layer
        out = self.fc_layers(out)
        
        return out.squeeze(-1)

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, patience, epochs):
    """
    Train and validate the model with early stopping.
    Returns the best validation loss and the history.
    """
    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    # Initialize the early stopping parameters
    epochs_no_improve = 0
    best_val_loss = float('inf')
    best_model = None

    # Define the history to store training and validation loss
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': []
    }

    # Start training
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            # Clear the previous gradients
            optimizer.zero_grad()
            # Forward propagation and calculate the loss
            outputs = model(features)
            loss = criterion(outputs, labels)
            # Backward propagation and clip the gradients to avoid exploding gradients
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            # Update the parameters of the model
            optimizer.step()
            # Update the training loss
            train_loss += loss.item()

        # Update the average training loss
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        # Start the validation loop
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                # Forward propagation and calculate the loss
                outputs = model(features)
                loss = criterion(outputs, labels)
                # Update the validation loss
                val_loss += loss.item()

        # Update the average validation loss and epoch
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        history['epoch'].append(epoch + 1)

        # Update the learning rate after validation
        scheduler.step(val_loss)

        # Check for early stopping and save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    return best_val_loss, best_model, history

def evaluate_model(model, data_loader):
    """
    Evaluate the model on a dataset and calculate performance metrics.
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    total_loss /= len(data_loader)
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    metrics = {
        'RMSE': np.sqrt(mean_squared_error(targets, predictions)),
        'MAE': mean_absolute_error(targets, predictions),
        'MAPE': mean_absolute_percentage_error(targets, predictions),
        'R²': r2_score(targets, predictions)
    }

    return predictions, targets, metrics

def objective(trial):
    """
    Objective function for Optuna to minimize validation loss
    """
    # Suggest values for the hyperparameters
    hyperparams = {
        "SEQUENCE_LENGTH": trial.suggest_categorical("SEQUENCE_LENGTH", [72, 144, 216, 288]),
        "HIDDEN_SIZE": trial.suggest_categorical("HIDDEN_SIZE", [32, 64, 128, 256]),
        "NUM_LAYERS": trial.suggest_int("NUM_LAYERS", 1, 4),
        "DROPOUT": trial.suggest_float("DROPOUT", 0.1, 0.5, step=0.1),
        "BATCH_SIZE": trial.suggest_categorical("BATCH_SIZE", [32, 64, 128]),
        "LEARNING_RATE": trial.suggest_float("LEARNING_RATE", 1e-4, 1e-2, log=True),
        "WEIGHT_DECAY": trial.suggest_float("WEIGHT_DECAY", 1e-6, 1e-3, log=True),
    }
    
    # Log the hyperparameters for this trial
    print(f"\nTrial {trial.number}: Testing hyperparameters: {hyperparams}")
    
    # Create datasets with the suggested sequence length
    train_dataset = BatteryDataset(df_train_scaled, hyperparams["SEQUENCE_LENGTH"])
    val_dataset = BatteryDataset(df_val_scaled, hyperparams["SEQUENCE_LENGTH"])
    
    # Create dataloaders with the suggested batch size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=hyperparams["BATCH_SIZE"], 
        shuffle=True, 
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=hyperparams["BATCH_SIZE"], 
        shuffle=False, 
        pin_memory=torch.cuda.is_available()
    )
    
    # Initialize model with suggested hyperparameters
    model = SOHLSTM(
        input_size=3,  # Voltage, current, temperature
        hidden_size=hyperparams["HIDDEN_SIZE"],
        num_layers=hyperparams["NUM_LAYERS"],
        dropout=hyperparams["DROPOUT"]
    ).to(device)
    
    # Initialize optimizer with suggested learning rate and weight decay
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=hyperparams["LEARNING_RATE"],
        weight_decay=hyperparams["WEIGHT_DECAY"]
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Train and validate the model
    best_val_loss, _, _ = train_and_validate(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        patience=base_hyperparams["PATIENCE"],
        epochs=base_hyperparams["EPOCHS"]
    )
    
    return best_val_loss

def train_final_model(best_params):
    """
    Train the final model using the best hyperparameters found by Optuna
    """
    # Update the base hyperparameters with the best ones
    final_hyperparams = base_hyperparams.copy()
    final_hyperparams.update(best_params)
    
    # Save the final hyperparameters
    with open(save_dir / "best_hyperparameters.json", "w") as f:
        json.dump(final_hyperparams, f, indent=4)
    
    # Create datasets and dataloaders with the best hyperparameters
    train_dataset = BatteryDataset(df_train_scaled, final_hyperparams["SEQUENCE_LENGTH"])
    val_dataset = BatteryDataset(df_val_scaled, final_hyperparams["SEQUENCE_LENGTH"])
    test_dataset = BatteryDataset(df_test_scaled, final_hyperparams["SEQUENCE_LENGTH"])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=final_hyperparams["BATCH_SIZE"], 
        shuffle=True, 
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=final_hyperparams["BATCH_SIZE"], 
        shuffle=False, 
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=final_hyperparams["BATCH_SIZE"], 
        shuffle=False, 
        pin_memory=torch.cuda.is_available()
    )
    
    # Initialize model with the best hyperparameters
    model = SOHLSTM(
        input_size=3,
        hidden_size=final_hyperparams["HIDDEN_SIZE"],
        num_layers=final_hyperparams["NUM_LAYERS"],
        dropout=final_hyperparams["DROPOUT"]
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = total_params * 4 / (1024 ** 2)  # 4 bytes per parameter, converted to MB
    
    print('-' * 70)
    print(f'Final model architecture:\n{model}')
    print('-' * 70)
    print(f'Total parameters: {total_params}')
    print(f'Estimated model size: {model_size:.2f} MB')
    print('-' * 70)
    
    # Initialize optimizer with the best hyperparameters
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=final_hyperparams["LEARNING_RATE"],
        weight_decay=final_hyperparams["WEIGHT_DECAY"]
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Train with more epochs for final model
    final_epochs = final_hyperparams["EPOCHS"] * 2  # Double the epochs for final training
    print(f"Training final model with {final_epochs} epochs...")
    
    # Train and validate the model
    _, best_model_state, history = train_and_validate(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        patience=final_hyperparams["PATIENCE"] * 2,  # Double the patience for final training
        epochs=final_epochs
    )
    
    # Save the best model and history
    torch.save(best_model_state, save_dir / "best_model.pth")
    
    # Save the training history
    history_df = pd.DataFrame(history)
    history_df.to_parquet(save_dir / "training_history.parquet", index=False)
    
    # Load the best model for evaluation
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    print("\nEvaluating the best model on the test set...")
    predictions, targets, metrics = evaluate_model(model, test_loader)
    
    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save test results
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
    
    # Plot learning curves
    plt.figure(figsize=(12, 6))
    plt.plot(history['epoch'], history['train_loss'], 'b-', label='Training Loss')
    plt.plot(history['epoch'], history['val_loss'], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "learning_curves.png")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel('Actual SOH')
    plt.ylabel('Predicted SOH')
    plt.title(f'Predictions vs Actual (R² = {metrics["R²"]:.4f}, RMSE = {metrics["RMSE"]:.4f})')
    plt.grid(True)
    plt.savefig(save_dir / "predictions_vs_actual.png")
    
    return model, metrics, history_df

def main():
    # Set the seed for reproducibility
    set_seed(42)
    
    # Load and prepare data (global variables for use in objective function)
    global df_train_scaled, df_val_scaled, df_test_scaled
    
    data_dir = Path("../01_Datenaufbereitung/Output/Calculated/")
    print("Loading data...")
    df_train, df_val, df_test = load_data(data_dir)
    
    print("Scaling data...")
    df_train_scaled, df_val_scaled, df_test_scaled = scale_data(df_train, df_val, df_test)
    
    # Create Optuna study
    print(f"Starting hyperparameter optimization with {args.trials} trials...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)
    
    # Get best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n" + "="*60)
    print(f"Best hyperparameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best validation loss: {best_value:.6f}")
    print("="*60 + "\n")
    
    # Save study results
    study_results = {
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": args.trials,
        "timestamp": timestamp
    }
    
    with open(save_dir / "study_results.json", "w") as f:
        json.dump(study_results, f, indent=4)
    
    # Save optimization history visualization
    plt.figure(figsize=(12, 8))
    
    # Plot optimization history
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(save_dir / "optimization_history.png")
    
    # Plot parameter importances if we have enough trials
    if args.trials >= 10:
        plt.figure(figsize=(12, 8))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(save_dir / "param_importances.png")
    
    # Train and evaluate final model with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    final_model, test_metrics, history_df = train_final_model(best_params)
    
    print("\nHyperparameter tuning completed!")
    print(f"Results saved to: {save_dir}")

if __name__ == "__main__":
    main()
