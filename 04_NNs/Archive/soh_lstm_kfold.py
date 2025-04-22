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
from datetime import datetime
from collections import deque
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from tqdm import tqdm

# Set configurations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = Path(__file__).parent / f"models/cell_kfold_{timestamp}"
save_dir.mkdir(exist_ok=True, parents=True)
hyperparams = {
    "MODEL": "LSTM K-fold evaluation for trial 18",
    "SEQUENCE_LENGTH": 864,
    "HIDDEN_SIZE": 64,
    "NUM_LAYERS": 5,
    "DROPOUT": 0.5,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.001,
    "EPOCHS": 200,
    "PATIENCE": 20,
    "WEIGHT_DECAY": 1e-05,
    "K_FOLDS": 4,  # 4 folds for cross-validation with 12 training cells
    "NUM_TEST_CELLS": 3,  # Reserve 3 cells for final testing
    "device": str(device)
}

# Main execution
def main():
    # Save hyperparameters to a JSON file
    hyperparams_path = save_dir / "hyperparameters.json"
    with open(hyperparams_path, "w") as f: 
        json.dump(hyperparams, f, indent=4)
    
    # Set the seed and device
    set_seed(42)
    torch.cuda.empty_cache()
    print(f'Using device: {device}\n')

    # ==================== Data Loading and Preprocessing ====================
    # Load all cell data
    data_dir = Path("../01_Datenaufbereitung/Output/Calculated/")
    cell_dataframes = load_cell_data(data_dir)
    
    # Split cells into training and testing sets
    train_cells, test_cells = split_cells_train_test(cell_dataframes)
    
    # Scale data (fit only on training cells)
    train_cells_scaled, test_cells_scaled = scale_cells_data(train_cells, test_cells)
    
    # Create datasets
    train_cell_datasets = {}
    for cell_id, df in train_cells_scaled.items():
        train_cell_datasets[cell_id] = BatteryDataset(df, hyperparams["SEQUENCE_LENGTH"])
    
    test_cell_datasets = {}
    for cell_id, df in test_cells_scaled.items():
        test_cell_datasets[cell_id] = BatteryDataset(df, hyperparams["SEQUENCE_LENGTH"])
    
    # ==================== K-fold Cross-Validation ====================
    # Perform K-fold CV on training cells
    kfold_results = perform_cell_kfold_cv(train_cell_datasets)
    
    # Save K-fold results
    kfold_results_df = pd.DataFrame(kfold_results)
    kfold_results_path = save_dir / "kfold_results.csv"
    kfold_results_df.to_csv(kfold_results_path, index=False)
    
    # ==================== Final Model Training ====================
    # Train the final model on all training cells
    final_model = train_final_model(train_cell_datasets)
    
    # ==================== Final Model Testing ====================
    # Test the final model on the held-out test cells
    test_model_on_cells(final_model, test_cell_datasets)

# Set seed for reproducibility
def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_cell_data(data_dir: Path, resample='10min'):
    """
    Load data for each cell from the directory.
    Returns a dictionary with cell_id as key and processed dataframe as value.
    """
    # Get all parquet files and sort them
    parquet_files = sorted(
        [f for f in data_dir.glob('*.parquet') if f.is_file()],
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    cell_dataframes = {}

    for file in parquet_files:
        # Extract cell ID from filename
        cell_id = file.stem.split('_')[1]
        
        # Read and process file
        df = pd.read_parquet(file)
        
        # Keep only needed columns
        columns_to_keep = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 
                          'Temperature[°C]', 'SOC_ZHU', 'SOH_ZHU']
        df_processed = df[columns_to_keep].copy()
        df_processed.dropna(inplace=True)
        
        # Process time column
        df_processed['Testtime[s]'] = df_processed['Testtime[s]'].round().astype(int)
        start_date = pd.Timestamp("2023-02-02")
        df_processed['Datetime'] = pd.date_range(
            start=start_date,
            periods=len(df_processed),
            freq='s'
        )
        
        # Resample data
        df_sampled = df_processed.resample(resample, on='Datetime').mean().reset_index(drop=False)
        df_sampled["cell_id"] = cell_id
        
        # Store in dictionary
        cell_dataframes[cell_id] = df_sampled
        
        print(f"Processed cell {cell_id}, shape: {df_sampled.shape}")
    
    print(f"\nLoaded data for {len(cell_dataframes)} cells")
    return cell_dataframes

def split_cells_train_test(cell_dataframes):
    """
    Split cell dataframes into training and testing sets.
    Reserves a specific number of cells for final testing.
    """
    # Get all cell IDs
    all_cell_ids = list(cell_dataframes.keys())
    
    # Determine number of test cells
    num_test_cells = hyperparams["NUM_TEST_CELLS"]
    
    # Randomly select test cells
    random.seed(42)  # For reproducibility
    test_cell_ids = random.sample(all_cell_ids, num_test_cells)
    train_cell_ids = [cell_id for cell_id in all_cell_ids if cell_id not in test_cell_ids]
    
    # Create dictionaries with train and test cells
    train_cells = {cell_id: cell_dataframes[cell_id] for cell_id in train_cell_ids}
    test_cells = {cell_id: cell_dataframes[cell_id] for cell_id in test_cell_ids}
    
    print(f"Training cells ({len(train_cells)}): {sorted(train_cell_ids)}")
    print(f"Testing cells ({len(test_cells)}): {sorted(test_cell_ids)}")
    
    return train_cells, test_cells

def scale_cells_data(train_cells, test_cells, scaler_type='standard'):
    """
    Scale features across all cells.
    The scaler is fit only on training cells and applied to both training and test cells.
    """
    features_to_scale = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
    
    # Combine all training dataframes for fitting the scaler
    combined_train_df = pd.concat(list(train_cells.values()), ignore_index=True)
    
    # Create scaled copies
    train_cells_scaled = {cell_id: df.copy() for cell_id, df in train_cells.items()}
    test_cells_scaled = {cell_id: df.copy() for cell_id, df in test_cells.items()}
    
    # Process the features based on the scaler type
    if scaler_type == 'standard':
        scaler = StandardScaler()
        
        # Fit only on training data
        scaler.fit(combined_train_df[features_to_scale])
        
        # Transform both training and testing data
        for cell_id in train_cells_scaled:
            train_cells_scaled[cell_id][features_to_scale] = scaler.transform(
                train_cells[cell_id][features_to_scale]
            )
        
        for cell_id in test_cells_scaled:
            test_cells_scaled[cell_id][features_to_scale] = scaler.transform(
                test_cells[cell_id][features_to_scale]
            )
        
        print('Features scaled using StandardScaler\n')
    
    elif scaler_type == 'minmax':
        # Use different scaling for each feature
        voltage_scaler = MinMaxScaler(feature_range=(0, 1))
        current_scaler = MinMaxScaler(feature_range=(-1, 1))
        temperature_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit on training data
        voltage_scaler.fit(combined_train_df[['Voltage[V]']])
        current_scaler.fit(combined_train_df[['Current[A]']])
        temperature_scaler.fit(combined_train_df[['Temperature[°C]']])
        
        # Transform training data
        for cell_id in train_cells_scaled:
            train_cells_scaled[cell_id]['Voltage[V]'] = voltage_scaler.transform(
                train_cells[cell_id][['Voltage[V]']]
            )
            train_cells_scaled[cell_id]['Current[A]'] = current_scaler.transform(
                train_cells[cell_id][['Current[A]']]
            )
            train_cells_scaled[cell_id]['Temperature[°C]'] = temperature_scaler.transform(
                train_cells[cell_id][['Temperature[°C]']]
            )
        
        # Transform testing data
        for cell_id in test_cells_scaled:
            test_cells_scaled[cell_id]['Voltage[V]'] = voltage_scaler.transform(
                test_cells[cell_id][['Voltage[V]']]
            )
            test_cells_scaled[cell_id]['Current[A]'] = current_scaler.transform(
                test_cells[cell_id][['Current[A]']]
            )
            test_cells_scaled[cell_id]['Temperature[°C]'] = temperature_scaler.transform(
                test_cells[cell_id][['Temperature[°C]']]
            )
        
        print('Features scaled using MinMaxScaler (Voltage: [0, 1], Current: [-1, 1], Temperature: [0, 1])\n')
    
    return train_cells_scaled, test_cells_scaled

class BatteryDataset(Dataset):
    def __init__(self, df, sequence_length=60):
        self.sequence_length = sequence_length
        
        # Get the features and labels
        features_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        label_col = 'SOH_ZHU'
        features = torch.tensor(df[features_cols].values, dtype=torch.float32)
        labels = torch.tensor(df[label_col].values, dtype=torch.float32)
        
        # Create the sequence data
        n_samples = len(df) - sequence_length
        if n_samples > 0:
            self.features = torch.zeros((n_samples, sequence_length, len(features_cols)), dtype=torch.float32)
            self.labels = torch.zeros(n_samples, dtype=torch.float32)
            
            # Create the sequence window data for each sample
            for i in range(n_samples):
                self.features[i] = features[i:i+sequence_length]
                self.labels[i] = labels[i+sequence_length]
        else:
            # Handle the case where df is smaller than sequence_length
            print(f"Warning: DataFrame with {len(df)} rows is smaller than sequence_length {sequence_length}")
            self.features = torch.zeros((0, sequence_length, len(features_cols)), dtype=torch.float32)
            self.labels = torch.zeros(0, dtype=torch.float32)

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
        # Initialize hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        out = lstm_out[:, -1, :]

        # Through the fully connected layer
        out = self.fc_layers(out)
        
        return out.squeeze(-1)

def perform_cell_kfold_cv(cell_datasets):
    """
    Perform K-fold cross-validation at the cell level.
    Each fold contains a set of cells for validation, with the rest used for training.
    """
    k_folds = hyperparams['K_FOLDS']
    
    # Get all cell IDs and prepare for folding
    all_cell_ids = list(cell_datasets.keys())
    
    # Shuffle the cell IDs (with a fixed seed for reproducibility)
    random.seed(42)
    random.shuffle(all_cell_ids)
    
    # Divide cells into K folds
    fold_size = len(all_cell_ids) // k_folds
    cell_folds = []
    for i in range(k_folds):
        if i < k_folds - 1:
            cell_folds.append(all_cell_ids[i * fold_size:(i + 1) * fold_size])
        else:
            # Last fold gets any remaining cells
            cell_folds.append(all_cell_ids[i * fold_size:])
    
    # Initialize results tracking
    fold_results = []
    all_metrics = {
        'fold': [],
        'val_loss': [],
        'RMSE': [],
        'MAE': [],
        'MAPE': [],
        'R²': []
    }
    
    # Perform K-fold cross-validation
    print(f"\nStarting {k_folds}-fold cross-validation at the cell level...")
    
    for fold, val_cells in enumerate(cell_folds):
        print(f"\n{'-'*70}")
        print(f"FOLD {fold+1}/{k_folds}")
        print(f"Validation cells: {sorted(val_cells)}")
        print(f"{'-'*70}")
        
        # Create train and validation sets for this fold
        train_cells = [cell_id for cell_id in all_cell_ids if cell_id not in val_cells]
        print(f"Training cells: {sorted(train_cells)}")
        
        # Create combined datasets for this fold
        train_fold_data = []
        for cell_id in train_cells:
            train_fold_data.append(cell_datasets[cell_id])
        
        val_fold_data = []
        for cell_id in val_cells:
            val_fold_data.append(cell_datasets[cell_id])
        
        # Create DataLoaders
        train_loader = create_combined_dataloader(train_fold_data)
        val_loader = create_combined_dataloader(val_fold_data)
        
        # Initialize a fresh model for this fold
        model = SOHLSTM(
            input_size=3,  # voltage, current, temperature
            hidden_size=hyperparams['HIDDEN_SIZE'],
            num_layers=hyperparams['NUM_LAYERS'],
            dropout=hyperparams['DROPOUT']
        ).to(device)
        
        # Train the model for this fold
        fold_dir = save_dir / f"fold_{fold+1}"
        fold_dir.mkdir(exist_ok=True)
        
        best_val_loss, model, history = train_and_validate_model(
            model, train_loader, val_loader, fold_dir, fold_num=fold+1
        )
        
        # Evaluate on validation set
        predictions, targets, metrics = evaluate_model(model, val_loader)
        
        # Plot results for this fold
        plot_losses(history, fold_dir)
        plot_results(predictions, targets, metrics, fold_dir, fold_num=fold+1)
        
        # Store results for this fold
        all_metrics['fold'].append(fold+1)
        all_metrics['val_loss'].append(best_val_loss)
        all_metrics['RMSE'].append(metrics['RMSE'])
        all_metrics['MAE'].append(metrics['MAE'])
        all_metrics['MAPE'].append(metrics['MAPE'])
        all_metrics['R²'].append(metrics['R²'])
        
        fold_result = {
            'fold': fold+1,
            'val_loss': best_val_loss,
            **metrics
        }
        fold_results.append(fold_result)
        
        print(f"\nFold {fold+1} Results:")
        for key, value in fold_result.items():
            if key != 'fold':
                print(f"{key}: {value:.4f}")
    
    # Calculate average metrics across all folds
    avg_results = {
        'fold': 'Average',
        'val_loss': np.mean(all_metrics['val_loss']),
        'RMSE': np.mean(all_metrics['RMSE']),
        'MAE': np.mean(all_metrics['MAE']),
        'MAPE': np.mean(all_metrics['MAPE']),
        'R²': np.mean(all_metrics['R²'])
    }
    fold_results.append(avg_results)
    
    # Print summary
    print(f"\n{'-'*70}")
    print(f"CROSS-VALIDATION SUMMARY")
    print(f"{'-'*70}")
    for key, value in avg_results.items():
        if key != 'fold':
            print(f"Average {key}: {value:.4f}")
    
    # Identify best fold
    best_fold_idx = np.argmin(all_metrics['val_loss'])
    best_fold = all_metrics['fold'][best_fold_idx]
    print(f"\nBest fold: {best_fold} with validation loss: {all_metrics['val_loss'][best_fold_idx]:.4f}")
    
    return fold_results

def create_combined_dataloader(datasets):
    """Create a DataLoader from multiple datasets."""
    if not datasets:
        return None
    
    combined_dataset = CombinedBatteryDataset(datasets)
    return DataLoader(
        combined_dataset,
        batch_size=hyperparams['BATCH_SIZE'],
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

class CombinedBatteryDataset(Dataset):
    """Dataset that combines multiple BatteryDataset instances."""
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = [0]
        for length in self.lengths:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
        
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        # Find which dataset the index belongs to
        dataset_idx = 0
        while dataset_idx < len(self.datasets) and idx >= self.cumulative_lengths[dataset_idx + 1]:
            dataset_idx += 1
        
        # Adjust the index for the specific dataset
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]

def train_and_validate_model(model, train_loader, val_loader, save_dir, fold_num=None):
    """
    Train and validate the model with early stopping.
    """
    # Create paths for saving models
    best_model_path = save_dir / 'best_model.pth'
    last_model_path = save_dir / 'last_model.pth'
    history_path = save_dir / 'train_history.csv'
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams['LEARNING_RATE'],
        weight_decay=hyperparams['WEIGHT_DECAY']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Initialize early stopping parameters
    epochs_no_improve = 0
    best_val_loss = float('inf')
    best_model_state = None
    
    # Initialize history
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': []
    }
    
    # Start training
    fold_label = f"Fold {fold_num} " if fold_num is not None else ""
    print(f'\nStart training {fold_label}...')
    
    for epoch in range(hyperparams['EPOCHS']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{hyperparams["EPOCHS"]}', leave=False) as pbar:
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                
                # Clear previous gradients
                optimizer.zero_grad()
                
                # Forward pass and loss calculation
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                pbar.update(1)
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Update validation loss
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Display progress
        prefix = f'[{fold_label}] ' if fold_num is not None else ''
        print(f'{prefix}Epoch {epoch + 1}/{hyperparams["EPOCHS"]} | '
              f'Train Loss: {train_loss:.3e} | '
              f'Val Loss: {val_loss:.3e} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            torch.save(best_model_state, best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= hyperparams['PATIENCE']:
                print(f'{prefix}Early stopping triggered after {epoch + 1} epochs!')
                break
    
    # Save the last model state
    torch.save(model.state_dict(), last_model_path)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)
    
    # Load the best model state
    model.load_state_dict(best_model_state)
    
    return best_val_loss, model, history_df

def train_final_model(cell_datasets):
    """
    Train the final model on all training cells.
    """
    print(f"\n{'-'*70}")
    print("TRAINING FINAL MODEL ON ALL TRAINING CELLS")
    print(f"{'-'*70}")
    
    # Combine all training datasets
    all_training_data = list(cell_datasets.values())
    train_loader = create_combined_dataloader(all_training_data)
    
    # Initialize fresh model for final training
    final_model = SOHLSTM(
        input_size=3,
        hidden_size=hyperparams['HIDDEN_SIZE'],
        num_layers=hyperparams['NUM_LAYERS'],
        dropout=hyperparams['DROPOUT']
    ).to(device)
    
    # Create final model save directory
    final_model_dir = save_dir / "final_model"
    final_model_dir.mkdir(exist_ok=True)
    final_model_path = final_model_dir / 'final_model.pth'
    
    # Define training parameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=hyperparams['LEARNING_RATE'],
        weight_decay=hyperparams['WEIGHT_DECAY']
    )
    
    # Train the final model
    print("\nTraining final model on all training cells...")
    final_model.train()
    
    history = {'epoch': [], 'train_loss': []}
    
    for epoch in range(hyperparams['EPOCHS']):
        epoch_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{hyperparams["EPOCHS"]}', leave=False) as pbar:
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                
                # Clear previous gradients
                optimizer.zero_grad()
                
                # Forward pass and loss calculation
                outputs = final_model(features)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1)
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                pbar.update(1)
        
        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_loss)
        
        # Display progress
        print(f'Epoch {epoch + 1}/{hyperparams["EPOCHS"]} | Training Loss: {epoch_loss:.3e}')
        
        # Stop at half the epochs for final model (optional early stopping)
        if epoch >= hyperparams['EPOCHS'] // 2 and epoch_loss < 1e-5:
            print(f"Training converged early after {epoch + 1} epochs")
            break
    
    # Save the final model
    torch.save(final_model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_loss'], marker='o')
    plt.title('Final Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(final_model_dir / 'training_loss.png')
    plt.close()
    
    return final_model

def evaluate_model(model, data_loader):
    """
    Evaluate the model on a dataset and calculate performance metrics.
    """
    model.eval()
    criterion = nn.MSELoss()
    all_predictions = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Update metrics
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Average loss
    avg_loss = total_loss / len(data_loader)
    
    # Concatenate results
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    # Calculate metrics
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(targets, predictions)),
        'MAE': mean_absolute_error(targets, predictions),
        'MAPE': mean_absolute_percentage_error(targets, predictions),
        'R²': r2_score(targets, predictions)
    }

    return predictions, targets, metrics

def plot_losses(history_df, save_dir):
    """
    Plot training and validation losses from the training history.
    """
    results_dir = save_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history_df['epoch'], history_df['train_loss'], label='Training Loss', 
            color='#2e78cc', marker='o', markersize=4, lw=2)
    
    if 'val_loss' in history_df.columns:
        ax.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss', 
                color='#e74c3c', marker='o', markersize=4, lw=2)
    
    # Format y-axis ticks in scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3,-3))  # Force 1e-3 notation
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_fontsize(12)
    
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.title('Training and Validation Losses over Epochs', fontsize=18)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.tight_layout()
    
    file_path = results_dir / "train_val_loss.png"
    plt.savefig(file_path)
    plt.close()

def plot_results(predictions, targets, metrics, save_dir, fold_num=None, cell_id=None):
    """
    Plot various visualizations of model predictions vs actual values.
    """
    results_dir = save_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Set label for plots
    if fold_num is not None:
        plot_label = f"Fold {fold_num} "
    elif cell_id is not None:
        plot_label = f"Cell {cell_id} "
    else:
        plot_label = ""
    
    # 1. Actual vs Predicted values over samples
    plt.figure(figsize=(12, 6))
    plt.plot(targets, label='Actual SOH', color='blue', lw=2)
    plt.plot(predictions, label='Predicted SOH', color='red', alpha=0.7)
    plt.title(f'{plot_label}Actual vs. Predicted SOH values\n'
              f'(RMSE={metrics["RMSE"]:.4f}, MAE={metrics["MAE"]:.4f}, '
              f'MAPE={metrics["MAPE"]:.4f}, R²={metrics["R²"]:.4f})')
    plt.xlabel('Sample Index')
    plt.ylabel('SOH')
    plt.legend()
    plt.grid(color='lightgrey', linewidth=0.5)
    plt.tight_layout()
    
    file_path = results_dir / "prediction.png"
    plt.savefig(file_path)
    plt.close()

    # 2. Scatter plot of Actual vs Predicted values
    plt.figure(figsize=(10, 8))
    
    # Plot perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    # Plot scatter points
    plt.scatter(targets, predictions, alpha=0.6, color='#2e78cc')
    
    plt.title(f'{plot_label}Actual vs. Predicted SOH values\n'
              f'(RMSE={metrics["RMSE"]:.4f}, MAE={metrics["MAE"]:.4f}, '
              f'MAPE={metrics["MAPE"]:.4f}, R²={metrics["R²"]:.4f})')
    plt.xlabel('Actual SOH')
    plt.ylabel('Predicted SOH')
    plt.grid(color='lightgrey', linewidth=0.5)
    plt.tight_layout()
    
    file_path = results_dir / "scatter_predict.png"
    plt.savefig(file_path)
    plt.close()

    # 3. Prediction errors distribution
    plt.figure(figsize=(10, 6))
    errors = targets - predictions
    
    # Plot histogram with kernel density estimation
    sns.histplot(errors, kde=True, color='#2e78cc')
    
    plt.axvline(x=0, color='r', linestyle='--', lw=2)
    plt.title(f'{plot_label}Prediction Errors Distribution')
    plt.xlabel('Prediction Errors')
    plt.ylabel('Density')
    plt.grid(color='lightgrey', linewidth=0.5)
    plt.tight_layout()
    
    file_path = results_dir / "errors_hist.png"
    plt.savefig(file_path)
    plt.close()

    # 4. Actual values vs Prediction errors
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, errors, alpha=0.6, color='#e74c3c')
    plt.axhline(y=0, color='b', linestyle='--', lw=2)
    plt.title(f'{plot_label}Actual SOH vs. Prediction Errors')
    plt.xlabel('Actual SOH')
    plt.ylabel('Prediction Errors')
    plt.grid(color='lightgrey', linewidth=0.5)
    plt.tight_layout()
    
    file_path = results_dir / "errors_scatter.png"
    plt.savefig(file_path)
    plt.close()
    
    # Print metrics
    print(f"\n{plot_label}Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / "metrics.csv", index=False)
    
    # Return metrics for further processing
    return metrics

def test_model_on_cells(model, test_cell_datasets):
    """
    Test the model on each test cell individually and calculate metrics.
    Also evaluate on all test cells combined.
    """
    print(f"\n{'-'*70}")
    print("TESTING FINAL MODEL ON HELD-OUT TEST CELLS")
    print(f"{'-'*70}")
    
    # Test results storage
    test_results = []
    all_predictions = []
    all_targets = []
    
    # Test on each cell individually
    for cell_id, dataset in test_cell_datasets.items():
        print(f"\nTesting on cell {cell_id}...")
        
        # Skip empty datasets
        if len(dataset) == 0:
            print(f"Skipping cell {cell_id} - insufficient data")
            continue
        
        # Create DataLoader for this cell
        test_loader = DataLoader(
            dataset,
            batch_size=hyperparams['BATCH_SIZE'],
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )
        
        # Evaluate model on this cell
        predictions, targets, metrics = evaluate_model(model, test_loader)
        
        # Store predictions and targets for combined evaluation
        all_predictions.append(predictions)
        all_targets.append(targets)
        
        # Store cell results
        cell_result = {
            'cell_id': cell_id,
            **metrics
        }
        test_results.append(cell_result)
        
        # Create cell result directory
        cell_dir = save_dir / f"test_cell_{cell_id}"
        cell_dir.mkdir(exist_ok=True)
        
        # Plot results for this cell
        plot_results(predictions, targets, metrics, cell_dir, cell_id=cell_id)
    
    # Evaluate on all test cells combined
    if all_predictions and all_targets:
        all_predictions_array = np.concatenate(all_predictions)
        all_targets_array = np.concatenate(all_targets)
        
        # Calculate metrics on all test data
        combined_metrics = {
            'RMSE': np.sqrt(mean_squared_error(all_targets_array, all_predictions_array)),
            'MAE': mean_absolute_error(all_targets_array, all_predictions_array),
            'MAPE': mean_absolute_percentage_error(all_targets_array, all_predictions_array),
            'R²': r2_score(all_targets_array, all_predictions_array)
        }
        
        # Create combined results directory
        combined_dir = save_dir / "test_combined"
        combined_dir.mkdir(exist_ok=True)
        
        # Plot combined results
        plot_results(all_predictions_array, all_targets_array, combined_metrics, combined_dir)
        
        # Add combined results
        test_results.append({
            'cell_id': 'Combined',
            **combined_metrics
        })
    
    # Save all test results to CSV
    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(save_dir / "test_results.csv", index=False)
    
    # Print summary
    print(f"\n{'-'*70}")
    print("TEST RESULTS SUMMARY")
    print(f"{'-'*70}")
    for result in test_results:
        if result['cell_id'] == 'Combined':
            print(f"\nCombined Test Results:")
        else:
            print(f"\nCell {result['cell_id']} Results:")
        
        for key, value in result.items():
            if key != 'cell_id':
                print(f"{key}: {value:.6f}")

# This conditional executes the main function when the script is run directly
if __name__ == "__main__":
    main()