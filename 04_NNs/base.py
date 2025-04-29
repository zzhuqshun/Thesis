import os
from pathlib import Path
import random
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create save directory
save_dir = Path(__file__).parent / "models/LSTM" / "without_early_stopping"
# save_dir = Path("models/LSTM/2025-04-22_17-16-56")
save_dir.mkdir(exist_ok=True, parents=True)

# Model hyperparameters
hyperparams = {
    "INFO": [
        "Model:SOH_LSTM",
        "LSTM(10 min resampling)",
        "val_id:['01', '15', '17']",
        "test_id:[random]",
        "Standard scaled ['Voltage[V]', 'Current[A]', 'Temperature[°C]', 'SOH_ZHU']"
    ],
    "SEQUENCE_LENGTH": 1008,
    "HIDDEN_SIZE": 128,
    "NUM_LAYERS": 3,
    "DROPOUT": 0.5,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 0.0,
    "EPOCHS": 500,
    "PATIENCE": 500,
    "device": str(device)
}

def main():
    """Main function to run the SOH prediction pipeline"""
    # Save hyperparameters to a JSON file
    hyperparams_path = save_dir / "hyperparameters.json"
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparams, f, indent=4)
    
    # Set random seed for reproducibility
    set_seed(42)
    print(f'Using device: {device}\n')

    # ==================== Data Preprocessing ====================
    # Load data
    data_dir = Path("../01_Datenaufbereitung/Output/Calculated/")
    df_train, df_val, df_test = load_data(data_dir)

    # Scale the data
    df_train_scaled, df_val_scaled, df_test_scaled, scaler = scale_data(df_train, df_val, df_test)

    # Create datasets and data loaders
    train_dataset = BatteryDataset(df_train_scaled, hyperparams["SEQUENCE_LENGTH"])
    val_dataset = BatteryDataset(df_val_scaled, hyperparams["SEQUENCE_LENGTH"])
    test_dataset = BatteryDataset(df_test_scaled, hyperparams["SEQUENCE_LENGTH"])

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['BATCH_SIZE'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['BATCH_SIZE'], shuffle=False)

    # ==================== Model Initialization ====================
    # Initialize the LSTM model
    model = SOHLSTM(
        input_size=3,  # Voltage, current, temperature
        hidden_size=hyperparams['HIDDEN_SIZE'],
        num_layers=hyperparams['NUM_LAYERS'],
        dropout=hyperparams['DROPOUT']
    ).to(device)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model architecture:\n{model}')
    print(f'Total parameters: {total_params}')

    # Define model save paths
    save_path = {
        'best': save_dir / 'best_soh_model.pth',
        'last': save_dir / 'last_soh_model.pth',
        'history': save_dir / 'train_history.parquet'
    }

    # Choose whether to train a new model or load an existing one
    TRAINING_MODE = True
    
    if TRAINING_MODE:
        # Train and validate the model
        history, _ = train_and_validate_model(model, train_loader, val_loader, save_path)
    else:
        # Load the model
        model_path = save_path['best']
        if os.path.exists(model_path):
            print(f"\nLoading model from {model_path}...")
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print("Model loaded successfully!")
        else:
            print(f"\nWarning: Model file {model_path} does not exist.")
            exit(1)

    # ==================== Model Evaluation ====================
    print("\nEvaluating the model on the testing set...")
    predictions, targets, metrics = evaluate_model(model, test_loader)

    # Print the evaluation metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # ==================== Results Visualization ====================
    results_dir = save_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    if save_path['history'].exists():
        plot_losses(pd.read_parquet(save_path['history']), results_dir)
    
    # 将scaler传递给绘图函数
    plot_predictions(predictions, targets, metrics, df_test_scaled, results_dir, scaler)

# Set seed for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_data(data_dir: Path, resample='10min'):
    """
    Load and prepare battery data, splitting into train, validation, and test sets
    
    Args:
        data_dir: Directory containing parquet files
        resample: Time interval for resampling data
        
    Returns:
        Tuple of three dataframes (df_train, df_val, df_test)
    """
    # Get all parquet files
    parquet_files = sorted(
        [f for f in data_dir.glob('*.parquet') if f.is_file()],
        key=lambda x: int(x.stem.split('_')[-1])
    )

    # Randomly select test file
    test_file = random.choice(parquet_files)
    remaining_files = [f for f in parquet_files if f != test_file]
    
    # Set fixed validation cell IDs
    val_cell_ids = ['01', '15', '17']
    
    # Find validation files based on cell IDs from remaining files
    val_files = [f for f in remaining_files if f.stem.split('_')[1] in val_cell_ids]
    
    # Remaining files are for training
    train_files = [f for f in remaining_files if f not in val_files]

    def process_file(file_path: Path):
        """Process a single parquet file into a resampled dataframe"""
        df = pd.read_parquet(file_path)
        columns_to_keep = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 
                          'Temperature[°C]', 'SOH_ZHU']
        df_processed = df[columns_to_keep].copy()
        df_processed.dropna(inplace=True)
        
        # Create datetime column for resampling
        df_processed['Testtime[s]'] = df_processed['Testtime[s]'].round().astype(int)
        start_date = pd.Timestamp("2023-02-02")
        df_processed['Datetime'] = pd.date_range(
            start=start_date,
            periods=len(df_processed),
            freq='s'
        )
        
        # Resample data to reduce size
        df_sampled = df_processed.resample(resample, on='Datetime').mean().reset_index(drop=False)
        df_sampled["cell_id"] = file_path.stem.split('_')[1]
        
        return df_sampled, file_path.name

    # Process files for each dataset
    test_data = [process_file(test_file)]
    val_data = [process_file(f) for f in val_files]
    train_data = [process_file(f) for f in train_files]

    # Print dataset information
    print(f"Training files: {[t[1] for t in train_data]}")
    print(f"Validation files: {[v[1] for v in val_data]}")
    print(f"Testing file: {test_data[0][1]}")

    # Combine data
    df_train = pd.concat([t[0] for t in train_data], ignore_index=True)
    df_val = pd.concat([v[0] for v in val_data], ignore_index=True)
    df_test = test_data[0][0]

    print(f"\nTraining data shape: {df_train.shape}")
    print(f"Validation data shape: {df_val.shape}")
    print(f"Testing data shape: {df_test.shape}\n")

    return df_train, df_val, df_test

def scale_data(df_train, df_val, df_test):
    """
    Scale features using StandardScaler fitted on training data
    
    Args:
        df_train, df_val, df_test: DataFrames containing features to scale
        
    Returns:
        Scaled versions of the input dataframes
    """
    features_to_scale = ['Voltage[V]', 'Current[A]', 'Temperature[°C]', 'SOH_ZHU']

    # Create copies of the dataframes
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(df_train[features_to_scale])

    # Transform all datasets
    df_train_scaled[features_to_scale] = scaler.transform(df_train[features_to_scale])
    df_val_scaled[features_to_scale] = scaler.transform(df_val[features_to_scale])
    df_test_scaled[features_to_scale] = scaler.transform(df_test[features_to_scale])

    print('Features scaled using StandardScaler\n')
    
    return df_train_scaled, df_val_scaled, df_test_scaled, scaler

class BatteryDataset(Dataset):
    """Dataset for battery SOH prediction with sequence data"""
    def __init__(self, df, sequence_length):
        self.sequence_length = sequence_length
        
        # Extract features and labels
        features_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        label_col = 'SOH_ZHU'
        features = torch.tensor(df[features_cols].values, dtype=torch.float32)
        labels = torch.tensor(df[label_col].values, dtype=torch.float32)
        
        # Create sequence data efficiently
        n_samples = len(df) - sequence_length
        self.features = torch.zeros((n_samples, sequence_length, len(features_cols)), dtype=torch.float32)
        self.labels = torch.zeros(n_samples, dtype=torch.float32)
        
        # Build sequences using tensor slicing
        for i in range(n_samples):
            self.features[i] = features[i:i+sequence_length]
            self.labels[i] = labels[i+sequence_length]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SOHLSTM(nn.Module):
    """LSTM model for SOH prediction"""
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

        # Fully connected layers for prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size]
            
        Returns:
            Model prediction of shape [batch_size]
        """
        # Initialize hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out shape: [batch_size, seq_len, hidden_size]

        # Take only the output from the last time step
        out = lstm_out[:, -1, :]  # shape: [batch_size, hidden_size]

        # Pass through fully connected layers
        out = self.fc_layers(out)  # shape: [batch_size, 1]
        
        return out.squeeze(-1)  # shape: [batch_size]

def train_and_validate_model(model, train_loader, val_loader, save_path):
    """
    Train and validate the model with early stopping
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        save_path: Dict with paths for saving model and history
        
    Returns:
        history: Training history dictionary
        best_val_loss: Best validation loss achieved
    """
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=hyperparams['LEARNING_RATE'], 
        weight_decay=hyperparams['WEIGHT_DECAY']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Early stopping variables
    epochs_no_improve = 0
    best_val_loss = float('inf')

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': []
    }

    # Training loop
    print('\nStart training...')
    for epoch in range(hyperparams['EPOCHS']):
        # Training phase
        model.train()
        train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{hyperparams["EPOCHS"]}', leave=False) as pbar:
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                
                train_loss += loss.item()
                pbar.update(1)

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        history['epoch'].append(epoch + 1)

        # Update learning rate
        scheduler.step(val_loss)

        # Print progress
        print(f'Epoch {epoch + 1}/{hyperparams["EPOCHS"]} | '
              f'Training Loss: {train_loss:.3e} | '
              f'Validation Loss: {val_loss:.3e} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # Check for early stopping and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path['best'])
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= hyperparams['PATIENCE']:
                print(f'Early stopping triggered after {epoch + 1} epochs!')
                break
        
    # Save the final model and training history
    torch.save(model.state_dict(), save_path['last'])
    history_df = pd.DataFrame(history)
    history_df.to_parquet(save_path['history'], index=False)

    return history, best_val_loss

def evaluate_model(model, data_loader):
    """
    Evaluate model performance on a dataset
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation data
        
    Returns:
        predictions: Model predictions
        targets: Ground truth values
        metrics: Dictionary of performance metrics
    """
    model.eval()
    criterion = nn.MSELoss()
    
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Concatenate batches
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    # Calculate performance metrics
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(targets, predictions)),
        'MAE': mean_absolute_error(targets, predictions),
        'MAPE': mean_absolute_percentage_error(targets, predictions),
        'R²': r2_score(targets, predictions)
    }

    return predictions, targets, metrics

def plot_losses(history_df, results_dir):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Training Loss', 
            marker='o', markersize=4, lw=2)
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss', 
            marker='o', markersize=4, lw=2)
    
    plt.title('Training and Validation Losses')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(results_dir / "train_val_loss.png")
    plt.close()

def plot_predictions(predictions, targets, metrics, df_test_scaled, results_dir, scaler):
    """
    Create visualizations of model predictions and errors
    
    Args:
        predictions: Model predictions (scaled)
        targets: Ground truth values (scaled)
        metrics: Dictionary of performance metrics
        df_test_scaled: Scaled test data with datetime information
        results_dir: Directory to save plots
        scaler: StandardScaler object used for data scaling
    """
    # 转换标准化的预测和目标值回原始尺度
    # 创建一个与scaler拟合时相同格式的数组
    soh_idx = 3  # 假设SOH_ZHU是第4列, 索引为3
    
    # 创建一个零矩阵，形状为(预测数量, 特征数量)
    inverse_predictions = np.zeros((len(predictions), 4))
    inverse_targets = np.zeros((len(targets), 4))
    
    # 只在SOH列（索引为3）放入预测和目标值
    inverse_predictions[:, soh_idx] = predictions
    inverse_targets[:, soh_idx] = targets
    
    # 使用scaler进行反向转换
    inverse_predictions = scaler.inverse_transform(inverse_predictions)
    inverse_targets = scaler.inverse_transform(inverse_targets)
    
    # 提取SOH列的值
    unscaled_predictions = inverse_predictions[:, soh_idx]
    unscaled_targets = inverse_targets[:, soh_idx]
    
    # 重新计算反向转换后的指标
    unscaled_metrics = {
        'RMSE': np.sqrt(mean_squared_error(unscaled_targets, unscaled_predictions)),
        'MAE': mean_absolute_error(unscaled_targets, unscaled_predictions),
        'R²': r2_score(unscaled_targets, unscaled_predictions),
        'MAPE': mean_absolute_percentage_error(unscaled_targets, unscaled_predictions)
    }

    # Plot 1: Time series of actual vs predicted values (using unscaled values)
    plt.figure(figsize=(10, 6))
    datetime_vals = df_test_scaled['Datetime'].iloc[hyperparams['SEQUENCE_LENGTH']:].values
    plt.plot(datetime_vals, unscaled_targets, label='Actual SOH', lw=2)
    plt.plot(datetime_vals, unscaled_predictions, label='Predicted SOH', alpha=0.7)
    
    plt.title(f'Actual vs. Predicted SOH Values\n'
              f'RMSE={unscaled_metrics["RMSE"]:.4f}, MAE={unscaled_metrics["MAE"]:.4f}, '
              f'MAPE={unscaled_metrics["MAPE"]:.4f}, R²={unscaled_metrics["R²"]:.4f}')
    plt.xlabel('Datetime')
    plt.ylabel('SOH')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    plt.savefig(results_dir / "prediction_time_series.png")
    plt.close()

    # Plot 2: Scatter plot of actual vs predicted values (using unscaled values)
    plt.figure(figsize=(8, 8))
    plt.scatter(unscaled_targets, unscaled_predictions, alpha=0.5)
    
    # Add diagonal line (perfect prediction)
    min_val = min(unscaled_targets.min(), unscaled_predictions.min())
    max_val = max(unscaled_targets.max(), unscaled_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title('Actual vs. Predicted SOH Values')
    plt.xlabel('Actual SOH')
    plt.ylabel('Predicted SOH')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(results_dir / "prediction_scatter.png")
    plt.close()

if __name__ == "__main__":
    main()