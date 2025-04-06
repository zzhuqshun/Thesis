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
save_dir = Path(__file__).parent / f"models/trial_18_{timestamp}" # 10minresample
# save_dir = Path(__file__).parent / f"models/optuna_tryout/best/01"
save_dir.mkdir(exist_ok=True)
hyperparams = {
    "MODEL" : "LSTM reasmple mean 10min",
    "SEQUENCE_LENGTH": 864,
    "HIDDEN_SIZE": 64,
    "NUM_LAYERS": 5,
    "DROPOUT": 0.5,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-3,
    "EPOCHS": 200,
    "PATIENCE": 20,
    "WEIGHT_DECAY": 1e-05,
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f'Using device: {device}\n')

    # ==================== Data Preprocessing ====================
    # Load data
    data_dir = Path("../01_Datenaufbereitung/Output/Calculated/")
    df_train, df_val, df_test = load_data(data_dir)

    # Scale the data
    df_train_scaled, df_val_scaled, df_test_scaled = scale_data(df_train, df_val, df_test, scaler_type='standard')

    # Create the DataSets and DataLoaders
    train_dataset = BatteryDataset(df_train_scaled, hyperparams["SEQUENCE_LENGTH"])
    val_dataset = BatteryDataset(df_val_scaled, hyperparams["SEQUENCE_LENGTH"])
    test_dataset = BatteryDataset(df_test_scaled, hyperparams["SEQUENCE_LENGTH"])

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['BATCH_SIZE'], shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['BATCH_SIZE'], shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['BATCH_SIZE'], shuffle=False, pin_memory=torch.cuda.is_available())

    # ==================== Model Initialization ====================
    # Initialize the LSTM model
    model = SOHLSTM(
        input_size=3,  # The number of features (voltage, current, temperature)
        hidden_size=hyperparams['HIDDEN_SIZE'], 
        num_layers=hyperparams['NUM_LAYERS'], 
        dropout=hyperparams['DROPOUT']
    ).to(device)
    
    # Count the total parameters and calculate the model size
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = total_params * 4 / (1024 ** 2)  # 4 bytes/per parameter, convert to MB

    # Print the model architecture
    print('-' * 70)
    print(f'Model architecture:\n{model}')
    print('-' * 70)
    print(f'Total parameters: {total_params}')
    print(f'Estimated model size: {model_size:.2f} MB')
    print('-' * 70)

    # ==================== Model Training ====================
    # Define the model path and training history path
    save_path = {
        'best': save_dir / 'best_soh_model.pth',
        'last': save_dir / 'last_soh_model.pth',
        'history': save_dir / 'train_history.parquet'
    }

    # Define TRAINING_MODE to control if the model should be trained or loaded
    TRAINING_MODE = True
    # Define which trained model to load and evaluate
    LOAD_MODEL_TYPE = 'best'  # 'best' or 'last'
 
    if TRAINING_MODE:
        # Train and validate the model
        train_and_validate_model(model, train_loader, val_loader, save_path)
    else:
        # Load the model
        selected_model_path = save_path['best'] if LOAD_MODEL_TYPE == 'best' else save_path['last']
        if os.path.exists(selected_model_path):
            print(f"\nLoading {LOAD_MODEL_TYPE} model from {selected_model_path}...")
            model.load_state_dict(torch.load(selected_model_path, map_location=device))
            print("Model loaded successfully!")
        else:
            print(f"\nWarning: Model file {selected_model_path} does not exist.")
            print("Make sure you've trained and saved the model or set TRAINING_MODE=True.")
            exit(1)

    # ==================== Model Evaluation ====================
    # Evaluate the model on the test set
    print("\nEvaluating the model on the testing set...")
    predictions, targets, metrics = evaluate_model(model, test_loader)

    # Print the evaluation metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # ==================== Results Visualization ====================
    if save_path['history'].exists():
        plot_losses(pd.read_parquet(save_path['history']), save_dir)
    plot_results(predictions, targets, metrics, df_test_scaled, save_dir)

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

def load_data(data_dir: Path, resample = '10min'):
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
        # df_sampled = df_processed.iloc[::600].reset_index(drop=True)
        df_sampled = df_processed.resample(resample, on='Datetime').mean().reset_index(drop=False)
        
        df_sampled["cell_id"] = file_path.stem.split('_')[1]
        
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


def scale_data(df_train, df_val, df_test, scaler_type='standard'):
    """
    Scaling the features using StandardScaler or MinMaxScaler.
    params:
        df_train (pd.DataFrame): The training data
        df_val (pd.DataFrame): The validation data
        df_test (pd.DataFrame): The testing data
        scaler_type (str): The type of scaler to use ('standard' or 'minmax')
    return:
        tuple: The scaled training, validation and testing dataframes
    """
    features_to_scale = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']

    # Create a copy of the scaled dataframes
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()

    # Process the features based on the scaler type
    if scaler_type == 'standard':
        scaler = StandardScaler()

        # Fit the features only on the training data
        scaler.fit(df_train[features_to_scale])

        # Transform the features on the training, validation and testing dataframes
        df_train_scaled[features_to_scale] = scaler.transform(df_train[features_to_scale])
        df_val_scaled[features_to_scale] = scaler.transform(df_val[features_to_scale])
        df_test_scaled[features_to_scale] = scaler.transform(df_test[features_to_scale])

        print('Features scaled using StandardScaler\n')

    elif scaler_type == 'minmax':
        # Use a different scaling method for each feature
        voltage_scaler = MinMaxScaler(feature_range=(0, 1))  # Voltage: [0, 1]
        current_scaler = MinMaxScaler(feature_range=(-1, 1))  # Current: [-1, 1]
        temperature_scaler = MinMaxScaler(feature_range=(0, 1))  # Temperature: [0, 1]

        # Fit the features only on the training data
        voltage_scaler.fit(df_train[['Voltage[V]']])
        current_scaler.fit(df_train[['Current[A]']])
        temperature_scaler.fit(df_train[['Temperature[°C]']])

        # Transform the features on the training, validation and testing dataframes
        df_train_scaled['Voltage[V]'] = voltage_scaler.transform(df_train[['Voltage[V]']])
        df_train_scaled['Current[A]'] = current_scaler.transform(df_train[['Current[A]']])
        df_train_scaled['Temperature[°C]'] = temperature_scaler.transform(df_train[['Temperature[°C]']])

        df_val_scaled['Voltage[V]'] = voltage_scaler.transform(df_val[['Voltage[V]']])
        df_val_scaled['Current[A]'] = current_scaler.transform(df_val[['Current[A]']])
        df_val_scaled['Temperature[°C]'] = temperature_scaler.transform(df_val[['Temperature[°C]']])

        df_test_scaled['Voltage[V]'] = voltage_scaler.transform(df_test[['Voltage[V]']])
        df_test_scaled['Current[A]'] = current_scaler.transform(df_test[['Current[A]']])
        df_test_scaled['Temperature[°C]'] = temperature_scaler.transform(df_test[['Temperature[°C]']])

        print('Features scaled using MinMaxScaler (Voltage: [0, 1], Current: [-1, 1], Temperature: [0, 1])\n')
    
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
    
# class BatteryDataset(Dataset):
#     def __init__(self, df, sequence_length=60):
#         self.sequence_length = sequence_length
        
#         # Get the features and labels
#         features_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
#         label_col = 'SOH_ZHU'
        
#         features_list = []
#         labels_list = []
#         cell_ids = []  # 可选：存储每个窗口对应的 cell id
#         for cell, group in df.groupby('cell_id'):
#             # 确保数据按时间顺序排列
#             group = group.sort_values('Datetime')
#             features_tensor = torch.tensor(group[features_cols].values, dtype=torch.float32)
#             labels_tensor = torch.tensor(group[label_col].values, dtype=torch.float32)
#             n_samples = len(group) - sequence_length
#             if n_samples <= 0:
#                 continue  # 如果该 cell 数据不足一个窗口则跳过
#             for i in range(n_samples):
#                 features_list.append(features_tensor[i:i+sequence_length])
#                 labels_list.append(labels_tensor[i+sequence_length])
#                 cell_ids.append(cell)
                
#         self.features = torch.stack(features_list)
#         self.labels = torch.tensor(labels_list, dtype=torch.float32)
#         self.cell_ids = cell_ids  # 如果后续需要用到 cell id，可以返回或进一步处理

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]
    
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
    
    def get_params(self):
        """Get model parameters as a flat vector"""
        params = []
        for param in self.parameters():
            params.append(param.view(-1))
        return torch.cat(params)

def train_and_validate_model(model, train_loader, val_loader, save_path):
    """
    Train and validate the model with early stopping.
    
    Parameters:
        model: The PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        save_path: Dict with paths for saving the model and history
        
    Returns:
        history: Training history dictionary
    """
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['LEARNING_RATE'], weight_decay=hyperparams['WEIGHT_DECAY'])
    # Define learning rate scheduler and early stopping
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Initialize the early stopping parameters
    epochs_no_improve = 0
    best_val_loss = float('inf')

    # Define the history to store training and validation loss
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': []
    }

    # Start training
    print('\nStart training...')
    for epoch in range(hyperparams['EPOCHS']):
        # ==================== Training phase ====================
        model.train()
        train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{hyperparams["EPOCHS"]}', leave=False) as pbar:
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
                # Update the training loss and progress bar
                train_loss += loss.item()
                pbar.update(1)

        # Update the average training loss
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # ==================== Validation phase ====================
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

        # Display the training and validation loss and learning rate
        print(f'Epoch {epoch + 1}/{hyperparams["EPOCHS"]} | '
              f'Training Loss: {train_loss:.3e} | '
              f'Validation Loss: {val_loss:.3e} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # Check for early stopping and save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path['best'])
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= hyperparams['PATIENCE']:
                print(f'Early stopping triggered after {epoch + 1} epochs!')
                break
        
    # Save the last model (whether early stopping or not)
    torch.save(model.state_dict(), save_path['last'])
    print(f'\nlast model saved to {save_path["last"]}')
    # Save the training history
    history_df = pd.DataFrame(history)
    history_df.to_parquet(save_path['history'], index=False)
    print(f'Training history saved to {save_path["history"]}')

    return history, best_val_loss

def evaluate_model(model, data_loader):
    """
    Evaluate the model on a dataset and calculate performance metrics.
    
    Parameters:
        model: The trained PyTorch model
        data_loader: DataLoader for the evaluation data
        
    Returns:
        predictions: Model predictions
        targets: Actual values
        metrics: Dictionary with performance metrics
    """
    # Evaluate the model
    model.eval()

    # Define the loss function (MSE)
    criterion = nn.MSELoss()
    total_loss = 0.0

    # Initialize the lists to store the actual and predicted values
    all_predictions = []
    all_targets = []

    # Start the evaluation loop
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            # Forward propagation and calculate the loss
            outputs = model(features)
            loss = criterion(outputs, labels)
            # Update the total loss
            total_loss += loss.item()

            # Append actual and predicted values to the lists
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Average the total loss
    total_loss /= len(data_loader)

    # Concatenate the lists to numpy arrays
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    # Calculate the metrics (RMSE, MAE, MAPE, R²)
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(targets, predictions)),
        'MAE': mean_absolute_error(targets, predictions),
        'MAPE': mean_absolute_percentage_error(targets, predictions),
        'R²': r2_score(targets, predictions)
    }

    return predictions, targets, metrics

def plot_losses(history_df: pd.DataFrame, save_dir):
    """
    Plot training and validation losses from the training history.
    
    Parameters:
        history_df (pd.DataFrame): DataFrame containing training metrics
    """
    results_dir = save_dir / "results"
    results_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(history_df['epoch'], history_df['train_loss'], label='Training Loss', 
            color='#2e78cc', marker='o', markersize=4, lw=2)
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
    file_path = os.path.join(results_dir, "train_val_loss.png")
    if not os.path.exists(file_path):
        plt.savefig(file_path)
    plt.close()

import os

def plot_results(predictions, targets, metrics, df_test_scaled, save_dir):
    # 如果文件夹不存在，则创建
    results_dir = save_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # 1. 绘制基于时间的实际与预测 SOH 值
    plt.figure(figsize=(10, 6))
    datetime_vals = df_test_scaled['Datetime'].iloc[hyperparams['SEQUENCE_LENGTH']:].values
    plt.plot(datetime_vals, targets, label='Actual SOH', color='blue', lw=2)
    plt.plot(datetime_vals, predictions, label='Predicted SOH', color='red', alpha=0.5)
    plt.title('Actual vs. Predicted SOH values\n'
              f'(RMSE={metrics["RMSE"]:.4f}, MAE={metrics["MAE"]:.4f}, '
              f'MAPE={metrics["MAPE"]:.4f}, R²={metrics["R²"]:.4f})')
    plt.xlabel('Datetime')
    plt.ylabel('SOH')
    plt.legend()
    plt.grid(color='lightgrey', linewidth=0.5)
    plt.gcf().autofmt_xdate()  # 自动格式化日期标签
    plt.tight_layout()
    file_path = os.path.join(results_dir, "prediction.png")
    if not os.path.exists(file_path):
        plt.savefig(file_path)
    plt.close()

    # 2. 绘制实际值与预测值的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.title('Actual vs. Predicted SOH values\n'
              f'(RMSE={metrics["RMSE"]:.4f}, MAE={metrics["MAE"]:.4f}, '
              f'MAPE={metrics["MAPE"]:.4f}, R²={metrics["R²"]:.4f})')
    plt.xlabel('Actual SOH')
    plt.ylabel('Predicted SOH')
    plt.grid(color='lightgrey', linewidth=0.5)
    plt.tight_layout()
    file_path = os.path.join(results_dir, "scatter_predict.png")
    if not os.path.exists(file_path):
        plt.savefig(file_path)
    plt.close()

    # 3. 绘制预测误差分布直方图（带核密度估计）
    plt.figure(figsize=(10, 6))
    errors = targets - predictions
    sns.histplot(errors, kde=True)
    plt.title('Prediction Errors Distribution')
    plt.xlabel('Prediction Errors')
    plt.ylabel('Density')
    plt.grid(color='lightgrey', linewidth=0.5)
    plt.tight_layout()
    file_path = os.path.join(results_dir, "errors_hist.png")
    if not os.path.exists(file_path):
        plt.savefig(file_path)
    plt.close()

    # 4. 绘制实际值与预测误差的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.title('Actual SOH vs. Prediction Errors')
    plt.xlabel('Actual SOH')
    plt.ylabel('Prediction Errors')
    plt.grid(color='lightgrey', linewidth=0.5)
    plt.tight_layout()
    file_path = os.path.join(results_dir, "errors_scatter.png")
    if not os.path.exists(file_path):
        plt.savefig(file_path)
    plt.close()



# This conditional executes the main function when the script is run directly
if __name__ == "__main__":
    main()