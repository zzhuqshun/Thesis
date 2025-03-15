import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import random
from collections import deque
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from tqdm import tqdm

# Set configurations
SEQUENCE_LENGTH = 144
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 20

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_data(data_dir: Path): 
    """
    Load data for training, validation and testing.
    """
    folders = sorted(
        [p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith('MGFarm_18650')],
        key=lambda x: int(x.name.split('_')[-1][1:])
    )
    
    test_folder = random.choice(folders)
    train_val_folders = [f for f in folders if f != test_folder]
    val_folders = random.sample(train_val_folders, len(train_val_folders) // 5)
    train_folders = [f for f in train_val_folders if f not in val_folders]

    def process_cell_data(folder):
        """Internal function to process each cell data."""
        df = pd.read_parquet(folder / 'df_LZ.parquet')
        
        # Only keep the important columns to reduce the memory usage
        columns_to_keep = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 
                           'Temperature[°C]', 'SOC_LZ', 'SOH_LZ']
        df_processed = df[columns_to_keep].copy()
        
        # Optimize the time processing and add the datetime column
        df_processed['Testtime[s]'] = df_processed['Testtime[s]'].round().astype(int)
        start_date = pd.Timestamp("2023-02-02")
        df_processed['Datetime'] = pd.date_range(start=start_date, periods=len(df_processed), freq='s')
        
        # Sample the data to reduce the size
        return df_processed.iloc[::600].reset_index(drop=True), folder.name
    
    # Process the data for training, validation and testing sets
    test_data = [process_cell_data(test_folder)]
    val_data = [process_cell_data(f) for f in val_folders]
    train_data = [process_cell_data(f) for f in train_folders]
    
    # Print the cell names for training, validation and testing sets
    print(f'Training cells: {[t[1] for t in train_data]}')
    print(f'Validation cells: {[v[1] for v in val_data]}')
    print(f'Testing cell: {test_data[0][1]}')
    
    # Concatenate the dataframes
    df_train = pd.concat([t[0] for t in train_data])
    df_val = pd.concat([v[0] for v in val_data])
    df_test = test_data[0][0]
    
    print(f'\nTraining dataframe shape: {df_train.shape}')
    print(f'Validation dataframe shape: {df_val.shape}')
    print(f'Testing dataframe shape: {df_test.shape}\n')
    
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
        label_col = 'SOH_LZ'
        features = torch.tensor(df[features_cols].values, dtype=torch.float32)
        labels = torch.tensor(df[label_col].values, dtype=torch.float32)
        
        # Create the sequence data using efficient preallocated memory method
        n_samples = len(df) - sequence_length
        self.features = torch.zeros((n_samples, sequence_length, len(features_cols)), dtype=torch.float32)
        self.labels = torch.zeros(n_samples, dtype=torch.float32)
        
        # Create the sequence window data for each sample using tensor slicing
        for i in range(n_samples):
            self.features[i] = features[i:i+sequence_length]
            self.labels[i] = labels[i+sequence_length-1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
# 1. Original LSTM model
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

        # Only take the output from the final time
        out = lstm_out[:, -1, :]  # The shape: [batch_size, hidden_size]

        # Through the fully connected layer
        out = self.fc_layers(out)
        
        return out.squeeze(-1)
    
# 2. Bidirectional LSTM model (BiLSTM)
class SOHBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(SOHBiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # Enable bidirectional LSTM layer
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layer (Output of BiLSTM is 2*hidden_size)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size),  # Batch normalization layer
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        batch_size = x.size(0)
        # The hidden state for BiLSTM is 2*hidden_size
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        
        # Forward propagate BiLSTM
        bilstm_out, _ = self.bilstm(x, (h0, c0))
        
        # Only take the output from the final time step
        out = bilstm_out[:, -1, :]  # The shape: [batch_size, 2*hidden_size]
        
        # Through the fully connected layer
        out = self.fc_layers(out)
        
        return out.squeeze(-1)
    
# 3. CNN-LSTM model
class SOHCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(SOHCNNLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN layers (for local feature extraction)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # LSTM layer
        self.lstm_layers = nn.LSTM(
            input_size=64,  # The number of output channels from the CNN layers
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
        batch_size = x.size(0)
        
        # Convert the input dimensions to fit Conv1d: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        
        # Apply the CNN layers
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        
        # Convert back to fit LSTM dimensions: [batch, features, seq_len] -> [batch, seq_len, features]
        x = x.permute(0, 2, 1)
        
        # Initialize the LSTM hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm_layers(x, (h0, c0))
        
        # Only take the output from the final time step
        out = lstm_out[:, -1, :]
        
        # Through the fully connected layer
        out = self.fc_layers(out)
        
        return out.squeeze(-1)
    
# 4. CNN-LSTM with Attention Mechanism model
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        # Attention weights calculation layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, lstm_output):  # lstm_output: [batch_size, seq_len, hidden_size]
        # Calculate the attention weights
        attn_weights = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        
        # Obtain normalized attention weights using softmax
        attn_weights = F.softmax(attn_weights, dim=1)  # Softmax along the seq_len dimension
        
        # Calculate the context vector
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_output)  # [batch_size, 1, hidden_size]
        context = context.squeeze(1)  # [batch_size, hidden_size]
        
        return context, attn_weights

class SOHCNNLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(SOHCNNLSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN layers (for local feature extraction)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=64,  # The number of output channels from the CNN layers
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_size)
        
        # Fully connected layer
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Convert the input dimensions to fit Conv1d: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        
        # Apply the CNN layers
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        
        # Convert back to fit LSTM dimensions: [batch, features, seq_len] -> [batch, seq_len, features]
        x = x.permute(0, 2, 1)
        
        # Initialize the LSTM hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply the attention mechanism
        attn_output, attn_weights = self.attention(lstm_out)
        
        # Through the fully connected layer
        out = self.fc_layers(attn_output)
        
        return out.squeeze(-1), attn_weights

def train_and_validate_model(model, train_loader, val_loader, save_path):
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Define learning rate scheduler and early stopping
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

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
    for epoch in range(EPOCHS):
        # ==================== Training phase ====================
        model.train()
        train_loss = 0.0

        # with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{EPOCHS}') as pbar:
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{EPOCHS}', leave=False) as pbar:
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                # Clear the previous gradients
                optimizer.zero_grad()
                # Forward propagation and calculate the loss
                outputs = model(features)
                # Check if the output is a tuple (to handle models with attention weights)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
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
                # Check if the output is a tuple (to handle models with attention weights)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
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
        print(f'Epoch {epoch + 1}/{EPOCHS} | '
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
            if epochs_no_improve >= PATIENCE:
                print(f'Eealy stopping triggered after {epoch + 1} epochs!')
                break
        
    # Save the final model (whether early stopping or not)
    torch.save(model.state_dict(), save_path['final'])
    print(f'\nFinal model saved to {save_path["final"]}')
    # Save the training history
    history_df = pd.DataFrame(history)
    history_df.to_parquet(save_path['history'], index=False)
    print(f'Training history saved to {save_path["history"]}')

    return history

def evaluate_model(model, data_loader):
    # Evaluate the model
    model.eval()

    # Define the loss function (MSE)
    criterion = nn.MSELoss()
    total_loss = 0.0

    # Initialize the lists to store the actual and predicted values
    all_predictions = []
    all_actuals = []
    all_attention_weights = []  # Store the attention weights if available

    # Start the evaluation loop
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            # Forward propagation and calculate the loss
            outputs = model(features)
            # Check if the output is a tuple (to handle models with attention weights)
            if isinstance(outputs, tuple):
                attention_weights = outputs[1]  # Get the attention weights
                all_attention_weights.append(attention_weights.cpu().numpy())
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            # Update the total loss
            total_loss += loss.item()

            # Append actual and predicted values to the lists
            all_predictions.append(outputs.cpu().numpy())
            all_actuals.append(labels.cpu().numpy())

    # Average the total loss
    total_loss /= len(data_loader)

    # Concatenate the lists to numpy arrays
    predictions = np.concatenate(all_predictions)
    actuals = np.concatenate(all_actuals)

    # Calculate the metrics (RMSE, MAE, MAPE, R²)
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
        'MAE': mean_absolute_error(actuals, predictions),
        'MAPE': mean_absolute_percentage_error(actuals, predictions),
        'R²': r2_score(actuals, predictions)
    }

    if all_attention_weights:
        return predictions, actuals, metrics, all_attention_weights
    else:
        return predictions, actuals, metrics

def plot_losses(history_df: pd.DataFrame):
    """
    Plot training and validation losses from the training history.
    Parameters:
        history_df (pd.DataFrame): DataFrame containing training metrics
    """
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
    # plt.show()

def visualize_attention(attention_weights, sample_idx=0, head_idx=0):
    """
    Visualize the attention weights.
    Args:
        attention_weights: Attention weighting list
        sample_idx: Index of samples to be visualized
        head_idx: Header Indexing in Multiattention
    """
    # Check if the attention weights are available
    if not attention_weights:
        print("No attention weights available for visualization")
        return
    
    # Get the attention weights for the specified sample and head
    sample_weights = attention_weights[sample_idx]
    
    if len(sample_weights.shape) == 4:  # [batch, heads, seq, seq]
        attn = sample_weights[0, head_idx]  # Take the first batch and head
    else:  # [batch, seq, seq]
        attn = sample_weights[0]  # Take the first batch
    
    # Plot the attention weights
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='viridis')
    plt.colorbar()
    plt.title(f'Attention Weights (Sample {sample_idx}, Head {head_idx})')
    plt.xlabel('Sequence Position (Key)')
    plt.ylabel('Sequence Position (Query)')
    plt.tight_layout()
    # plt.show()

def plot_results(predictions, actuals, metrics, df_test_scaled):
    fig = plt.figure(figsize=(15, 12))

    # 1. Plot the actual vs. predicted SOH values
    ax1 = fig.add_subplot(221)
    # Ensure the alignment of datetime and predictions by skipping the first SEQUENCE_LENGTH offsets
    datetime = df_test_scaled['Datetime'].iloc[SEQUENCE_LENGTH:].values
    ax1.plot(datetime, actuals, label='Actual SOH', color='blue', ls='-', lw=2)
    ax1.plot(datetime, predictions, label='Predicted SOH', color='red', alpha=0.5)
    ax1.set_title('Actual vs. Predicted SOH values over Datetime')
    ax1.set_xlabel('Datetime')
    ax1.set_ylabel('SOH')
    ax1.legend()
    ax1.grid(color='lightgrey', linewidth=0.5)

    # Format the x-axis to display the appropriate date labels
    fig.autofmt_xdate()  # Auto-adjust the date labels to prevent overlap

    # 2. Plot the scatter plot of the actual vs. predicted SOH values
    ax2 = fig.add_subplot(222)
    ax2.scatter(actuals, predictions, alpha=0.5)
    ax2.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    ax2.set_title('Actual vs. Predicted SOH values\n' +
                  f'(RMSE={metrics["RMSE"]:.4f}, MAE={metrics["MAE"]:.4f}, MAPE={metrics["MAPE"]:.4f}, R²={metrics["R²"]:.4f})')
    ax2.set_xlabel('Actual SOH')
    ax2.set_ylabel('Predicted SOH')
    ax2.grid(color='lightgrey', linewidth=0.5)

    # 3. Plot the error distribution
    ax3 = fig.add_subplot(223)
    errors = actuals - predictions
    sns.histplot(errors, kde=True, ax=ax3)
    ax3.set_title('Prediction Errors Distribution')
    ax3.set_xlabel('Prediction Errors')
    ax3.set_ylabel('Density')
    ax3.grid(color='lightgrey', linewidth=0.5)

    # 4. Plot the scatter plot of the actual vs. prediction errors
    ax4 = fig.add_subplot(224)
    ax4.scatter(actuals, errors, alpha=0.5)
    ax4.axhline(y=0, color='r', linestyle='--', lw=2)
    ax4.set_title('Actual SOH vs. Prediction Errors')
    ax4.set_xlabel('Actual SOH')
    ax4.set_ylabel('Prediction Errors')
    ax4.grid(color='lightgrey', linewidth=0.5)

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set the seed and device
    set_seed(6)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f'Using device: {device}\n')

    # ==================== Data Preprocessing ====================
    # Load data
    data_dir = Path('/home/l1qun/Documents/MA/DLModels/TEST/MGFarm_18650_Dataframes/')
    df_train, df_val, df_test = load_data(data_dir)

    # Scale the data
    df_train_scaled, df_val_scaled, df_test_scaled = scale_data(df_train, df_val, df_test, scaler_type='standard')

    # Create the DataSets and DataLoaders
    train_dataset = BatteryDataset(df_train_scaled, SEQUENCE_LENGTH)
    val_dataset = BatteryDataset(df_val_scaled, SEQUENCE_LENGTH)
    test_dataset = BatteryDataset(df_test_scaled, SEQUENCE_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())

    # ==================== Model Initialization ====================
    # Initialize the model
    # 选择模型类型：'lstm', 'bilstm', 'cnn_lstm', 'cnn_lstm_attention'
    MODEL_TYPE = 'cnn_lstm'  

    if MODEL_TYPE == 'lstm':
        # Option 1: Original LSTM model
        model = SOHLSTM(
            input_size=3,  # The number of features (voltage, current, temperature)
            hidden_size=64, 
            num_layers=2, 
            dropout=0.2
        ).to(device)
    elif MODEL_TYPE == 'bilstm':
        # Option 2: Bidirectional LSTM model
        model = SOHBiLSTM(
            input_size=3,
            hidden_size=32, 
            num_layers=2, 
            dropout=0.2
        ).to(device)
    elif MODEL_TYPE == 'cnn_lstm':
        # Option 3: CNN-LSTM model
        model = SOHCNNLSTM(
            input_size=3,
            hidden_size=64, 
            num_layers=2,
            dropout=0.2
        ).to(device)
    elif MODEL_TYPE == 'cnn_lstm_attention':
        # Option 4: CNN-LSTM with Attention Mechanism model
        model = SOHCNNLSTMAttention(
            input_size=3,
            hidden_size=64, 
            num_layers=2,
            dropout=0.2,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}. Choose from 'lstm', 'bilstm', 'cnn_lstm', 'cnn_lstm_attention'.")
    
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
    save_dir = Path(os.path.dirname(__file__)) / 'models'
    save_dir.mkdir(exist_ok=True)
    save_path = {
        'best': save_dir / 'best_soh_model.pth',
        'final': save_dir / 'final_soh_model.pth',
        'history': save_dir / 'train_history.parquet'
    }

    # Define TRAINING_MODE to control if the model should be trained or loaded
    TRAINING_MODE = True
    # Define which trained model to load and evaluate
    LOAD_MODEL_TYPE = 'final'  # 'best' or 'final'

    if TRAINING_MODE:
        # Train and validate the model
        history = train_and_validate_model(model, train_loader, val_loader, save_path)
    else:
        # Load the model according to the MODEL_TYPE
        selected_model_path = save_path['best'] if LOAD_MODEL_TYPE == 'best' else save_path['final']
        # selected_model_path = '/home/l1qun/Documents/MA/DLModels/TEST/LSTM/models/benchmark/cnnlstm/final_soh_model.pth'
        # Load one of the trained models
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
    eval_results = evaluate_model(model, test_loader)

    # Check if the attention weights are available
    if len(eval_results) == 4:  # Contains attention weights
        predictions, actuals, metrics, attention_weights = eval_results
        print("Attention weights captured for visualization")
        # Visualize the attention weights
        if attention_weights is not None and len(attention_weights) > 0:
            visualize_attention(attention_weights)
    else:  # No attention weights
        predictions, actuals, metrics = eval_results

    # Print the evaluation metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # ==================== Results Visualization ====================
    if save_path['history'].exists():
        plot_losses(pd.read_parquet(save_path['history']))
    plot_results(predictions, actuals, metrics, df_test_scaled)