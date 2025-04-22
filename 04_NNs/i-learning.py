import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
from pathlib import Path
from datetime import datetime
import copy
from tqdm import tqdm

# ===============================================================
# Main Function for EWC Incremental Learning
# ===============================================================

def main():
    """Main function for incremental learning with EWC"""
    # Setup directories and device
    save_dir = Path(__file__).parent / "models/EWC" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define hyperparameters
    hyperparams = {
        "SEQUENCE_LENGTH": 1008,  # Length of input sequence
        "HIDDEN_SIZE": 128,       # Size of LSTM hidden units
        "NUM_LAYERS": 3,          # Number of LSTM layers
        "DROPOUT": 0.5,           # Dropout rate
        "BATCH_SIZE": 32,         # Training batch size
        "LEARNING_RATE": 1e-4,    # Initial learning rate
        "EPOCHS": 200,            # Maximum number of epochs
        "PATIENCE": 20,           # Early stopping patience
        "WEIGHT_DECAY": 0.0,      # L2 regularization strength
        "EWC_LAMBDA": 5000,       # EWC regularization strength
    }
    
    # Save hyperparameters
    with open(save_dir / "hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=4)
    
    print(f"Using device: {device}")
    print(f"Running Elastic Weight Consolidation (EWC) for incremental learning\n")

    # Set random seed for reproducibility
    set_seed(42)

    # Load and prepare data
    data_dir = Path("../01_Datenaufbereitung/Output/Calculated/")  # Update with your data directory
    print("Loading data from:", data_dir)
    df_base, df_update1, df_update2, df_test = load_and_prepare_data(data_dir)
    
    # Normalize data using StandardScaler
    print("Normalizing data...")
    df_base_scaled, df_update1_scaled, df_update2_scaled, df_test_scaled = scale_data(
        df_base, df_update1, df_update2, df_test
    )
    
    # Split into train/validation sets by cell
    print("Splitting data into train/validation sets...")
    df_base_train, df_base_val = split_by_cell(df_base_scaled, "Base", val_cells=1)
    df_update1_train, df_update1_val = split_by_cell(df_update1_scaled, "Update1", val_cells=1)
    df_update2_train, df_update2_val = split_by_cell(df_update2_scaled, "Update2", val_cells=1)

    # Create datasets for training and validation
    print("Creating datasets...")
    base_train_dataset = BatteryDataset(df_base_train, hyperparams["SEQUENCE_LENGTH"])
    base_val_dataset = BatteryDataset(df_base_val, hyperparams["SEQUENCE_LENGTH"])
    update1_train_dataset = BatteryDataset(df_update1_train, hyperparams["SEQUENCE_LENGTH"])
    update1_val_dataset = BatteryDataset(df_update1_val, hyperparams["SEQUENCE_LENGTH"])
    update2_train_dataset = BatteryDataset(df_update2_train, hyperparams["SEQUENCE_LENGTH"])
    update2_val_dataset = BatteryDataset(df_update2_val, hyperparams["SEQUENCE_LENGTH"])

    # Create data loaders
    print("Creating data loaders...")
    base_train_loader = DataLoader(base_train_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    base_val_loader = DataLoader(base_val_dataset, batch_size=hyperparams["BATCH_SIZE"])
    update1_train_loader = DataLoader(update1_train_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    update1_val_loader = DataLoader(update1_val_dataset, batch_size=hyperparams["BATCH_SIZE"])
    update2_train_loader = DataLoader(update2_train_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    update2_val_loader = DataLoader(update2_val_dataset, batch_size=hyperparams["BATCH_SIZE"])
    
    # Split test set into 3 segments for each phase
    print("Splitting test set into segments...")
    timestamps = df_test_scaled['Datetime'].values

    # Create time boundaries to divide test set into 3 equal parts
    time_boundary1 = timestamps[len(timestamps) // 3]  
    time_boundary2 = timestamps[2 * len(timestamps) // 3]  

    # Create boolean masks for each segment
    test_idx1 = df_test_scaled['Datetime'] < time_boundary1
    test_idx2 = (df_test_scaled['Datetime'] >= time_boundary1) & (df_test_scaled['Datetime'] < time_boundary2)
    test_idx3 = df_test_scaled['Datetime'] >= time_boundary2

    # Create test datasets for each segment
    test_dataset1 = BatteryDataset(df_test_scaled[test_idx1], hyperparams["SEQUENCE_LENGTH"])
    test_dataset2 = BatteryDataset(df_test_scaled[test_idx2], hyperparams["SEQUENCE_LENGTH"])
    test_dataset3 = BatteryDataset(df_test_scaled[test_idx3], hyperparams["SEQUENCE_LENGTH"])

    # Create test data loaders
    base_test_loader = DataLoader(test_dataset1, batch_size=hyperparams["BATCH_SIZE"])
    update1_test_loader = DataLoader(test_dataset2, batch_size=hyperparams["BATCH_SIZE"])
    update2_test_loader = DataLoader(test_dataset3, batch_size=hyperparams["BATCH_SIZE"])

    # ================= EWC Incremental Learning =================
    print("=" * 80)
    print("Method: Elastic Weight Consolidation (EWC)")
    print("=" * 80)
    
    # Initialize the LSTM model for SOH prediction
    input_size = 3  # Voltage, Current, Temperature
    base_model = SOHLSTM(
        input_size=input_size,
        hidden_size=hyperparams["HIDDEN_SIZE"],
        num_layers=hyperparams["NUM_LAYERS"],
        dropout=hyperparams["DROPOUT"]
    ).to(device)
    
    # Phase 1: Train on base data
    print("\nPhase 1: Training on base data...")
    base_model, base_history = train_model(
        model=base_model,
        train_loader=base_train_loader,
        val_loader=base_val_loader,
        epochs=hyperparams["EPOCHS"],
        lr=hyperparams["LEARNING_RATE"],
        weight_decay=hyperparams["WEIGHT_DECAY"],
        patience=hyperparams["PATIENCE"]
    )
    
    # Save base model and training history
    torch.save(base_model.state_dict(), save_dir / "ewc_base_model.pt")
    base_history_df = pd.DataFrame(base_history)
    base_history_df.to_parquet(save_dir / "ewc_base_history.parquet", index=False)
    
    # Evaluate base model on test data
    print("\nEvaluating base model...")
    base_pred, base_targets, base_metrics, _ = evaluate_model(base_model, base_test_loader)
    print(f"Base model metrics: {base_metrics}")
    
    # Phase 2: Initialize EWC after training on base data
    print("\nInitializing EWC...")
    ewc = EWC(base_model, device, lambda_ewc=hyperparams["EWC_LAMBDA"])
    
    # Compute Fisher information matrix using base data
    print("Computing Fisher information matrix...")
    ewc.compute_fisher(base_train_loader)
    
    # Update optimal parameters for EWC
    ewc.update_optimal_params()
    
    # Train on update1 data with EWC regularization
    print("\nPhase 2: Training on update1 data with EWC regularization...")
    base_model, update1_history = train_model(
        model=base_model,
        train_loader=update1_train_loader,
        val_loader=update1_val_loader,
        epochs=hyperparams["EPOCHS"],
        lr=hyperparams["LEARNING_RATE"],
        weight_decay=hyperparams["WEIGHT_DECAY"],
        patience=hyperparams["PATIENCE"],
        ewc=ewc  # Use EWC for regularization
    )
    
    # Save model after first update
    torch.save(base_model.state_dict(), save_dir / "ewc_update1_model.pt")
    update1_history_df = pd.DataFrame(update1_history)
    update1_history_df.to_parquet(save_dir / "ewc_update1_history.parquet", index=False)
    
    # Evaluate after first update
    print("\nEvaluating after first update...")
    update1_pred, update1_targets, update1_metrics, _ = evaluate_model(base_model, update1_test_loader)
    print(f"Update 1 metrics: {update1_metrics}")
    
    # Phase 3: Update Fisher information matrix for second update
    print("\nUpdating Fisher information matrix...")
    # Combine base and update1 data for Fisher calculation
    combined_loader = DataLoader(
        ConcatDataset([base_train_dataset, update1_train_dataset]),
        batch_size=hyperparams["BATCH_SIZE"],
        shuffle=True
    )
    ewc.compute_fisher(combined_loader)
    ewc.update_optimal_params()
    
    # Train on update2 data with updated EWC regularization
    print("\nPhase 3: Training on update2 data with EWC regularization...")
    base_model, update2_history = train_model(
        model=base_model,
        train_loader=update2_train_loader,
        val_loader=update2_val_loader,
        epochs=hyperparams["EPOCHS"],
        lr=hyperparams["LEARNING_RATE"],
        weight_decay=hyperparams["WEIGHT_DECAY"],
        patience=hyperparams["PATIENCE"],
        ewc=ewc  # Use updated EWC for regularization
    )
    
    # Save final model and training history
    torch.save(base_model.state_dict(), save_dir / "ewc_update2_model.pt")
    update2_history_df = pd.DataFrame(update2_history)
    update2_history_df.to_parquet(save_dir / "ewc_update2_history.parquet", index=False)
    
    # Evaluate after second update
    print("\nEvaluating after second update...")
    update2_pred, update2_targets, update2_metrics, _ = evaluate_model(base_model, update2_test_loader)
    print(f"Update 2 metrics: {update2_metrics}")
    
    # Plot results
    results_dir = Path(save_dir) / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot prediction results with metrics
    plot_results(save_dir, "Parameter based(EWC)", df_test_scaled, hyperparams["SEQUENCE_LENGTH"],
                 base_pred, update1_pred, update2_pred, base_metrics, update1_metrics, update2_metrics)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    
    # Plot base training and validation loss
    plt.plot(base_history["epoch"], base_history["train_loss"], label="Base Training Loss")
    plt.plot(base_history["epoch"], base_history["val_loss"], label="Base Validation Loss")
    
    # Plot update1 training and validation loss (shifted epochs)
    epoch_shift = max(base_history["epoch"])
    plt.plot([e + epoch_shift for e in update1_history["epoch"]], 
             update1_history["train_loss"], label="Update1 Training Loss")
    plt.plot([e + epoch_shift for e in update1_history["epoch"]], 
             update1_history["val_loss"], label="Update1 Validation Loss")
    
    # Plot update2 training and validation loss (shifted epochs)
    epoch_shift += max(update1_history["epoch"])
    plt.plot([e + epoch_shift for e in update2_history["epoch"]], 
             update2_history["train_loss"], label="Update2 Training Loss")
    plt.plot([e + epoch_shift for e in update2_history["epoch"]], 
             update2_history["val_loss"], label="Update2 Validation Loss")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title("EWC Training and Validation Loss")
    
    plt.savefig(results_dir / "ewc_learning_curves.png")
    plt.close()
    
    print("\nAll results saved to:", results_dir)
    print("EWC Incremental Learning Finished!")

# ===============================================================
# Utility Functions
# ===============================================================

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ===============================================================
# Data Processing Functions
# ===============================================================

class BatteryDataset(Dataset):
    """Dataset for battery SOH prediction"""
    def __init__(self, df, sequence_length):
        self.sequence_length = sequence_length
        
        # Features and target
        self.features = torch.tensor(df[['Voltage[V]', 'Current[A]', 'Temperature[°C]']].values, dtype=torch.float32)
        self.targets = torch.tensor(df['SOH_ZHU'].values, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length - 1]
        return x, y

def load_and_prepare_data(data_dir, resample='10min'):
    """Load and prepare battery datasets for incremental learning"""
    parquet_files = sorted(
        [f for f in data_dir.glob('*.parquet') if f.is_file()],
        key=lambda x: int(x.stem.split('_')[-1])
    )

    # Random assignment of files to different sets
    test_file = random.choice(parquet_files)
    remaining_files = [f for f in parquet_files if f != test_file]
    random.shuffle(remaining_files)

    # Assign files to different sets
    base_files = remaining_files[:4]
    update1_files = remaining_files[4:9]
    update2_files = remaining_files[9:14]
    
    def process_file(file_path):
        df = pd.read_parquet(file_path)
        columns_to_keep = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 
                          'Temperature[°C]', 'SOC_ZHU', 'SOH_ZHU']
        df_processed = df[columns_to_keep].copy()
        df_processed.dropna(inplace=True)
        
        df_processed['Testtime[s]'] = df_processed['Testtime[s]'].round().astype(int)
        start_date = pd.Timestamp("2023-02-02")
        df_processed['Datetime'] = pd.date_range(
            start=start_date,
            periods=len(df_processed),
            freq='s'
        )
        
        df_sampled = df_processed.resample(resample, on='Datetime').mean().reset_index(drop=False)
        df_sampled["cell_id"] = file_path.stem.split('_')[1]
        return df_sampled, file_path.name
    
    test_data = process_file(test_file)
    base_data = [process_file(f) for f in base_files]
    update1_data = [process_file(f) for f in update1_files]
    update2_data = [process_file(f) for f in update2_files]
    
    print(f"Test cell: {test_data[1]}")
    print(f"Base training cells: {[t[1] for t in base_data]}")
    print(f"Update 1 cells: {[u[1] for u in update1_data]}")
    print(f"Update 2 cells: {[u[1] for u in update2_data]}")

    df_test = test_data[0]
    df_base = pd.concat([t[0] for t in base_data], ignore_index=True)
    df_update1 = pd.concat([u[0] for u in update1_data], ignore_index=True)
    df_update2 = pd.concat([u[0] for u in update2_data], ignore_index=True)
    
    print(f"\nBase training data shape: {df_base.shape}")
    print(f"Update 1 data shape: {df_update1.shape}")
    print(f"Update 2 data shape: {df_update2.shape}")
    print(f"Test data shape: {df_test.shape}\n")
    
    return df_base, df_update1, df_update2, df_test

def scale_data(df_base, df_update1, df_update2, df_test):
    """Normalize datasets using StandardScaler fitted on base data"""
    features_to_scale = ['Voltage[V]', 'Current[A]', 'Temperature[°C]', 'SOH_ZHU']
    
    df_base_scaled = df_base.copy()
    df_update1_scaled = df_update1.copy()
    df_update2_scaled = df_update2.copy()
    df_test_scaled = df_test.copy()
    
    scaler = StandardScaler()
    scaler.fit(df_base[features_to_scale])
    
    df_base_scaled[features_to_scale] = scaler.transform(df_base[features_to_scale])
    df_test_scaled[features_to_scale] = scaler.transform(df_test[features_to_scale])
    df_update1_scaled[features_to_scale] = scaler.transform(df_update1[features_to_scale])
    df_update2_scaled[features_to_scale] = scaler.transform(df_update2[features_to_scale])
    
    print('Features scaled using StandardScaler fitted on base training data.\n')
    
    return df_base_scaled, df_update1_scaled, df_update2_scaled, df_test_scaled

def split_by_cell(df, name, val_cells=1, seed=42):
    """Split dataset into training and validation sets based on cell_id"""
    np.random.seed(seed)
    cell_ids = df['cell_id'].unique().tolist()
    np.random.shuffle(cell_ids)
    
    val_ids = cell_ids[:val_cells]
    train_ids = cell_ids[val_cells:]
    df_train = df[df['cell_id'].isin(train_ids)].reset_index(drop=True)
    df_val = df[df['cell_id'].isin(val_ids)].reset_index(drop=True)
    
    print(f"{name} - Training cells: {train_ids}")
    print(f"{name} - Validation cells: {val_ids}")
    return df_train, df_val

# ===============================================================
# Model Definition
# ===============================================================

class SOHLSTM(nn.Module):
    """LSTM model for SOH prediction"""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(SOHLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take only the last output
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out

# ===============================================================
# Elastic Weight Consolidation (EWC)
# ===============================================================

class EWC:
    """
    Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.
    
    EWC adds a regularization term to the loss function that penalizes changes to parameters
    that were important for previous tasks, based on the Fisher information matrix.
    """
    
    def __init__(self, model, device, lambda_ewc=5000):
        """Initialize EWC"""
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        
        # Initialize dictionaries for Fisher information and optimal parameters
        self.fisher = {}
        self.optimal_params = {}
        
        # Initialize parameters
        for n, p in self.model.named_parameters():
            self.fisher[n] = torch.zeros_like(p, device=self.device)
            self.optimal_params[n] = p.clone().detach()
    
    def compute_fisher(self, data_loader, num_samples=None):
        """Compute the Fisher Information Matrix using the data loader"""
        self.model.train()
        
        # Initialize parameter gradients
        for n, p in self.model.named_parameters():
            self.fisher[n] = torch.zeros_like(p, device=self.device)
        
        criterion = nn.MSELoss()
        sample_count = 0
        
        for features, labels in data_loader:
            if num_samples is not None and sample_count >= num_samples:
                break
                
            features, labels = features.to(self.device), labels.to(self.device)
            sample_count += features.size(0)
            
            # Forward and backward pass
            self.model.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Update Fisher information with squared gradients
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.pow(2).detach()
        
        # Normalize Fisher information by number of samples
        if sample_count > 0:
            for n in self.fisher:
                self.fisher[n] /= sample_count
    
    def update_optimal_params(self):
        """Update the optimal parameter values after training on a task"""
        for n, p in self.model.named_parameters():
            self.optimal_params[n] = p.clone().detach()
    
    def ewc_loss(self):
        """Compute the EWC regularization loss"""
        loss = 0
        for n, p in self.model.named_parameters():
            if n not in self.fisher:
                continue
            loss += (self.fisher[n] * (p - self.optimal_params[n]).pow(2)).sum()
        
        return self.lambda_ewc * 0.5 * loss

# ===============================================================
# Training and Evaluation Functions
# ===============================================================

def train_model(model, train_loader, val_loader, epochs, lr=1e-4, weight_decay=1e-4, patience=10, ewc=None):
    """Train model with optional EWC regularization"""
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', leave=False) as pbar:
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Add EWC regularization if available
                if ewc is not None:
                    loss += ewc.ewc_loss()
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                
                train_loss += loss.item()
                pbar.update(1)

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.3e} | "
              f"Val Loss: {val_loss:.3e} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Record metrics for current epoch
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Save best model and check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
    
    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def evaluate_model(model, data_loader):
    """Evaluate model performance on test data"""
    model.eval()
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    total_loss = 0.0

    all_predictions, all_targets = [], []
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
    metrics = calc_metrics(predictions, targets)

    return predictions, targets, metrics, total_loss

def calc_metrics(predictions, targets):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    return metrics

def plot_results(save_dir, method_name, df_test, seq_len,
                base_pred, update1_pred, update2_pred, 
                base_metrics, update1_metrics, update2_metrics):
    """Plot results with metrics for each phase of incremental learning"""
    # Extract x-coordinates and true target values
    sequence_length = seq_len
    datetime_vals = df_test['Datetime'].iloc[sequence_length:].values
    true_vals = df_test['SOH_ZHU'].iloc[sequence_length:].values

    # Convert predictions to numpy arrays
    base_pred = np.array(base_pred)
    update1_pred = np.array(update1_pred)
    update2_pred = np.array(update2_pred)
    
    # Determine segment boundaries based on prediction lengths
    n_base = len(base_pred)
    n_update1 = len(update1_pred)
    n_update2 = len(update2_pred)
    
    # Split true values and dates for each phase
    base_true = true_vals[:n_base]
    update1_true = true_vals[n_base:n_base+n_update1]
    update2_true = true_vals[n_base+n_update1:n_base+n_update1+n_update2]
    
    x_base = datetime_vals[:n_base]
    x_update1 = datetime_vals[n_base:n_base+n_update1]
    x_update2 = datetime_vals[n_base+n_update1:n_base+n_update1+n_update2]
    
    # Concatenate predictions for the overall curve
    all_pred = np.concatenate([base_pred, update1_pred, update2_pred])
    
    # Create the main plot
    plt.figure(figsize=(15, 6))
    plt.plot(datetime_vals[:len(true_vals)], true_vals, label='True Values')
    plt.plot(datetime_vals[:len(all_pred)], all_pred, label='Predicted Values')
    
    # Function to annotate each segment with metrics
    def annotate_segment(x_segment, seg_true, seg_pred, metrics, phase_name):
        if len(x_segment) == 0:
            return
            
        # Use middle of segment for annotation position
        mid_x = x_segment[len(x_segment) // 2]
        # Use mean of true and predicted values for y-position
        y_mean = np.mean(np.concatenate([seg_true, seg_pred]))
        
        # Format metrics text
        text = (f"{phase_name}\n"
                f"RMSE: {metrics['RMSE']:.4f}\n"
                f"MAE: {metrics['MAE']:.4f}\n"
                f"MAPE: {metrics['MAPE']:.2f}%\n"
                f"R²: {metrics['R²']:.4f}")
                
        plt.text(mid_x, y_mean, text, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7),
                 horizontalalignment='center', verticalalignment='center')
    
    # Add annotations for each phase
    annotate_segment(x_base, base_true, base_pred, base_metrics, "Base Model")
    annotate_segment(x_update1, update1_true, update1_pred, update1_metrics, "Update 1")
    annotate_segment(x_update2, update2_true, update2_pred, update2_metrics, "Update 2")
    
    # Set plot labels and title
    plt.xlabel("Datetime")
    plt.ylabel("SOH")
    plt.title(f"{method_name} - True vs Predicted SOH Across Phases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Create output directory and save figure
    results_dir = Path(save_dir) / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(results_dir / f"{method_name}_true_pred_plot.png")
    plt.close()

    # Create metrics comparison bar chart
    plt.figure(figsize=(12, 6))
    metrics_names = ['RMSE', 'MAE', 'MAPE', 'R²']
    
    # Set up bar positions
    x = np.arange(len(metrics_names))
    width = 0.25
    
    # Plot bars for each phase
    plt.bar(x - width, [base_metrics[m] for m in metrics_names], width, label='Base Model')
    plt.bar(x, [update1_metrics[m] for m in metrics_names], width, label='Update 1')
    plt.bar(x + width, [update2_metrics[m] for m in metrics_names], width, label='Update 2')
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title(f'{method_name} - Performance Metrics Comparison')
    plt.xticks(x, metrics_names)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(results_dir / f"{method_name}_metrics_comparison.png")
    plt.close()

    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()