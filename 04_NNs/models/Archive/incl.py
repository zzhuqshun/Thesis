import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Import Progressive Neural Network
from models.Archive.pnn import ProgressiveNN

# ===============================================================
# Main Function for Incremental Learning using PNN
# ===============================================================

def main():
    # Setup directories and device
    save_dir = Path(__file__).parent / 'models' / 'incLearning_PNN' / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    hyperparams = {
        'SEQUENCE_LENGTH': 432,
        'HIDDEN_SIZE': 32,
        'NUM_LAYERS': 2,
        'DROPOUT': 0.1,
        'BATCH_SIZE': 16,
        'LEARNING_RATE': 1e-4,
        'EPOCHS': 200,
        'PATIENCE': 20,
        'WEIGHT_DECAY': 0.0,
    }
    with open(save_dir / 'hyperparams.json', 'w') as f:
        json.dump(hyperparams, f, indent=4)

    print(f'Using device: {device}')
    set_seed(42)

    # Load data
    data_dir = Path('../01_Datenaufbereitung/Output/Calculated/')
    (df_base_train_scaled, df_base_val_scaled,
     df_update1_train_scaled, df_update1_val_scaled,
     df_update2_train_scaled, df_update2_val_scaled,
     df_test_base_scaled, df_test_update1_scaled, df_test_update2_scaled) = \
        load_and_prepare_data(data_dir, resample='10min')

    # Create datasets and loaders
    base_train_loader = DataLoader(BatteryDataset(df_base_train_scaled, hyperparams['SEQUENCE_LENGTH']),
                                   batch_size=hyperparams['BATCH_SIZE'], shuffle=True)
    base_val_loader = DataLoader(BatteryDataset(df_base_val_scaled, hyperparams['SEQUENCE_LENGTH']),
                                 batch_size=hyperparams['BATCH_SIZE'])
    
    update1_train_loader = DataLoader(BatteryDataset(df_update1_train_scaled, hyperparams['SEQUENCE_LENGTH']),
                                      batch_size=hyperparams['BATCH_SIZE'], shuffle=True)
    update1_val_loader = DataLoader(BatteryDataset(df_update1_val_scaled, hyperparams['SEQUENCE_LENGTH']),
                                    batch_size=hyperparams['BATCH_SIZE'])
    
    update2_train_loader = DataLoader(BatteryDataset(df_update2_train_scaled, hyperparams['SEQUENCE_LENGTH']),
                                      batch_size=hyperparams['BATCH_SIZE'], shuffle=True)
    update2_val_loader = DataLoader(BatteryDataset(df_update2_val_scaled, hyperparams['SEQUENCE_LENGTH']),
                                    batch_size=hyperparams['BATCH_SIZE'])

    base_test_loader = DataLoader(BatteryDataset(df_test_base_scaled, hyperparams['SEQUENCE_LENGTH']),
                                  batch_size=hyperparams['BATCH_SIZE'])
    update1_test_loader = DataLoader(BatteryDataset(df_test_update1_scaled, hyperparams['SEQUENCE_LENGTH']),
                                     batch_size=hyperparams['BATCH_SIZE'])
    update2_test_loader = DataLoader(BatteryDataset(df_test_update2_scaled, hyperparams['SEQUENCE_LENGTH']),
                                     batch_size=hyperparams['BATCH_SIZE'])

    # ================= PNN Incremental Learning =================
    print('='*80)
    print('Method: Progressive Neural Network (PNN)')
    print('='*80)

    # Initialize PNN
    input_size = 3  # Voltage, Current, Temperature
    pnn_model = ProgressiveNN(input_size=input_size,
                              hidden_size=hyperparams['HIDDEN_SIZE'],
                              num_layers=hyperparams['NUM_LAYERS'],
                              dropout=hyperparams['DROPOUT']).to(device)

    # Phase 1: train on base data
    print('\nPhase 1: Training column 0 on base data...')
    pnn_model, base_history = train_model(pnn_model, base_train_loader, base_val_loader,
                                          hyperparams['EPOCHS'], hyperparams['LEARNING_RATE'],
                                          hyperparams['WEIGHT_DECAY'], hyperparams['PATIENCE'])
    torch.save(pnn_model.state_dict(), save_dir / 'pnn_column0.pt')
    base_pred, base_targets, base_metrics, _ = evaluate_pnn_model(pnn_model, base_test_loader, task_id=0)
    print(f'Column 0 metrics: {base_metrics}')

    # Phase 2: add column 1 and train
    print('\nPhase 2: Adding column 1 and training on update1 data...')
    pnn_model.add_column()
    pnn_model, update1_history = train_model(pnn_model, update1_train_loader, update1_val_loader,
                                             hyperparams['EPOCHS'], hyperparams['LEARNING_RATE'],
                                             hyperparams['WEIGHT_DECAY'], hyperparams['PATIENCE'])
    torch.save(pnn_model.state_dict(), save_dir / 'pnn_column1.pt')
    update1_pred, update1_targets, update1_metrics, _ = evaluate_pnn_model(pnn_model, update1_test_loader, task_id=1)
    print(f'Column 1 metrics: {update1_metrics}')

    # Phase 3: add column 2 and train
    print('\nPhase 3: Adding column 2 and training on update2 data...')
    pnn_model.add_column()
    pnn_model, update2_history = train_model(pnn_model, update2_train_loader, update2_val_loader,
                                             hyperparams['EPOCHS'], hyperparams['LEARNING_RATE'],
                                             hyperparams['WEIGHT_DECAY'], hyperparams['PATIENCE'])
    torch.save(pnn_model.state_dict(), save_dir / 'pnn_column2.pt')
    update2_pred, update2_targets, update2_metrics, _ = evaluate_pnn_model(pnn_model, update2_test_loader, task_id=2)
    print(f'Column 2 metrics: {update2_metrics}')

    print('\nPNN Incremental Learning Finished! Models and results saved to: ', save_dir)

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


def calc_metrics(predictions, targets):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    return {'RMSE': rmse, 'MAE': mae, 'R²': r2}

def plot_history(history, task_id, save_dir):
    """Plot training and validation loss curves"""
    plt.figure()
    epochs = history['epoch']
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f'Task {task_id} Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f'loss_task{task_id}.png')
    plt.close()

def evaluate_pnn_model(model, data_loader, task_id):
    """Evaluate a PNN model for a specific task/column"""
    model.eval()
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    total_loss = 0.0
    all_pred, all_targets = [], []
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features, task_id=task_id)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_pred.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    total_loss /= len(data_loader)
    predictions = np.concatenate(all_pred)
    targets = np.concatenate(all_targets)
    metrics = calc_metrics(predictions, targets)
    return predictions, targets, metrics, total_loss

# ===============================================================
# Dataset Definition
# ===============================================================

class BatteryDataset(Dataset):
    """Dataset for battery SOH prediction"""
    def __init__(self, df, sequence_length):
        self.sequence_length = sequence_length
        self.features = torch.tensor(df[['Voltage[V]', 'Current[A]', 'Temperature[°C]']].values, dtype=torch.float32)
        self.targets = torch.tensor(df['SOH_ZHU'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length - 1]
        return x, y

# ===============================================================
# Data Loading and Preparation
# ===============================================================

def load_and_prepare_data(data_dir, resample='10min'):
    """
    Load and prepare battery datasets for incremental learning based on SOH degradation rates.
    
    Dataset division strategy:
    1. Split cells into three categories: normal (SOH > 0.8), fast (0.65 < SOH < 0.8), faster (SOH < 0.65)
    2. Use cell 17 as test set, divided by SOH ranges:
       - SOH 1.0-0.9: base test
       - SOH 0.9-0.8: update1 test
       - SOH < 0.8: update2 test
    3. Training sets:
       - Base: 7 normal cells (5 train, 2 val)
       - Update1: base's 2 val + 1 normal + 2 fast as train, 1 fast as val
       - Update2: update1's val + 2 faster as train, remaining faster as val
    """
    parquet_files = sorted(
        [f for f in data_dir.glob('*.parquet') if f.is_file()],
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    # Process files to get final SOH values and categorize cells
    cell_info = []
    for fp in parquet_files:
        cell_id = fp.stem.split('_')[1]
        
        # Skip cell 17 as it's reserved for testing
        if cell_id == '17':
            test_file = fp
            continue
            
        # Read the file to get SOH values
        df = pd.read_parquet(fp)
        initial_soh = df['SOH_ZHU'].iloc[0]
        final_soh = df['SOH_ZHU'].iloc[-1]
        
        # Categorize cells based on final SOH
        if final_soh > 0.8:
            category = 'normal'
        elif 0.65 < final_soh <= 0.8:
            category = 'fast'
        else:
            category = 'faster'
            
        cell_info.append({
            'file': fp,
            'cell_id': cell_id,
            'initial_soh': initial_soh,
            'final_soh': final_soh,
            'category': category
        })
    
    # Group cells by category
    normal_cells = [c for c in cell_info if c['category'] == 'normal']
    fast_cells = [c for c in cell_info if c['category'] == 'fast']
    faster_cells = [c for c in cell_info if c['category'] == 'faster']
    
    # Set random seed for reproducible cell selection
    random.seed(42)
    
    # Assign cells to different training phases
    # Base phase: 7 normal cells (5 train, 2 val)
    if len(normal_cells) < 7:
        print(f"Warning: Not enough normal cells ({len(normal_cells)}). Need at least 7.")
        # If not enough, supplement with fastest degrading cells from next category
        if len(normal_cells) + len(fast_cells) >= 7:
            fast_cells_sorted = sorted(fast_cells, key=lambda x: x['final_soh'], reverse=True)
            normal_cells.extend(fast_cells_sorted[:7-len(normal_cells)])
            # Remove the cells that were moved to normal
            fast_cells = fast_cells_sorted[7-len(normal_cells):]
        else:
            print("Error: Not enough cells for base training.")
            
    # Randomly select 7 normal cells
    selected_normal = random.sample(normal_cells, min(7, len(normal_cells)))
    
    # Base train (5) and val (2)
    base_train_cells = selected_normal[:5]
    base_val_cells = selected_normal[5:7]
    
    # Remaining normal cells
    remaining_normal = [c for c in normal_cells if c not in selected_normal]
    
    # Update1: base val (2) + 1 normal + 2 fast as train, 1 fast as val
    update1_train_normal = remaining_normal[:1] if remaining_normal else []
    
    if len(fast_cells) < 3:
        print(f"Warning: Not enough fast cells ({len(fast_cells)}). Need at least 3.")
    
    update1_train_fast = fast_cells[:2] if len(fast_cells) >= 2 else fast_cells
    update1_val_fast = fast_cells[2:3] if len(fast_cells) >= 3 else []
    
    # Combine cells for update1
    update1_train_cells = base_val_cells + update1_train_normal + update1_train_fast
    update1_val_cells = update1_val_fast
    
    # Update2: update1 val + 2 faster as train, remaining faster as val
    update2_train_faster = faster_cells[:2] if len(faster_cells) >= 2 else faster_cells
    update2_val_faster = faster_cells[2:] if len(faster_cells) >= 3 else []
    
    # Combine cells for update2
    update2_train_cells = update1_val_cells + update2_train_faster
    update2_val_cells = update2_val_faster
    
    # Print assignment summary
    print("Cell Assignment Summary:")
    print(f"Normal cells ({len(normal_cells)}): {[c['cell_id'] for c in normal_cells]}")
    print(f"Fast cells ({len(fast_cells)}): {[c['cell_id'] for c in fast_cells]}")
    print(f"Faster cells ({len(faster_cells)}): {[c['cell_id'] for c in faster_cells]}")
    print("\nTraining Sets:")
    print(f"Base train: {[c['cell_id'] for c in base_train_cells]}")
    print(f"Base val: {[c['cell_id'] for c in base_val_cells]}")
    print(f"Update1 train: {[c['cell_id'] for c in update1_train_cells]}")
    print(f"Update1 val: {[c['cell_id'] for c in update1_val_cells]}")
    print(f"Update2 train: {[c['cell_id'] for c in update2_train_cells]}")
    print(f"Update2 val: {[c['cell_id'] for c in update2_val_cells]}")
    
    # Function to process files
    def process_file(file_path):
        """Process a single parquet file"""
        df = pd.read_parquet(file_path)
        columns_to_keep = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 
                           'Temperature[°C]', 'SOH_ZHU']
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
        return df_sampled
    
    # Process training and validation files
    df_base_train = pd.concat([process_file(c['file']) for c in base_train_cells], ignore_index=True)
    df_base_val = pd.concat([process_file(c['file']) for c in base_val_cells], ignore_index=True)
    
    df_update1_train = pd.concat([process_file(c['file']) for c in update1_train_cells], ignore_index=True)
    df_update1_val = pd.concat([process_file(c['file']) for c in update1_val_cells], ignore_index=True) if update1_val_cells else pd.DataFrame()
    
    df_update2_train = pd.concat([process_file(c['file']) for c in update2_train_cells], ignore_index=True) if update2_train_cells else pd.DataFrame()
    df_update2_val = pd.concat([process_file(c['file']) for c in update2_val_cells], ignore_index=True) if update2_val_cells else pd.DataFrame()
    
    # Process test file
    df_test_full = process_file(test_file)
    
    # Split test set based on SOH values
    df_test_base = df_test_full[df_test_full['SOH_ZHU'] >= 0.9].reset_index(drop=True)
    df_test_update1 = df_test_full[(df_test_full['SOH_ZHU'] < 0.9) & (df_test_full['SOH_ZHU'] >= 0.8)].reset_index(drop=True)
    df_test_update2 = df_test_full[df_test_full['SOH_ZHU'] < 0.8].reset_index(drop=True)
    
    # Normalize data using StandardScaler
    scaler = StandardScaler()
    feature_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
    scaler.fit(df_base_train[feature_cols])
    
    def apply_scaling(df):
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.transform(df[feature_cols])
        return df_scaled

    # Base
    df_base_train_scaled = apply_scaling(df_base_train)
    df_base_val_scaled   = apply_scaling(df_base_val)

    # Update1
    df_update1_train_scaled = apply_scaling(df_update1_train)
    df_update1_val_scaled   = apply_scaling(df_update1_val)

    # Update2
    df_update2_train_scaled = apply_scaling(df_update2_train)
    df_update2_val_scaled   = apply_scaling(df_update2_val)

    # Test 三个分段
    df_test_base_scaled    = apply_scaling(df_test_base)
    df_test_update1_scaled = apply_scaling(df_test_update1)
    df_test_update2_scaled = apply_scaling(df_test_update2)

    # Print dataset sizes
    print(f"\nDataset sizes:")
    print(f"Base train: {len(df_base_train_scaled)} samples, val: {len(df_base_val_scaled)} samples")
    print(f"Update1 train: {len(df_update1_train_scaled)} samples, val: {len(df_update1_val_scaled)} samples")
    print(f"Update2 train: {len(df_update2_train_scaled)} samples, val: {len(df_update2_val_scaled)} samples")

    print("\nTest Sets:")
    print(f"Test cell: {test_file.stem.split('_')[1]}")
    print(f"Base test (SOH ≥ 0.9): {len(df_test_base)} samples")
    print(f"Update1 test (0.8 ≤ SOH < 0.9): {len(df_test_update1)} samples")
    print(f"Update2 test (SOH < 0.8): {len(df_test_update2)} samples")
    
    return (df_base_train_scaled, df_base_val_scaled, 
            df_update1_train_scaled, df_update1_val_scaled, 
            df_update2_train_scaled, df_update2_val_scaled,  
            df_test_base_scaled, df_test_update1_scaled, df_test_update2_scaled)

# ===============================================================
# Training Function
# ===============================================================

def train_model(model, train_loader, val_loader, epochs, lr, weight_decay, patience):
    """Train model for one task"""
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', leave=False) as pbar:
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

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        history['epoch'].append(epoch+1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.3e} | Val Loss: {val_loss:.3e}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_state is not None:
        for k, v in best_state.items():
            best_state[k] = v.to(device)
        model.load_state_dict(best_state)
    return model, history

if __name__ == '__main__':
    main()
