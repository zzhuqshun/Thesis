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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create save directory with descriptive name
save_dir = Path(__file__).parent / "models/LSTM" / "test_for_incl" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir.mkdir(exist_ok=True, parents=True)

# Model hyperparameters - centralized configuration for easy adjustment
hyperparams = {
    "INFO": [
        "Model: SOH_LSTM",
        "Data: 10 min resampling of battery cycle data",
        "Degradation categories: normal, fast, faster",
        "Data split: Train ['03','01','21','05','27'], Validation ['23'], Test ['17'](SOH>=0.9)",
        "Features: Standard scaled voltage, current, temperature"
    ],
    "SEQUENCE_LENGTH": 864,  
    "HIDDEN_SIZE": 256,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.4,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-6,
    "RESAMPLE": '10min',
    "EPOCHS": 100,
    "PATIENCE": 10,
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
    # Load and prepare data
    data_dir = Path("../01_Datenaufbereitung/Output/Calculated/")
    df_train, df_val, df_test = load_data(data_dir, resample=hyperparams["RESAMPLE"])
    
    # Scale features using StandardScaler
    df_train_scaled, df_val_scaled, df_test_scaled = scale_data(df_train, df_val, df_test)
    
    # Create datasets and data loaders
    train_dataset = BatteryDataset(df_train_scaled, hyperparams["SEQUENCE_LENGTH"])
    val_dataset = BatteryDataset(df_val_scaled, hyperparams["SEQUENCE_LENGTH"])
    test_dataset = BatteryDataset(df_test_scaled, hyperparams["SEQUENCE_LENGTH"])

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['BATCH_SIZE'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['BATCH_SIZE'], shuffle=False)

    # ==================== Model Initialization ====================
    # Initialize the LSTM model for SOH prediction
    model = SOHLSTM(
        input_size=3,  # Features: voltage, current, temperature
        hidden_size=hyperparams['HIDDEN_SIZE'],
        num_layers=hyperparams['NUM_LAYERS'],
        dropout=hyperparams['DROPOUT']
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model architecture:\n{model}')
    print(f'Total trainable parameters: {total_params:,}')

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
        history, best_val_loss = train_and_validate_model(model, train_loader, val_loader, save_path)
        print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")
    else:
        # Load a previously trained model
        model_path = save_path['last']
        if os.path.exists(model_path):
            print(f"\nLoading model from {model_path}...")
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully!")
        else:
            print(f"\nError: Model file {model_path} does not exist.")
            return

    # ==================== Model Evaluation ====================
    print("\nEvaluating the model on the testing set...")
    predictions, targets, metrics = evaluate_model(model, test_loader)

    # Print the evaluation metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # ==================== Results Visualization ====================
    results_dir = save_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot training and validation loss curves
    if save_path['history'].exists():
        plot_losses(pd.read_parquet(save_path['history']), results_dir)
    
    # Plot predictions vs actual values
    plot_predictions(predictions, targets, df_test_scaled, results_dir)
    plot_prediction_scatter(predictions, targets, results_dir, cell_id="17")

    

def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries
    
    Args:
        seed: Integer seed value
    """
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
    Load battery cell data files, compute degradation rates, categorize and split into train/validation/test sets
    using fixed cell IDs for testing.

    Args:
        data_dir: Directory containing battery data parquet files
        resample: Time interval for resampling the data

    Returns:
        Tuple of three DataFrames: (train, validation, test)
    """
    # 1. List and sort parquet files by the numeric cell ID
    parquet_files = sorted(
        [f for f in data_dir.glob('*.parquet') if f.is_file()],
        key=lambda x: int(x.stem.split('_')[-1])
    )

    def process_file(fp: Path):
        """Process a single battery cell data file"""
        raw = pd.read_parquet(fp)[
            ['Testtime[s]', 'SOH_ZHU', 'Voltage[V]', 'Current[A]', 'Temperature[°C]']
        ].dropna().copy()
        raw['Testtime[s]'] = raw['Testtime[s]'].round().astype(int)
        raw['Datetime'] = pd.to_datetime(
            raw['Testtime[s]'], unit='s', origin=pd.Timestamp("2023-02-02")
        )
        df_s = raw.set_index('Datetime').resample(resample).mean().reset_index()
        df_s['cell_id'] = fp.stem.split('_')[1]
        return raw, df_s

    # 2. Process all files
    records = []
    for fp in parquet_files:
        raw_df, df_s = process_file(fp)
        cid = df_s['cell_id'].iloc[0]
        records.append({'file': fp, 'raw': raw_df, 'df': df_s, 'cell_id': cid})

    # 3. Define fixed splits
    train_ids = ['03', '01', '21', '05', '27']
    val_ids   = ['23']
    test_ids  = ['17']

    # 4. Build DataFrames
    df_train = pd.concat([r['df'] for r in records if r['cell_id'] in train_ids], ignore_index=True)
    df_val   = pd.concat([r['df'] for r in records if r['cell_id'] in val_ids], ignore_index=True)
    # Test only SOH_ZHU > 0.9
    df_test_all = pd.concat([r['df'] for r in records if r['cell_id'] in test_ids], ignore_index=True)
    df_test = df_test_all[df_test_all['SOH_ZHU'] >= 0.9].reset_index(drop=True)

    # 5. Debug logs
    print(f"Train IDs: {train_ids}, Val IDs: {val_ids}, Test IDs: {test_ids}")
    print(f"Train shape: {df_train.shape}, Val shape: {df_val.shape}, Test shape (SOH>0.9): {df_test.shape}")

    return df_train, df_val, df_test



def scale_data(df_train, df_val, df_test):
    """
    Scale features using StandardScaler fitted on training data
    
    Args:
        df_train: Training data DataFrame
        df_val: Validation data DataFrame
        df_test: Test data DataFrame
        
    Returns:
        Tuple of scaled DataFrames (train_scaled, val_scaled, test_scaled)
    """
    features_to_scale = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']

    # Create copies of the dataframes to avoid modifying originals
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()

    # Fit scaler on training data only to prevent data leakage
    feature_scaler = StandardScaler()
    feature_scaler.fit(df_train[features_to_scale])
    
    # Transform all datasets using the fitted scaler
    df_train_scaled[features_to_scale] = feature_scaler.transform(df_train[features_to_scale])
    df_val_scaled[features_to_scale] = feature_scaler.transform(df_val[features_to_scale])
    df_test_scaled[features_to_scale] = feature_scaler.transform(df_test[features_to_scale])
    
    print('Features scaled with StandardScaler')
    
    return df_train_scaled, df_val_scaled, df_test_scaled


class BatteryDataset(Dataset):
    """Dataset for battery SOH prediction using sequence data"""
    def __init__(self, df, sequence_length):
        """
        Initialize the dataset by creating sequences from the input DataFrame
        
        Args:
            df: DataFrame containing the battery data
            sequence_length: Number of time steps in each input sequence
        """
        self.sequence_length = sequence_length
        
        # Extract features and target variable
        features_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        label_col = 'SOH_ZHU'
        
        # Convert to PyTorch tensors
        features = torch.tensor(df[features_cols].values, dtype=torch.float32)
        labels = torch.tensor(df[label_col].values, dtype=torch.float32)
        
        # Create sequence data efficiently
        n_samples = len(df) - sequence_length
        self.features = torch.zeros((n_samples, sequence_length, len(features_cols)), dtype=torch.float32)
        self.labels = torch.zeros(n_samples, dtype=torch.float32)
        
        # Build sequences using tensor slicing for efficiency
        for i in range(n_samples):
            self.features[i] = features[i:i+sequence_length]
            self.labels[i] = labels[i+sequence_length]  # Predict the SOH at the end of sequence

    def __len__(self):
        """Return the number of sequences in the dataset"""
        return len(self.features)

    def __getitem__(self, idx):
        """Return a specific sequence and its target SOH value"""
        x = self.features[idx]
        soh = self.labels[idx]
        y =(1.0-soh) * 100.0  # Convert to percentage
        return x, y


class SOHLSTM(nn.Module):
    """LSTM model for battery State of Health (SOH) prediction"""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        """
        Initialize the LSTM model for SOH prediction
        
        Args:
            input_size: Number of input features (voltage, current, temperature)
            hidden_size: Size of LSTM hidden layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
        """
        super(SOHLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer for sequence modeling
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Expect input as [batch, seq, feature]
            dropout=dropout if num_layers > 1 else 0  # Apply dropout between LSTM layers
        )

        # Fully connected layers for regression prediction
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
        # Initialize hidden and cell states with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out shape: [batch_size, seq_len, hidden_size]
        
        # Take only the output from the last time step
        last_time_step = lstm_out[:, -1, :]  # shape: [batch_size, hidden_size]
        
        # Pass through fully connected layers
        out = self.fc_layers(last_time_step)  # shape: [batch_size, 1]
        
        return out.squeeze(-1)  # shape: [batch_size]


def train_and_validate_model(model, train_loader, val_loader, save_path):
    """
    Train and validate the model with early stopping.

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
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Early stopping variables
    epochs_no_improve = 0
    best_val_loss = float('inf')

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'epoch': []}

    print('\nStart training...')
    for epoch in range(hyperparams['EPOCHS']):
        # === Training phase ===
        model.train()
        train_loss = 0.0

        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch+1}/{hyperparams["EPOCHS"]}',
                  leave=False) as pbar:
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(features)  # [batch_size]
                loss = criterion(outputs, labels)

                # Backward + optimize
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                train_loss += loss.item()
                pbar.update(1)

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # === Validation phase ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)

        history['val_loss'].append(val_loss)
        history['epoch'].append(epoch + 1)

        # Scheduler step
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{hyperparams["EPOCHS"]} | '
              f'Train Loss: {train_loss:.3e} | Val Loss: {val_loss:.3e} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # Early stopping & saving best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path['best'])
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= hyperparams['PATIENCE']:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Save final model and history
    last_model_state = model.state_dict()
    torch.save(last_model_state, save_path['last'])
    print(f"Final model saved to {save_path['last']}")
    print(f"Best model saved to {save_path['best']}")
    pd.DataFrame(history).to_parquet(save_path['history'], index=False)
    

    return history, best_val_loss

def plot_losses(history_df, results_dir):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(history_df['epoch'], history_df['train_loss'], label='Training Loss', 
            marker='o', markersize=4, lw=2)
    plt.semilogy(history_df['epoch'], history_df['val_loss'], label='Validation Loss', 
            marker='o', markersize=4, lw=2)
    
    plt.title('Training and Validation Losses')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(results_dir / "train_val_loss.png")
    plt.close()

def evaluate_model(model, test_loader):
    """
    评估 LSTM模型在测试数据集上的性能
    
    Args:
        model: 训练好的PyTorch模型
        test_dataset: BatteryDataset对象
        batch_size: 用于评估的batch size
        device: 计算设备
        
    Returns:
        predictions: 模型预测值数组
        targets: 真实值数组
        cell_ids: 对应每个样本的cell_id
        metrics: 性能指标字典
    """
    model.eval()
    # 重置模型的隐藏状态
    model.hidden_states = {}
    
    all_preds, all_tgts = [], []
    
    with torch.no_grad():
        for feats, labs in test_loader:
            feats, labs = feats.to(device), labs.to(device)
            # 始终使用测试cell_id的状态
            out = model(feats)
            all_preds.extend(out.cpu().numpy())
            all_tgts.extend(labs.cpu().numpy())

    preds_drop = np.array(all_preds)
    targs_drop = np.array(all_tgts)

    # 转回 SOH
    preds_soh = 1.0 - preds_drop / 100.0
    targs_soh = 1.0 - targs_drop / 100.0

    metrics = {
        'RMSE': np.sqrt(mean_squared_error(targs_soh, preds_soh)),
        'MAE' : mean_absolute_error   (targs_soh, preds_soh),
        'R²'  : r2_score              (targs_soh, preds_soh)
    }
    return preds_soh, targs_soh, metrics

def plot_predictions(predictions, targets, df_test_scaled, results_dir):
    """
    为测试电池创建SOH预测曲线对比图
    
    Args:
        predictions: 模型预测值数组
        targets: 真实值数组
        df_test_scaled: 带有'Datetime'和'cell_id'列的测试集DataFrame
        results_dir: 保存图像的目录
        cell_ids: 与predictions/targets顺序对应的cell_id
    """
    # 由于只有一个cell_id，我们可以简化代码
    test_cell_id = '17'  # 所有cell_ids都应该相同
    
    # 构造用于绘图的数据框
    df_plot = pd.DataFrame({
        'Datetime': df_test_scaled['Datetime'].iloc[hyperparams['SEQUENCE_LENGTH']:].values,
        'predicted_SOH': predictions,
        'actual_SOH': targets
    })
    
    # 计算cell特定的指标
    cell_metrics = {
        'RMSE': np.sqrt(mean_squared_error(df_plot['actual_SOH'], df_plot['predicted_SOH'])),
        'MAE': mean_absolute_error(df_plot['actual_SOH'], df_plot['predicted_SOH']),
        'R²': r2_score(df_plot['actual_SOH'], df_plot['predicted_SOH'])
    }
    
    # 创建更详细的可视化
    plt.figure(figsize=(12, 7))
    
    # 绘制实际和预测的SOH
    plt.plot(df_plot['Datetime'], df_plot['actual_SOH'], 
            label='Actual SOH', lw=2.5, color='#3366CC')
    plt.plot(df_plot['Datetime'], df_plot['predicted_SOH'], 
            label='Predicted SOH', lw=2, color='#FF9900')
    
    # 添加带有指标的标题
    plt.title(f'Battery Cell {test_cell_id} - Actual vs Predicted SOH\n'
             f"Evaluate Metrics: RMSE={cell_metrics['RMSE']:.4f}, MAE={cell_metrics['MAE']:.4f}, R²={cell_metrics['R²']:.4f}",
             fontsize=13)
    
    # 设置轴标签和格式
    plt.xlabel('Date')
    plt.ylabel('State of Health (SOH)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.gcf().autofmt_xdate()  # 自动格式化日期标签
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(results_dir / f"prediction_cell_{test_cell_id}.png", dpi=300)
    plt.close()
    
    print(f"Prediction plot saved for cell {test_cell_id}")

def plot_prediction_scatter(predictions, targets, results_dir, cell_id="17"):
    """
    创建实际SOH值与预测SOH值的散点图，并显示性能指标
    
    Args:
        predictions: 模型预测值数组
        targets: 真实值数组
        results_dir: 保存图像的目录
        cell_id: 电池ID
    """
    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    # 创建散点图
    plt.figure(figsize=(10, 8))
    
    # 绘制散点图
    plt.scatter(targets, predictions, alpha=0.7, color='#0066CC', s=20)
    
    # 添加理想预测线（y=x线）
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    # 设置标题和标签
    plt.title(f'Actual vs. Predicted SOH values\n(RMSE={rmse:.4f}, MAE={mae:.4f},  R²={r2:.4f})', 
              fontsize=14)
    plt.xlabel('Actual SOH', fontsize=12)
    plt.ylabel('Predicted SOH', fontsize=12)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置轴范围稍微扩展一点
    buffer = (max_val - min_val) * 0.02  # 2%的缓冲区
    plt.xlim(min_val - buffer, max_val + buffer)
    plt.ylim(min_val - buffer, max_val + buffer)
    
    # 确保图形是正方形的
    plt.axis('square')
    
    # 添加紧凑布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(results_dir / f"prediction_scatter_cell_{cell_id}.png", dpi=300)
    plt.close()
    
    print(f"Prediction scatter plot saved for cell {cell_id}")
    
if __name__ == "__main__":
    main()