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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create save directory
save_dir = Path(__file__).parent / "models/Stateful_LSTM" / "seq6days"
save_dir.mkdir(exist_ok=True, parents=True)

# Model hyperparameters
hyperparams = {
    "INFO": [
        "Model: SOH_stateful_LSTM",
        "Data: 10-minute resampling of battery cycle data",
        "Degradation categories: normal, fast, faster",
        "Data split: Train (11 cells), Validation (3 cells), Test (1 cell)",
        "Features: Standard scaled voltage, current, temperature"
    ],
    "SEQUENCE_LENGTH": 864,    # Number of time steps in each sequence
    "HIDDEN_SIZE": 256,        # Size of LSTM hidden layers
    "NUM_LAYERS": 2,           # Number of LSTM layers
    "DROPOUT": 0.4,            # Dropout rate for regularization
    "BATCH_SIZE": 32,          # Batch size for training
    "LEARNING_RATE": 1e-4,     # Learning rate for optimizer
    "WEIGHT_DECAY": 1e-6,       # L2 regularization
    "EPOCHS": 100,             # Maximum number of training epochs
    "PATIENCE": 10,            # Early stopping patience    "device": str(device)      # Computation device
}


def main():
    """Main function to run the SOH prediction pipeline with stateful LSTM"""
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
    df_train_scaled, df_val_scaled, df_test_scaled = scale_data(df_train, df_val, df_test)

    # Create datasets and data loaders
    train_dataset = BatteryDataset(df_train_scaled, hyperparams["SEQUENCE_LENGTH"])
    val_dataset = BatteryDataset(df_val_scaled, hyperparams["SEQUENCE_LENGTH"])
    test_dataset = BatteryDataset(df_test_scaled, hyperparams["SEQUENCE_LENGTH"])

    # 为stateful LSTM创建特殊的数据加载器，确保相同cell_id的样本在一起
    train_loader = create_cell_ordered_dataloader(train_dataset, batch_size=hyperparams['BATCH_SIZE'])
    val_loader = create_cell_ordered_dataloader(val_dataset, batch_size=hyperparams['BATCH_SIZE'])

    # ==================== Model Initialization ====================
    # 初始化stateful LSTM模型
    model = StatefulSOHLSTM(
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
        'best': save_dir / 'best_stateful_soh_model.pth',
        'last': save_dir / 'last_stateful_soh_model.pth',
        'history': save_dir / 'train_history.parquet'
    }

    # Choose whether to train a new model or load an existing one
    TRAINING_MODE = True
    
    if TRAINING_MODE:
        # Train and validate the model with stateful LSTM
        history, _ = train_and_validate_stateful_model(model, train_loader, val_loader, save_path)
    else:
        # Load the model
        model_path = save_path['last']
        if os.path.exists(model_path):
            print(f"\nLoading model from {model_path}...")
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print("Model loaded successfully!")
        else:
            print(f"\nWarning: Model file {model_path} does not exist.")
            exit(1)

    # ==================== Model Evaluation ====================
    print("\nEvaluating the model on the testing set...")
    predictions, targets, cell_ids, metrics = evaluate_stateful_model(
    model, test_dataset, hyperparams['BATCH_SIZE'], device
)

    # Print the evaluation metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # ==================== Results Visualization ====================
    results_dir = save_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
        
    if save_path['history'].exists():
        plot_losses(pd.read_parquet(save_path['history']), results_dir)
    
    plot_predictions_by_cell(predictions, targets, df_test_scaled, results_dir, cell_ids)
    plot_prediction_scatter(predictions, targets, results_dir, cell_id="17")

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
    Load battery cell data files, compute degradation rates, categorize cells,
    and split into train/validation/test sets.

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
        # Read and clean data
        raw = pd.read_parquet(fp)[
            ['Testtime[s]', 'SOH_ZHU', 'Voltage[V]', 'Current[A]', 'Temperature[°C]']
        ].dropna().copy()
        
        # Round and convert timestamps
        raw['Testtime[s]'] = raw['Testtime[s]'].round().astype(int)
        
        # Convert to datetime using Testtime offset from a reference
        raw['Datetime'] = pd.to_datetime(
            raw['Testtime[s]'], unit='s', origin=pd.Timestamp("2023-02-02")
        )
        
        # Resample to uniform time interval
        df_s = raw.set_index('Datetime').resample(resample).mean().reset_index()
        df_s['cell_id'] = fp.stem.split('_')[1]
        
        return raw, df_s

    # 2. Process files and compute degradation rate on raw data
    records = []
    for fp in parquet_files:
        raw_df, df_s = process_file(fp)
        cid = df_s['cell_id'].iloc[0]
        
        # Compute degradation rate (SOH loss per day)
        total_days = (raw_df['Datetime'].iloc[-1] - raw_df['Datetime'].iloc[0]).total_seconds() / (3600*24)
        rate = (raw_df['SOH_ZHU'].iloc[0] - raw_df['SOH_ZHU'].iloc[-1]) / total_days
        
        records.append({'file': fp, 'df': df_s, 'cell_id': cid, 'rate': rate})
    
    rates_df = pd.DataFrame(records)[['cell_id', 'rate']]

    # 3. Categorize cells into degradation rate bins
    q1, q2 = rates_df['rate'].quantile([0.33, 0.66])
    rates_df['category'] = pd.cut(
        rates_df['rate'], 
        bins=[-np.inf, q1, q2, np.inf], 
        labels=['normal', 'fast', 'faster']
    )

    # 4. Stratified sampling for validation and test sets
    random.seed(42)
    val_ids, test_ids = [], ['17']  # Cell 17 is designated as test cell
    
    # Select one validation cell from each degradation category
    # Ensure we don't select the test cell for validation
    for cat in ['normal', 'fast', 'faster']:
        cat_ids = rates_df.loc[rates_df['category'] == cat, 'cell_id'].tolist()
        # Remove test cell ID from candidate list if present
        cat_ids = [cid for cid in cat_ids if cid not in test_ids]
        val_ids.append(random.choice(cat_ids))
    
    # Remaining cells go to training
    train_ids = [i for i in rates_df['cell_id'] if i not in val_ids + test_ids]

    # 5. Aggregate DataFrames by set
    df_train = pd.concat([r['df'] for r in records if r['cell_id'] in train_ids], ignore_index=True)
    df_val = pd.concat([r['df'] for r in records if r['cell_id'] in val_ids], ignore_index=True)
    
    # Fixed: Proper test set creation
    df_test = pd.concat([r['df'] for r in records if r['cell_id'] in test_ids], ignore_index=True)

    # 6. Log information about the dataset split
    print("Degradation categories and rates:\n", rates_df, "\n")
    print(f"Train IDs: {train_ids}, Val IDs: {val_ids}, Test IDs: {test_ids}")
    print(f"Shapes -> Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}\n")

    return df_train, df_val, df_test

def scale_data(df_train, df_val, df_test):
    """
    Scale features using StandardScaler fitted on training data
    
    Args:
        df_train, df_val, df_test: DataFrames containing features to scale
        
    Returns:
        Scaled versions of the input dataframes
    """
    features_to_scale = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']

    # Create copies of the dataframes
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()

    # Fit scaler on training data only
    feature_scaler = StandardScaler()
    feature_scaler.fit(df_train[features_to_scale])
    df_train_scaled[features_to_scale] = feature_scaler.transform(df_train[features_to_scale])
    df_val_scaled[features_to_scale] = feature_scaler.transform(df_val[features_to_scale])
    df_test_scaled[features_to_scale] = feature_scaler.transform(df_test[features_to_scale])


    print('Features scaled with StandardScaler')
    
    return df_train_scaled, df_val_scaled, df_test_scaled

class BatteryDataset(Dataset):
    """Dataset for battery SOH prediction with sequence data"""
    def __init__(self, df, sequence_length):
        self.sequence_length = sequence_length
        
        # 为stateful LSTM保存cell_id信息
        self.cell_ids = []
        
        # Extract features and labels
        features_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        label_col = 'SOH_ZHU'
        features = torch.tensor(df[features_cols].values, dtype=torch.float32)
        labels = torch.tensor(df[label_col].values, dtype=torch.float32)
        
        # 创建cell_id到索引的映射
        if 'cell_id' in df.columns:
            cell_ids = df['cell_id'].values
        else:
            # 如果没有cell_id，创建一个伪cell_id
            cell_ids = np.array(['default'] * len(df))
        
        # Create sequence data efficiently
        n_samples = len(df) - sequence_length
        self.features = torch.zeros((n_samples, sequence_length, len(features_cols)), dtype=torch.float32)
        self.labels = torch.zeros(n_samples, dtype=torch.float32)
        
        # Build sequences using tensor slicing
        for i in range(n_samples):
            self.features[i] = features[i:i+sequence_length]
            self.labels[i] = labels[i+sequence_length]
            # 保存对应的cell_id
            self.cell_ids.append(cell_ids[i+sequence_length])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.cell_ids[idx]

def create_cell_ordered_dataloader(dataset, batch_size):
    """创建按电池ID排序的DataLoader，确保相同cell_id的样本在一起批处理"""
    # 按cell_id对样本进行分组
    indices_by_cell = {}
    for idx in range(len(dataset)):
        cell_id = dataset.cell_ids[idx]
        if cell_id not in indices_by_cell:
            indices_by_cell[cell_id] = []
        indices_by_cell[cell_id].append(idx)
    
    # 为每个电池创建批次
    all_batches = []
    for cell_id, indices in indices_by_cell.items():
        for i in range(0, len(indices), batch_size):
            batch = indices[i:min(i + batch_size, len(indices))]
            all_batches.append(batch)
    
    # 随机打乱电池的顺序，但保持每个电池内部顺序不变
    random.shuffle(all_batches)
    
    # 创建自定义批次采样器
    class CustomBatchSampler(torch.utils.data.Sampler):
        def __init__(self, batches):
            self.batches = batches
        
        def __iter__(self):
            for batch in self.batches:
                yield batch
        
        def __len__(self):
            return len(self.batches)
    
    # 创建DataLoader
    return torch.utils.data.DataLoader(
        dataset, 
        batch_sampler=CustomBatchSampler(all_batches)
    )

class StatefulSOHLSTM(nn.Module):
    """Stateful LSTM model for SOH prediction"""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(StatefulSOHLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 存储每个电池的隐藏状态
        self.hidden_states = {}

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

    def init_hidden(self, cell_id, batch_size):
        """初始化特定电池的隐藏状态"""
        self.hidden_states[cell_id] = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        )
    
    def detach_hidden(self, cell_id):
        """分离特定电池的隐藏状态梯度，防止梯度爆炸"""
        if cell_id in self.hidden_states:
            self.hidden_states[cell_id] = (
                self.hidden_states[cell_id][0].detach(),
                self.hidden_states[cell_id][1].detach()
            )

    def forward(self, x, cell_id):
        """
        前向传播，使用特定电池的隐藏状态
        
        Args:
            x: 输入张量 [batch_size, sequence_length, input_size]
            cell_id: 电池ID
            
        Returns:
            模型预测值 [batch_size]
        """
        batch_size = x.size(0)
        
        # 如果该电池没有隐藏状态或批次大小变化，初始化新的隐藏状态
        if cell_id not in self.hidden_states or self.hidden_states[cell_id][0].size(1) != batch_size:
            self.init_hidden(cell_id, batch_size)
        
        # 使用该电池的隐藏状态进行前向传播
        lstm_out, hidden = self.lstm(x, self.hidden_states[cell_id])
        
        # 更新隐藏状态
        self.hidden_states[cell_id] = hidden
        
        # 只取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        
        # 通过全连接层
        out = self.fc_layers(out)
        
        return out.squeeze(-1)  # shape: [batch_size]

def train_and_validate_stateful_model(model, train_loader, val_loader, save_path):
    """
    Train and validate the stateful LSTM model with early stopping
    
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
    print('\nStart training with stateful LSTM...')
    for epoch in range(hyperparams['EPOCHS']):
        # Training phase
        model.train()
        train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{hyperparams["EPOCHS"]}', leave=False) as pbar:
            for features, labels, cell_ids in train_loader:
                features, labels = features.to(device), labels.to(device)
                cell_id = cell_ids[0]  # 批次内所有样本应该来自同一个电池
                
                optimizer.zero_grad()
                
                # 前向传播（使用特定电池的隐藏状态）
                outputs = model(features, cell_id)
                loss = criterion(outputs, labels)
                
                # 反向传播前分离该电池的隐藏状态
                model.detach_hidden(cell_id)
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                
                train_loss += loss.item()
                pbar.update(1)

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        model.hidden_states = {}
                
        val_loss = 0.0
        with torch.no_grad():
            for features, labels, cell_ids in val_loader:
                features, labels = features.to(device), labels.to(device)
                cell_id = cell_ids[0]
                
                outputs = model(features, cell_id)
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
              f'Validation Loss: {val_loss:.3e} ')

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

def evaluate_stateful_model(model, test_dataset, batch_size, device):
    """
    评估stateful LSTM模型在测试数据集上的性能
    
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
    
    # 创建按时间顺序的DataLoader (不打乱顺序)
    seq_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 获取唯一的cell_id (在测试集中只有一个: "17")
    test_cell_id = test_dataset.cell_ids[0]
    print(f"Evaluating cell_id: {test_cell_id}")
    
    all_preds, all_tgts, all_cids = [], [], []
    
    with torch.no_grad():
        for feats, labs, _ in seq_loader:
            feats, labs = feats.to(device), labs.to(device)
            # 始终使用测试cell_id的状态
            out = model(feats, test_cell_id)
            all_preds.extend(out.cpu().numpy())
            all_tgts.extend(labs.cpu().numpy())
            all_cids.extend([test_cell_id] * len(labs))  # 填充对应的cell_id
    
    preds = np.array(all_preds)
    tgts = np.array(all_tgts)
    
    # 计算性能指标
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(tgts, preds)),
        'MAE': mean_absolute_error(tgts, preds),
        'R²': r2_score(tgts, preds)
    }
    
    print(f"Evaluation metrics for cell {test_cell_id}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return preds, tgts, all_cids, metrics

def plot_predictions_by_cell(predictions, targets, df_test_scaled, results_dir, cell_ids):
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
    test_cell_id = cell_ids[0]  # 所有cell_ids都应该相同
    
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