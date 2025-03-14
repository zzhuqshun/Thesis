from data_processing import *

from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna
import random

import numpy as np
from tqdm import tqdm
import copy

##########################################################
# Data loading
##########################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "01_Datenaufbereitung/Output/Calculated/"
all_data = load_data(data_dir)

train_df, val_df, test_df = split_data(all_data, train=13, val=1, test=1, parts = 1)
train_scaled, val_scaled, test_scaled = scale_data(train_df, val_df, test_df)

# Hyperparameters
seq_len = 120  # Length of input sequence
pred_len = 48   # Length of prediction sequence
batch_size = 64 # Increased batch size
hidden_dim = 80 # Increased hidden dimension
num_layers = 5  # Reduced number of layers
dropout = 0.5   # Adjusted dropout

class BatteryCellDataset(Dataset):
    def __init__(self, df, seq_len, pred_len, is_train=True):
        """
        df: DataFrame 包含列:
            - SOH_ZHU
            - Current[A]
            - Voltage[V]
            - Temperature[°C]
            - cell_id
            - Testtime[h]
        is_train: 是否是训练模式
        """
        self.df = df
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.is_train = is_train
        
        # 获取所有唯一的 cell_id
        self.unique_cells = df['cell_id'].unique()
        
        # 预处理每个cell的数据点
        self.samples = []
        for cell_id in self.unique_cells:
            cell_data = df[df['cell_id'] == cell_id]
            cell_data = cell_data.sort_values('Testtime[h]').reset_index(drop=True)
            
            if is_train:
                # 训练模式：使用较小的stride生成batch训练样本
                stride = max(pred_len // 4, 1)  # 可以根据需要调整stride大小
                for i in range(0, len(cell_data) - seq_len - pred_len + 1, stride):
                    self.samples.append({
                        'cell_id': cell_id,
                        'start_idx': i
                    })
            else:
                # 验证/测试模式：只添加每个电池的起始点
                self.samples.append({
                    'cell_id': cell_id,
                    'start_idx': 0
                })
    
    def shuffle_samples(self):
        """仅在训练模式下打乱样本"""
        if self.is_train:
            random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        cell_id = sample['cell_id']
        start_idx = sample['start_idx']
        
        cell_data = self.df[self.df['cell_id'] == cell_id]
        features = ['SOH_ZHU', 'Current[A]', 'Voltage[V]', 'Temperature[°C]']
        sequence = cell_data[features].iloc[start_idx:start_idx + self.seq_len + self.pred_len].values
        
        X_seed = sequence[:self.seq_len]
        X_future = sequence[self.seq_len:]
        Y_target = sequence[self.seq_len:, 0]
        
        return torch.FloatTensor(X_seed), torch.FloatTensor(X_future), torch.FloatTensor(Y_target), cell_id

def create_data_loaders(train_df, val_df, seq_len, pred_len, batch_size):
    """创建训练和验证数据加载器"""
    train_dataset = BatteryCellDataset(train_df, seq_len, pred_len, is_train=True)
    val_dataset = BatteryCellDataset(val_df, seq_len, pred_len, is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # 使用dataset级别的shuffle
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证保持顺序
        drop_last=False
    )
    
    return train_loader, val_loader

class LSTMSOH(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=80, num_layers=2, dropout=0.1, pred_len=10):
        super(LSTMSOH, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pred_len = pred_len
        
        # Encoder LSTM
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout)
        
        # Decoder LSTM
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout)
        
        # Projection layer
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor, future_features: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            future_features: Future known features (batch_size, pred_len, input_dim-1)
        Returns:
            predictions: Predictions of shape (batch_size, pred_len)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, 
                        dtype=x.dtype, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, 
                        dtype=x.dtype, device=device)
        
        # Encode the input sequence
        _, (hidden, cell) = self.encoder(x, (h0, c0))
        
        # Initialize decoder input with the last step of input sequence
        decoder_input = x[:, -1:, :]
        
        # Store predictions
        predictions = []
        
        # Decode step by step
        for t in range(self.pred_len):
            # Get prediction for current step
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            current_pred = self.fc(decoder_output[:, -1:, :])  # Shape: (batch_size, 1, 1)
            predictions.append(current_pred.squeeze(-1).squeeze(-1)) # Shape: (batch_size,)
            
            # Prepare next input
            if future_features is not None:
                # Combine prediction with known future features
                next_input = torch.cat([
                    current_pred,
                    future_features[:, t:t+1, :]  # Known features for next step
                ], dim=-1)
            else:
                # If no future features provided, use zeros
                next_features = torch.zeros(batch_size, 1, x.size(-1)-1, device=device)
                next_input = torch.cat([current_pred, next_features], dim=-1)
            
            decoder_input = next_input
        
        # Stack predictions along time dimension
        predictions = torch.stack(predictions, dim=1)  # Shape: (batch_size, pred_len)
        
        return predictions


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'val_mae': [], 'val_rmse': [], 'val_r2': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        # -----------------------------
        # 1) Training Loop (Batch mode)
        # -----------------------------
        model.train()
        train_loader.dataset.shuffle_samples()
        train_losses = []
        
        for X_seed, X_future, Y_target, _ in train_loader:
            X_seed = X_seed.to(device)
            X_future = X_future.to(device)
            Y_target = Y_target.to(device)
            
            future_features = X_future[:, :, 1:]
            predictions = model(X_seed, future_features)
            loss = criterion(predictions, Y_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        mean_train_loss = np.mean(train_losses)
        
        # -----------------------------
        # 2) Validation Loop (Singlepass mode)
        # -----------------------------
        model.eval()
        val_metrics = {}
        
        with torch.no_grad():
            for cell_id in val_loader.dataset.unique_cells:
                cell_data = val_loader.dataset.df[val_loader.dataset.df['cell_id'] == cell_id].copy()
                features = ['SOH_ZHU', 'Current[A]', 'Voltage[V]', 'Temperature[°C]']
                data_array = cell_data[features].values
                preds = np.full(len(data_array), np.nan)
                
                # 对每个电池进行连续预测
                for i in range(seq_len, len(data_array) - pred_len + 1, pred_len):
                    input_seq = data_array[i - seq_len : i]
                    future_features = data_array[i:i + pred_len, 1:]
                    
                    x_t = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
                    future_t = torch.tensor(future_features, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    predictions = model(x_t, future_t)
                    
                    preds[i:i + pred_len] = predictions.cpu().numpy().squeeze()
                    # 更新数据用于下一次预测
                    data_array[i:i + pred_len, 0] = predictions.cpu().numpy().squeeze()
                
                # 计算每个电池的指标
                valid_mask = ~np.isnan(preds)
                cell_metrics = {
                    'mae': mean_absolute_error(cell_data[features[0]][valid_mask], preds[valid_mask]),
                    'rmse': np.sqrt(mean_squared_error(cell_data[features[0]][valid_mask], preds[valid_mask])),
                    'r2': r2_score(cell_data[features[0]][valid_mask], preds[valid_mask])
                }
                val_metrics[cell_id] = cell_metrics
        
        # 计算平均验证指标
        mean_val_metrics = {
            'mae': np.mean([m['mae'] for m in val_metrics.values()]),
            'rmse': np.mean([m['rmse'] for m in val_metrics.values()]),
            'r2': np.mean([m['r2'] for m in val_metrics.values()])
        }
        
        # Early Stopping 检查
        if mean_val_metrics['mae'] < best_val_loss:
            best_val_loss = mean_val_metrics['mae']
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # 更新历史记录
        history['val_mae'].append(mean_val_metrics['mae'])
        history['val_rmse'].append(mean_val_metrics['rmse'])
        history['val_r2'].append(mean_val_metrics['r2'])
        history['train_loss'].append(mean_train_loss)

        # 更新进度条描述
        pbar.set_postfix({
            'MAE': f"{mean_val_metrics['mae']:.5e}",
            'R2': f"{mean_val_metrics['r2']:.5f}",
            'RMSE': f"{mean_val_metrics['rmse']:.5e}"
        })
    
    return history, best_model_state


# Create data loaders
train_loader, val_loader = create_data_loaders(
    train_scaled, val_scaled, 
    seq_len, pred_len, 
    batch_size
)

# Create model and optimizer
model = LSTMSOH(
    input_dim=4, 
    hidden_dim=hidden_dim, 
    num_layers=num_layers, 
    dropout=dropout, 
    pred_len=pred_len
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(), 
    lr=5e-5,  # Increased learning rate
    weight_decay=1e-4
)

# Train model with progress bars
history, best_model_state = train_model(
    model, criterion, optimizer, 
    train_loader, val_loader, 
    num_epochs=50,  # Reduced number of epochs
    patience=5      # Reduced patience for early stopping
)

# Save best model
torch.save(best_model_state, "best_model.pth")


def evaluate_singlepass(model, df, seq_len=100, pred_len=50):
    model.eval()
    features = ['SOH_ZHU', 'Current[A]', 'Voltage[V]', 'Temperature[°C]']
    data_array = df[features].values.copy()
    preds = np.full(len(data_array), np.nan)
    
    with torch.no_grad():
        for i in range(seq_len, len(data_array) - pred_len + 1, pred_len):
            # 获取输入序列
            input_seq = data_array[i - seq_len : i]
            # 获取未来特征
            future_features = data_array[i:i + pred_len, 1:]
            
            # 转换为张量并添加批次维度
            x_t = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            future_t = torch.tensor(future_features, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 进行多步预测
            predictions = model(x_t, future_t)
            
            # 保存预测结果
            preds[i:i + pred_len] = predictions.cpu().numpy().squeeze()
            
            # 更新数据数组中的SOH值，用于下一个预测窗口
            data_array[i:i + pred_len, 0] = predictions.cpu().numpy().squeeze()
    
    return preds

# Evaluate on test set
model.load_state_dict(best_model_state)
all_preds = evaluate_singlepass(model, test_scaled, seq_len=seq_len, pred_len=pred_len)
all_targets = test_scaled['SOH_ZHU'].values

# 只使用非NaN的预测值进行评估
valid_mask = ~np.isnan(all_preds)
valid_preds = all_preds[valid_mask]
valid_targets = all_targets[valid_mask]

r2 = r2_score(valid_targets, valid_preds)
mae = mean_absolute_error(valid_targets, valid_preds)
rmse = np.sqrt(mean_squared_error(valid_targets, valid_preds))
    
print("\nTest Set Metrics:")
print(f"R2: {r2:.5f}")
print(f"MAE: {mae:.5e}")
print(f"RMSE: {rmse:.5e}")