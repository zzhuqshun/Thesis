from data_processing import *

from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna
import os
import numpy as np
from tqdm import tqdm
import copy
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################
# Data loading
##########################################################

data_dir = "01_Datenaufbereitung/Output/Calculated/"
all_data = load_data(data_dir)

train_df, val_df, test_df = split_data(all_data, train=13, val=1, test=1,parts = 1)
train_scaled, val_scaled, test_scaled = scale_data(train_df, val_df, test_df)

##########################################################
# Dataset definition    
##########################################################
class SequenceDataset(Dataset):
    def __init__(self, df, seed_len=36, pred_len=5, is_train=True):
        """
        Args:
            df: DataFrame containing SOH and other features
            seed_len: Length of input sequence
            pred_len: Length of prediction sequence
            is_train: If True, use sliding window with stride=1, else use non-overlapping windows
        """
        self.seed_len = seed_len
        self.pred_len = pred_len
        self.is_train = is_train
        
        # Get unique cell IDs
        self.cell_ids = df['cell_id'].unique()
        
        # Store data for each cell separately
        self.cell_data = {}
        self.samples = []
        
        for cell_id in self.cell_ids:
            # Get data for this cell
            cell_df = df[df['cell_id'] == cell_id]
            # Sort by time
            cell_df = cell_df.sort_values('Testtime[h]')
            # Store features
            self.cell_data[cell_id] = cell_df[['SOH_ZHU', 'Current[A]', 'Voltage[V]', 'Temperature[°C]']].values
            
            data_len = len(self.cell_data[cell_id])
            
            if is_train:
                # Training mode: use sliding window with stride=1
                for i in range(0, data_len - seed_len - pred_len + 1):
                    self.samples.append({
                        'cell_id': cell_id,
                        'start_idx': i
                    })
            else:
                # Validation/Test mode: use non-overlapping windows
                for i in range(0, data_len - seed_len - pred_len + 1, pred_len):
                    self.samples.append({
                        'cell_id': cell_id,
                        'start_idx': i
                    })

    def shuffle(self):
        """Shuffle samples for training"""
        if self.is_train:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cell_id = sample['cell_id']
        start_idx = sample['start_idx']
        
        # Get data block for this sample
        data = self.cell_data[cell_id]
        block = data[start_idx : start_idx + self.seed_len + self.pred_len]
        
        # Split into input and target sequences
        x_seed = block[:self.seed_len]          # (seed_len, 4)
        x_future = block[self.seed_len:]        # (pred_len, 4)
        y_target = x_future[:, 0]               # (pred_len,)
        
        return (
            torch.tensor(x_seed, dtype=torch.float32),
            torch.tensor(x_future, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32),
            cell_id
        )


##########################################################  
# Model definition  
##########################################################
class LSTMSOH(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=3, dropout=0.1, pred_len=50):
        super(LSTMSOH, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pred_len = pred_len
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers, batch_first=True, 
            dropout=dropout 
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers, batch_first=True, 
            dropout=dropout 
        )
        
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
            current_pred = self.fc(decoder_output[:, -1:, :]) # Shape: (batch_size, 1)
            predictions.append(current_pred.squeeze(-1))  # Shape: (batch_size,)
            
            # Prepare next input
            if future_features is not None:
                # Combine prediction with known future features
                next_input = torch.cat([
                    current_pred.squeeze(1),  # Predicted SOH for next step
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


##########################################################
# Training process
##########################################################      
def evaluate_continuous(model, data, seed_len, pred_len, device):
    """连续预测整个序列"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        # 初始种子序列
        current_sequence = data[:seed_len]
        
        # 对剩余序列进行预测
        for i in range(seed_len, len(data) - pred_len + 1, pred_len):
            x_seed = torch.FloatTensor(current_sequence[-seed_len:]).unsqueeze(0).to(device)
            future_features = torch.FloatTensor(data[i:i + pred_len, 1:]).unsqueeze(0).to(device)
            
            # 预测下一个窗口
            pred = model(x_seed, future_features)
            pred = pred.cpu().numpy().squeeze()
            
            # 保存预测结果和目标值
            predictions.append(pred)
            targets.append(data[i:i + pred_len, 0])
            
            # 更新序列，加入预测值和实际特征
            new_sequence = np.column_stack((
                pred,
                data[i:i + pred_len, 1:]
            ))
            current_sequence = np.vstack((current_sequence, new_sequence))
    
    return np.concatenate(predictions), np.concatenate(targets)

def train_model(model, criterion, optimizer, train_loader, val_dataset, num_epochs=10, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    history = {
        'train_loss': [], 
        'val_mae': [], 'val_rmse': [], 'val_r2': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # -----------------------------
        # 1) Training Loop
        # -----------------------------
        model.train()
        train_loader.dataset.shuffle()
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
        history['train_loss'].append(mean_train_loss)
        
        # -----------------------------
        # 2) Validation Loop (自回归)
        # -----------------------------
        model.eval()
        
        # 对验证集中的电池进行连续预测
        cell_id = val_dataset.cell_ids[0]  # 只有一个电池
        cell_data = val_dataset.cell_data[cell_id]
        
        val_predictions, val_targets = evaluate_continuous(
            model, cell_data, seq_length, pred_length, device
        )
        
        # 计算验证指标
        val_mae = mean_absolute_error(val_targets, val_predictions)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
        val_r2 = r2_score(val_targets, val_predictions)
        
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {mean_train_loss:.4e}")
        print(f"Val Metrics: MAE: {val_mae:.4e} | RMSE: {val_rmse:.4e} | R2: {val_r2:.4f}")
        
        # Early Stopping based on MAE
        if val_mae < best_val_loss:
            best_val_loss = val_mae
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    return history, best_model_state

def evaluate_model(model, test_dataset):
    model.eval()
    
    # 获取测试电池数据
    cell_id = test_dataset.cell_ids[0]  # 只有一个电池
    cell_data = test_dataset.cell_data[cell_id]
    
    # 进行连续预测
    predictions, targets = evaluate_continuous(
        model, cell_data, seq_length, pred_length, device
    )
    
    # 计算指标
    metrics = {
        'r2': r2_score(targets, predictions),
        'mae': mean_absolute_error(targets, predictions),
        'rmse': np.sqrt(mean_squared_error(targets, predictions))
    }
    
    print("\nTest Metrics:")
    print(f"R2: {metrics['r2']:.5f}")
    print(f"MAE: {metrics['mae']:.5e}")
    print(f"RMSE: {metrics['rmse']:.5e}")
    
    return predictions, targets, metrics

seq_length = 13
pred_length = 10
batch_size = 16 

# Create datasets with appropriate modes
train_dataset = SequenceDataset(train_scaled, seed_len=seq_length, pred_len=pred_length, is_train=True)
val_dataset = SequenceDataset(val_scaled, seed_len=seq_length, pred_len=pred_length, is_train=False)

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=False,  # We'll use dataset-level shuffling
    drop_last=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    drop_last=False  # Keep all validation samples
)


model = LSTMSOH(
    input_dim=4,
    hidden_dim=128,
    num_layers=3,
    dropout=0.3,
    pred_len=pred_length  # 使用之前定义的pred_length
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-3
)

# 训练模型
history, best_model_state = train_model(
    model, criterion, optimizer,
    train_loader, val_dataset,
    num_epochs=100,
    patience=10
)

# 保存最佳模型
torch.save(best_model_state, "best_model.pth")


##########################################################
# Evaluation
##########################################################  

# 加载最佳模型进行评估
model.load_state_dict(best_model_state)
test_dataset = SequenceDataset(test_scaled, seed_len=seq_length, pred_len=pred_length, is_train=False)

# 评估模型
predictions, targets, metrics = evaluate_model(model, test_dataset)

# 绘制预测结果
plt.figure(figsize=(12, 6))
plt.plot(targets, label='Ground Truth')
plt.plot(predictions, label='Predicted')
plt.title(f"SOH Prediction - Test Set\nR2: {metrics['r2']:.5f} | MAE: {metrics['mae']:.5e} | RMSE: {metrics['rmse']:.5e}")
plt.xlabel('Time Steps')
plt.ylabel('SOH')
plt.legend()
plt.tight_layout()
plt.show()