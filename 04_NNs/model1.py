import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###########################################
# 1) 数据加载与预处理 (示例)
###########################################
from data_processing import load_data, split_data, scale_data

data_dir = os.path.abspath("01_Datenaufbereitung/Output/Calculated")
print(f"Loading data from: {data_dir}")
all_data = load_data(data_dir)

# 示例: 13:1:1 划分 + parts=5，视项目需要
train_df, val_df, test_df = split_data(all_data, train=13, val=1, test=1, parts=5)
train_scaled, val_scaled, test_scaled = scale_data(train_df, val_df, test_df)


###########################################
# 2) Dataset：返回协变量、历史SOH和目标SOH，支持调度采样
###########################################
class SequenceDataset(Dataset):
    """
    返回协变量、历史SOH和目标SOH，支持调度采样。
    """
    def __init__(self, df, seed_len=24, pred_len=1):
        super().__init__()
        self.seed_len = seed_len
        self.pred_len = pred_len
        
        # 分离协变量和目标
        self.covariates = df[['Current[A]', 'Voltage[V]', 'Temperature[°C]']].values
        self.soh = df['SOH_ZHU'].values

    def __len__(self):
        return len(self.covariates) - (self.seed_len + self.pred_len)

    def __getitem__(self, idx):
        # 协变量序列
        x_covariates = self.covariates[idx : idx + self.seed_len]
        # SOH历史序列
        x_soh = self.soh[idx : idx + self.seed_len]
        # 目标SOH序列
        y_seq = self.soh[idx + self.seed_len : idx + self.seed_len + self.pred_len]

        return {
            'covariates': torch.tensor(x_covariates, dtype=torch.float32),
            'soh_history': torch.tensor(x_soh, dtype=torch.float32),
            'target': torch.tensor(y_seq, dtype=torch.float32)
        }


###########################################
# 3) DataLoader
###########################################
seed_len  = 24  # 过去多少步
pred_len  = 5   # 未来预测多少步
batch_size = 16

train_dataset = SequenceDataset(train_scaled, seed_len=seed_len, pred_len=pred_len)
val_dataset   = SequenceDataset(val_scaled,   seed_len=seed_len, pred_len=pred_len)
test_dataset  = SequenceDataset(test_scaled,  seed_len=seed_len, pred_len=pred_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=True)


###########################################
# 4) 模型定义：单向LSTM -> 全连接 -> 输出pred_len步
###########################################
class LSTMSOH(nn.Module):
    """
    模型：仅对协变量进行编码，直接预测未来 pred_len 步SOH。
    """
    def __init__(self, covariate_dim=3, hidden_dim=64, num_layers=2, dropout=0.2, pred_len=5):
        super(LSTMSOH, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        
        # 协变量编码器
        self.covariate_encoder = nn.LSTM(
            input_size=covariate_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        # SOH历史编码器
        self.soh_encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len)
        )
        
    def forward(self, covariates, soh_history=None, teacher_forcing_ratio=0.0):
        batch_size = covariates.size(0)
        
        # 编码协变量（始终使用）
        h0_cov = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=covariates.device)
        c0_cov = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=covariates.device)
        cov_out, _ = self.covariate_encoder(covariates, (h0_cov, c0_cov))
        
        # 获取协变量最后时刻的特征
        cov_features = cov_out[:, -1, :]  # (batch_size, hidden_dim)
        
        if soh_history is not None and teacher_forcing_ratio > 0:
            # 编码SOH历史
            soh_history = soh_history.unsqueeze(-1)
            h0_soh = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=covariates.device)
            c0_soh = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=covariates.device)
            soh_out, _ = self.soh_encoder(soh_history, (h0_soh, c0_soh))
            
            # 获取SOH历史最后时刻的特征，并按比例使用
            soh_features = soh_out[:, -1, :] * teacher_forcing_ratio
        else:
            # 不使用SOH历史时，该部分特征为0
            soh_features = torch.zeros_like(cov_features)
        
        # 组合特征
        combined_features = torch.cat([cov_features, soh_features], dim=1)
        
        # 预测
        predictions = self.fc(combined_features)
        
        return predictions


###########################################
# 5) 训练函数：无 Teacher Forcing
###########################################
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=50, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r2': []}
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # 计算teacher forcing比例（从0.8逐渐减小到0）
        teacher_forcing_ratio = max(0, 0.8 * (1 - epoch / (num_epochs * 0.7)))
        
        # Training Loop
        model.train()
        train_losses = []
        
        for batch in train_loader:
            covariates = batch['covariates'].to(device)
            soh_history = batch['soh_history'].to(device)
            targets = batch['target'].to(device)
            
            predictions = model(
                covariates=covariates,
                soh_history=soh_history,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            
            loss = criterion(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        mean_train_loss = np.mean(train_losses)
        history['train_loss'].append(mean_train_loss)
        
        # Validation Loop
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                covariates = batch['covariates'].to(device)
                targets = batch['target'].to(device)
                
                # 验证时不使用teacher forcing
                predictions = model(
                    covariates=covariates,
                    soh_history=None,
                    teacher_forcing_ratio=0.0
                )
                
                val_loss = criterion(predictions, targets)
                val_losses.append(val_loss.item())
                
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # 计算验证指标
        mean_val_loss = np.mean(val_losses)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets.flatten(), all_preds.flatten())
        
        history['val_loss'].append(mean_val_loss)
        history['val_mae'].append(mae)
        history['val_rmse'].append(rmse)
        history['val_r2'].append(r2)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] "
              f"TF Ratio: {teacher_forcing_ratio:.2f} | "
              f"Train Loss: {mean_train_loss:.4e} | "
              f"Val Loss: {mean_val_loss:.4e} | "
              f"MAE: {mae:.4e} | "
              f"RMSE: {rmse:.4e} | "
              f"R2: {r2:.4f}")
        
        # Early Stopping
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    return history, best_model_state


###########################################
# 6) 初始化模型并训练
###########################################
model = LSTMSOH(
    covariate_dim=3,   # 电流、电压、温度
    hidden_dim=80,
    num_layers=2,
    dropout=0.2,
    pred_len=pred_len  # 一次性预测未来 pred_len 步
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

history, best_state = train_model(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    patience=15
)

# 保存最佳模型
torch.save(best_state, "best_model_direct_multistep.pth")

###########################################
# 7) 可视化训练曲线
###########################################
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.semilogy(history['train_loss'], label='Train Loss')
plt.semilogy(history['val_loss'],   label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()


###########################################
# 8) 测试阶段：与验证保持一致
###########################################
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            covariates = batch['covariates'].to(device)
            targets    = batch['target'].to(device)
            
            preds = model(covariates)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_preds   = np.concatenate(all_preds,   axis=0)  # shape (N, pred_len)
    all_targets = np.concatenate(all_targets, axis=0)  # shape (N, pred_len)
    
    mae  = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2   = r2_score(all_targets.flatten(), all_preds.flatten())
    
    print(f"Test R2: {r2:.4f} | MAE: {mae:.4e} | RMSE: {rmse:.4e}")
    return {
        'predictions': all_preds,
        'targets':     all_targets,
        'mae':         mae,
        'rmse':        rmse,
        'r2':          r2
    }

# 加载最佳模型权重
model.load_state_dict(best_state)
model.to(device)

metrics = evaluate_model(model, test_loader, device)
all_preds = metrics['predictions']
all_targets = metrics['targets']

# 简单可视化 (只看预测第1步)
plt.figure(figsize=(10,5))
plt.plot(all_targets[:, 0], label="True SOH (first step)")
plt.plot(all_preds[:, 0],   label="Pred SOH (first step)")
plt.title("Direct Multi-step SOH Prediction - Test")
plt.xlabel("Samples")
plt.ylabel("SOH (scaled?)")
plt.legend()
plt.tight_layout()
plt.show()
