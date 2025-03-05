from data_processing import *

from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna

import numpy as np
from tqdm import tqdm
import copy

##########################################################
# Data loading
##########################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "../01_Datenaufbereitung/Output/Calculated/"
all_data = load_data(data_dir)

train_df, val_df, test_df = split_data(all_data, train=13, val=1, test=1,parts = 5)
train_scaled, val_scaled, test_scaled = scale_data(train_df, val_df, test_df)

##########################################################
# Dataset definition    
##########################################################
class SequenceDataset(Dataset):
    def __init__(self, df, seed_len=36, pred_len=5):
        self.seed_len = seed_len
        self.pred_len = pred_len
        # (目标 + 外生) 例如 [SOH_ZHU, Current, Voltage, Temperature]
        self.data_all = df[['SOH_ZHU', 'Current[A]', 'Voltage[V]', 'Temperature[°C]']].values

    def __len__(self):
        return len(self.data_all) - (self.seed_len + self.pred_len) + 1

    def __getitem__(self, idx):
        block = self.data_all[idx : idx + self.seed_len + self.pred_len]
        # block shape: (seed_len + pred_len, 4)

        # 历史: [0 : seed_len], 未来: [seed_len : seed_len + pred_len]
        x_seed = block[:self.seed_len]          # (seed_len, 4)
        x_future = block[self.seed_len:]        # (pred_len, 4)

        # 目标只取这 pred_len 行的第 0 列
        y_target = x_future[:, 0]  # shape (pred_len, )
        return (
            torch.tensor(x_seed, dtype=torch.float32),
            torch.tensor(x_future, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32)
        )

# Using ground truth of SOH and 3 cova riances
seq_length=13
pred_length= 10  
batch_size=16
train_dataset = SequenceDataset(train_scaled, seed_len=seq_length, pred_len=pred_length)
val_dataset = SequenceDataset(val_scaled, seed_len=seq_length, pred_len=pred_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


##########################################################  
# Model definition  
##########################################################
class LSTMSOH(nn.Module):
    def __init__(self, input_dim = 4, hidden_dim = 128, num_layers = 3, dropout = 0.3):
        super(LSTMSOH, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout= dropout)
        # Attention layer: project hidden state at each time step to a scalar attention weight
        # self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, dtype=x.dtype, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, dtype=x.dtype, device=x.device)
        lstm_out, _ = self.lstm(x,(h0,c0))  # lstm_out shape: (batch_size, seq_len, hidden_dim)
        
        # # # Compute attention scores and normalize them
        # attn_scores = self.attention(lstm_out)  # shape: (batch_size, seq_len, 1)
        # attn_weights = torch.softmax(attn_scores, dim=1)  # softmax over seq_len
        
        # # # Compute the context vector as the weighted sum of LSTM outputs
        # context = torch.sum(attn_weights * lstm_out, dim=1)  # shape: (batch_size, hidden_dim)
        # out = self.fc(context)  # Final prediction, shape: (batch_size, 1)
        
        out = self.fc(lstm_out[:,-1,:]) # (batch_size, hidden_dim) -> (batch_size, 1)
        
        return out.squeeze(-1) #(batch_size,)


##########################################################
# Training process
##########################################################      
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, patience=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    history = {'train_loss': [],'val_loss': [],'val_mae': [],'val_rmse': [],'val_r2': []}
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # -----------------------------
        # 1) Training Loop (pure autoregressive)
        # -----------------------------
        model.train()
        train_losses = []
        
        for X_seed, X_future, Y_target in train_loader:
            X_seed = X_seed.to(device)# X_seed shape: (batch_size, seed_len, 4)       # [SOH, Current, Voltage, Temp]
            X_future = X_future.to(device)# X_future shape: (batch_size, pred_len, 4) 
            Y_target = Y_target.to(device)# Y_target shape: (batch_size, pred_len)
                        
            current_seq = X_seed.clone()  # 初始输入(含目标+外生), shape (batch_size, seed_len, 4)
            preds_steps = []

            for t in range(pred_len):
                pred = model(current_seq)  # -> shape (batch_size,)
                preds_steps.append(pred.unsqueeze(1))

                # 取下一时刻的未来特征(含目标列): shape (batch_size, 4)
                next_frame = X_future[:, t, :].clone()
                # 替换目标列 (SOH) 为模型预测
                next_frame[:, 0] = pred

                # 滑动窗口
                current_seq = torch.cat([current_seq[:, 1:, :], next_frame.unsqueeze(1)], dim=1)
            preds_steps = torch.cat(preds_steps, dim=1)
            loss = criterion(preds_steps, Y_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        mean_train_loss = np.mean(train_losses)
        history['train_loss'].append(mean_train_loss)
        
        # -----------------------------
        # 2) Validation Loop (pure autoregressive)
        # -----------------------------
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_seed, X_future, Y_target in val_loader:
                # 将数据转移到 device 上
                X_seed = X_seed.to(device)
                X_future = X_future.to(device)
                Y_target = Y_target.to(device)
                
                # 当前序列初始化为历史seed数据（含目标和外生变量）
                current_seq = X_seed.clone()
                preds_steps = []
                
                for t in range(Y_target.shape[1]):  # 这里 pred_len = Y_target.shape[1]
                    pred = model(current_seq)  # 输出 shape: (batch_size,)
                    preds_steps.append(pred.unsqueeze(1))
                    
                    # 取出未来时刻的外生变量（含目标列）
                    next_frame = X_future[:, t, :].clone()
                    # 将目标位置（假设为第0列）替换为当前预测
                    next_frame[:, 0] = pred
                    # 更新序列：移除最旧的一帧，添加新的预测帧
                    current_seq = torch.cat([current_seq[:, 1:, :], next_frame.unsqueeze(1)], dim=1)
                
                preds_steps = torch.cat(preds_steps, dim=1)  # (batch_size, pred_len)
                val_loss = criterion(preds_steps, Y_target)
                val_losses.append(val_loss.item())
                
                all_preds.append(preds_steps.cpu().numpy())
                all_targets.append(Y_target.cpu().numpy()) 

        mean_val_loss = np.mean(val_losses)
        history['val_loss'].append(mean_val_loss)
        
        # Calculate overall MAE, RMSE, R2
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 使用sklearn.metrics中的函数
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds)) 
        r2 = r2_score(all_targets, all_preds)
        
        history['val_mae'].append(mae)
        history['val_rmse'].append(rmse)
        history['val_r2'].append(r2)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {mean_train_loss:.4e} | Val Loss: {mean_val_loss:.4e} | "
              f"MAE: {mae:.4e} | RMSE: {rmse:.4e} | R2: {r2:.4f}")
        
        # -----------------------------
        # 3) Early Stopping
        # -----------------------------
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return history, best_model_state

model = LSTMSOH().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1.4296450393279462e-05, weight_decay=0.00012)

history, best_model_state = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100, patience=10)


    
##########################################################
# Hyperparameter Optimization
##########################################################
# def objective(trial):
#     # Suggest hyperparameters
#     hidden_size = trial.suggest_int('hidden_size', 32, 256, step = 16)
#     num_layers = trial.suggest_int('num_layers', 2, 5)
#     learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
#     dropout = trial.suggest_float('dropout', 0.1,0.5)
#     weight_decay= trial.suggest_float('weight_decay',1e-5,1e-1, log=True)
    
#     seed_len = trial.suggest_int('seed_len', 12, 128)
#     pred_len = trial.suggest_int('pred_len', 1, 20)
#     batch_size = trial.suggest_int('batch_size', 16, 64, step = 8)
    
#     train_dataset = SequenceDataset(train_scaled, seed_len=seed_len, pred_len=pred_len)
#     val_dataset = SequenceDataset(val_scaled, seed_len=seed_len, pred_len=pred_len)
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
#     # Instantiate model with suggested hyperparameters
#     model = LSTMSOH(input_dim=4, hidden_dim=hidden_size, num_layers=num_layers, dropout=dropout).type(torch.float32).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #L2 regularization
#     history, _ = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs = 100, patience = 10)

#     # Extract last validation loss
#     last_val_loss = history['val_loss'][-1]
#     return last_val_loss

## Optuna study
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)

# # Extract best trial
# best_trial = study.best_trial
# print(f"Best trial: {best_trial}")

# best_hyperparams = study.best_trial.params
# print('Best hyperparameters:', best_hyperparams)

##########################################################
# Evaluation
##########################################################  

# predict_autoregressive
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_seed, X_future, Y_target in test_loader:
            X_seed = X_seed.to(device)        # shape: (batch_size, seed_len, 4)
            X_future = X_future.to(device)    # shape: (batch_size, pred_len, 4)
            Y_target = Y_target.to(device)    # shape: (batch_size, pred_len)
            
            batch_size, seed_len, num_features = X_seed.shape
            pred_len = Y_target.shape[1]
            
            current_seq = X_seed.clone()  # 初始输入：历史的seed数据
            preds_steps = []
            
            for t in range(pred_len):
                # 预测下一时刻目标值
                pred = model(current_seq)  # 输出 shape: (batch_size,)
                preds_steps.append(pred.unsqueeze(1))
                
                # 从X_future中获取当前时刻的外生变量（含目标列）
                next_frame = X_future[:, t, :].clone()  # shape: (batch_size, num_features)
                # 将目标位置（假设第0列）替换为预测值
                next_frame[:, 0] = pred
                # 更新输入序列：移除最早的时间步，添加新的预测帧
                current_seq = torch.cat([current_seq[:, 1:, :], next_frame.unsqueeze(1)], dim=1)
            
            preds_steps = torch.cat(preds_steps, dim=1)  # (batch_size, pred_len)
            all_preds.append(preds_steps.cpu().numpy())
            all_targets.append(Y_target.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    r2 = r2_score(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    
    print(f"R2:{r2:.5f} | MAE: {mae:.5e} | RMSE:{rmse:.5e}")
    
    return all_preds, all_targets, r2, mae, rmse


# Load best model   
model_path = '04_NNs/Results/LSTM/C,V,T/00/best_model.pth'
model = LSTMSOH().to(device)
model.load_state_dict(torch.load(model_path))

seed_len = 13
pred_len = 10
batch_size = 16

test_dataset = SequenceDataset(test_scaled, seed_len=seed_len, pred_len=pred_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

all_preds, all_targets, r2, mae, rmse = evaluate_model(model, test_loader)

plt.figure(figsize=(10, 5))
plt.plot(all_targets[:, 0], label="Ground Truth SOH")
plt.plot(all_preds[:, 0], label="Predicted SOH")
plt.title("Autoregressive SOH-Estimation - Test ")
plt.text(0.5, 0.95, f"R2:{r2:.5f} | MAE: {mae:.5e}| RMSE:{rmse:.5e}", 
         horizontalalignment='center', transform=plt.gca().transAxes)
print(f"R2:{r2:.5f} | MAE: {mae:.5e}| RMSE:{rmse:.5e}")
plt.xlabel("Samples")
plt.ylabel("SOH")
plt.legend()
plt.tight_layout()
plt.show()
