# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib widget
from data_processing import *
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import optuna

# %%
### Load data
data_dir = "../01_Datenaufbereitung/Output/Calculated/"
all_data = load_data(data_dir)


# %%
train_df, val_df, test_df = split_data(all_data, train=13, val=1, test=1,parts = 5)
train_scaled, val_scaled, test_scaled = scale_data(train_df, val_df, test_df)

# %%
### Visualize data
# visualize_data(all_data)
# inspect_data_ranges(all_data)
# inspect_data_ranges(train_scaled)
# plot_dataset_soh(train_df, "Train")
# plot_dataset_soh(val_df, "Validation")
# plot_dataset_soh(test_df, "Test")

# %%
class SequenceDataset(Dataset):
    def __init__(self, df, seed_len=36, pred_len=10,
                 feature_cols=["Current[A]", "Voltage[V]", "Temperature[°C]"],
                 target_col="SOH_ZHU"):
        self.seed_len = seed_len
        self.pred_len = pred_len
        self.features = df[feature_cols].values
        self.targets = df[target_col].values

    def __len__(self):
        return len(self.features) - (self.seed_len + self.pred_len) + 1

    def __getitem__(self, idx):
        # X: 前 seed_len 个时间步 (batch_size, seed_len, num_features)
        x_seq = self.features[idx : idx + self.seed_len]
        # Y: 后续 pred_len 个目标 (batch_size, pred_len)
        y_seq = self.targets[idx + self.seed_len : idx + self.seed_len + self.pred_len]
        
        x = torch.tensor(x_seq, dtype=torch.float32)
        y = torch.tensor(y_seq, dtype=torch.float32)
        return x, y


# Using ground truth of SOH and 3 covariances
seq_len=36
batch_size=16
train_cols = ["SOH_ZHU","Current[A]", "Voltage[V]", "Temperature[°C]"]
# feature_cols=["Current[A]", "Voltage[V]", "Temperature[°C]"]
# train_cols = ["Current[A]", "Voltage[V]", "Temperature[°C]", "Q_sum", "EFC", 'InternalResistance[Ohms]']
# feature_cols=["SOH_ZHU", "Current[A]", "Voltage[V]", "Temperature[°C]", "Q_sum", "EFC", 'InternalResistance[Ohms]']

train_dataset = SequenceDataset(train_scaled, feature_cols=train_cols)
val_dataset = SequenceDataset(val_scaled, feature_cols=train_cols)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)




# %%
x, y = val_dataset[0]
print("First sample X shape:", x.shape)
print("First sample y:", y)

# %%
class LSTMAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super(LSTMAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout= dropout)
        # Attention layer: project hidden state at each time step to a scalar attention weight
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, dtype=x.dtype, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, dtype=x.dtype, device=x.device)
        lstm_out, _ = self.lstm(x,(h0,c0))  # lstm_out shape: (batch_size, seq_len, hidden_dim)
        
        # # Compute attention scores and normalize them
        attn_scores = self.attention(lstm_out)  # shape: (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # softmax over seq_len
        
        # # Compute the context vector as the weighted sum of LSTM outputs
        context = torch.sum(attn_weights * lstm_out, dim=1)  # shape: (batch_size, hidden_dim)
        out = self.fc(context )  # Final prediction, shape: (batch_size, 1)
        return out.squeeze(-1)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import copy

def train_model(model, 
                                        criterion, 
                                        optimizer, 
                                        train_loader, 
                                        val_loader, 
                                        num_epochs=10, 
                                        patience=10):
    """
    Train the model using Scheduled Sampling.
    
    Assumptions:
      - Each training batch consists of:
          X_batch: tensor of shape (batch_size, seed_len, num_features)
          Y_batch: tensor of shape (batch_size, pred_len) containing the ground truth for the autoregressive steps.
      - The model, given an input sequence of shape (batch_size, seed_len, num_features),
        outputs the prediction for the next time step (shape: (batch_size,)).
      - The target variable is assumed to be at index 0 in the input features.
    
    Scheduled Sampling:
      - At each autoregressive step, with probability p_teacher, use the ground truth value as the next input.
        With probability (1 - p_teacher), use the model's prediction.
      - p_teacher decays with epoch (here we use a linear decay from 1 to 0).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'val_r2': []
    }
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Define target column index in the input features (adjust if needed)
    target_idx = 0
    
    for epoch in range(num_epochs):
        # Compute teacher forcing probability (linearly decay from 1 to 0)
        p_teacher = max(0, 1 - epoch / num_epochs)
        model.train()
        train_losses = []
        
        for X_batch, Y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False):
            # Assume X_batch: (batch_size, seed_len, num_features)
            #        Y_batch: (batch_size, pred_len)
            X_batch = X_batch.to(device)   # seed input sequence
            Y_batch = Y_batch.to(device)   # ground truth for autoregressive steps
            
            batch_size, seed_len, num_features = X_batch.shape
            pred_len = Y_batch.shape[1]
            
            # current_seq will be updated autoregressively
            current_seq = X_batch.clone()
            preds_steps = []
            
            # Unroll for pred_len steps
            for t in range(pred_len):
                # Predict the next value given the current sequence
                pred = model(current_seq)  # output shape: (batch_size,)
                preds_steps.append(pred.unsqueeze(1))  # shape: (batch_size, 1)
                
                # Decide, for each sample, whether to use teacher forcing or the prediction
                # Generate a mask: 1 means use ground truth, 0 means use prediction.
                teacher_mask = (torch.rand(batch_size, device=device) < p_teacher).float()
                
                # The ground truth for this time step is Y_batch[:, t]
                gt_next = Y_batch[:, t]
                # Next value for the target feature: if teacher forcing then use ground truth, else use prediction.
                next_value = teacher_mask * gt_next + (1 - teacher_mask) * pred
                
                # Construct the next input vector:
                # For simplicity，我们假设除目标变量外，其余特征直接用 X_batch 中对应时刻的值（或者保持不变）。
                # 这里我们取 X_batch 中第 t+seed_len 时刻的数值（如果存在），否则保持原样。
                # 如果你的数据不提供未来其他特征，可直接不更新其他特征或使用固定值。
                if seed_len + t < X_batch.shape[1]:
                    # 如果 X_batch 时间步足够，则采用真实值作为其他特征
                    next_input = X_batch[:, seed_len + t, :].clone()
                else:
                    # 否则，保持当前最后一帧的其他特征不变
                    next_input = current_seq[:, -1, :].clone()
                
                # 将目标变量更新为 next_value
                next_input[:, target_idx] = next_value
                
                # 更新 current_seq：删除第一个时间步，添加新的预测时间步
                current_seq = torch.cat([current_seq[:, 1:, :], next_input.unsqueeze(1)], dim=1)
            
            # 将所有预测步拼接，形状变为 (batch_size, pred_len)
            preds_steps = torch.cat(preds_steps, dim=1)
            loss = criterion(preds_steps, Y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        mean_train_loss = np.mean(train_losses)
        history['train_loss'].append(mean_train_loss)
        
        # Validation loop using fully autoregressive
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val = X_val.to(device)
                Y_val = Y_val.to(device)
                batch_size, seed_len, num_features = X_val.shape
                pred_len = Y_val.shape[1]

                current_seq = X_val.clone()
                preds_steps = []
                for t in range(pred_len):
                    pred = model(current_seq)
                    preds_steps.append(pred.unsqueeze(1))
                    
                    next_value = pred

                    if seed_len + t < X_val.shape[1]:
                        next_input = X_val[:, seed_len + t, :].clone()
                    else:
                        next_input = current_seq[:, -1, :].clone()
                    next_input[:, target_idx] = next_value
                    current_seq = torch.cat([current_seq[:, 1:, :], next_input.unsqueeze(1)], dim=1)
                    
                preds_steps = torch.cat(preds_steps, dim=1)
                loss = criterion(preds_steps, Y_val)
                val_losses.append(loss.item())
                all_preds.append(preds_steps.cpu().numpy())
                all_targets.append(Y_val.cpu().numpy())
        
        mean_val_loss = np.mean(val_losses)
        history['val_loss'].append(mean_val_loss)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets)**2))
        ss_res = np.sum((all_targets - all_preds)**2)
        ss_tot = np.sum((all_targets - np.mean(all_targets))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        history['val_mae'].append(mae)
        history['val_rmse'].append(rmse)
        history['val_r2'].append(r2)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {mean_train_loss:.4e} | Val Loss: {mean_val_loss:.4e} | "
              f"MAE: {mae:.4e} | RMSE: {rmse:.4e} | R2: {r2:.4f} | p_teacher: {p_teacher:.2f}")
        
        # Early stopping check
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} because validation loss did not improve.")
                break

    return history, best_model_state
    



# %%
model = LSTMAttention(input_dim=4, hidden_dim=128, num_layers=3, dropout= 0.3)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.MSELoss()
history, best_state = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=50, patience=10)

# %%
# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()


# %%

# # Example usage:
# # Assume LSTMAttentionModel, train_loader, and val_loader have been defined elsewhere.
# def objective(trial):
#     # Suggest hyperparameters
#     hidden_size = trial.suggest_int('hidden_size', 10, 100)
#     num_layers = trial.suggest_int('num_layers', 1, 5)
#     learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

#     # Instantiate model with suggested hyperparameters
#     model = LSTMAttention(input_dim=6, hidden_dim=hidden_size, num_layers=num_layers).type(torch.float32).to(device)

#     # Define your loss function and optimizer with suggested hyperparameters
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     # Call your train and validate function
#     history, best_state = train_model(model, criterion, optimizer, train_loader, val_loader)

#     # Extract last validation loss
#     last_val_loss = history['val_loss'][-1]
#     return last_val_loss

#     # Optuna study
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=10)

# # Extract best trial
# best_trial = study.best_trial
# print(f"Best trial: {best_trial}")

# best_hyperparams = study.best_trial.params
# print('Best hyperparameters:', best_hyperparams)

# %%
def predict_autoregressive(model, df, seq_len=36,
                           input_cols=["SOH_ZHU", "Current[A]", "Voltage[V]", "Temperature[°C]"],
                           target_col="SOH_ZHU",
                           device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Autoregressive prediction: Instead of using the true SOH for the next step,
    use the model's last predicted SOH.
    
    Procedure:
      - Copy the specified input columns into an array.
      - Use the first `seq_len` true values as the initial sequence.
      - From index seq_len onward, update the target column with the model's prediction for subsequent predictions.
    
    Parameters:
      model: Trained PyTorch model.
      df (pd.DataFrame): DataFrame containing the data.
      seq_len (int): Sequence length.
      input_cols (list): List of columns used for autoregressive prediction, which must include the target column (e.g., "SOH_ZHU").
      target_col (str): Name of the target column, default "SOH_ZHU".
      device: Device to use (CPU or GPU).
    
    Returns:
      preds (np.ndarray): An array of predictions with the same length as df (the first seq_len values are NaN).
    """
    model.eval()
    # Copy the data for the specified input columns, ensuring the original df is not modified
    data_array = df[input_cols].values.copy()
    preds = np.full(len(data_array), np.nan)
    
    # Find the index of the target column within input_cols
    target_idx = input_cols.index(target_col)
    
    # Save the true target values (for metric calculations)
    y_true = df[target_col].values.copy()
    
    with torch.no_grad():
        # Loop over the data such that we have a full sequence of length seq_len
        for i in range(seq_len, len(data_array)):
            # Prepare the input sequence of shape (seq_len, num_features)
            input_seq = data_array[i - seq_len: i]
            x_t = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get the predicted SOH value from the model
            pred_soh = model(x_t).item()
            preds[i] = pred_soh
            
            # Write the predicted value back to data_array at the target column, to be used for future predictions
            data_array[i, target_idx] = pred_soh

    # Filter out the first seq_len values (which are NaN) for metric calculations
    valid_mask = ~np.isnan(preds)
    y_pred_valid = preds[valid_mask]
    y_true_valid = y_true[valid_mask]

    mae = np.mean(np.abs(y_pred_valid - y_true_valid))
    rmse = np.sqrt(np.mean((y_pred_valid - y_true_valid)**2))
    
    ss_res = np.sum((y_true_valid - y_pred_valid)**2)
    ss_tot = np.sum((y_true_valid - np.mean(y_true_valid))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # Print the metrics
    print(f"Autoregressive prediction metrics: MAE:  {mae:.4e} |  RMSE: {rmse:.4e} | R2:   {r2:.4e}")

    return preds

preds_target = predict_autoregressive(model, test_scaled, seq_len=36)


# %%
true_target = test_scaled['SOH_ZHU'].values  # "echter" SOH (unskalierter, da 0..1)
timeidx  = test_scaled['Testtime[h]'].values

# scaler_soh = RobustScaler()
# train_scaled['SOH_ZHU'] = scaler_soh.fit_transform(train_df[['SOH_ZHU']])

# preds_unscaled = scaler_soh.inverse_transform(preds_target.reshape(-1, 1)).flatten()
# true_unscaled = scaler_soh.inverse_transform(true_target.reshape(-1, 1)).flatten()

plt.figure(figsize=(10,5))
plt.plot(timeidx, true_target, label="Ground Truth SOH")
plt.plot(timeidx, preds_target, label="Predicted SOH (autoregressive)")
plt.title(f"Autoregressive SOH-Vorhersage - Test")
plt.xlabel("Time")
plt.ylabel("SOH")
plt.legend()
plt.tight_layout()

# %%



