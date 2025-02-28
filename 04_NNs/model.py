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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "../01_Datenaufbereitung/Output/Calculated/"
all_data = load_data(data_dir)

train_df, val_df, test_df = split_data(all_data, train=13, val=1, test=1,parts = 5)
train_scaled, val_scaled, test_scaled = scale_data(train_df, val_df, test_df)

class SequenceDataset(Dataset):
    def __init__(self, df, seed_len = 36, pred_len = 5):
        self.seed_len = seed_len
        self.pred_len = pred_len
        self.data = df[["SOH_ZHU",'Current[A]', 'Voltage[V]','Temperature[°C]']].values

    def __len__(self):
        return len(self.data) - (self.seed_len + self.pred_len)

    def __getitem__(self, idx):
        # X: (batch_size, seq_len, num_features)
        x_seq = self.data[idx : idx + self.seed_len]
        # Y: (batch_size, pred_len)
        y_seq = self.data[idx + self.seed_len : idx + self.seed_len + self.pred_len, 0]

        x = torch.tensor(x_seq, dtype=torch.float32)
        y = torch.tensor(y_seq, dtype=torch.float32)
        return x, y



# Using ground truth of SOH and 3 covariances
seq_length=72
batch_size=32
train_dataset = SequenceDataset(train_scaled, seed_len=seq_length)
val_dataset = SequenceDataset(val_scaled, seed_len=seq_length)
test_dataset = SequenceDataset(test_scaled, seed_len=seq_length)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

class LSTMSOH(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
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

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, patience=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    history = {'train_loss': [],'val_loss': [],'val_mae': [],'val_rmse': [],'val_r2': []}
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    target_idx = 0
    
    for epoch in range(num_epochs):
        # -----------------------------
        # 1) Training Loop (pure autoregressive)
        # -----------------------------
        model.train()
        train_losses = []
        
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)  # shape: (batch_size, seed_len, num_features)
            Y_batch = Y_batch.to(device)  # shape: (batch_size, pred_len)
            
            batch_size, seed_len, num_features = X_batch.shape
            pred_len = Y_batch.shape[1]
            
            # current_seq als autoregressive rolling window
            current_seq = X_batch.clone()  # (batch_size, seed_len, num_features)
            preds_steps = []
            
            for t in range(pred_len):
                # 1) Predicting the next time step with the current sequence
                pred = model(current_seq)  # (batch_size,)
                preds_steps.append(pred.unsqueeze(1))  # -> (batch_size, 1)
                
                # 2) Write pred back to current_seq's target column, ready for the next time step.
                # - If there are other features for the next moment, replace them with the true values from the X_batch.
                # - Otherwise, leave the last feature unchanged.
                if seed_len + t < X_batch.shape[1]:
                    # Explain that X_batch also provides other characteristics of this moment in time
                    next_input = X_batch[:, seed_len + t, :].clone()
                else:
                    # exceed seed_len or no more features, only the current last frame will be kept.
                    next_input = current_seq[:, -1, :].clone()
                
                # Replace target columns with model predictions
                next_input[:, target_idx] = pred
                
                # 3) Move the window: remove the top frame, put in a new prediction frame
                #   current_seq[:, 1:, :] -> drop the last step
                #   next_input.unsqueeze(1) -> (batch_size, 1, num_features)
                current_seq = torch.cat([current_seq[:, 1:, :], next_input.unsqueeze(1)], dim=1)
            
            preds_steps = torch.cat(preds_steps, dim=1)
            loss = criterion(preds_steps, Y_batch)
            
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
            for X_val, Y_val in val_loader:
                X_val = X_val.to(device)
                Y_val = Y_val.to(device)
                batch_size, seed_len, num_features = X_val.shape
                pred_len = Y_val.shape[1]
                
                current_seq = X_val.clone()
                preds_steps = []
                
                for t in range(pred_len):
                    pred = model(current_seq)  # (batch_size,)
                    preds_steps.append(pred.unsqueeze(1))
                    
                    if seed_len + t < X_val.shape[1]:
                        next_input = X_val[:, seed_len + t, :].clone()
                    else:
                        next_input = current_seq[:, -1, :].clone()
                    
                    next_input[:, target_idx] = pred
                    current_seq = torch.cat([current_seq[:, 1:, :], next_input.unsqueeze(1)], dim=1)
                
                preds_steps = torch.cat(preds_steps, dim=1)  # (batch_size, pred_len)
                val_loss = criterion(preds_steps, Y_val)
                val_losses.append(val_loss.item())
                
                all_preds.append(preds_steps.cpu().numpy())
                all_targets.append(Y_val.cpu().numpy())
        
        mean_val_loss = np.mean(val_losses)
        history['val_loss'].append(mean_val_loss)
        
        # Calculate overall MAE, RMSE, R2
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets)**2))
        
        ss_res = np.sum((all_targets - all_preds)**2)
        ss_tot = np.sum((all_targets - np.mean(all_targets))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        
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

    


# Example usage:
def objective(trial):
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 32, 256, step = 16)
    num_layers = trial.suggest_int('num_layers', 2, 5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    dropout = trial.suggest_float('dropout', 0.1,0.5)
    weight_decay= trial.suggest_float('weight_decay',1e-5,1e-1, log=True)
    
    seed_len = trial.suggest_int('seed_len', 12, 128)
    pred_len = trial.suggest_int('pred_len', 1, 20)
    batch_size = trial.suggest_int('batch_size', 16, 64, step = 8)
    
    train_dataset = SequenceDataset(train_scaled, seed_len=seed_len, pred_len=pred_len)
    val_dataset = SequenceDataset(val_scaled, seed_len=seed_len, pred_len=pred_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    # Instantiate model with suggested hyperparameters
    model = LSTMSOH(input_dim=4, hidden_dim=hidden_size, num_layers=num_layers, dropout=dropout).type(torch.float32).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #L2 regularization
    history, _ = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs = 100, patience = 10)

    # Extract last validation loss
    last_val_loss = history['val_loss'][-1]
    return last_val_loss

    # Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Extract best trial
best_trial = study.best_trial
print(f"Best trial: {best_trial}")

best_hyperparams = study.best_trial.params
print('Best hyperparameters:', best_hyperparams)



