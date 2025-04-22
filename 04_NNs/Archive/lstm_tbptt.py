import random
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
import matplotlib.pyplot as plt

# ========== Reproducibility & Device Setup ==========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Setup and Hyperparameters ==========
model_dir = Path(__file__).parent / "models" / datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir.mkdir(parents=True, exist_ok=True)

hyperparams = {
    "INPUT_DIM": 3,
    "HIDDEN_DIM": 32,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.5,
    "BATCH_SIZE": 64,
    "TBPTT_LEN": 144,
    "LR": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    "MAX_EPOCHS": 100,
    "PATIENCE": 10,
    "RESAMPLE_RULE": "10min",
    "MODEL_DIR": str(model_dir)
}
# Save hyperparameters
with open(model_dir / "hyperparameters.json", "w") as fp:
    json.dump(hyperparams, fp, indent=4)

# ========== Data Preparation ==========
def load_and_scale(data_dir: Path, rule: str, feature_cols: list, label_col: str):
    """
    Load parquet files, split into train/val/test, resample and scale features.
    Prints cell IDs for each split for traceability.
    """
    print("[Step] Loading and scaling data...")
    files = sorted(data_dir.glob("*.parquet"), key=lambda x: int(x.stem.split("_")[-1]))
    test_file = random.choice(files)
    rest = [f for f in files if f != test_file]
    val_files = random.sample(rest, max(1, len(rest) // 5))
    train_files = [f for f in rest if f not in val_files]

    # Extract cell IDs
    train_ids = [f.stem.split("_")[1] for f in train_files]
    val_ids = [f.stem.split("_")[1] for f in val_files]
    test_id = test_file.stem.split("_")[1]
    print(f"Train cell IDs: {train_ids}")
    print(f"Validation cell IDs: {val_ids}")
    print(f"Test cell ID: {test_id}\n")

    def process(fp: Path):
        df = pd.read_parquet(fp)[feature_cols + [label_col]].dropna()
        df["Datetime"] = pd.date_range("2023-02-02", periods=len(df), freq="s")
        df = df.set_index("Datetime").resample(rule).mean().reset_index()
        df["cell_id"] = fp.stem.split("_")[1]
        return df

    df_train = pd.concat([process(f) for f in train_files], ignore_index=True)
    df_val = pd.concat([process(f) for f in val_files], ignore_index=True)
    df_test = process(test_file)

    scaler = StandardScaler().fit(df_train[feature_cols])
    for df in (df_train, df_val, df_test):
        df[feature_cols] = scaler.transform(df[feature_cols])

    print(f"Training dataframe shape: {df_train.shape}")
    print(f"Validation dataframe shape: {df_val.shape}")
    print(f"Testing dataframe shape: {df_test.shape}\n")
    return df_train, df_val, df_test

class StreamCellDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list, label_col: str):
        self.groups = [g.sort_values("Datetime") for _, g in df.groupby("cell_id")]
        self.feature_cols = feature_cols
        self.label_col = label_col

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx: int):
        df = self.groups[idx]
        x = torch.tensor(df[self.feature_cols].values, dtype=torch.float32)
        y = torch.tensor(df[self.label_col].values, dtype=torch.float32)
        return x, y

# Pad sequences and return lengths for masking
def collate_fn(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    xs_p = pad_sequence(xs, batch_first=True)
    ys_p = pad_sequence(ys, batch_first=True)
    return xs_p, ys_p, lengths

# ========== Model Definition ==========
class SOHLSTM_TBPTT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, hidden=None):
        if hidden is None:
            batch = x.size(0)
            layers = self.lstm.num_layers
            h0 = x.new_zeros(layers, batch, self.lstm.hidden_size)
            c0 = x.new_zeros(layers, batch, self.lstm.hidden_size)
        else:
            h0, c0 = hidden
        out, (hn, cn) = self.lstm(x, (h0, c0))
        preds = (
            self.fc(out.contiguous().view(-1, out.size(-1)))
            .view(x.size(0), -1)
        )
        return preds, (hn, cn)

# ========== Training & Evaluation ==========
def train_tbptt(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, hp: dict):
    print("[Step] Starting TBPTT training...")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["LR"], weight_decay=hp["WEIGHT_DECAY"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss(reduction="none")
    best_val, patience_cnt = float("inf"), 0

    train_losses, val_losses = [], []
    for epoch in range(1, hp["MAX_EPOCHS"] + 1):
        model.train()
        total_train = 0.0
        for x_batch, y_batch, lengths in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            lengths = lengths.to(device)
            hidden = None
            batch_loss, wn = 0.0, 0
            L = x_batch.size(1)
            for start in range(0, L, hp["TBPTT_LEN"]):
                xw = x_batch[:, start:start + hp["TBPTT_LEN"]]
                yw = y_batch[:, start:start + hp["TBPTT_LEN"]]
                mask = (torch.arange(xw.size(1), device=device)[None, :] < (lengths[:, None] - start)).float()
                preds, hidden = model(xw, hidden)
                hidden = (hidden[0].detach(), hidden[1].detach())
                loss = (criterion(preds, yw) * mask).sum() / mask.sum()
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                batch_loss += loss.item(); wn += 1
            total_train += batch_loss / wn
        avg_tr = total_train / len(train_loader)
        train_losses.append(avg_tr)

        # Validation
        model.eval(); total_val=0.0
        with torch.no_grad():
            for x_batch, y_batch, lengths in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                lengths=lengths.to(device); hidden=None; batch_loss, wn =0.0,0; L=x_batch.size(1)
                for start in range(0, L, hp["TBPTT_LEN"]):
                    xw=x_batch[:, start:start+hp["TBPTT_LEN"]]
                    yw=y_batch[:, start:start+hp["TBPTT_LEN"]]
                    mask=(torch.arange(xw.size(1),device=device)[None,:]<(lengths[:,None]-start)).float()
                    preds,hidden=model(xw,hidden); hidden=(hidden[0].detach(),hidden[1].detach())
                    loss=(criterion(preds,yw)*mask).sum()/mask.sum()
                    batch_loss+=loss.item();wn+=1
                total_val+=batch_loss/wn
        avg_vl=total_val/len(val_loader)
        val_losses.append(avg_vl)
        print(f"Epoch {epoch}|Train:{avg_tr:.4e}|Val:{avg_vl:.4e}")
        scheduler.step(avg_vl)
        if avg_vl<best_val:
            best_val,patience_cnt=avg_vl,0
            torch.save(model.state_dict(),Path(hp["MODEL_DIR"]) / "best.pth")
        else:
            patience_cnt+=1
            if patience_cnt>=hp["PATIENCE"]:
                print("Early stopping")
                break
    # Plot loss curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.tight_layout()
    loss_fig = Path(hp["MODEL_DIR"]) / 'train_val_loss.png'
    plt.savefig(loss_fig)
    plt.close()
    print(f"[Step] Training completed. Loss curve saved to {loss_fig}\n")


def test_tbptt(model: nn.Module, df_test: pd.DataFrame, feature_cols: list, label_col: str, hp: dict):
    """
    Load best model, run TBPTT on test set, compute metrics and return preds, y_true, datetime.
    """
    # Use weights_only to avoid unpickle risks
    state = torch.load(Path(hp["MODEL_DIR"]) / "best.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    x = torch.tensor(df_test[feature_cols].values, dtype=torch.float32, device=device).unsqueeze(0)
    y_true = df_test[label_col].values
    hidden = None
    warmup = hp["TBPTT_LEN"]
    with torch.no_grad():
        _, hidden = model(x[:, :warmup], hidden)
        hidden = (hidden[0].detach(), hidden[1].detach())
    preds = []
    seq_len = x.size(1)
    for start in range(warmup, seq_len, hp["TBPTT_LEN"]):
        xw = x[:, start:start + hp["TBPTT_LEN"]]
        out, hidden = model(xw, hidden)
        hidden = (hidden[0].detach(), hidden[1].detach())
        # Detach before numpy conversion
        preds.append(out.squeeze(0).detach().cpu().numpy())
    preds = np.concatenate(preds)
    y_eval = y_true[warmup : warmup + len(preds)]
    rmse = np.sqrt(mean_squared_error(y_eval, preds))
    mae = mean_absolute_error(y_eval, preds)
    mape = mean_absolute_percentage_error(y_eval, preds)
    r2 = r2_score(y_eval, preds)
    print(f"RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}, R2={r2:.4f}")
    dates = df_test['Datetime'].iloc[warmup : warmup + len(preds)]
    return preds, y_eval, dates

# ========== Main ==========
if __name__ == "__main__":
    base = Path(__file__).parent.parent / "01_Datenaufbereitung" / "Output" / "Calculated"
    feature_cols = ["Voltage[V]", "Current[A]", "Temperature[°C]"]
    label_col = "SOH_ZHU"
    df_tr, df_val, df_test = load_and_scale(base, hyperparams["RESAMPLE_RULE"], feature_cols, label_col)

    # Store globals for use in train_tbptt
    df_test_global = df_test
    feature_cols_global = feature_cols
    label_col_global = label_col

    tr_dl = DataLoader(
        StreamCellDataset(df_tr, feature_cols, label_col),
        batch_size=hyperparams["BATCH_SIZE"],
        shuffle=True,
        collate_fn=collate_fn
    )
    vl_dl = DataLoader(
        StreamCellDataset(df_val, feature_cols, label_col),
        batch_size=hyperparams["BATCH_SIZE"],
        shuffle=False,
        collate_fn=collate_fn
    )
    model = SOHLSTM_TBPTT(
        hyperparams["INPUT_DIM"],
        hyperparams["HIDDEN_DIM"],
        hyperparams["NUM_LAYERS"],
        hyperparams["DROPOUT"]
    ).to(device)

    train_tbptt(model, tr_dl, vl_dl, hyperparams)
    
    print("[Step] Starting testing...")
    preds, y_eval, dates = test_tbptt(model, df_test_global, feature_cols_global, label_col_global, hyperparams)
    # Plot predictions vs true
    plt.figure()
    plt.plot(dates, y_eval, label='True SOH')
    plt.plot(dates, preds, label='Predicted SOH')
    plt.xlabel('Datetime')
    plt.ylabel('SOH')
    plt.title('True vs Predicted SOH')
    plt.legend()
    plt.tight_layout()
    pred_fig = Path(hyperparams["MODEL_DIR"]) / 'pred_vs_true.png'
    plt.savefig(pred_fig)
    plt.close()
    print(f"[Step] Testing completed. Prediction plot saved to {pred_fig}\n")