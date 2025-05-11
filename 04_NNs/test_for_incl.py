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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------------------------------------------------------- 
# Global configuration & helper functions
# ----------------------------------------------------------------------------- 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_dir = Path(__file__).parent / "models/LSTM/test_for_incl" / "MinMAx" / "2025-05-10_10-18-14"
save_dir.mkdir(exist_ok=True, parents=True)

hyperparams = {
    "INFO": [
        "Model: SOH_LSTM",
        "Data: 10-min resampling of battery cycle data",
        "Degradation categories: normal, fast, faster",
        "Data split: Train['01','03','05','21','27'], Validation ['23'], Test ['17']",
        "Features: MinMax-scaled voltage & temperature (0-1), current (-1-1)",
    ],
    "SEQUENCE_LENGTH": 864,
    "HIDDEN_SIZE": 256,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.4,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 3e-4,
    "WEIGHT_DECAY": 3e-06,
    "RESAMPLE": "10min",
    "EPOCHS": 100,
    "PATIENCE": 10,
    "device": str(device),
}

def main():
    """Run the SOH-prediction pipeline."""
    hyperparams_path = save_dir / "hyperparameters.json"
    with open(hyperparams_path, "w", encoding="utf-8") as f:
        json.dump(hyperparams, f, indent=4)

    set_seed(42)
    print(f"Using device: {device}\n")

    data_dir = Path("../01_Datenaufbereitung/Output/Calculated/")
    df_train, df_val, df_test = load_data(data_dir, resample=hyperparams["RESAMPLE"])
    df_train_s, df_val_s, df_test_s = scale_data(df_train, df_val, df_test)

    train_ds = BatteryDataset(df_train_s, hyperparams["SEQUENCE_LENGTH"])
    val_ds = BatteryDataset(df_val_s, hyperparams["SEQUENCE_LENGTH"])
    test_ds = BatteryDataset(df_test_s, hyperparams["SEQUENCE_LENGTH"])

    train_loader = DataLoader(train_ds, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hyperparams["BATCH_SIZE"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=hyperparams["BATCH_SIZE"], shuffle=False)

    model = SOHLSTM(
        input_size=3,
        hidden_size=hyperparams["HIDDEN_SIZE"],
        num_layers=hyperparams["NUM_LAYERS"],
        dropout=hyperparams["DROPOUT"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["LEARNING_RATE"],
        weight_decay=hyperparams["WEIGHT_DECAY"]
    )

    model_paths = {
        "best": save_dir / "best_soh_model.pth",
        "last": save_dir / "last_soh_model.pth",
        "history": save_dir / "train_history.parquet",
        "checkpoint": save_dir / "checkpoint.pth",
    }

    TRAINING_MODE = False

    if TRAINING_MODE:
        if os.path.exists(model_paths["checkpoint"]):
            print(f"\nLoading checkpoint from {model_paths['checkpoint']}...")
            checkpoint = torch.load(
                model_paths["checkpoint"],
                map_location=device,
                weights_only=True
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch_start = checkpoint["epoch"]
            best_val_loss = checkpoint["best_val"]
            print(f"Resuming from epoch {epoch_start} with best validation loss: {best_val_loss:.4f}")
        else:
            print("\nNo checkpoint found, starting training from scratch.")
            epoch_start = 0
            best_val_loss = None

        history, best_val_loss = train_and_validate_model(
            model,
            train_loader,
            val_loader,
            model_paths,
            start_epoch=epoch_start,
            initial_best_val=best_val_loss
        )
        print(f"\nBest validation loss: {best_val_loss:.4f}")

    else:
        print("\nLoading pre-trained model for testing...")
        if os.path.exists(model_paths["best"]):
            model.load_state_dict(
                torch.load(model_paths["best"], map_location=device, weights_only=True)
            )
            print("Model loaded successfully!")
        else:
            print("Error: No pre-trained model found!")
            return

    print("\nEvaluating the model on the test set …")
    preds, tgts, metrics = evaluate_model(model, test_loader)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    results_dir = save_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    if model_paths["history"].exists():
        plot_losses(pd.read_parquet(model_paths["history"]), results_dir)

    plot_predictions(preds, tgts, metrics, df_test_s, results_dir)

    with open(hyperparams_path, "w", encoding="utf-8") as f:
        json.dump(hyperparams, f, indent=4)
    print(f"\nUpdated run information written to {hyperparams_path.relative_to(Path.cwd())}")

# ----------------------------------------------------------------------------- 
# Utility functions 
# ----------------------------------------------------------------------------- 

def set_seed(seed: int = 42):
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
    train_ids = ['01', '03', '25', '21', '27']
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
    df_train_s = df_train.copy()
    df_val_s = df_val.copy()
    df_test_s = df_test.copy()

    vt_cols = ["Voltage[V]", "Temperature[°C]"]
    c_col   = ["Current[A]"]

    vt_scaler = MinMaxScaler((0, 1)).fit(df_train[vt_cols])
    c_scaler  = MinMaxScaler((-1, 1)).fit(df_train[c_col])

    df_train_s[vt_cols] = vt_scaler.transform(df_train[vt_cols])
    df_val_s[vt_cols]   = vt_scaler.transform(df_val[vt_cols])
    df_test_s[vt_cols]  = vt_scaler.transform(df_test[vt_cols])

    df_train_s[c_col] = c_scaler.transform(df_train[c_col])
    df_val_s[c_col]   = c_scaler.transform(df_val[c_col])
    df_test_s[c_col]  = c_scaler.transform(df_test[c_col])

    print("Voltage & temperature scaled to [0,1]; current to [-1,1]\n")
    return df_train_s, df_val_s, df_test_s

class BatteryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int):
        self.seq_len = seq_len
        feats_cols = ["Voltage[V]", "Current[A]", "Temperature[°C]"]
        label_col  = "SOH_ZHU"

        feats  = torch.tensor(df[feats_cols].values, dtype=torch.float32)
        labels = torch.tensor(df[label_col].values,  dtype=torch.float32)

        n = len(df) - seq_len
        self.X = torch.zeros((n, seq_len, len(feats_cols)), dtype=torch.float32)
        self.y = torch.zeros(n, dtype=torch.float32)

        for i in range(n):
            self.X[i] = feats[i : i + seq_len]
            self.y[i] = labels[i + seq_len]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SOHLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        batch = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch, self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1]).squeeze(-1)

def train_and_validate_model(
    model,
    train_loader,
    val_loader,
    paths,
    start_epoch: int = 0,
    initial_best_val: float = None
):
    """Train and validate the model."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["LEARNING_RATE"],
        weight_decay=hyperparams["WEIGHT_DECAY"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val = initial_best_val if initial_best_val is not None else float("inf")
    patience_ctr = 0
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    print(f"\nStart training from epoch {start_epoch+1} …")
    for epoch in range(start_epoch+1, hyperparams["EPOCHS"]+1):
        model.train()
        running_train = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch}/{hyperparams['EPOCHS']}", leave=False) as pbar:
            for X, y in pbar:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_train += loss.item()

        train_loss = running_train / len(train_loader)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                running_val += criterion(model(X), y).item()

        val_loss = running_val / len(val_loader)
        scheduler.step(val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch}/{hyperparams['EPOCHS']}  |  "
            f"train {train_loss:.3e}  val {val_loss:.3e}  "
            f"lr {optimizer.param_groups[0]['lr']:.1e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), paths["best"])
        else:
            patience_ctr += 1
            if patience_ctr >= hyperparams["PATIENCE"]:
                print(f"Early stopping at epoch {epoch}")
                break

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val": best_val
        }, paths["checkpoint"])

    torch.save(model.state_dict(), paths["last"])
    pd.DataFrame(history).to_parquet(paths["history"], index=False)
    return history, best_val

def evaluate_model(model, dl):
    model.eval()
    criterion = nn.MSELoss()
    preds, tgts = [], []

    with torch.no_grad():
        for X, y in dl:
            X, y = X.to(device), y.to(device)
            p = model(X)
            preds.extend(p.cpu().numpy())
            tgts.extend(y.cpu().numpy())

    preds = np.asarray(preds)
    tgts  = np.asarray(tgts)
    return preds, tgts, {
        "RMSE": np.sqrt(mean_squared_error(tgts, preds)),
        "MAE" : mean_absolute_error(tgts, preds),
        "R²"  : r2_score(tgts, preds),
    }

def plot_losses(hist_df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(10, 6))
    plt.semilogy(hist_df["epoch"], hist_df["train_loss"], label="train")
    plt.semilogy(hist_df["epoch"], hist_df["val_loss"],   label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png")
    plt.close()

def plot_predictions(preds, tgts, metrics, df_test_s, out_dir: Path):
    date_index = df_test_s["Datetime"].iloc[hyperparams["SEQUENCE_LENGTH"]:].values
    plt.figure(figsize=(12, 6))
    plt.plot(date_index, tgts, label="actual")
    plt.plot(date_index, preds, label="predicted", alpha=0.7)
    plt.xlabel("date")
    plt.ylabel("SOH")
    plt.legend()
    plt.grid(True)
    plt.title(
        f"Predictions vs. Actual SOH\n"
        f"RMSE: {metrics['RMSE']:.4f}, "
        f"MAE: {metrics['MAE']:.4f}, "
        f"R²: {metrics['R²']:.4f}"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_timeseries.png")
    plt.close()

def plot_prediction_scatter(preds, tgts, out_dir: Path):
    plt.figure(figsize=(8, 8))
    plt.scatter(tgts, preds, alpha=0.6)
    lims = [min(tgts.min(), preds.min()), max(tgts.max(), preds.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("actual SOH")
    plt.ylabel("predicted SOH")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_scatter.png")
    plt.close()

if __name__ == "__main__":
    main()
