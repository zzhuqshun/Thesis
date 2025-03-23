import argparse
import os
import json
import copy
import random
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ====== 这里是关键，导入 CATS.py 的 Model ======
from CATS import Model as CATSModel

# ---- 如果你需要用到 run.py 里 argparse 的各种参数写法，也可以拷过来一起用 ----
# 这里只是示例演示，所以只保留最主要参数
parser = argparse.ArgumentParser()
parser.add_argument("--run_folder", type=str, default=None,
                    help="Folder path to save models and logs.")

# CATS 常用参数
parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
parser.add_argument("--pred_len", type=int, default=24, help="prediction sequence length")
parser.add_argument("--dec_in", type=int, default=4, help="number of input channels (features)")
parser.add_argument("--d_model", type=int, default=128, help="dimension of the model")
parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
parser.add_argument("--d_ff", type=int, default=256, help="dimension of FCN")
parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
parser.add_argument("--d_layers", type=int, default=3, help="num of decoder layers for CATS")
parser.add_argument("--QAM_start", type=float, default=0.1, help="start prob of QueryAdaptiveMasking")
parser.add_argument("--QAM_end", type=float, default=0.3, help="end prob of QueryAdaptiveMasking")
parser.add_argument("--patch_len", type=int, default=24, help="patch length in CATS")
parser.add_argument("--stride", type=int, default=24, help="patch stride in CATS")
parser.add_argument("--query_independence", action="store_true", help="whether to share query across dimension")
# 也可以根据需要添加更多 CATS 参数

args = parser.parse_args()

# 如果没有传 run_folder，则生成一个默认目录（可选）
if args.run_folder is None:
    args.run_folder = os.path.join(
        "models",
        datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
os.makedirs(args.run_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 你原先用的超参，这里演示保留（部分做了合并），也可以都合并到 argparse 中
hyperparams = {
    "MODEL": "CATS-based SOH predictor",
    "SEQUENCE_LENGTH": args.seq_len,      # 统一到 CATS 的 seq_len
    "PREDICT_LENGTH": args.pred_len,      # 统一到 CATS 的 pred_len
    "HIDDEN_SIZE": args.d_model,          # 对应 CATS 的 d_model
    "DROPOUT": args.dropout,
    "BATCH_SIZE": 64,
    "LEARNING_RATE": 1e-4,
    "EPOCHS": 100,
    "PATIENCE": 10,
    "WEIGHT_DECAY": 1e-5
}

# 将参数保存到 run_folder 里（可选）
with open(os.path.join(args.run_folder, "hyperparameters.json"), "w") as f:
    json.dump(hyperparams, f, indent=4)

# =========== 数据部分，你的原始逻辑保持不变，如下 ===========
from data_processing import load_data, split_data, scale_data
data_dir = "../01_Datenaufbereitung/Output/Calculated/"
all_data = load_data(data_dir)
train_df, val_df, test_df = split_data(all_data, train=13, val=1, test=1, parts=1)
train_scaled, val_scaled, test_scaled = scale_data(train_df, val_df, test_df)

# 自定义数据集
class CellDataset(Dataset):
    def __init__(self, df, sequence_length=60, pred_len=1, stride=1):
        features_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]', 'EFC']  # 4 个特征
        label_col = 'SOH_ZHU'
        cell_id_col = 'cell_id'
        
        self.sequence_length = sequence_length
        self.pred_len = pred_len
        self.features = []
        self.labels = []
        
        for cell_id in df[cell_id_col].unique():
            cell_data = df[df[cell_id_col] == cell_id].sort_index()
            cell_features = torch.tensor(cell_data[features_cols].values, dtype=torch.float32)
            cell_labels = torch.tensor(cell_data[label_col].values, dtype=torch.float32)
            n_samples = len(cell_data) - sequence_length - pred_len + 1
            
            if n_samples > 0:
                for i in range(0, n_samples, stride):
                    feature_window = cell_features[i:i + sequence_length]
                    label_window = cell_labels[i + sequence_length : i + sequence_length + pred_len]
                    self.features.append(feature_window)  # (seq_len, 4)
                    self.labels.append(label_window)       # (pred_len,)
        
        self.features = torch.stack(self.features)  # shape: [N, seq_len, 4]
        self.labels = torch.stack(self.labels)      # shape: [N, pred_len]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = CellDataset(train_scaled,
                            sequence_length=hyperparams["SEQUENCE_LENGTH"],
                            pred_len=hyperparams["PREDICT_LENGTH"])
val_dataset = CellDataset(val_scaled,
                          sequence_length=hyperparams["SEQUENCE_LENGTH"],
                          pred_len=hyperparams["PREDICT_LENGTH"])
test_dataset = CellDataset(test_scaled,
                           sequence_length=hyperparams["SEQUENCE_LENGTH"],
                           pred_len=hyperparams["PREDICT_LENGTH"])

train_loader = DataLoader(train_dataset, batch_size=hyperparams["BATCH_SIZE"],
                          shuffle=True, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=hyperparams["BATCH_SIZE"],
                        shuffle=False, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_dataset, batch_size=hyperparams["BATCH_SIZE"],
                         shuffle=False, pin_memory=torch.cuda.is_available())


# =========== 用 CATS 替代原先的 LSTM ===========

# 封装一个 PyTorch nn.Module，用来对接后续训练循环
# 你也可以不封装，直接把 CATSModel(args) 实例化并 forward
class CATSWrapper(nn.Module):
    def __init__(self, args):
        super(CATSWrapper, self).__init__()
        self.cats_model = CATSModel(args)  # 直接使用 CATS.py 里的 Model 类

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, num_features]
        CATS 默认期望 [batch_size, input_len, channel].
        这里 x 本身就是 (batch, seq_len, 4)，正好满足要求。
        CATS 输出: [batch_size, pred_len, channel].
        """
        out = self.cats_model(x)  # => shape [batch, pred_len, 4] 当 dec_in=4 时
        return out

# 实例化 CATS
model = CATSWrapper(args).to(device)

# ====== 训练与验证 ======
def train_and_validation(model, train_loader, val_loader, hyperparams):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["LEARNING_RATE"],
        weight_decay=hyperparams["WEIGHT_DECAY"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    epochs_no_improve = 0
    best_val_loss = float('inf')

    history = {"train_loss": [], "val_loss": [], "epoch": []}

    print("\nStart training...")
    for epoch in range(hyperparams["EPOCHS"]):
        # ---- Training phase ----
        model.train()
        train_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{hyperparams["EPOCHS"]}', leave=False) as pbar:
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()

                # forward
                outputs = model(features)  # [batch_size, pred_len, 4]

                # 如果只需要用其中一个通道来做回归（例如通道 0），则:
                # outputs_for_loss = outputs[..., 0]
                # 让它 shape match labels => [batch_size, pred_len]
                # 也可以做更复杂的映射，这里示例取第 0 通道
                outputs_for_loss = outputs[..., 0]

                loss = criterion(outputs_for_loss, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                train_loss += loss.item()
                pbar.update(1)
        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # ---- Validation phase ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)  # [batch_size, pred_len, 4]
                outputs_for_loss = outputs[..., 0]
                loss = criterion(outputs_for_loss, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)
        history["epoch"].append(epoch + 1)

        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}/{hyperparams["EPOCHS"]} | '
              f'Train Loss: {train_loss:.3e} | Val Loss: {val_loss:.3e} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model = copy.deepcopy(model)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= hyperparams["PATIENCE"]:
                print(f'Early stopping triggered after {epoch+1} epochs!')
                break

    last_model = copy.deepcopy(model)
    history_df = pd.DataFrame(history)
    # 保存日志
    history_df.to_parquet(os.path.join(args.run_folder, 'history.parquet'), index=False)
    torch.save(best_model.state_dict(), os.path.join(args.run_folder, 'best_cats.pth'))
    torch.save(last_model.state_dict(), os.path.join(args.run_folder, 'last_cats.pth'))

    return history_df, best_model, last_model

# ====== 测试 / 评估 ======
def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)        # [batch_size, pred_len, 4]
            outputs_for_loss = outputs[..., 0]  # 只拿通道 0 做回归
            predictions.extend(outputs_for_loss.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    mae = mean_absolute_error(true_labels, predictions)
    rmse = np.sqrt(mean_squared_error(true_labels, predictions))
    r2 = r2_score(true_labels, predictions)

    print("Evaluation Metrics:")
    print(f"MAE:  {mae:.4e}")
    print(f"RMSE: {rmse:.4e}")
    print(f"R2:   {r2:.4f}")
    return predictions, true_labels, mae, rmse, r2

def main():
    # 训练
    history, best_model, last_model = train_and_validation(model, train_loader, val_loader, hyperparams)
    # 测试评估
    evaluate_model(best_model, test_loader)

if __name__ == "__main__":
    main()
