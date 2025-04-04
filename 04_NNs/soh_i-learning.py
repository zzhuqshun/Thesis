import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import pandas as pd
import numpy as np
from pathlib import Path
import random
import json
import copy
from datetime import datetime
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from soh_lstm import SOHLSTM, BatteryDataset, set_seed


def main(model_type="structure_based(PNN)"):
    """
        - "structure_based": Progressive Neural Network
        - "parameter_based": Elastic Weight Consolidation (EWC)
        - "data_based": Replay Buffer
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(__file__).parent / f"models/incremental_learning_{timestamp}"
    save_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hyperparams = {
        "MODEL_TYPE": model_type,
        "SEQUENCE_LENGTH": 1008,
        "HIDDEN_SIZE": 128,
        "NUM_LAYERS": 3,
        "DROPOUT": 0.4,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 1e-4,
        "EPOCHS": 100,
        "PATIENCE": 10,
        "WEIGHT_DECAY": 1e-4,
        # "REPLAY_BUFFER_SIZE": 5000,   
        # "EWC_LAMBDA": 5000,          
        # "Fisher_n_batches": 32,      
        "device": str(device)
    }
    
    with open(save_dir / "hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=4)
    
    print(f"Using device: {device}")
    print(f"Current model_type: {model_type}\n")

    set_seed(42)

    data_dir = Path("../01_Datenaufbereitung/Output/Calculated/")
    # 加载数据
    df_base, df_update1, df_update2, df_test = load_and_prepare_data(data_dir)
    # 归一化数据
    (df_base_scaled, df_update1_scaled,
     df_update2_scaled, df_test_scaled) = scale_data(
        df_base, df_update1, df_update2, df_test
    )
    
    df_base_train, df_base_val = split_by_cell(df_base_scaled, "Base", val_cells=1)
    df_update1_train, df_update1_val = split_by_cell(df_update1_scaled, "Update1", val_cells=1)
    df_update2_train, df_update2_val = split_by_cell(df_update2_scaled, "Update2", val_cells=1)

    # 构建对应的 Dataset
    base_train_dataset = BatteryDataset(df_base_train, hyperparams["SEQUENCE_LENGTH"])
    base_val_dataset = BatteryDataset(df_base_val, hyperparams["SEQUENCE_LENGTH"])
    update1_train_dataset = BatteryDataset(df_update1_train, hyperparams["SEQUENCE_LENGTH"])
    update1_val_dataset = BatteryDataset(df_update1_val, hyperparams["SEQUENCE_LENGTH"])
    update2_train_dataset = BatteryDataset(df_update2_train, hyperparams["SEQUENCE_LENGTH"])
    update2_val_dataset = BatteryDataset(df_update2_val, hyperparams["SEQUENCE_LENGTH"])
    test_dataset = BatteryDataset(df_test_scaled, hyperparams["SEQUENCE_LENGTH"])


    # 构建 Dataloader
    base_train_loader = DataLoader(base_train_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    base_val_loader = DataLoader(base_val_dataset, batch_size=hyperparams["BATCH_SIZE"])
    update1_train_loader = DataLoader(update1_train_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    update1_val_loader = DataLoader(update1_val_dataset, batch_size=hyperparams["BATCH_SIZE"])
    update2_train_loader = DataLoader(update2_train_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    update2_val_loader = DataLoader(update2_val_dataset, batch_size=hyperparams["BATCH_SIZE"])
    
    n = len(test_dataset)
    seg1 = Subset(test_dataset, list(range(0, n // 3)))
    seg2 = Subset(test_dataset, list(range(n // 3, 2 * n // 3)))
    seg3 = Subset(test_dataset, list(range(2 * n // 3, n)))
    base_test = DataLoader(seg1, batch_size=hyperparams["BATCH_SIZE"])
    update1_test = DataLoader(seg2, batch_size=hyperparams["BATCH_SIZE"])
    update2_test = DataLoader(seg3, batch_size=hyperparams["BATCH_SIZE"])

    # ========== 4. 选择不同的增量学习方法 ==========
    if model_type == "structure_based(PNN)":
        """
        Progressive Neural Network
        """
        print("=" * 80)
        print("Method: Progressive Neural Networks (PNN)")
        print("=" * 80)

        # 1) 初始化 ProgressiveNN
        input_size = 3  # Voltage, Current, Temperature
        pnn_model = ProgressiveNN(
            input_size=input_size,
            hidden_size=hyperparams["HIDDEN_SIZE"],
            num_layers=hyperparams["NUM_LAYERS"],
            dropout=hyperparams["DROPOUT"]
        ).to(device)

        # 2) 训练 Base Column
        print("\nTraining base column...")
        pnn_model, _ = train_model(
            model=pnn_model,
            train_loader=base_train_loader,
            val_loader=base_val_loader,
            epochs=hyperparams["EPOCHS"],
            lr=hyperparams["LEARNING_RATE"],
            weight_decay=hyperparams["WEIGHT_DECAY"],
            patience=hyperparams["PATIENCE"]
        )

        print("\nEvaluating base model...")
        base_pred, _, base_metrics = evaluate_pnn(pnn_model, base_test, task_id=0)
        print(f"Base model metrics: {base_metrics}")

        print("\nAdding column for first update...")
        task1_idx = pnn_model.add_column()
        print("Training column for first update...")
        pnn_model, _ = train_model(
            model=pnn_model,
            train_loader=update1_train_loader,
            val_loader=update1_val_loader,
            epochs=hyperparams["EPOCHS"] // 2,
            lr=hyperparams["LEARNING_RATE"],
            weight_decay=hyperparams["WEIGHT_DECAY"],      
            patience=hyperparams["PATIENCE"]
        )

        print("\nEvaluating after first update...")
        update1_pred, _, update1_metrics = evaluate_pnn(pnn_model, update1_test, task_id=task1_idx)
        print(f"Update 1 metrics: {update1_metrics}")

        # 5) 第二次增量
        print("\nAdding column for second update...")
        task2_idx = pnn_model.add_column()
        print("Training column for second update...")
        pnn_model,_ = train_model(
            model=pnn_model,
            train_loader=update2_train_loader,
            val_loader=update2_val_loader,
            epochs=hyperparams["EPOCHS"] // 2,
            lr=hyperparams["LEARNING_RATE"],
            patience=hyperparams["PATIENCE"]
        )

        print("\nEvaluating after second update...")
        update2_pred, _, update2_metrics = evaluate_pnn(pnn_model, update2_test, task_id=task2_idx)
        print(f"Update 2 metrics: {update2_metrics}")

        # 保存并画图
        plot_results(save_dir, "Structure based(PNN)", df_test_scaled, hyperparams["SEQUENCE_LENGTH"], 
                     base_pred, update1_pred, update2_pred, base_metrics, update1_metrics, update2_metrics)
        print("PNN Incremental Learning Finished!")

    elif model_type != "structure_based(PNN)":
        # initialize the model and train on base data
        input_size = 3  # Voltage, Current, Temperature
        base_model = SOHLSTM(
            input_size=input_size,
            hidden_size=hyperparams["HIDDEN_SIZE"],
            num_layers=hyperparams["NUM_LAYERS"],
            dropout=hyperparams["DROPOUT"]
        ).to(device)
        
        base_model, _ = train_model(
            model=base_model,
            train_loader=base_train_loader,
            val_loader=base_val_loader,
            epochs=hyperparams["EPOCHS"],
            lr=hyperparams["LEARNING_RATE"],
            patience=hyperparams["PATIENCE"]
        )
        
        # if model_type == "parameter_based(EWC)":
        #     """
        #     Elastic Weight Consolidation, EWC
        #     """
        #     print("=" * 80)
        #     print("Method: Elastic Weight Consolidation (EWC)")
        #     print("=" * 80)

        #     ewc = EWC(
        #         model=base_model,
        #         dataset=ConcatDataset([base_dataset, base_val_dataset]),
        #         lambda_param=hyperparams["EWC_LAMBDA"]
        #     )
        #     ewc.register_task()  

        #     print("\nUpdating with first batch of new data using EWC...")
        #     ewc_model, _ = train_model(
        #         model=ewc_model,
        #         train_loader=update1_train_loader,
        #         val_loader=update1_val_loader,
        #         epochs=hyperparams["EPOCHS"] // 2,
        #         lr=hyperparams["LEARNING_RATE"] / 10,
        #         patience=hyperparams["PATIENCE"],
        #         ewc=ewc
        #     )

        #     print("\nEvaluating after first update...")
        #     _, _, update1_metrics, _ = evaluate_model(ewc_model, test_loader)
        #     print(f"Update 1 metrics: {update1_metrics}")

        #     # 注册第二个任务
        #     ewc.register_task()

        #     # 5) 第二次增量
        #     print("\nUpdating with second batch of new data using EWC...")
        #     ewc_model, _ = train_model(
        #         model=ewc_model,
        #         train_loader=update2_loader,
        #         val_loader=base_val_loader,
        #         epochs=hyperparams["EPOCHS"] // 2,
        #         lr=hyperparams["LEARNING_RATE"] / 10,
        #         patience=hyperparams["PATIENCE"],
        #         ewc=ewc
        #     )

        #     print("\nEvaluating after second update...")
        #     _, _, update2_metrics, _ = evaluate_model(ewc_model, test_loader)
        #     print(f"Update 2 metrics: {update2_metrics}")

        #     # 保存并画图
        #     ewc_metrics_df = plot_results(save_dir, "ewc", base_metrics, update1_metrics, update2_metrics)
        #     print("EWC Incremental Learning Finished!")

        # elif model_type == "data_based(Replay Buffer)":
        #     """
        #     数据增量 (Replay Buffer)
        #     """
        #     print("=" * 80)
        #     print("Method: Experience Replay")
        #     print("=" * 80)

        #     # 1) 初始化模型
        #     input_size = 3
        #     replay_model = SOHLSTM(
        #         input_size=input_size,
        #         hidden_size=hyperparams["HIDDEN_SIZE"],
        #         num_layers=hyperparams["NUM_LAYERS"],
        #         dropout=hyperparams["DROPOUT"]
        #     ).to(device)

        #     # 2) 初始化 Replay Buffer
        #     replay_buffer = ReplayBuffer(max_size=hyperparams["REPLAY_BUFFER_SIZE"])

        #     # 3) 训练 Base
        #     print("\nTraining base model...")
        #     replay_model, _ = train_model(
        #         model=replay_model,
        #         train_loader=base_loader,
        #         val_loader=base_val_loader,
        #         epochs=hyperparams["EPOCHS"],
        #         lr=hyperparams["LEARNING_RATE"],
        #         patience=hyperparams["PATIENCE"],
        #         replay_buffer=replay_buffer  # 训练过程中会把旧数据存到 buffer
        #     )

        #     print("\nEvaluating base model...")
        #     _, _, base_metrics, _ = evaluate_model(replay_model, test_loader)
        #     print(f"Base model metrics: {base_metrics}")

        #     # 4) 第一次增量
        #     print("\nUpdating with first batch of new data using Replay Buffer...")
        #     replay_model, _ = train_model(
        #         model=replay_model,
        #         train_loader=update1_loader,
        #         val_loader=base_val_loader,
        #         epochs=hyperparams["EPOCHS"] // 2,
        #         lr=hyperparams["LEARNING_RATE"] / 10,
        #         patience=hyperparams["PATIENCE"],
        #         replay_buffer=replay_buffer,
        #         replay_ratio=0.3  # 每次训练时 30% 的数据来自“旧任务”
        #     )

        #     print("\nEvaluating after first update...")
        #     _, _, update1_metrics, _ = evaluate_model(replay_model, test_loader)
        #     print(f"Update 1 metrics: {update1_metrics}")

        #     # 5) 第二次增量
        #     print("\nUpdating with second batch of new data using Replay Buffer...")
        #     replay_model, _ = train_model(
        #         model=replay_model,
        #         train_loader=update2_loader,
        #         val_loader=base_val_loader,
        #         epochs=hyperparams["EPOCHS"] // 2,
        #         lr=hyperparams["LEARNING_RATE"] / 10,
        #         patience=hyperparams["PATIENCE"],
        #         replay_buffer=replay_buffer,
        #         replay_ratio=0.3
        #     )

        #     print("\nEvaluating after second update...")
        #     _, _, update2_metrics, _ = evaluate_model(replay_model, test_loader)
        #     print(f"Update 2 metrics: {update2_metrics}")

        #     # 保存并画图
        #     replay_metrics_df = plot_results(save_dir, "replay", base_metrics, update1_metrics, update2_metrics)
        #     print("Replay Buffer Incremental Learning Finished!")

    else:
        print(f"Unknown model_type: {model_type}. 请选择 'structure_based', 'parameter_based' 或 'data_based'。")

    print("\nAll Done.")


#####################################################################
#                       下面是所需的类和函数定义
#####################################################################
class ProgressiveNN(nn.Module):
    """
    结构增量: Progressive Neural Network
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(ProgressiveNN, self).__init__()
        
        self.columns = nn.ModuleList()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 第一个列
        self.add_column(lateral_connections=False)
        
    def add_column(self, lateral_connections=True):
        column_idx = len(self.columns)
        new_column = nn.ModuleDict()
        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        
        # LSTM
        if column_idx == 0 or not lateral_connections:
            # 没有 lateral connections
            new_column['lstm'] = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0
            ).to(device)
        else:
            total_lateral_size = self.hidden_size * column_idx
            new_column['lstm'] = nn.LSTM(
                input_size=self.input_size + total_lateral_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0
            ).to(device)
            # lateral adapters
            lateral_adapters = nn.ModuleList()
            for i in range(column_idx):
                adapter = nn.Linear(self.hidden_size, self.hidden_size).to(device)
                lateral_adapters.append(adapter)
            new_column['lateral_adapters'] = lateral_adapters
        
        # FC
        new_column['fc_layers'] = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, 1)
        ).to(device)
        
        self.columns.append(new_column)
        return column_idx
    
    def forward(self, x, task_id=None):
        if task_id is None:
            task_id = len(self.columns) - 1  # 默认用最新的

        batch_size = x.size(0)
        lstm_outputs = []
        
        for col_idx in range(task_id + 1):
            column = self.columns[col_idx]
            
            if col_idx == 0:
                h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
                c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
                lstm_out, _ = column['lstm'](x, (h0, c0))
                lstm_outputs.append(lstm_out)
            else:
                lateral_inputs = []
                for prev_idx in range(col_idx):
                    prev_output = lstm_outputs[prev_idx].detach()
                    adapted_output = column['lateral_adapters'][prev_idx](prev_output)
                    lateral_inputs.append(adapted_output)
                combined_input = torch.cat([x] + lateral_inputs, dim=2)
                
                h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
                c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
                lstm_out, _ = column['lstm'](combined_input, (h0, c0))
                lstm_outputs.append(lstm_out)
        
        final_lstm_out = lstm_outputs[task_id][:, -1, :]
        out = self.columns[task_id]['fc_layers'](final_lstm_out)
        
        return out.squeeze(-1)


class EWC:
    """
    参数增量: Elastic Weight Consolidation (EWC)
    """
    def __init__(self, model, dataset, lambda_param=5000):
        self.model = model
        self.dataset = dataset
        self.lambda_param = lambda_param
        self.fisher_information = {}
        self.optimal_params = {}
        
    def compute_fisher_information(self, n_batches=32):
        self.model.eval()
        loader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        fisher_information = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        count = 0
        for batch_idx, (features, labels) in enumerate(loader):
            if batch_idx >= n_batches:
                break
            count += 1

            features, labels = features.to(next(self.model.parameters()).device), labels.to(next(self.model.parameters()).device)
            self.model.zero_grad()
            outputs = self.model(features)
            loss = F.mse_loss(outputs, labels)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_information[n] += p.grad.detach() ** 2
        
        # 取平均
        for n in fisher_information:
            fisher_information[n] /= max(count, 1)
            
        return fisher_information
    
    def register_task(self):
        # 记录最优参数
        self.optimal_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        # 计算 Fisher 信息
        self.fisher_information = self.compute_fisher_information()
        
    def ewc_loss(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher_information and p.requires_grad:
                loss += (self.fisher_information[n] * (p - self.optimal_params[n]) ** 2).sum()
        return self.lambda_param * loss / 2


class ReplayBuffer:
    """
    数据增量: 经验回放 (Replay Buffer)
    """
    def __init__(self, max_size=5000):
        self.buffer = deque(maxlen=max_size)
        
    def add_batch(self, features, labels):
        for i in range(len(features)):
            self.buffer.append((features[i], labels[i]))
    
    def get_batch(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        features = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        return features, labels
    
    def __len__(self):
        return len(self.buffer)


def load_and_prepare_data(data_dir: Path, resample='10min'):
    """
    加载并返回 base / update1 / update2 / test 四份数据
    """
    parquet_files = sorted(
        [f for f in data_dir.glob('*.parquet') if f.is_file()],
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    if len(parquet_files) < 15:
        raise ValueError(f"Need at least 15 cell files, but found only {len(parquet_files)}")
    
    # 随机分配
    random.shuffle(parquet_files)
    test_file = parquet_files[0]
    base_files = parquet_files[1:5]
    update1_files = parquet_files[5:10]
    update2_files = parquet_files[10:15]
    
    def process_file(file_path: Path):
        df = pd.read_parquet(file_path)
        columns_to_keep = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 
                           'Temperature[°C]', 'SOC_ZHU', 'SOH_ZHU']
        df_processed = df[columns_to_keep].copy()
        df_processed.dropna(inplace=True)
        
        df_processed['Testtime[s]'] = df_processed['Testtime[s]'].round().astype(int)
        start_date = pd.Timestamp("2023-02-02")
        df_processed['Datetime'] = pd.date_range(
            start=start_date,
            periods=len(df_processed),
            freq='s'
        )
        
        df_sampled = df_processed.resample(resample, on='Datetime').mean().reset_index(drop=False)
        df_sampled["cell_id"] = file_path.stem.split('_')[1]
        return df_sampled, file_path.name
    
    test_data = process_file(test_file)
    base_data = [process_file(f) for f in base_files]
    update1_data = [process_file(f) for f in update1_files]
    update2_data = [process_file(f) for f in update2_files]
    
    print(f"Test cell: {test_data[1]}")
    print(f"Base training cells: {[t[1] for t in base_data]}")
    print(f"Update 1 cells: {[u[1] for u in update1_data]}")
    print(f"Update 2 cells: {[u[1] for u in update2_data]}")

    df_test = test_data[0]
    df_base = pd.concat([t[0] for t in base_data], ignore_index=True)
    df_update1 = pd.concat([u[0] for u in update1_data], ignore_index=True)
    df_update2 = pd.concat([u[0] for u in update2_data], ignore_index=True)
    
    print(f"\nBase training data shape: {df_base.shape}")
    print(f"Update 1 data shape: {df_update1.shape}")
    print(f"Update 2 data shape: {df_update2.shape}")
    print(f"Test data shape: {df_test.shape}\n")
    
    return df_base, df_update1, df_update2, df_test

def split_by_cell(df, name, val_cells=1, seed=42):
    """
    根据 cell_id 划分训练集和验证集。
    参数:
        df: 包含 cell_id 列的 DataFrame
        val_cells: 用作验证的 cell 数量（默认为1个）
        seed: 随机种子，保证可重复性
    返回:
        df_train, df_val: 分别为训练集和验证集
    """
    np.random.seed(seed)
    cell_ids = df['cell_id'].unique().tolist()
    np.random.shuffle(cell_ids)
    # 验证集选取前 val_cells 个 cell，其余用于训练
    val_ids = cell_ids[:val_cells]
    train_ids = cell_ids[val_cells:]
    df_train = df[df['cell_id'].isin(train_ids)].reset_index(drop=True)
    df_val = df[df['cell_id'].isin(val_ids)].reset_index(drop=True)
    print(f"{name} - Training cells: {train_ids}")
    print(f"{name} - Validation cells: {val_ids}")
    return df_train, df_val


def scale_data(df_base, df_update1, df_update2, df_test):
    """
    用Base数据的StandardScaler来对所有数据做归一化
    """
    features_to_scale = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
    
    df_base_scaled = df_base.copy()
    df_update1_scaled = df_update1.copy()
    df_update2_scaled = df_update2.copy()
    df_test_scaled = df_test.copy()
    
    scaler = StandardScaler()
    update1_scaler = StandardScaler()
    update2_scaler = StandardScaler()
    scaler.fit(df_base[features_to_scale])
    update1_scaler.fit(df_update1[features_to_scale])
    update2_scaler.fit(df_update2[features_to_scale])
    
    df_base_scaled[features_to_scale] = scaler.transform(df_base[features_to_scale])
    df_test_scaled[features_to_scale] = scaler.transform(df_test[features_to_scale])
    
    df_update1_scaled[features_to_scale] = update1_scaler.transform(df_update1[features_to_scale])
    df_update2_scaled[features_to_scale] = update2_scaler.transform(df_update2[features_to_scale])
    
    print('Features scaled using StandardScaler fitted on base training data.\n')
    
    return df_base_scaled, df_update1_scaled, df_update2_scaled, df_test_scaled


def train_model(model, train_loader, val_loader, epochs, lr=1e-4, weight_decay=1e-4, patience=10,
                ewc=None, replay_buffer=None, replay_ratio=0.3):
    """
    通用的模型训练函数，可选择传 EWC 或 Replay Buffer。
    同时返回训练好的模型和训练历史记录 history，
    history 包含每个 epoch 的 train_loss 和 val_loss。
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    # 初始化训练历史记录字典
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', leave=False) as pbar:
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                
                # 如果有 replay_buffer，将旧任务数据混入当前批次
                if replay_buffer is not None and len(replay_buffer) > 0:
                    replay_batch_size = int(features.shape[0] * replay_ratio)
                    if replay_batch_size > 0:
                        old_feat, old_lbl = replay_buffer.get_batch(replay_batch_size)
                        old_feat, old_lbl = old_feat.to(device), old_lbl.to(device)
                        features = torch.cat([features, old_feat], dim=0)
                        labels = torch.cat([labels, old_lbl], dim=0)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # 如果有 EWC，就加上 EWC 正则项
                if ewc is not None:
                    loss += ewc.ewc_loss()
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                
                train_loss += loss.item()
                pbar.update(1)

                # 每个 batch 加到 replay_buffer 里
                if replay_buffer is not None:
                    features_cpu = features.detach().cpu()
                    labels_cpu = labels.detach().cpu()
                    replay_buffer.add_batch(features_cpu, labels_cpu)

        train_loss /= len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.3e} | "
              f"Val Loss: {val_loss:.3e} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 将本 epoch 的指标记录到 history 中
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history



def evaluate_model(model, data_loader):
    """
    通用的模型评估函数（非PNN）
    """
    model.eval()
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    total_loss = 0.0

    all_predictions, all_targets = [], []
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    total_loss /= len(data_loader)
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    metrics = calc_metrics(predictions, targets)

    return predictions, targets, metrics, total_loss


def evaluate_pnn(model, data_loader, task_id):
    """
    专门评估PNN指定task_id的输出
    """
    model.eval()
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    total_loss = 0.0

    all_predictions, all_targets = [], []
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features, task_id=task_id)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    total_loss /= len(data_loader)
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    metrics = calc_metrics(predictions, targets)

    return predictions, targets, metrics


def calc_metrics(predictions, targets):
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    mape = mean_absolute_percentage_error(targets, predictions) * 100
    r2 = r2_score(targets, predictions)
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    }
    return metrics

def plot_results(save_dir, method_name, df_test, seq_len,
                base_pred, update1_pred, update2_pred, 
                base_metrics, update1_metrics, update2_metrics):
    """
    绘制连续的真实值和预测值曲线，并在每个阶段中标注相应的metrics指标。
    
    参数:
        save_dir: 保存结果的目录（Path对象或字符串）
        method_name: 方法名称，用于生成文件名
        df_test: 测试数据的DataFrame，需包含 'Datetime' 和 'target' 列，
                 x坐标和真实值均从 df_test 中获取（从 SEQUENCE_LENGTH 之后开始）
        base_pred: base阶段的预测值（list或numpy数组）
        update1_pred: update1阶段的预测值
        update2_pred: update2阶段的预测值
        base_metrics, update1_metrics, update2_metrics: 每个阶段对应的指标字典，
            包含 'RMSE', 'MAE', 'MAPE', 'R²' 键，数值为float
    """
    # 从 df_test 中获取 x 坐标和真实目标值
    # 此处 SEQUENCE_LENGTH 建议由全局变量 hyperparams 或其他方式提供
    sequence_length = seq_len
    datetime_vals = df_test['Datetime'].iloc[sequence_length:].values
    true_vals = df_test['SOH_ZHU'].iloc[sequence_length:].values

    # 转换预测值为 numpy 数组
    base_pred = np.array(base_pred)
    update1_pred = np.array(update1_pred)
    update2_pred = np.array(update2_pred)
    
    # 根据预测结果的长度确定各阶段在测试集中的位置
    n_base = len(base_pred)
    n_update1 = len(update1_pred)
    n_update2 = len(update2_pred)
    
    # 分割出各阶段对应的真实值和日期
    base_true = true_vals[:n_base]
    update1_true = true_vals[n_base:n_base+n_update1]
    update2_true = true_vals[n_base+n_update1:n_base+n_update1+n_update2]
    
    x_base = datetime_vals[:n_base]
    x_update1 = datetime_vals[n_base:n_base+n_update1]
    x_update2 = datetime_vals[n_base+n_update1:n_base+n_update1+n_update2]
    
    # 合并三个阶段的预测值，用于整体曲线的绘制
    all_pred = np.concatenate([base_pred, update1_pred, update2_pred])
    
    # 绘制整体真实值和预测值曲线
    plt.figure(figsize=(15, 6))
    plt.plot(datetime_vals, true_vals, label='True Values', marker='o', linestyle='-')
    plt.plot(datetime_vals, all_pred, label='Predicted Values', marker='x', linestyle='--')
    
    # 定义在指定阶段区域添加注释的内部函数
    def annotate_segment(x_segment, seg_true, seg_pred, metrics, phase_name):
        # 取当前阶段中间位置的 x 坐标（datetime 值直接取中间元素）
        mid_x = x_segment[len(x_segment) // 2]
        # 取该阶段真实值和预测值的均值作为纵坐标位置
        y_mean = np.mean(np.concatenate([seg_true, seg_pred]))
        text = (f"{phase_name}\n"
                f"RMSE: {metrics['RMSE']:.2f}\n"
                f"MAE: {metrics['MAE']:.2f}\n"
                f"MAPE: {metrics['MAPE']:.2f}\n"
                f"R²: {metrics['R²']:.2f}")
        plt.text(mid_x, y_mean, text, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7),
                 horizontalalignment='center', verticalalignment='center')
    
    # 添加各阶段的指标注释
    annotate_segment(x_base, base_true, base_pred, base_metrics, "Base Model")
    annotate_segment(x_update1, update1_true, update1_pred, update1_metrics, "Update 1")
    annotate_segment(x_update2, update2_true, update2_pred, update2_metrics, "Update 2")
    
    plt.xlabel("Datetime")
    plt.ylabel("SOH")
    plt.title(f"{method_name} - True vs Predicted SOH Across Phases")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 保存图片
    results_dir = Path(save_dir) / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(results_dir / f"{method_name}_true_pred_plot.png")
    plt.close()

    return print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()  
