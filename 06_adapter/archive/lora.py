from __future__ import annotations
import json
import os
import time
import random
import copy
from pathlib import Path
from datetime import datetime
import logging
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ===============================================================
# Configuration Class
# ===============================================================
class Config:
    """Configuration class for continual learning experiments (LoRA version)"""
    def __init__(self, **kwargs):
        # Training mode: 'joint' for baseline, 'incremental' for continual learning
        self.MODE = 'incremental'

        # Directory structure
        self.BASE_DIR = Path.cwd() / "lora_tuning"
        self.DATA_DIR = Path('../01_Datenaufbereitung/Output/Calculated/')

        # Model hyperparameters
        self.SEQUENCE_LENGTH = 720  # Input sequence length for LSTM
        self.HIDDEN_SIZE = 128      # LSTM hidden state size
        self.NUM_LAYERS = 2         # Number of LSTM layers
        self.DROPOUT = 0.3          # Dropout rate

        # Training hyperparameters
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-4
        self.EPOCHS = 200
        self.PATIENCE = 20          # Early stopping patience
        self.WEIGHT_DECAY = 1e-6

        # Data preprocessing
        self.SCALER = "RobustScaler"
        self.RESAMPLE = '10min'     # Time series resampling frequency
        self.ALPHA = 0.1            # Smoothing factor for predictions

        # Continual Learning parameters
        self.NUM_TASKS = 3          # Number of incremental tasks

        # LoRA parameters
        self.USE_LORA = True        # Whether to switch to LoRA after Task0
        self.LORA_RANK = 8          # Low‑rank dimension r
        self.LORA_ALPHA = 32.0      # Scaling factor α
        self.FREEZE_BACKBONE = True # Whether to freeze frozen layers

        # Random seed for reproducibility
        self.SEED = 42

        # Dataset splits for joint training (baseline)
        self.joint_datasets = {
            'train_ids': ['03', '05', '07', '09', '11', '15', '21', '23', '25', '27', '29'],
            'val_ids': ['01', '19', '13'],
            'test_id': '17'
        }

        # Incremental split (same策略与之前一致)
        self.incremental_datasets = self._create_incremental_splits()

        # Experiment metadata
        self.Info = {
            "method": "LoRA Fine‑tuning" if self.USE_LORA else "Pure Fine‑tuning",
            "lora_rank": self.LORA_RANK if self.USE_LORA else None,
            "lora_alpha": self.LORA_ALPHA if self.USE_LORA else None,
            "freeze_backbone": self.FREEZE_BACKBONE,
            "resample": self.RESAMPLE,
            "scaler": "RobustScaler - fit on base train",
            "smooth_alpha": self.ALPHA,
            "num_tasks": self.NUM_TASKS
        }

        # GPU information
        if torch.cuda.is_available():
            gpu_list = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        else:
            gpu_list = ["CPU"]
        self.Info["gpu_model"] = gpu_list

        # Override
        for k, v in kwargs.items():
            setattr(self, k, v)

    # --- same _create_incremental_splits as before (omitted for brevity) ---
    def _create_incremental_splits(self):
        random.seed(self.SEED)
        normal_cells = ['03', '05', '07', '27']
        fast_cells = ['21', '23', '25']
        faster_cells = ['09', '11', '15', '29']
        task0_normal = random.sample(normal_cells, 3)
        task0_fast = random.sample(fast_cells, 1)
        task0_faster = random.sample(faster_cells, 1)
        task0_train_ids = task0_normal + task0_fast + task0_faster
        remaining = [c for c in normal_cells if c not in task0_normal] + \
                    [c for c in fast_cells if c not in task0_fast] + \
                    [c for c in faster_cells if c not in task0_faster]
        random.shuffle(remaining)
        task1_train_ids = remaining[:3]
        task2_train_ids = remaining[3:6]
        return {
            'task0_train_ids': task0_train_ids,
            'task0_val_ids': ['01'],
            'task1_train_ids': task1_train_ids,
            'task1_val_ids': ['19'],
            'task2_train_ids': task2_train_ids,
            'task2_val_ids': ['13'],
            'test_id': '17'
        }

    # save / load 同旧版 (省略)
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}, f, indent=4)

# ===============================================================
# Visualization / Dataset / DataProcessor 保持不变 (与之前一致)          
# ===============================================================

# ===============================================================
# LoRA building blocks
# ===============================================================
class LoRALinear(nn.Module):
    """Wrap a nn.Linear with LoRA low‑rank adapters"""
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 32.):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_dim, out_dim = linear.in_features, linear.out_features
        # LoRA parameters
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        # Freeze original
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        result = self.linear(x)
        lora = F.linear(x, torch.matmul(self.B, self.A)) * self.scaling
        return result + lora

# ===============================================================
# Backbone LSTM
# ===============================================================
class SOHLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

# ===============================================================
# LSTM + LoRA  (只在 fc[0] 注入 LoRA)
# ===============================================================
class SOHLSTMWithLoRA(nn.Module):
    def __init__(self, base: SOHLSTM, rank=8, alpha=32., freeze_backbone=True):
        super().__init__()
        self.lstm = base.lstm
        if freeze_backbone:
            for p in self.lstm.parameters():
                p.requires_grad = False
        # 替换 fc[0]
        orig_fc0: nn.Linear = base.fc[0]
        self.fc0 = LoRALinear(orig_fc0, rank=rank, alpha=alpha)
        self.fc_rest = nn.Sequential(*base.fc[1:])

    def get_trainable_parameters(self):
        return list(self.fc0.parameters())

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = self.fc0(h)
        return self.fc_rest(h).squeeze(-1)

# ===============================================================
# Trainer / utilities 与之前 Adapter 版基本一致，差异：
#   * detect SOHLSTMWithLoRA 而非 Adapter
# ===============================================================
# (为节省篇幅，只展示与 LoRA 相关的关键改动)

class Trainer:
    def __init__(self, model, device, config, task_dir=None):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.task_dir = Path(task_dir) if task_dir else None
        if self.task_dir:
            self.task_dir.mkdir(parents=True, exist_ok=True)

    def _select_params(self):
        if isinstance(self.model, SOHLSTMWithLoRA):
            return self.model.get_trainable_parameters()
        else:
            return [p for p in self.model.parameters() if p.requires_grad]

    def train_task(self, train_loader, val_loader, task_id):
        opt = torch.optim.Adam(self._select_params(), lr=self.config.LEARNING_RATE,
                               weight_decay=self.config.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)
        best_val, no_imp, best_state = float('inf'), 0, None
        history = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'lr', 'time']}
        for epoch in tqdm.tqdm(range(self.config.EPOCHS), desc=f"Task{task_id}"):
            t0 = time.time()
            self.model.train()
            tot = 0.
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = F.mse_loss(self.model(x), y)
                loss.backward()
                nn.utils.clip_grad_norm_(self._select_params(), 1.0)
                opt.step()
                tot += loss.item() * x.size(0)
            train_loss = tot / len(train_loader.dataset)
            # validation
            self.model.eval(); val_tot = 0.
            with torch.no_grad():
                for x, y in val_loader:
                    val_tot += F.mse_loss(self.model(x.to(self.device)), y.to(self.device)).item() * x.size(0)
            val_loss = val_tot / len(val_loader.dataset)
            # record
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(opt.param_groups[0]['lr'])
            history['time'].append(time.time() - t0)
            logger.info("Epoch %d train=%.3e val=%.3e", epoch, train_loss, val_loss)
            sched.step(val_loss)
            if val_loss < best_val:
                best_val, no_imp = val_loss, 0
                best_state = copy.deepcopy(self.model.state_dict())
                if self.task_dir:
                    torch.save({'model_state': best_state}, self.task_dir / f"task{task_id}_best.pt")
            else:
                no_imp += 1
                if no_imp >= self.config.PATIENCE:
                    logger.info("Early stop at %d", epoch)
                    break
        if best_state:
            self.model.load_state_dict(best_state)
        return history

    # evaluate 与旧版一致 (略)

# ===============================================================
# incremental_training 修改点：Task1 时切到 LoRA
# ===============================================================

def incremental_training(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dp = DataProcessor(config.DATA_DIR, config.RESAMPLE, config)
    data = dp.prepare_incremental_data(config.incremental_datasets)
    loaders = create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    model: nn.Module = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    inc_dir = config.BASE_DIR / "incremental_training"; inc_dir.mkdir(parents=True, exist_ok=True)
    for task_idx in range(config.NUM_TASKS):
        if config.USE_LORA and task_idx == 1:  # 训练完 Task0 再注入 LoRA
            model = SOHLSTMWithLoRA(model,
                                    rank=config.LORA_RANK,
                                    alpha=config.LORA_ALPHA,
                                    freeze_backbone=config.FREEZE_BACKBONE).to(device)
            logger.info("Switched to LoRA model (rank=%d, alpha=%.1f)", config.LORA_RANK, config.LORA_ALPHA)
        task_dir = inc_dir / f"task{task_idx}"
        trainer = Trainer(model, device, config, task_dir)
        trainer.train_task(loaders[f"task{task_idx}_train"], loaders[f"task{task_idx}_val"], task_idx)
        # 不涉及 EWC/SI 等正则, 本例纯微调/LoRA
    # 评估函数复用 (略)

# ===============================================================
# Main (与旧版类似)
# ===============================================================
if __name__ == '__main__':
    cfg = Config()
    Path(cfg.BASE_DIR).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(cfg.BASE_DIR / 'train.log', 'w', 'utf-8')])
    random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    incremental_training(cfg)
