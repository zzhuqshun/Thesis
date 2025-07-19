from __future__ import annotations
import json
import os
import time
import random
import copy
from pathlib import Path
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tqdm

logger = logging.getLogger(__name__)

# ===============================================================
# Configuration
# ===============================================================
class Config:
    def __init__(self, **kwargs):
        # Training Mode
        self.MODE = 'joint'  # 'joint' or 'incremental'
        self.BASE_DIR = Path('model/joint_training')  # Base directory for outputs
        # Model parameters
        self.SEQUENCE_LENGTH = 720
        self.HIDDEN_SIZE = 128
        self.NUM_LAYERS = 2
        self.DROPOUT = 0.3
        
        # Training parameters
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-4
        self.EPOCHS = 200
        self.PATIENCE = 20
        self.WEIGHT_DECAY = 1e-6
        
        # Data processing
        self.SCALER = "RobustScaler"

        self.FEATURES_COLS = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        self.SEED = 42
        self.RESAMPLE = '10min'
        
        # Incremental learning parameters
        # self.LWF_ALPHA = [0.0, 1.92, 0.57]
        # self.EWC_LAMBDA = [7780.1555769014285, 141.35935551752303, 1000.0]
        
        self.LWF_ALPHA = [0.0, 0.0, 0.0]
        self.EWC_LAMBDA = [0.0, 0.0, 0.0]
        
        # self.LWF_ALPHA = [0.0, 0.0, 0.0]
        # self.EWC_LAMBDA = [106.55897372508032, 388.9925746974312, 1000.0]
        
        # Dataset splits
        # Joint
        self.dataset_joint = {
            "train": ['03', '05', '07', '09', '11', '15', '21', '23', '25', '27', '29'],
            "val": ['01', '19', '13'],
            "test": ['17']
        }
        
        self.dataset_incl = {
            "base": {
                "train": ['03', '05', '07', '27'],
                "val": ['01']
            },
            "update1": {
                "train": ['21', '23', '25'],
                "val": ['19']
            },
            "update2": {
                "train": ['09', '11', '15', '29'],
                "val": ['13']
            },
            "test": ['17']
        }
        
        # Update with any provided kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def save(self, path):
    # 把 Path 类型转成字符串
        serializable_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                serializable_dict[k] = str(v)
            else:
                serializable_dict[k] = v
        with open(path, 'w') as f:
            json.dump(serializable_dict, f, indent=4)

# ===============================================================
# Dataset
# ===============================================================
class BatteryDataset(Dataset):
    def __init__(self, df, seq_len):
        feats = df[['Voltage[V]','Current[A]','Temperature[°C]']].values
        self.X = torch.tensor(feats, dtype=torch.float32)
        self.y = torch.tensor(df['SOH_ZHU'].values, dtype=torch.float32)
        self.seq_len = seq_len
    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, idx):
        return (
            self.X[idx:idx+self.seq_len],
            self.y[idx+self.seq_len]
        )
    
def create_dataloaders(datasets, config):
    seq_len = config.SEQUENCE_LENGTH
    batch_size = config.BATCH_SIZE
    loaders = {}
    for key, df in datasets.items():
        if not df.empty and ('train' in key or 'val' in key or 'test' in key):
            ds = BatteryDataset(df, seq_len)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=('train' in key))
            loaders[key] = loader
    return loaders

# ===============================================================
# Data Processing
# ===============================================================
class DataProcessor:
    def __init__(self, data_dir, config):
        self.data_dir = Path(data_dir)
        self.config = config
        self.scaler = RobustScaler()
        self.feature_cols = config.FEATURES_COLS
    
    def process_file(self, filepath):
        df = pd.read_parquet(filepath)[
            ['Testtime[s]', 'Voltage[V]', 'Current[A]', 'Temperature[°C]', 'SOH_ZHU']
        ]
        df = df.dropna().reset_index(drop=True)
        df['Testtime[s]'] = df['Testtime[s]'].round().astype(int)
        df['Datetime'] = pd.date_range('2023-02-02', periods=len(df), freq='s')
        df = df.set_index('Datetime').resample(self.config.RESAMPLE).mean().reset_index()
        df['cell_id'] = filepath.stem.split('_')[-1]
        return df
    
    def prepare_data(self):
        files = sorted(self.data_dir.glob('*.parquet'))
        file_map = {f.stem.split('_')[-1]: f for f in files}
        datasets = {}

        if self.config.MODE == 'joint':
            
            logger.info("[DATA] Mode: JOINT")
            logger.info("[DATA] Resample rate: %s", self.config.RESAMPLE)
            all_train_ids = self.config.dataset_joint['train']
            all_val_ids = self.config.dataset_joint['val']
            
            logger.info("  (Split) Train IDs: %s", all_train_ids)
            logger.info("  (Split) Val IDs: %s", all_val_ids)
            dfs_train = [self.process_file(file_map[id]) for id in all_train_ids]
            datasets['joint_train'] = pd.concat(dfs_train, ignore_index=True)
            logger.info("  (Split) joint_train: %d samples", len(datasets['joint_train']))
            dfs_val = [self.process_file(file_map[id]) for id in all_val_ids]
            datasets['joint_val'] = pd.concat(dfs_val, ignore_index=True)
            logger.info("  (Split) joint_val: %d samples", len(datasets['joint_val']))
            
            scaler_data = datasets['joint_train'][self.feature_cols]
            test_id = self.config.dataset_incl['test'][0]
            df_test = self.process_file(file_map[test_id])
            datasets['test_full'] = df_test
            logger.info("  (Split) joint_test_full: %d samples", len(datasets['test_full']))
            
        elif self.config.MODE == 'incremental':
            logger.info("[DATA] Mode: INCREMENTAL")
            logger.info("[DATA] Resample rate: %s", self.config.RESAMPLE)
            summary = []
            for phase in ['base', 'update1', 'update2']:
                for split in ['train', 'val']:
                    ids = self.config.dataset_incl[phase].get(split, [])
                    key = f"{phase}_{split}"
                    if ids:
                        dfs = [self.process_file(file_map[id]) for id in ids]
                        datasets[key] = pd.concat(dfs, ignore_index=True)
                        size = len(datasets[key])
                    else:
                        size = 0
                    summary.append(f"{key}: {size} samples, ids={ids}")
            logger.info("  (Split) Details:\n    %s", "\n    ".join(summary))
            scaler_data = datasets['base_train'][self.feature_cols]
            test_id = self.config.dataset_incl['test'][0]
            df_test = self.process_file(file_map[test_id])
            datasets['test_full'] = df_test
            datasets['test_base'] = df_test[df_test['SOH_ZHU'] >= 0.9].reset_index(drop=True)
            datasets['test_update1'] = df_test[(df_test['SOH_ZHU'] < 0.9) & (df_test['SOH_ZHU'] >= 0.8)].reset_index(drop=True)
            datasets['test_update2'] = df_test[df_test['SOH_ZHU'] < 0.8].reset_index(drop=True)
            logger.info("  (Split) Test splits: test_full(%d), test_base(%d), test_update1(%d), test_update2(%d)",
                        len(df_test), len(datasets['test_base']), len(datasets['test_update1']), len(datasets['test_update2']))
        else:
            raise ValueError(f"Unknown MODE: {self.config.MODE}. Should be 'joint' or 'incremental'.")

        logger.info("[DATA] Fitting scaler: %s", self.config.SCALER)
        self.scaler.fit(scaler_data)
        logger.info("  (Scaler) Scaler centers: %s", self.scaler.center_)
        logger.info("  (Scaler) Scaler scales: %s", self.scaler.scale_)
        logger.info("  (Scaler) Features scaled: %s", self.feature_cols)

        for key, df in datasets.items():
            if not df.empty:
                df_scaled = df.copy()
                df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
                datasets[key] = df_scaled
        logger.info("[DATA] Preparation complete. Keys: %s", list(datasets.keys()))
        return datasets

# ===============================================================
# Model
# ===============================================================
class SOHLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.fc(h).squeeze(-1)

# ===============================================================
# EWC Implementation
# ===============================================================
class EWC:
    def __init__(self, model, dataloader, device, lam):
        self.device = device
        self.lam = lam
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(model, dataloader)
    
    def _compute_fisher(self, model, dataloader):
        model_copy = copy.deepcopy(model).to(self.device)
        model_copy.train()
        for m in model_copy.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
            if isinstance(m, nn.LSTM):
                m.dropout = 0.0
        fisher = {n: torch.zeros_like(p, device=self.device)
                 for n, p in model_copy.named_parameters() if p.requires_grad}
        n_samples = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            model_copy.zero_grad()
            out = model_copy(x)
            loss = F.mse_loss(out, y)
            loss.backward()
            batch_size = x.size(0)
            n_samples += batch_size
            with torch.no_grad():
                for n, p in model_copy.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.pow(2) * batch_size
        for n in fisher:
            fisher[n] /= float(n_samples)
        del model_copy
        return fisher
    
    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += self.lam * (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss

# ===============================================================
# Trainer
# ===============================================================
class Trainer:
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.ewc_tasks = []
        self.old_model = None
    
    def train_task(self, train_loader, val_loader, task_id, apply_ewc=True, alpha_lwf=0.0):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.5, patience=5
        )
        best_val_loss = float('inf')
        no_improvement = 0
        history = []
        for epoch in tqdm.tqdm(range(self.config.EPOCHS), desc=f"Task {task_id}"):
            self.model.train()
            train_metrics = self._train_epoch(train_loader, optimizer, apply_ewc, alpha_lwf)
            self.model.eval()
            val_loss = self._validate(val_loader)
            scheduler.step(val_loss)
            history.append({
                'epoch': epoch,
                'train_loss': train_metrics['total'],
                'val_loss': val_loss,
                'task_loss': train_metrics['task'],
                'kd_loss': train_metrics['kd'],
                'ewc_loss': train_metrics['ewc'],
                'lr': optimizer.param_groups[0]['lr']
            })
            logger.info("[TRAIN] Epoch %d: Train Loss: %.4e, Val Loss: %.4e, Task: %.4e, KD: %.4e, EWC: %.4e",
                        epoch, train_metrics['total'], val_loss, train_metrics['task'], train_metrics['kd'], train_metrics['ewc'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                no_improvement += 1
                if no_improvement >= self.config.PATIENCE:
                    logger.info("[TRAIN] Early stopping at epoch %d", epoch)
                    break
        self.model.load_state_dict(self.best_model_state)
        return history
    
    def _train_epoch(self, loader, optimizer, apply_ewc, alpha_lwf):
        total_loss = 0
        task_loss = 0
        kd_loss = 0
        ewc_loss = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            
            y_pred = self.model(x)
            loss_task = F.mse_loss(y_pred, y)
            
            loss_kd = torch.tensor(0., device=self.device)
            if alpha_lwf > 0 and self.old_model is not None:
                with torch.no_grad():
                    y_old = self.old_model(x)
                loss_kd = F.mse_loss(y_pred, y_old)
                
            loss_ewc = torch.tensor(0., device=self.device)
            if apply_ewc and self.ewc_tasks:
                loss_ewc = sum(task.penalty(self.model) for task in self.ewc_tasks)
                
            loss = loss_task + alpha_lwf * loss_kd + loss_ewc
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            optimizer.step()
            
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            task_loss += loss_task.item() * batch_size
            kd_loss += loss_kd.item() * batch_size
            ewc_loss += loss_ewc.item() * batch_size
        dataset_size = len(loader.dataset)
        return {
            'total': total_loss / dataset_size,
            'task': task_loss / dataset_size,
            'kd': kd_loss / dataset_size,
            'ewc': ewc_loss / dataset_size
        }
    
    def _validate(self, loader):
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                loss = F.mse_loss(y_pred, y)
                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
        total_loss /= len(loader.dataset)
        return total_loss 
    
    def consolidate(self, loader, lam):
        ewc = EWC(self.model, loader, self.device, lam)
        self.ewc_tasks.append(ewc)
        self.old_model = copy.deepcopy(self.model).to(self.device)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)
    
    def evaluate(self, loader):
        self.model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y_pred = self.model(x)
                preds.extend(y_pred.cpu().numpy())
                targets.extend(y.cpu().numpy())
        preds = np.array(preds)
        targets = np.array(targets)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(targets, preds)),
            'MAE': mean_absolute_error(targets, preds),
            'R2': r2_score(targets, preds),
            'NEG_MAE': -mean_absolute_error(targets, preds)  # Add negative MAE
        }
        return metrics, preds, targets
    
    def evaluate_smooth(self, loader, alpha=0.1):
        _, preds, targets = self.evaluate(loader)
        preds_smooth = pd.Series(preds).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        smooth_metrics = {
            'RMSE': np.sqrt(mean_squared_error(targets, preds_smooth)),
            'MAE': mean_absolute_error(targets, preds_smooth),
            'R2': r2_score(targets, preds_smooth),
            'NEG_MAE': -mean_absolute_error(targets, preds_smooth)  # Add negative MAE
        }
        return smooth_metrics, preds_smooth, targets
# ===============================================================
# Visualization
# ===============================================================
def plot_losses(history, save_path):
    df = pd.DataFrame(history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.semilogy(df['epoch'], df['train_loss'], label='Train')
    ax1.semilogy(df['epoch'], df['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Training Progress')
    ax2.plot(df['epoch'], df['task_loss'], label='Task Loss')
    ax2.plot(df['epoch'], df['kd_loss'], label='KD Loss')
    ax2.plot(df['epoch'], df['ewc_loss'], label='EWC Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Loss Components')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_predictions(preds, targets, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # 计算指标
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    metric_text = f"MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}"

    # 时序图
    plt.figure(figsize=(10, 6))
    plt.plot(targets, label='Actual', alpha=0.7)
    plt.plot(preds, label='Predicted', alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel('SOH')
    plt.legend()
    plt.grid(True)
    plt.title('SOH Predictions\n' + metric_text)  # 标题下加指标
    plt.tight_layout()
    plt.savefig(save_dir / 'timeseries.png')
    plt.close()

    # 散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel('Actual SOH')
    plt.ylabel('Predicted SOH')
    plt.grid(True)
    plt.title('Prediction Scatter Plot\n' + metric_text)
    plt.tight_layout()
    plt.savefig(save_dir / 'scatter.png')
    plt.close()

# ===============================================================
# Main Pipeline
# ===============================================================
def setup_logging(base_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console)
    log_file = base_dir / 'experiment.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

def run_joint_training(config, base_dir, device):
    logger.info("=" * 60)
    logger.info("[JOINT] Starting Joint Training")
    logger.info("=" * 60)
    processor = DataProcessor('../../01_Datenaufbereitung/Output/Calculated/', config)
    datasets = processor.prepare_data()
    loaders = create_dataloaders(datasets, config)

    model = SOHLSTM(len(config.FEATURES_COLS), config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
    logger.info("  (MODEL) SOHLSTM: in=%d, hidden=%d, layers=%d, drop=%.2f",
                len(config.FEATURES_COLS), config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
    trainer = Trainer(model, device, config)
    logger.info("  (TRAIN) Start training on joint dataset ...")
    history = trainer.train_task(
        loaders['joint_train'],
        loaders['joint_val'],
        task_id=0,
        apply_ewc=False,
        alpha_lwf=0.0
    )
    joint_dir = base_dir / 'joint_training'
    joint_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(joint_dir / 'history.csv', index=False)
    
    plot_losses(history, joint_dir / 'training_curves.png')
    logger.info("  (TRAIN) Finished. Loss curves saved: %s", joint_dir / 'training_curves.png')
    metrics, preds, targets = trainer.evaluate(loaders['test_full'])
    logger.info("  (EVAL) [Test] RMSE: %.4e, MAE: %.4e, R2: %.4f",
                metrics['RMSE'], metrics['MAE'], metrics['R2'])
    plot_predictions(preds, targets, joint_dir / 'predictions')
    logger.info("  (EVAL) Prediction plots saved: %s", joint_dir / 'predictions')
    torch.save(trainer.model.state_dict(), joint_dir / 'model.pt')
    logger.info("  (MODEL) Checkpoint saved: %s", joint_dir / 'model.pt')
    logger.info("=" * 60)
    logger.info("[JOINT] Joint training finished.")
    logger.info("=" * 60)

def run_incremental_learning(config: Config, base_dir: Path, device: torch.device):
    """Incremental (continual) training pipeline with GEM-style metric evaluation using negative MAE.

    Flow
    ----
    1. Prepare data loaders (base/update1/update2 + all test splits)
    2. For each task:
       2.1 Train one task (optionally EWC & LWF)
       2.2 Consolidate Fisher information (EWC)
       2.3 Evaluate on current task and test_full only
    3. After all tasks: compute ACC, BWT, FWT according to GEM definitions using negative MAE
    4. Save final metrics and predictions
    """

    logger.info("=" * 60)
    logger.info("[INCREMENTAL] Starting Incremental Learning with -MAE evaluation")
    logger.info("=" * 60)

    # ─────────────────── Data ───────────────────
    processor = DataProcessor('../../01_Datenaufbereitung/Output/Calculated/', config)
    datasets  = processor.prepare_data()
    loaders   = create_dataloaders(datasets, config)
    
    # Make sure the root incremental folder exists
    (base_dir / 'incremental').mkdir(parents=True, exist_ok=True)

    # ─────────────────── Model / Trainer ───────────────────
    model   = SOHLSTM(len(config.FEATURES_COLS),
                      config.HIDDEN_SIZE,
                      config.NUM_LAYERS,
                      config.DROPOUT)
    logger.info("  (MODEL) SOHLSTM: in=%d, hidden=%d, layers=%d, drop=%.2f",
                len(config.FEATURES_COLS), config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
    trainer = Trainer(model, device, config)

    # Task setup
    tasks = [
        ('task0', 'base',    0, False, config.EWC_LAMBDA[0], config.LWF_ALPHA[0]),
        ('task1', 'update1', 1, True,  config.EWC_LAMBDA[1], config.LWF_ALPHA[1]),
        ('task2', 'update2', 2, True,  config.EWC_LAMBDA[2], config.LWF_ALPHA[2]),
    ]

    # ───────── Baseline -MAE for FWT (random-init model) ─────────
    logger.info("  (FWT) Computing random-init baselines using -MAE …")
    fwt_baseline_neg_mae = {}
    with torch.no_grad():  # 不需要梯度计算
        baseline_model = SOHLSTM(len(config.FEATURES_COLS),
                                config.HIDDEN_SIZE,
                                config.NUM_LAYERS,
                                config.DROPOUT).to(device)
        baseline_trainer = Trainer(baseline_model, device, config)
        
        for ph in ['base', 'update1', 'update2']:
            tk = f'test_{ph}'
            if tk in loaders:
                m, _, _ = baseline_trainer.evaluate(loaders[tk])
                fwt_baseline_neg_mae[tk] = m['NEG_MAE']  # Use negative MAE
                logger.info("    (BASELINE) %s  -MAE=%.4f", tk, m['NEG_MAE'])
        
        # 显式清理
        del baseline_model
        del baseline_trainer
    
    # 强制释放GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    logger.info("  (FWT) Baseline computation complete, memory cleaned")

    # Containers for final metric computation
    R_matrix = {}  # R_{i,j}: performance of model after task i on task j (using -MAE)
    task_order = []  # 记录任务顺序
    process_metrics = []  # 记录每个任务的过程指标

    # ───────────────────── Main Training Loop ────────────────────────
    for task_idx, (tag, phase, tid, use_ewc, lam, alpha) in enumerate(tasks):
        logger.info("\n%s\n[TASK] %s  phase=%s  id=%d\n%s", "="*60, tag, phase, tid, "-"*60)

        task_order.append(tag)
        test_key    = f'test_{phase}'
        tr_loader   = loaders.get(f'{phase}_train')
        val_loader  = loaders.get(f'{phase}_val') or tr_loader
        te_loader   = loaders[test_key]
        full_loader = loaders['test_full']

        # ── Training ───────────────────────────────────────────────
        logger.info("    (TRAIN) Starting training for %s ...", tag)
        hist = trainer.train_task(tr_loader, val_loader, tid,
                                  apply_ewc=use_ewc, alpha_lwf=alpha)
        
        # Save training history
        out_dir = base_dir / 'incremental' / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(hist).to_csv(out_dir / 'history.csv', index=False)
        plot_losses(hist, out_dir / 'training_curves.png')

        # ── Consolidate Fisher (EWC) ───────────────────────────────
        logger.info("    (EWC) Consolidate with λ = %.4f", lam)
        trainer.consolidate(tr_loader, lam)

        # ── Evaluate only on current task and test_full ───────────
        # Current task evaluation
        met_current, preds_current, tgts_current = trainer.evaluate(te_loader)
        logger.info("    (EVAL) Current Task (%s): -MAE=%.4f, MAE=%.4f, R²=%.4f", 
                    tag, met_current['NEG_MAE'], met_current['MAE'], met_current['R2'])
        
        # Test full evaluation
        met_full, preds_full, tgts_full = trainer.evaluate(full_loader)
        logger.info("    (EVAL) Test Full: -MAE=%.4f, MAE=%.4f, R²=%.4f", 
                    met_full['NEG_MAE'], met_full['MAE'], met_full['R2'])
        
        # Save predictions for current task
        plot_predictions(preds_current, tgts_current, out_dir / 'predictions_current')
        plot_predictions(preds_full, tgts_full, out_dir / 'predictions_full')
        
        # Save model checkpoint
        torch.save(trainer.model.state_dict(), out_dir / 'model.pt')
        
        # Save detailed predictions as CSV
        pred_df_current = pd.DataFrame({
            'actual': tgts_current,
            'predicted': preds_current
        })
        pred_df_current.to_csv(out_dir / 'predictions_current.csv', index=False)
        
        pred_df_full = pd.DataFrame({
            'actual': tgts_full,
            'predicted': preds_full
        })
        pred_df_full.to_csv(out_dir / 'predictions_full.csv', index=False)
        
        # Save current task metrics
        task_metrics = {
            "task": tag,
            "current_task_NEG_MAE": float(met_current["NEG_MAE"]),
            "current_task_MAE":     float(met_current["MAE"]),
            "current_task_R2":      float(met_current["R2"]),
            "current_task_RMSE":    float(met_current["RMSE"]),
            "test_full_NEG_MAE":    float(met_full["NEG_MAE"]),
            "test_full_MAE":        float(met_full["MAE"]),
            "test_full_R2":         float(met_full["R2"]),
            "test_full_RMSE":       float(met_full["RMSE"]),
            "EWC_lambda":           float(lam),  
            "LWF_alpha":            float(alpha),  
        }
        process_metrics.append(task_metrics)
        
        # Save individual task metrics
        with open(out_dir / 'task_metrics.json', 'w') as f:
            json.dump(task_metrics, f, indent=4)

        # ── Store performance matrix for final metric computation ──
        # Evaluate on all tasks seen so far to build R_matrix (using -MAE)
        with torch.no_grad():      
            for j, (prev_tag, prev_phase, _, _, _, _) in enumerate(tasks[:task_idx + 1]):
                prev_test_key = f'test_{prev_phase}'
                if prev_test_key in loaders:
                    met_j, _, _ = trainer.evaluate(loaders[prev_test_key])
                    R_matrix[f"{task_idx},{j}"] = met_j['NEG_MAE']  # Use -MAE instead of R²
            
            for fut_idx in range(task_idx + 1, len(tasks)):
                fut_phase = tasks[fut_idx][1]
                fut_test_key = f'test_{fut_phase}'
                if fut_test_key in loaders:
                    met_fut, _, _ = trainer.evaluate(loaders[fut_test_key])
                    R_matrix[f"{task_idx},{fut_idx}"] = met_fut['NEG_MAE']  # Use -MAE
                    logger.info("    (FWT-PREP) R_%d,%d = %.4f",
                                task_idx, fut_idx, met_fut['NEG_MAE'])

    # ───────────────────── Final Metric Computation (GEM Style with -MAE) ────────────────────────
    logger.info("\n" + "="*60)
    logger.info("[FINAL METRICS] Computing ACC, BWT, FWT according to GEM definitions using -MAE")
    logger.info("="*60)

    num_tasks = len(tasks)
    
    # ACC (Average Accuracy): (1/T) * Σ R_{T-1,i} for i=0 to T-1
    acc_values = []
    for j in range(num_tasks):
        key = f"{num_tasks-1},{j}"  # Performance after final task on task j
        if key in R_matrix:
            acc_values.append(R_matrix[key])
    ACC = np.mean(acc_values) if acc_values else 0.0
    
    # BWT (Backward Transfer): (1/(T-1)) * Σ (R_{T-1,i} - R_{i,i}) for i=0 to T-2
    bwt_values = []
    for i in range(num_tasks - 1):  # i=0 to T-2
        final_perf_key = f"{num_tasks-1},{i}"  # R_{T-1,i}
        initial_perf_key = f"{i},{i}"  # R_{i,i}
        if final_perf_key in R_matrix and initial_perf_key in R_matrix:
            bwt_delta = R_matrix[final_perf_key] - R_matrix[initial_perf_key]
            bwt_values.append(bwt_delta)
    BWT = np.mean(bwt_values) if bwt_values else 0.0
    
    # FWT (Forward Transfer): (1/(T-1)) * Σ (R_{i-1,i} - b_i) for i=1 to T-1
    fwt_values = []
    for i in range(1, num_tasks):  # i=1 to T-1
        pretrain_perf_key = f"{i-1},{i}"  # R_{i-1,i}
        baseline_key = f"test_{tasks[i][1]}"  # baseline for task i
        if pretrain_perf_key in R_matrix and baseline_key in fwt_baseline_neg_mae:
            fwt_delta = R_matrix[pretrain_perf_key] - fwt_baseline_neg_mae[baseline_key]
            fwt_values.append(fwt_delta)
    FWT = np.mean(fwt_values) if fwt_values else 0.0
    
    # Final evaluation on test_full
    final_met_full, final_preds_full, final_tgts_full = trainer.evaluate(loaders['test_full'])
    
    # ───────── Log Final Results ─────────
    logger.info("\n" + "-"*60)
    logger.info("[FINAL RESULTS] (Using -MAE as primary metric)")
    logger.info("  ACC (Average Accuracy):     %.4f", ACC)
    logger.info("  BWT (Backward Transfer):    %.4f", BWT)
    logger.info("  FWT (Forward Transfer):     %.4f", FWT)
    logger.info("  Test Full -MAE:             %.4f", final_met_full['NEG_MAE'])
    logger.info("  Test Full MAE:              %.4f", final_met_full['MAE'])
    logger.info("  Test Full R²:               %.4f", final_met_full['R2'])
    logger.info("  Test Full RMSE:             %.4f", final_met_full['RMSE'])
    logger.info("-"*60)

    # ───────── Save Final Results ─────────
    final_metrics = {
        'ACC': float(ACC),
        'BWT': float(BWT),
        'FWT': float(FWT),
        'test_full_NEG_MAE': float(final_met_full['NEG_MAE']),
        'test_full_MAE': float(final_met_full['MAE']),
        'test_full_R2': float(final_met_full['R2']),
        'test_full_RMSE': float(final_met_full['RMSE']),
        'R_matrix': {k: float(v) for k, v in R_matrix.items()},
        'baseline_NEG_MAE': {k: float(v) for k, v in fwt_baseline_neg_mae.items()},
        'metric_type': 'NEG_MAE',  # Document which metric was used
        'note': 'All continual learning metrics (ACC, BWT, FWT) computed using negative MAE'
    }
    
    # Save final metrics
    final_dir = base_dir / 'incremental'
    with open(final_dir / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    # Save process metrics (all tasks)
    process_df = pd.DataFrame(process_metrics)
    process_df.to_csv(final_dir / 'process_metrics.csv', index=False)
    
    # Save final predictions
    plot_predictions(final_preds_full, final_tgts_full, final_dir / 'final_predictions')
    
    # Save final predictions as CSV
    final_pred_df = pd.DataFrame({
        'actual': final_tgts_full,
        'predicted': final_preds_full
    })
    final_pred_df.to_csv(final_dir / 'final_predictions.csv', index=False)
    
    logger.info("[INCREMENTAL] Final metrics saved → %s", final_dir / 'final_metrics.json')
    logger.info("=" * 60)


def main():
    config = Config()
    set_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = config.BASE_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(base_dir)
    config.save(base_dir / 'config.json')
    logger.info("[EXPERIMENT] Directory: %s", base_dir)
    logger.info("[EXPERIMENT] Device: %s", device)
    if config.MODE == 'joint':
        run_joint_training(config, base_dir, device)
    if config.MODE == 'incremental':
        run_incremental_learning(config, base_dir, device)
    logger.info("[EXPERIMENT] Completed!")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    main()