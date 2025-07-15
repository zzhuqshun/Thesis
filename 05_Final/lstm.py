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
        self.MODE = 'incremental'  # 'joint' or 'incremental'
        self.BASE_DIR = Path('model/tryouts/3inputs/incremental')
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
        self.GROUP_BY_CELL = False # Group data by cell_id
        self.FEATURES_COLS = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        self.SEED = 42
        self.RESAMPLE = '10min'
        
        # Incremental learning parameters
        self.LWF_ALPHA = [0.0, 1.9161084252463925, 0.5711627077804184]
        self.EWC_LAMBDA = [7780.1555769014285, 141.35935551752303, 1000.0]
        
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
    def __init__(self, df, config):
        self.seq_len = config.SEQUENCE_LENGTH
        self.samples = []
        self.config = config
        self.group_by_cell = config.GROUP_BY_CELL
        
        if self.group_by_cell:
            for cell_id, group in df.groupby('cell_id'):
                group = group.reset_index(drop=True)
                feats = group[config.FEATURES_COLS].values
                targets = group['SOH_ZHU'].values
                
                n_samples = max(0, len(feats) - self.seq_len)
                for i in range(n_samples):
                    x = torch.tensor(feats[i:i+self.seq_len], dtype=torch.float32)
                    y = torch.tensor(targets[i+self.seq_len], dtype=torch.float32)
                    self.samples.append((x, y))
        else:
            feats = df[config.FEATURES_COLS].values
            targets = df['SOH_ZHU'].values
            n_samples = max(0, len(feats) - self.seq_len)
            for i in range(n_samples):
                x = torch.tensor(feats[i:i+self.seq_len], dtype=torch.float32)
                y = torch.tensor(targets[i+self.seq_len], dtype=torch.float32)
                self.samples.append((x, y))
        
        logger.info("[DATA] Created dataset with %d samples", len(self.samples))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
def create_dataloaders(datasets, config):
    loaders = {}
    logger.info("[DATA] Grouping datasets by cell_id: %s", config.GROUP_BY_CELL)
    for key, df in datasets.items():
        if not df.empty and any(x in key for x in ['train', 'val', 'test']):
            dataset = BatteryDataset(df, config)
            shuffle = 'train' in key
            loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle)
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
            ['Testtime[s]', 'Voltage[V]', 'Current[A]', 'Temperature[°C]', 'EFC', 'SOH_ZHU']
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
            all_train_ids = ['03', '05', '07', '09', '11', '15', '21', '23', '25', '27', '29']
            all_val_ids = ['01', '19', '13']
            
            logger.info("  (Split) Train IDs: %s", all_train_ids)
            logger.info("  (Split) Val IDs: %s", all_val_ids)
            dfs = [self.process_file(file_map[id]) for id in all_train_ids]
            datasets['joint_train'] = pd.concat(dfs, ignore_index=True)
            logger.info("  (Split) joint_train: %d samples", len(datasets['joint_train']))
            dfs = [self.process_file(file_map[id]) for id in all_val_ids]
            datasets['joint_val'] = pd.concat(dfs, ignore_index=True)
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
            raise ValueError(f"Unknown MODE: {self.MODE}. Should be 'joint' or 'incremental'.")

        logger.info("[DATA] Fitting scaler: %s", self.config.SCALER)
        self.scaler.fit(scaler_data)
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
            logger.info("[TRAIN] Epoch %d: Train Loss: %.4e, Val Loss: %.4e, Task: %.4e, KD: %.4e, EWC: %.4e, LR: %.6e",
                        epoch, train_metrics['total'], val_loss, train_metrics['task'], train_metrics['kd'], train_metrics['ewc'], optimizer.param_groups[0]['lr'])
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
        n_samples = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            y_pred = self.model(x)
            loss_task = F.mse_loss(y_pred, y)
            loss_kd = torch.zeros(1, device=self.device)
            if alpha_lwf > 0 and self.old_model is not None:
                with torch.no_grad():
                    y_old = self.old_model(x)
                loss_kd = F.mse_loss(y_pred, y_old)
            loss_ewc = torch.zeros(1, device=self.device)
            if apply_ewc and self.ewc_tasks:
                loss_ewc = sum(task.penalty(self.model) for task in self.ewc_tasks)
            loss = loss_task + alpha_lwf * loss_kd + loss_ewc
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            batch_size = x.size(0)
            n_samples += batch_size
            total_loss += loss.item() * batch_size
            task_loss += loss_task.item() * batch_size
            kd_loss += loss_kd.item() * batch_size
            ewc_loss += loss_ewc.item() * batch_size
        return {
            'total': total_loss / n_samples,
            'task': task_loss / n_samples,
            'kd': kd_loss / n_samples,
            'ewc': ewc_loss / n_samples
        }
    
    def _validate(self, loader):
        total_loss = 0
        n_samples = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                loss = F.mse_loss(y_pred, y)
                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                n_samples += batch_size
        return total_loss / n_samples
    
    def consolidate(self, loader, lam):
        ewc = EWC(self.model, loader, self.device, lam)
        self.ewc_tasks.append(ewc)
        self.old_model = copy.deepcopy(self.model).to(self.device)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)
    
    def evaluate(self, loader, dataset_df=None):
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
            'R2': r2_score(targets, preds)
        }
        return metrics, preds, targets

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
    processor = DataProcessor('../01_Datenaufbereitung/Output/Calculated/', config)
    datasets = processor.prepare_data()
    loaders = create_dataloaders(datasets, config)
    logger.info("  (DATA) Dataloaders created for: %s", list(loaders.keys()))
    model = SOHLSTM(len(config.FEATURES_COLS), config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
    logger.info("  (MODEL) SOHLSTM initialized: input_size=4, hidden=%d, layers=%d, dropout=%.2f",
                config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
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

def run_incremental_learning(config, base_dir, device):
    logger.info("=" * 60)
    logger.info("[INCREMENTAL] Starting Incremental Learning")
    logger.info("=" * 60)
    processor = DataProcessor('../01_Datenaufbereitung/Output/Calculated/', config)
    datasets = processor.prepare_data()
    loaders = create_dataloaders(datasets, config)
    logger.info("  (DATA) Dataloaders created for: %s", list(loaders.keys()))
    model = SOHLSTM(len(config.FEATURES_COLS), config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
    logger.info("  (MODEL) SOHLSTM initialized: input_size=%d, hidden=%d, layers=%d, dropout=%.2f",
                len(config.FEATURES_COLS), config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
    trainer = Trainer(model, device, config)
    tasks = [
        ('task0', 'base', 0, False, config.EWC_LAMBDA[0], config.LWF_ALPHA[0]),
        ('task1', 'update1', 1, True, config.EWC_LAMBDA[1], config.LWF_ALPHA[1]),
        ('task2', 'update2', 2, True, config.EWC_LAMBDA[2], config.LWF_ALPHA[2])
    ]
    metrics_history = []
    task_performance = {}
    all_task_test_mae = {}
    all_task_test_results = {}
    logger.info("  (FWT) Computing FWT baselines (randomly initialized model)")
    baseline_model = SOHLSTM(len(config.FEATURES_COLS), config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    baseline_trainer = Trainer(baseline_model, device, config)
    fwt_baseline_mae = {}
    
    for phase in ['base', 'update1', 'update2']:
        test_key = f'test_{phase}'
        if test_key in loaders:
            metrics, _, _ = baseline_trainer.evaluate(loaders[test_key])
            fwt_baseline_mae[test_key] = metrics['MAE']
            logger.info("    (FWT BASELINE) [%s] MAE = %.4e", test_key, metrics['MAE'])
            
    for task_name, data_key, task_id, use_ewc, ewc_lambda, lwf_alpha in tasks:
        logger.info("\n" + "=" * 60)
        logger.info("[TASK] %s (phase=%s, id=%d)", task_name, data_key, task_id)
        logger.info("-" * 60)
        task_dir = base_dir / 'incremental' / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        train_loader = loaders.get(f'{data_key}_train')
        val_loader = loaders.get(f'{data_key}_val')
        
        if trainer.ewc_tasks:
            ewc_lambdas = {f"Task {i}": float(f"{ewc.lam:.2f}") for i, ewc in enumerate(trainer.ewc_tasks)}
            logger.info("  (EWC) Active penalty: %s", json.dumps(ewc_lambdas))
        else:
            logger.info("  (EWC) Active penalty: None")
        history = trainer.train_task(
            train_loader, val_loader, task_id,
            apply_ewc=use_ewc, alpha_lwf=lwf_alpha
        )
        plot_losses(history, task_dir / 'training_curves.png')
        logger.info("  (TRAIN) Finished. Loss curves saved: %s", task_dir / 'training_curves.png')
        
        if ewc_lambda > 0:
            logger.info("  (EWC) Consolidating penalty (λ = %.4f) after %s", ewc_lambda, task_name)
            trainer.consolidate(train_loader, ewc_lambda)
        test_mapping = {
            'task0': ['test_base'],
            'task1': ['test_base', 'test_update1'],
            'task2': ['test_base', 'test_update1', 'test_update2']
        }
        
        all_task_test_mae[task_name] = {}
        all_task_test_results[task_name] = {}
        test_results = {}
        for test_key in test_mapping[task_name]:
            if test_key in loaders:
                metrics, preds, targets = trainer.evaluate(loaders[test_key])
                test_results[test_key] = metrics
                all_task_test_mae[task_name][test_key] = metrics['MAE']
                all_task_test_results[task_name][test_key] = metrics
                logger.info("    (EVAL) [%s] RMSE: %.4e, MAE: %.4e, R2: %.4f",
                            test_key, metrics['RMSE'], metrics['MAE'], metrics['R2'])
                if test_key == f'test_{data_key}':
                    plot_predictions(preds, targets, task_dir / 'predictions')
                    logger.info("    (EVAL) Prediction plots saved: %s", task_dir / 'predictions')
        task_performance[task_name] = test_results
        
        bwt_sum, bwt_count, bwt_norm_list = 0.0, 0, []
        if task_id > 0:
            for prev_task_idx in range(task_id):
                prev_task_name = tasks[prev_task_idx][0]
                prev_test_key = f'test_{tasks[prev_task_idx][1]}'
                cur_mae = all_task_test_mae[task_name].get(prev_test_key, None)
                ori_mae = all_task_test_mae[prev_task_name].get(prev_test_key, None)
                if cur_mae is not None and ori_mae is not None:
                    bwt_val = cur_mae - ori_mae
                    bwt_norm = (cur_mae - ori_mae) / ori_mae
                    bwt_sum += bwt_val
                    bwt_count += 1
                    bwt_norm_list.append(bwt_norm)
                    logger.info("    (BWT) [%s] current: %.4e, original: %.4e, BWT: %.4e, BWT_NORM: %.4e",
                                prev_test_key, cur_mae, ori_mae, bwt_val, bwt_norm)
            avg_bwt = bwt_sum / bwt_count if bwt_count else 0.0
            avg_bwt_norm = np.mean(bwt_norm_list) if bwt_norm_list else 0.0
        else:
            avg_bwt, avg_bwt_norm = 0.0, 0.0
            
            
        fwt_sum, fwt_count = 0.0, 0
        if task_id > 0:
            this_test_key = f'test_{data_key}'
            if this_test_key in fwt_baseline_mae:
                for prev_task_idx in range(task_id):
                    prev_task_name = tasks[prev_task_idx][0]
                    prev_test_mae = all_task_test_mae[prev_task_name].get(this_test_key, None)
                    baseline_mae = fwt_baseline_mae[this_test_key]
                    if prev_test_mae is not None:
                        fwt_val = baseline_mae - prev_test_mae
                        fwt_sum += fwt_val
                        fwt_count += 1
                        logger.info("    (FWT) From [%s]: baseline: %.4e, prev: %.4e, FWT: %.4e",
                                    prev_task_name, baseline_mae, prev_test_mae, fwt_val)
            avg_fwt = fwt_sum / fwt_count if fwt_count else 0.0
        else:
            avg_fwt = 0.0
            
        acc = np.mean([m['MAE'] for m in test_results.values()]) if test_results else float('nan')
        
        
        logger.info("  (SUMMARY) %s -- ACC: %.4e, BWT: %.4e, BWT_NORM: %.4e, FWT: %.4e",
                    task_name, acc, avg_bwt, avg_bwt_norm, avg_fwt)
        torch.save(trainer.model.state_dict(), task_dir / 'model.pt')
        logger.info("  (MODEL) Checkpoint saved: %s", task_dir / 'model.pt')
        metrics_history.append({
            'task': task_name,
            'ACC': acc,
            'BWT': avg_bwt,
            'BWT_NORM': avg_bwt_norm,
            'FWT': avg_fwt,
            'test_results': test_results
        })
    metrics_path = base_dir / 'incremental' / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=4)
    logger.info("=" * 60)
    logger.info("[INCREMENTAL] All incremental learning metrics saved: %s", metrics_path)
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
