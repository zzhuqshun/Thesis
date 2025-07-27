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
    def __init__(self, **kwargs):
        # job_id = os.getenv('JOB_ID') or os.getenv('SLURM_JOB_ID')
        self.MODE = "incremental"  # 'joint' or 'incremental'
        self.BASE_DIR = Path.cwd() / "strategies" / "mas"
        self.DATA_DIR = Path('../../01_Datenaufbereitung/Output/Calculated/')
        
        # Model parameters
        self.SEQUENCE_LENGTH = 720
        self.HIDDEN_SIZE = 128
        self.NUM_LAYERS = 2
        self.DROPOUT = 0.3
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-4
        self.EPOCHS = 200
        self.PATIENCE = 20
        self.WEIGHT_DECAY = 1e-6
        self.SCALER = "RobustScaler"
        self.SEED = 42
        self.RESAMPLE = '10min'
        
        self.ALPHA = 0.1  # Smoothing factor for predictions
        
        # # trial 17
        # self.LWF_ALPHAS = [0.0, 1.5967913830605263, 0.7224937140102906]  # alpha0, alpha1, alpha2
        # self.MAS_LAMBDAS = [215.95446197298435, 588.671303985778, 0.0] # lambda0, lambda1, lambda2
        
        self.LWF_ALPHAS = [0.0, 0.0, 0.0]
        self.MAS_LAMBDAS = [215.95446197298435, 588.671303985778, 0.0] # lambda0, lambda1, lambda2

        # # trial 21
        # self.LWF_ALPHAS = [0.0, 0.30787546631146706, 1.261950525276332]  # alpha0, alpha1, alpha2
        # self.mas_LAMBDAS = [555.8803969466832, 1964.1799421675798, 0.0] # lambda0, lambda1, lambda2
        
        # self.LWF_ALPHAS = [0.0, 0.0, 0.0]
        # self.mas_LAMBDAS = [0.0, 0.0, 0.0]
        
        # Dataset IDs for joint training
        self.joint_datasets = {
            'train_ids': ['03', '05', '07', '09', '11', '15', '21', '23', '25', '27', '29'],
            'val_ids': ['01', '19', '13'],
            'test_id': '17'
        }
        
        # # Dataset IDs for incremental training -- cum val
        # self.incremental_datasets = {
        #     'task0_train_ids': ['03', '05', '07', '27'],
        #     'task0_val_ids': ['01'],
        #     'task1_train_ids': ['21', '23', '25'],
        #     'task1_val_ids': ['01','19'],
        #     'task2_train_ids': ['09', '11', '15', '29'],
        #     'task2_val_ids': ['01', '19', '13'],
        #     'test_id': '17'
        # }
        
        # Dataset IDs for incremental training
        self.incremental_datasets = {
            'task0_train_ids': ['03', '05', '07', '27'],
            'task0_val_ids': ['01'],
            'task1_train_ids': ['21', '23', '25'],
            'task1_val_ids': ['19'],
            'task2_train_ids': ['09', '11', '15', '29'],
            'task2_val_ids': ['13'],
            'test_id': '17'
        }
        
        self.Info = {
            "description": "Incremental learning with mas",
            "resample": self.RESAMPLE,
            "scaler": "RobustScaler - fit on base train",
            "smooth_alpha": self.ALPHA,
            "lwf_alphas": self.LWF_ALPHAS,
            "mas_lambdas": self.MAS_LAMBDAS,
        }
        
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                config_dict[k] = str(v)
            else:
                config_dict[k] = v
                
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

# ===============================================================
# Visualization Functions
# ===============================================================
class Visualizer:
    @staticmethod
    def plot_losses(history, out_dir):
        """Plot training and validation loss curves (without lr and training time), and loss components in one figure"""
        df = pd.DataFrame(history)
        out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        # Training and validation loss
        plt.semilogy(df['epoch'], df['train_loss'], label='Train Loss')
        plt.semilogy(df['epoch'], df['val_loss'], label='Val Loss')
        # Loss components
        plt.semilogy(df['epoch'], df['task_loss'], label='Task Loss', linestyle='--')
        if 'kd_loss' in df.columns:
            plt.semilogy(df['epoch'], df['kd_loss'], label='KD Loss', linestyle='--')
        if 'mas_loss' in df.columns:
            plt.semilogy(df['epoch'], df['mas_loss'], label='mas Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title('Training/Validation Loss and Loss Components')
        plt.tight_layout()
        plt.savefig(out_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_predictions(preds, tgts, metrics, out_dir, alpha=0.1):
        """Plot prediction vs actual and residuals as separate figures, including smooth predictions"""
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = np.arange(len(tgts))
        
        # Calculate smooth predictions
        preds_smooth = pd.Series(preds).ewm(alpha=alpha, adjust=False).mean().to_numpy()

        # Figure 1: Prediction vs Actual
        plt.figure(figsize=(12, 6))
        plt.plot(idx, tgts, label='Actual', color='tab:blue', alpha=0.8, linewidth=1.5)
        plt.plot(idx, preds, label='Predicted', color='tab:orange', alpha=0.7, linewidth=1)
        plt.plot(idx, preds_smooth, label='Predicted (Smooth)', color='tab:red', alpha=0.8, linewidth=1.2)
        plt.xlabel('Index')
        plt.ylabel('SOH')
        
        # Title with both original and smooth metrics
        title_text = (f"Predictions\n"
                     f"RMSE: {metrics['RMSE']:.4e}, MAE: {metrics['MAE']:.4e}, R²: {metrics['R2']:.4f}\n"
                     f"RMSE (smooth): {metrics['RMSE_smooth']:.4e}, MAE (smooth): {metrics['MAE_smooth']:.4e}, R² (smooth): {metrics['R2_smooth']:.4f}")
        plt.title(title_text)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / 'predictions.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 2: Residuals
        plt.figure(figsize=(12, 5))
        residuals = tgts - preds
        residuals_smooth = tgts - preds_smooth
        plt.plot(idx, residuals, color='tab:green', alpha=0.7, label='Residuals', linewidth=1)
        plt.plot(idx, residuals_smooth, color='tab:purple', alpha=0.7, label='Residuals (Smooth)', linewidth=1.2)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
        plt.xlabel('Index')
        plt.ylabel('Residuals')
        plt.title('Residuals (Actual - Predicted)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / 'residuals.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_prediction_scatter(preds, tgts, out_dir, alpha=0.1):
        """Plot prediction scatter plot and error distribution as two separate figures, including smooth predictions"""
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate smooth predictions
        preds_smooth = pd.Series(preds).ewm(alpha=alpha, adjust=False).mean().to_numpy()

        # Scatter plot: Predicted vs Actual
        plt.figure(figsize=(12, 5))
        
        # Original predictions scatter
        plt.subplot(1, 2, 1)
        plt.scatter(tgts, preds, alpha=0.6, label='Original')
        lims = [min(tgts.min(), preds.min()), max(tgts.max(), preds.max())]
        plt.plot(lims, lims, 'r--', label='Perfect Prediction')
        plt.xlabel('Actual SOH')
        plt.ylabel('Predicted SOH')
        plt.legend()
        plt.grid(True)
        plt.title('Prediction Scatter Plot (Original)')
        
        # Smooth predictions scatter
        plt.subplot(1, 2, 2)
        plt.scatter(tgts, preds_smooth, alpha=0.6, color='red', label='Smooth')
        lims = [min(tgts.min(), preds_smooth.min()), max(tgts.max(), preds_smooth.max())]
        plt.plot(lims, lims, 'r--', label='Perfect Prediction')
        plt.xlabel('Actual SOH')
        plt.ylabel('Predicted SOH (Smooth)')
        plt.legend()
        plt.grid(True)
        plt.title('Prediction Scatter Plot (Smooth)')
        
        plt.tight_layout()
        plt.savefig(out_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Error distribution
        plt.figure(figsize=(12, 4))
        
        # Original errors
        plt.subplot(1, 2, 1)
        errors = np.abs(tgts - preds)
        plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.title('Error Distribution (Original)')
        
        # Smooth errors
        plt.subplot(1, 2, 2)
        errors_smooth = np.abs(tgts - preds_smooth)
        plt.hist(errors_smooth, bins=30, alpha=0.7, edgecolor='black', color='red')
        plt.xlabel('Absolute Error (Smooth)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.title('Error Distribution (Smooth)')
        
        plt.tight_layout()
        plt.savefig(out_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

# ===============================================================
# Dataset & DataProcessor
# ===============================================================
class BatteryDataset(Dataset):
    def __init__(self, df, seq_len):
        feats = df[['Voltage[V]', 'Current[A]', 'Temperature[°C]']].values
        self.X = torch.tensor(feats, dtype=torch.float32)
        self.y = torch.tensor(df['SOH_ZHU'].values, dtype=torch.float32)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.X) - self.seq_len
    
    def __getitem__(self, idx):
        return (
            self.X[idx:idx + self.seq_len],
            self.y[idx + self.seq_len]
        )

class DataProcessor:
    def __init__(self, data_dir, resample='10min', config=None):
        self.data_dir = Path(data_dir)
        self.config = config
        self.resample = resample
        self.scaler = RobustScaler()

    def load_cell_data(self):
        files = sorted(self.data_dir.glob('*.parquet'),
                      key=lambda x: int(x.stem.split('_')[-1]))
        info = {fp.stem.split('_')[-1]: fp for fp in files}
        return info

    def process_file(self, fp):
        df = pd.read_parquet(fp)[[
            'Testtime[s]', 'Voltage[V]', 'Current[A]', 'Temperature[°C]', 'SOH_ZHU']]
        df = df.dropna().reset_index(drop=True)
        df['Testtime[s]'] = df['Testtime[s]'].round().astype(int)
        df['Datetime'] = pd.date_range('2023-02-02', periods=len(df), freq='s')
        df = df.set_index('Datetime').resample(self.resample).mean().reset_index()
        df['cell_id'] = fp.stem.split('_')[-1]
        return df

    def prepare_joint_data(self, datasets_config):
        """Prepare data for joint training"""
        info_map = self.load_cell_data()
        
        def build(ids):
            if not ids:
                return pd.DataFrame()
            dfs = [self.process_file(info_map[c]) for c in ids]
            return pd.concat(dfs, ignore_index=True)

        # Build datasets
        df_train = build(datasets_config['train_ids'])
        df_val = build(datasets_config['val_ids'])
        df_test = self.process_file(info_map[datasets_config['test_id']])

        logger.info("Joint training - Train IDs: %s, size: %d", 
                   datasets_config['train_ids'], len(df_train))
        logger.info("Joint training - Val IDs: %s, size: %d", 
                   datasets_config['val_ids'], len(df_val))
        logger.info("Joint training - Test ID: %s, size: %d", 
                   datasets_config['test_id'], len(df_test))

        # Fit scaler on training data
        feat_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        self.scaler.fit(df_train[feat_cols])
        logger.info("  (Scaler) Scaler centers: %s", self.scaler.center_)
        logger.info("  (Scaler) Scaler scales: %s", self.scaler.scale_)

        def scale_df(df):
            if df.empty:
                return df
            df2 = df.copy()
            df2[feat_cols] = self.scaler.transform(df2[feat_cols])
            return df2

        return {
            'train': scale_df(df_train),
            'val': scale_df(df_val),
            'test': scale_df(df_test)
        }

    def prepare_incremental_data(self, datasets_config):
        """Prepare data for incremental training"""
        info_map = self.load_cell_data()
        
        def build(ids):
            if not ids:
                return pd.DataFrame()
            dfs = [self.process_file(info_map[c]) for c in ids]
            return pd.concat(dfs, ignore_index=True)

        # Build phases
        df_0train   = build(datasets_config['task0_train_ids'])
        df_0val     = build(datasets_config['task0_val_ids'])
        df_1train   = build(datasets_config['task1_train_ids'])
        df_1val     = build(datasets_config['task1_val_ids'])
        df_2train   = build(datasets_config['task2_train_ids'])
        df_2val     = build(datasets_config['task2_val_ids'])
        
        # Test splits
        df_test = self.process_file(info_map[datasets_config['test_id']])
        df_test_0 = df_test[df_test['SOH_ZHU'] >= 0.9].reset_index(drop=True)
        df_test_1 = df_test[(df_test['SOH_ZHU'] < 0.9) & (df_test['SOH_ZHU'] >= 0.8)].reset_index(drop=True)
        df_test_2 = df_test[df_test['SOH_ZHU'] < 0.8].reset_index(drop=True)

        logger.info("Incremental training - Task 0 Train IDs: %s, size: %d", 
                    datasets_config['task0_train_ids'], len(df_0train))
        logger.info("Incremental training - Task 0 Val IDs: %s, size: %d", 
                    datasets_config['task0_val_ids'], len(df_0val))
        logger.info("Incremental training - Task 1 Train IDs: %s, size: %d", 
                    datasets_config['task1_train_ids'], len(df_1train))
        logger.info("Incremental training - Task 1 Val IDs: %s, size: %d", 
                    datasets_config['task1_val_ids'], len(df_1val))
        logger.info("Incremental training - Task 2 Train IDs: %s, size: %d", 
                    datasets_config['task2_train_ids'], len(df_2train))
        logger.info("Incremental training - Task 2 Val IDs: %s, size: %d", 
                    datasets_config['task2_val_ids'], len(df_2val))
        logger.info("Incremental training - Test ID: %s, size: %d", 
                    datasets_config['test_id'], len(df_test))
        logger.info("Incremental training - Test Task 0 size: %d", len(df_test_0))
        logger.info("Incremental training - Test Task 1 size: %d", len(df_test_1))
        logger.info("Incremental training - Test Task 2 size: %d", len(df_test_2))
        

        # Fit scaler on base training data
        feat_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        self.scaler.fit(df_0train[feat_cols])
        logger.info("  (Scaler) Scaler centers: %s", self.scaler.center_)
        logger.info("  (Scaler) Scaler scales: %s", self.scaler.scale_)

        def scale_df(df):
            if df.empty:
                return df
            df2 = df.copy()
            df2[feat_cols] = self.scaler.transform(df2[feat_cols])
            return df2

        return {
            'task0_train': scale_df(df_0train),
            'task0_val': scale_df(df_0val),
            'task1_train': scale_df(df_1train),
            'task1_val': scale_df(df_1val),
            'task2_train': scale_df(df_2train),
            'task2_val': scale_df(df_2val),
            'test_full': scale_df(df_test),
            'test_task0': scale_df(df_test_0),
            'test_task1': scale_df(df_test_1),
            'test_task2': scale_df(df_test_2)
        }

# ===============================================================
# Model & mas
# ===============================================================
class SOHLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        # h = self.norm(h)
        return self.fc(h).squeeze(-1)

# ===============================================================
# MAS Class: Memory Aware Synapses
# ===============================================================
class MAS:
    def __init__(self, model, dataloader, device, lam):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.lam = lam
        self.importance = self._compute_importance()

    def _compute_importance(self):
        model = self.model
        was_training = model.training
        model.train()
        
        imp = {n: torch.zeros_like(p, device=self.device)
               for n, p in model.named_parameters() if p.requires_grad}
        
        for x, _ in self.dataloader:
            x = x.to(self.device)
            model.zero_grad()
            out = model(x)
            (out.sum()).backward()
            
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        imp[n] += p.grad.abs()
                        
        if not was_training:
            model.eval()
        # normalize
        for n in imp:
            imp[n] /= float(len(self.dataloader))
        return imp

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += self.lam * (self.importance[n] * (p - self.params[n]).pow(2)).sum()
        return loss

# ===============================================================
# Trainer
# ===============================================================
class Trainer:
    def __init__(self, model, device, config, task_dir=None):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.mas_tasks = []
        self.old_model = None
        self.task_dir = Path(task_dir) if task_dir else None
        if self.task_dir:
            self.task_dir.mkdir(parents=True, exist_ok=True)

    def train_task(self, train_loader, val_loader, task_id, alpha_lwf=0.0):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.config.LEARNING_RATE,
                                     weight_decay=self.config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.5, patience=5)
        best_val = float('inf')
        no_imp = 0
        best_model_state = None

        history = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'lr', 'time', 'task_loss', 'kd_loss', 'mas_loss']}

        for epoch in tqdm.tqdm(range(self.config.EPOCHS), desc=f"Training Task {task_id}"):
            epoch_start = time.time()
            self.model.train()
            sum_task = sum_kd = sum_mas = train_loss = 0.0
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(x)
                task_loss = F.mse_loss(y_pred, y)
                #F.smooth_l1_loss(y_pred, y)
                
                kd_loss = torch.zeros((), device=self.device)
                if alpha_lwf > 0 and self.old_model is not None:
                    with torch.no_grad():
                        y_old = self.old_model(x)
                    kd_loss = F.mse_loss(y_pred, y_old)
                    
                mas_loss = torch.zeros((),device=self.device)
                if self.mas_tasks:
                    mas_loss = sum(m.penalty(self.model) for m in self.mas_tasks)
                    
                loss = task_loss + alpha_lwf*kd_loss + mas_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
                
                bs = x.size(0)
                sum_task += task_loss.item() * bs
                sum_kd += kd_loss.item() * bs
                sum_mas += mas_loss.item() * bs
                train_loss += loss.item() * bs

            n_train = len(train_loader.dataset)
            train_loss /= n_train
            task_mean = sum_task / n_train
            kd_mean = sum_kd / n_train
            mas_mean = sum_mas / n_train

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    val_loss += F.mse_loss(self.model(x), y).item() * x.size(0)
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            epoch_time = time.time() - epoch_start

            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['task_loss'].append(task_mean)
            history['kd_loss'].append(kd_mean)
            history['mas_loss'].append(mas_mean)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['time'].append(epoch_time)

            logger.info(
                "Epoch %03d | task %.4e | kd %.4e | mas %.4e | val %.4e | lr %.2e | %.2fs",
                epoch, task_mean, kd_mean, mas_mean, val_loss,
                optimizer.param_groups[0]['lr'], epoch_time
            )

            if val_loss < best_val:
                best_val = val_loss
                no_imp = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                # Save to task directory if available
                if self.task_dir:
                    torch.save({'model_state': best_model_state}, self.task_dir / f"task{task_id}_best.pt")
            else:
                no_imp += 1
                if no_imp >= self.config.PATIENCE:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return history

    def consolidate(self, loader, task_id=None, lam=0.0):
        self.mas_tasks.append(MAS(self.model, loader, self.device, lam))
        self.old_model = copy.deepcopy(self.model).to(self.device)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)

    def evaluate(self, loader, alpha=0.1, log = True):
        self.model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                preds.append(self.model(x).cpu().numpy())
                tgts.append(y.cpu().numpy().ravel())
        preds = np.concatenate(preds)
        tgts = np.concatenate(tgts)
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(tgts, preds)),
            'MAE': mean_absolute_error(tgts, preds),
            'R2': r2_score(tgts, preds)
        }
        
        preds_smooth = pd.Series(preds).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        metrics['RMSE_smooth'] = np.sqrt(mean_squared_error(tgts, preds_smooth))
        metrics['MAE_smooth'] = mean_absolute_error(tgts, preds_smooth)
        metrics['R2_smooth'] = r2_score(tgts, preds_smooth)
        
        if log:        
            logger.info("RMSE: %.4e, MAE: %.4e, R²: %.4f", metrics['RMSE'], metrics['MAE'], metrics['R2'])
        return preds, tgts, metrics

# ===============================================================
# Utilities
# ===============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_dataloaders(datasets, seq_len, batch_size):
    loaders = {}
    for key, df in datasets.items():
        if not df.empty and ('train' in key or 'val' in key or 'test' in key):
            ds = BatteryDataset(df, seq_len)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=('train' in key))
            loaders[key] = loader
    return loaders

def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = log_dir / 'train.log'
    
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str(log_path)
               for h in logger.handlers):
        log_f = logging.FileHandler(log_path, encoding='utf-8')
        log_f.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(log_f)
    
    return logger

# ===============================================================
# Main Training Functions
# ===============================================================
def joint_training(config):
    """Joint training on all data"""
    logger.info("==== Joint Training Phase ====")
    
    # Setup directories
    joint_dir = config.BASE_DIR / 'joint'
    ckpt_dir = joint_dir / 'checkpoints'
    results_dir = joint_dir / 'results'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    dp = DataProcessor(
        data_dir=config.DATA_DIR,
        resample=config.RESAMPLE,
        config=config
    )
    datasets = dp.prepare_joint_data(config.joint_datasets)
    loaders = create_dataloaders(datasets, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    
    # Initialize model and trainer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    trainer = Trainer(model, device, config, ckpt_dir)
    
    # Train
    history = trainer.train_task(
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        task_id=0,
        apply_mas=False
    )
    
    # Save training history and plots
    pd.DataFrame(history).to_csv(ckpt_dir / 'training_history.csv', index=False)
    Visualizer.plot_losses(history, results_dir)
    
    # Evaluate
    preds, tgts, metrics = trainer.evaluate(loader=loaders['test'], alpha=config.ALPHA)
    
    # Save visualizations
    Visualizer.plot_predictions(preds, tgts, metrics, results_dir, alpha=config.ALPHA)
    Visualizer.plot_prediction_scatter(preds, tgts, results_dir, alpha=config.ALPHA)
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv(results_dir / 'test_metrics.csv', index=False)
    
    logger.info("==== Joint Training Completed ====")

def incremental_training(config):
    """Incremental training with mas"""
    logger.info("==== Incremental Training Phase ====")
    
    # Setup directories
    inc_dir = config.BASE_DIR / 'incremental'
    inc_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    dp = DataProcessor(
        data_dir=config.DATA_DIR,
        resample=config.RESAMPLE,
        config=config
    )
    datasets = dp.prepare_incremental_data(config.incremental_datasets)
    loaders = create_dataloaders(datasets, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    
    # Initialize model and trainer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    trainer = Trainer(model, device, config, inc_dir)
    
    # Task configurations
    tasks = [
        ('task0', config.MAS_LAMBDAS[0], config.LWF_ALPHAS[0]),
        ('task1', config.MAS_LAMBDAS[1], config.LWF_ALPHAS[1]),
        ('task2', config.MAS_LAMBDAS[2], config.LWF_ALPHAS[2]),
    ]

    # ===============================================================
    # Model Training
    # ===============================================================
    for name, lam, alpha in tasks:
        tid = int(name[-1])
        set_seed(config.SEED+tid)
        logger.info("== Training %s (MAS λ=%.2f) ==", name, lam)
        loader_train = loaders[f"{name}_train"]
        loader_val = loaders[f"{name}_val"]
        history = trainer.train_task(loader_train, loader_val, tid, alpha_lwf=alpha)
        # save plots & history
        out = inc_dir/name/'results'
        out.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(history).to_csv(inc_dir/name/'training_history.csv',index=False)
        Visualizer.plot_losses(history, out)
        
        # consolidate MAS
        trainer.consolidate(loader_train,task_id=tid,lam=lam)
        
    logger.info("==== Incremental Training Completed ====")

    # ===============================================================
    # Evaluation Phase - Load each stage model and evaluate
    # ===============================================================
    logger.info("==== Starting Comprehensive Evaluation ====")
    
    # Create evaluation directory
    eval_dir = inc_dir / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Matrix to store R values (performance metric = -MAE)
    # R[i][j] = performance of model after learning task i on test set of task j
    num_tasks = len(tasks)
    R_matrix = np.zeros((num_tasks, num_tasks))
    
    metrics_summary = []
    
    for i, (trained_task_id, _, _, _) in enumerate(tasks):
        # Load model checkpoint from this stage
        checkpoint_path = inc_dir / f"task{i}" / f"task{i}_best.pt"
        if not checkpoint_path.exists():
            logger.error("Checkpoint not found: %s", checkpoint_path)
            continue
            
        # Create fresh model and load weights
        eval_model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        eval_model.load_state_dict(checkpoint['model_state'])
        eval_model.eval()
        
        # Create temporary trainer for evaluation
        eval_trainer = Trainer(eval_model, device, config, None)
        
        # Evaluate on test full
        full_preds, full_true, full_metrics = eval_trainer.evaluate(
            loaders['test_full'], alpha=config.ALPHA, log=False
        )
        Visualizer.plot_predictions(
            full_preds, full_true, full_metrics,
            eval_dir / f"task{i}_full"
        )
        
        metrics_summary.append({
        "trained_after_task": trained_task_id,
        "evaluated_on_task": "full",
        "trained_task_idx": i,
        "eval_task_idx": -1,
        **{k: full_metrics[k] for k in (
            "MAE","MAE_smooth","RMSE","RMSE_smooth","R2","R2_smooth"
        )},
        "R_value": -full_metrics["MAE"]
        })
        
        
        # Evaluate on all task test sets
        for j, (test_task_id, _, _, _) in enumerate(tasks):
            test_loader = loaders[f'test_{test_task_id}']
            # Get predictions and metrics
            _, _, metrics = eval_trainer.evaluate(test_loader, alpha=config.ALPHA, log=False)
            
            # Store R value (negative MAE for maximization perspective)
            R_matrix[i][j] = -metrics['MAE']
            
            # Store only essential metrics for CL calculation
            metrics_summary.append({
                "trained_after_task": trained_task_id,
                "evaluated_on_task": test_task_id,
                "trained_task_idx": i,
                "eval_task_idx": j,
                "MAE": metrics['MAE'],
                "MAE_smooth": metrics['MAE_smooth'],
                "RMSE": metrics['RMSE'],
                "RMSE_smooth": metrics['RMSE_smooth'],
                "R2": metrics['R2'],
                "R2_smooth": metrics['R2_smooth'],
                "R_value": R_matrix[i][j]
            })
    
    # ===============================================================
    # Calculate BWT, FWT, ACC according to literature
    # ===============================================================
    logger.info("==== Computing BWT, FWT, ACC Metrics ====")
    
    # Baseline performance calculation for FWT
    logger.info("Computing random initialization baselines for FWT calculation...")
    torch.manual_seed(config.SEED + 773)
    baseline_model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    baseline_trainer = Trainer(baseline_model, device, config, None)
    
    baseline_performance = np.zeros(num_tasks)
    for j in range(num_tasks):
        test_loader = loaders[f'test_task{j}']
        _, _, baseline_metrics = baseline_trainer.evaluate(test_loader, alpha=config.ALPHA, log=False)
        baseline_performance[j] = -baseline_metrics['MAE']
    
    # Calculate BWT, FWT, ACC
    BWT = np.mean([R_matrix[num_tasks-1, i] - R_matrix[i, i] for i in range(num_tasks-1)])
    
    FWT = np.mean([R_matrix[i-1, i] - baseline_performance[i] for i in range(1, num_tasks)])
    
    ACC = np.mean(R_matrix[num_tasks-1, :])
    
    # Create summary metrics
    continual_learning_metrics = {
        "BWT": BWT,
        "FWT": FWT, 
        "ACC": ACC,
    }
    
    logger.info("==== Continual Learning Metrics ====")
    logger.info("BWT: %.4f", BWT)
    logger.info("FWT: %.4f", FWT)
    logger.info("ACC: %.4f", ACC)
    
    # Print R matrix for inspection
    logger.info("==== R Matrix (Performance Matrix) ====")
    for i in range(num_tasks):
        row_str = " ".join([f"{R_matrix[i,j]:7.4f}" for j in range(num_tasks)])
        logger.info("Task %d: [%s]", i, row_str)
    
    # ===============================================================
    # Save Results to evaluation directory
    # ===============================================================
    
    # Save detailed metrics
    summary_df = pd.DataFrame(metrics_summary)
    summary_df.to_csv(eval_dir / 'detailed_evaluation_results.csv', index=False)
    
    # Save continual learning metrics
    cl_metrics_df = pd.DataFrame([continual_learning_metrics])
    cl_metrics_df.to_csv(eval_dir / 'continual_learning_metrics.csv', index=False)
    
    # Save R matrix
    r_matrix_df = pd.DataFrame(R_matrix, 
                              index=[f"after_task{i}" for i in range(num_tasks)],
                              columns=[f"eval_task{j}" for j in range(num_tasks)])
    r_matrix_df.to_csv(eval_dir / 'R_matrix.csv')
    
    logger.info("==== Incremental Training and Evaluation Completed ====")
    logger.info("All evaluation results saved to: %s", eval_dir)
    
    return continual_learning_metrics, R_matrix

        
# ===============================================================
# Main Pipeline
# ===============================================================
def main():
    config = Config()
    config.BASE_DIR.mkdir(parents=True,exist_ok=True)
    setup_logging(config.BASE_DIR)
    config.save(config.BASE_DIR/'config.json')
    set_seed(config.SEED)

    # Joint training
    if config.MODE == 'joint':
        joint_training(config)
    elif config.MODE == 'incremental':
        incremental_training(config)


if __name__ == '__main__':
    main()