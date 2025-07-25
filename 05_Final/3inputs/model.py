from __future__ import annotations
import json
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
        
        self.MODE = "incremental"  # 'joint' or 'incremental'
        self.BASE_DIR = Path.cwd() / "model" / "fine-tuning-ewc(no-eval)"
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
        self.LWF_ALPHAS = [0.0, 0.0, 0.0]  # alpha0, alpha1, alpha2
        self.EWC_LAMBDAS = [0.0, 0.0, 0.0] # lambda0, lambda1, lambda2
        
        # Dataset IDs for joint training
        self.joint_datasets = {
            'train_ids': ['03', '05', '07', '09', '11', '15', '21', '23', '25', '27', '29'],
            'val_ids': ['01', '19', '13'],
            'test_id': '17'
        }
        
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
            "description": "Incremental learning with EWC",
            "resample": self.RESAMPLE,
            "scaler": "RobustScaler - fit on base train",
            "smooth_alpha": self.ALPHA,
            "lwf_alphas": self.LWF_ALPHAS,
            "ewc_lambdas": self.EWC_LAMBDAS,
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
        if 'kd_loss' in df.columns and df['kd_loss'].sum() > 0:
            plt.semilogy(df['epoch'], df['kd_loss'], label='KD Loss', linestyle='--')
        if 'ewc_loss' in df.columns and df['ewc_loss'].sum() > 0:
            plt.semilogy(df['epoch'], df['ewc_loss'], label='EWC Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title('Training/Validation Loss and Loss Components')
        plt.tight_layout()
        plt.savefig(out_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_predictions(preds, tgts, metrics, out_dir):
        """Plot prediction vs actual and residuals in a single figure with two subplots"""
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = np.arange(len(tgts))

        _, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Subplot 1: Prediction vs Actual
        axs[0].plot(idx, tgts, label='Actual', color='tab:blue', alpha=0.8)
        axs[0].plot(idx, preds, label='Predicted', color='tab:orange', alpha=0.7)
        axs[0].set_ylabel('SOH')
        axs[0].set_title(f"Predictions\nRMSE: {metrics['RMSE']:.4e}, MAE: {metrics['MAE']:.4e}, R²: {metrics['R2']:.4f}")
        axs[0].legend()
        axs[0].grid(True)

        # Subplot 2: Residuals
        residuals = tgts - preds
        axs[1].plot(idx, residuals, color='tab:green', alpha=0.7, label='Residuals')
        axs[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
        axs[1].set_xlabel('Index')
        axs[1].set_ylabel('Residuals')
        axs[1].set_title('Residuals (Actual - Predicted)')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(out_dir / 'prediction_and_residuals.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_prediction_scatter(preds, tgts, out_dir):
        """Plot prediction scatter plot and error distribution as two separate figures"""
        out_dir.mkdir(parents=True, exist_ok=True)

        # Scatter plot: Predicted vs Actual
        plt.figure(figsize=(6, 6))
        plt.scatter(tgts, preds, alpha=0.6)
        lims = [min(tgts.min(), preds.min()), max(tgts.max(), preds.max())]
        plt.plot(lims, lims, 'r--', label='Perfect Prediction')
        plt.xlabel('Actual SOH')
        plt.ylabel('Predicted SOH')
        plt.legend()
        plt.grid(True)
        plt.title('Prediction Scatter Plot')
        plt.tight_layout()
        plt.savefig(out_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Error distribution
        plt.figure(figsize=(6, 4))
        errors = np.abs(tgts - preds)
        plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.title('Error Distribution')
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
# Model & EWC
# ===============================================================
class SOHLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
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

class EWC:
    def __init__(self, model, dataloader, device, lam):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher()
        self.lam = lam

    def _compute_fisher(self):
        model_copy = copy.deepcopy(self.model).to(self.device)
        model_copy.train()

        # Disable dropout for Fisher computation
        for m in model_copy.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
            if isinstance(m, nn.LSTM):
                m.dropout = 0.0

        fisher = {n: torch.zeros_like(p, device=self.device)
                 for n, p in model_copy.named_parameters() if p.requires_grad}

        n_processed = 0

        for x, y in self.dataloader:
            x, y = x.to(self.device), y.to(self.device)

            model_copy.zero_grad(set_to_none=True)
            out = model_copy(x)
            loss = F.mse_loss(out, y)
            loss.backward()

            bs = x.size(0)
            n_processed += bs

            with torch.no_grad():
                for n, p in model_copy.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.pow(2) * bs

        # Normalize
        for n in fisher:
            fisher[n] /= float(n_processed)

        del model_copy
        torch.cuda.empty_cache()

        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += self.lam * (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss

# ===============================================================
# Trainer
# ===============================================================
class Trainer:
    def __init__(self, model, device, config, checkpoint_dir):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.ewc_tasks = []
        self.old_model = None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_task(self, train_loader, val_loader, task_id, apply_ewc=True, alpha_lwf=0.0):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.config.LEARNING_RATE,
                                     weight_decay=self.config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.5, patience=5)
        best_val = float('inf')
        no_imp = 0
        best_model_state = None

        history = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'lr', 'time', 'task_loss', 'kd_loss', 'ewc_loss']}

        for epoch in tqdm.tqdm(range(self.config.EPOCHS), desc="Training"):
            epoch_start = time.time()
            self.model.train()
            sum_task = sum_kd = sum_ewc = train_loss = 0.0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(x)
                task_loss = F.mse_loss(y_pred, y)
                kd_loss = torch.zeros((), device=self.device)
                if alpha_lwf > 0 and self.old_model is not None:
                    with torch.no_grad():
                        y_old = self.old_model(x)
                    kd_loss = F.mse_loss(y_pred, y_old)
                ewc_loss = torch.zeros((), device=self.device)
                if apply_ewc and self.ewc_tasks:
                    ewc_loss = sum(t.penalty(self.model) for t in self.ewc_tasks)
                loss = task_loss + alpha_lwf * kd_loss + ewc_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
                bs = x.size(0)
                sum_task += task_loss.item() * bs
                sum_kd += kd_loss.item() * bs
                sum_ewc += ewc_loss.item() * bs
                train_loss += loss.item() * bs

            n_train = len(train_loader.dataset)
            train_loss /= n_train
            task_mean = sum_task / n_train
            kd_mean = sum_kd / n_train
            ewc_mean = sum_ewc / n_train

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
            history['ewc_loss'].append(ewc_mean)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['time'].append(epoch_time)

            logger.info(
                "Epoch %03d | task %.4e | kd %.4e | ewc %.4e | val %.4e | lr %.2e | %.2fs",
                epoch, task_mean, kd_mean, ewc_mean, val_loss,
                optimizer.param_groups[0]['lr'], epoch_time
            )

            if val_loss < best_val:
                best_val = val_loss
                no_imp = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                if self.checkpoint_dir:
                    torch.save({'model_state': best_model_state}, self.checkpoint_dir / f"task{task_id}_best.pt")
            else:
                no_imp += 1
                if no_imp >= self.config.PATIENCE:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return history

    def consolidate(self, loader, task_id=None, lam=0.0):
        self.ewc_tasks.append(EWC(self.model, loader, self.device, lam))
        if self.checkpoint_dir and task_id is not None:
            path = self.checkpoint_dir / f"task{task_id}_best.pt"
            if path.exists():
                state = torch.load(path, map_location=self.device)
                state['ewc_tasks'] = [
                    {
                        'params': {n: p.cpu() for n, p in e.params.items()},
                        'fisher': {n: f.cpu() for n, f in e.fisher.items()},
                        'lam': e.lam
                    }
                    for e in self.ewc_tasks
                ]
                torch.save(state, path)
        self.old_model = copy.deepcopy(self.model).to(self.device)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)

    def evaluate(self, loader, alpha=0.1):
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
        apply_ewc=False
    )
    
    # Save training history and plots
    pd.DataFrame(history).to_csv(ckpt_dir / 'training_history.csv', index=False)
    Visualizer.plot_losses(history, results_dir)
    
    # Evaluate
    preds, tgts, metrics = trainer.evaluate(loader=loaders['test'], alpha=config.ALPHA)
    
    # Save visualizations
    Visualizer.plot_predictions(preds, tgts, metrics, results_dir)
    Visualizer.plot_prediction_scatter(preds, tgts, results_dir)
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv(results_dir / 'test_metrics.csv', index=False)
    
    logger.info("==== Joint Training Completed ====")

def incremental_training(config):
    """Incremental training with EWC"""
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
        ('task0', False, config.EWC_LAMBDAS[0], config.LWF_ALPHAS[0]),
        ('task1', True, config.EWC_LAMBDAS[1], config.LWF_ALPHAS[1]), 
        ('task2', True, config.EWC_LAMBDAS[2], config.LWF_ALPHAS[2])
    ]
    metrics_summary = []
    for task_id, apply_ewc, lam, alpha_lwf in tasks:
        logger.info("==== Training task %s ====", task_id)
        
        task_dir = inc_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        task_idx = int(task_id[-1])
        set_seed(config.SEED + task_idx)  # Ensure different seed for each task
        train_loader = loaders[f'{task_id}_train']
        val_loader = loaders[f'{task_id}_val']
        
        history = trainer.train_task(
            train_loader=train_loader,
            val_loader=val_loader,
            task_id=task_idx,
            apply_ewc=apply_ewc,
            alpha_lwf=alpha_lwf
        )
        pd.DataFrame(history).to_csv(task_dir / f'{task_id}_history.csv', index=False)
        Visualizer.plot_losses(history, task_dir / 'results')
        logger.info("Consolidate lambda: %.4f", lam)
        logger.info("Applying LWF alpha: %.4f", alpha_lwf)
        trainer.consolidate(train_loader, task_id=task_idx, lam=lam)
        logger.info("==== Training task %s completed ====", task_id)
        
        # logger.info("==== Evaluating task %s ====", task_id)
        # test_loader = loaders[f'test_{task_id}']
        # preds, tgts, metrics = trainer.evaluate(test_loader)
        # logger.info("Current task %s metrics: RMSE=%.4e, MAE=%.4e, R²=%.4f",
        #             task_id, metrics['RMSE'], metrics['MAE'], metrics['R2'])
        # # Save metrics
        # metrics_summary.append({
        #     "task": task_id,
        #     "scope": "task",
        #     **metrics,
        # })
        # plots_dir = task_dir / 'results'
        # Visualizer.plot_predictions(preds, tgts, metrics, plots_dir)
        # Visualizer.plot_prediction_scatter(preds, tgts, plots_dir)
        
        # test_full_loader = loaders['test_full']
        # full_preds, full_tgts, full_metrics = trainer.evaluate(test_full_loader)
        
        # metrics_summary.append({
        #     "task": task_id,
        #     "scope": "full",
        #     **full_metrics,
        # })
        # logger.info("Full test metrics: RMSE=%.4e, MAE=%.4e, R²=%.4f",
        #             full_metrics['RMSE'], full_metrics['MAE'], full_metrics['R2'])
        # Visualizer.plot_predictions(full_preds, full_tgts, full_metrics, plots_dir)
        # Visualizer.plot_prediction_scatter(full_preds, full_tgts, plots_dir)
        # logger.info("==== Evaluation for task %s completed ====", task_id)
    
    summary_df = pd.DataFrame(metrics_summary)
    summary_df.to_csv(inc_dir / 'incremental_metrics_summary.csv', index=False)
    logger.info("==== Incremental Training Completed ====")
        
# ===============================================================
# Main Pipeline
# ===============================================================
def main():
    """Main training pipeline"""
    # Initialize configuration and logging
    config = Config()
    base_dir = config.BASE_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(base_dir)
    config.save(base_dir / 'config.json')
    set_seed(config.SEED)
    # Joint training
    if config.MODE == 'joint':
        joint_training(config)
    elif config.MODE == 'incremental':
        incremental_training(config)


if __name__ == '__main__':
    main()