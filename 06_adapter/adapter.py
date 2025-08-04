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
    """Configuration class for continual learning experiments"""
    def __init__(self, **kwargs):
        # Training mode: 'joint' for baseline, 'incremental' for continual learning
        self.MODE = 'joint'  
        
        # Directory structure
        self.BASE_DIR = Path.cwd() / "joint_mish"
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
        
        # Adapter parameters
        self.USE_ADAPTER = True     # Whether to use adapter
        self.ADAPTER_SIZE = 32    # Adapter bottleneck size
        self.FREEZE_BACKBONE = True # Whether to freeze LSTM backbone
        self.ACTIVATION = 'GELU'  # Activation function for adapter

        # Random seed for reproducibility
        self.SEED = 42
        
        # Dataset splits for joint training (baseline)
        self.joint_datasets = {
            'train_ids': ['03', '05', '07', '09', '11', '15', '21', '23', '25', '27', '29'],
            'val_ids': ['01', '19', '13'],
            'test_id': '17'
        }
        
        # Dataset splits for incremental learning
        # New strategy: Task0 mixed types, Task1&2 random sampling
        self.incremental_datasets = self._create_incremental_splits()
        
        # Experiment metadata
        self.Info = {
            "method": "Adapter Fine-tuning" if self.USE_ADAPTER else "Pure Fine-tuning",
            "adapter_size": self.ADAPTER_SIZE if self.USE_ADAPTER else None,
            "freeze_backbone": self.FREEZE_BACKBONE if self.USE_ADAPTER else None,
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
        
        # Override with any provided arguments
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def _create_incremental_splits(self):
        """
        Create incremental learning splits with mixed Task0 and random Task1&2
        
        Cell categorization by degradation speed:
        - Normal: ['03', '05', '07', '27'] 
        - Fast: ['21', '23', '25']
        - Faster: ['09', '11', '15', '29']
        - Test: '17' (Fast category)
        - Val: ['01', '19', '13'] (keep unchanged)
        """
        # Set seed for reproducible random sampling
        random.seed(self.SEED)
        
        # Define cell categories
        normal_cells = ['03', '05', '07', '27']
        fast_cells = ['21', '23', '25']  # '17' reserved for testing
        faster_cells = ['09', '11', '15', '29']
        
        # Task 0: Mixed types (3 normal + 1 fast + 1 faster)
        task0_normal = random.sample(normal_cells, 3)  # 3 normal cells
        task0_fast = random.sample(fast_cells, 1)      # 1 fast cell  
        task0_faster = random.sample(faster_cells, 1)  # 1 faster cell
        
        task0_train_ids = task0_normal + task0_fast + task0_faster
        
        # Remaining cells for Task1 & Task2
        remaining_cells = (
            [c for c in normal_cells if c not in task0_normal] +
            [c for c in fast_cells if c not in task0_fast] +
            [c for c in faster_cells if c not in task0_faster]
        )
        
        # Shuffle remaining cells for random assignment
        random.shuffle(remaining_cells)
        
        # Task 1: Random 3 cells
        task1_train_ids = remaining_cells[:3]
        
        # Task 2: Next random 3 cells  
        task2_train_ids = remaining_cells[3:6]
        
        logger.info("=== Data Split Strategy ===")
        logger.info("Task 0 (Mixed): %s", task0_train_ids)
        logger.info("  - Normal: %s", task0_normal)
        logger.info("  - Fast: %s", task0_fast) 
        logger.info("  - Faster: %s", task0_faster)
        logger.info("Task 1 (Random): %s", task1_train_ids)
        logger.info("Task 2 (Random): %s", task2_train_ids)
        logger.info("Validation IDs: ['01', '19', '13'] (unchanged)")
        logger.info("Test ID: '17' (Fast category)")
        
        return {
            'task0_train_ids': task0_train_ids,
            'task0_val_ids': ['01'],
            'task1_train_ids': task1_train_ids, 
            'task1_val_ids': ['19'],
            'task2_train_ids': task2_train_ids,
            'task2_val_ids': ['13'],
            'test_id': '17'
        }
    
    def save(self, path):
        """Save configuration to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, path):
        """Load configuration from JSON file"""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

# ===============================================================
# Visualization Functions
# ===============================================================
class Visualizer:
    """Utility class for creating training and evaluation plots"""
    
    @staticmethod
    def plot_losses(history, out_dir):
        """Plot training/validation losses"""
        df = pd.DataFrame(history)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(df['epoch'], df['train_loss'], label='Train Loss')
        plt.semilogy(df['epoch'], df['val_loss'], label='Val Loss')
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title('Training/Validation Loss')
        plt.tight_layout()
        plt.savefig(out_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_predictions(preds, tgts, metrics, out_dir, alpha=0.1):
        """Plot predictions vs actual values with metrics"""
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = np.arange(len(tgts))
        
        # Apply exponential smoothing to predictions
        preds_smooth = pd.Series(preds).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        
        plt.figure(figsize=(12, 6))
        plt.plot(idx, tgts, label='Actual')
        plt.plot(idx, preds, label='Predicted')
        plt.plot(idx, preds_smooth, label='Predicted (Smooth)')
        plt.xlabel('Index')
        plt.ylabel('SOH')
        
        title = (f"RMSE: {metrics['RMSE']:.4e}, MAE: {metrics['MAE']:.4e}, R2: {metrics['R2']:.4f}\n"
                 f"RMSE(s): {metrics['RMSE_smooth']:.4e}, MAE(s): {metrics['MAE_smooth']:.4e}, R2(s): {metrics['R2_smooth']:.4f}")
        plt.title('Predictions vs Actuals\n' + title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / 'predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_prediction_scatter(preds, tgts, out_dir, alpha=0.1):
        """Create scatter plots of predictions vs actuals"""
        out_dir.mkdir(parents=True, exist_ok=True)
        preds_smooth = pd.Series(preds).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        
        plt.figure(figsize=(12, 5))
        
        # Original predictions scatter
        plt.subplot(1, 2, 1)
        plt.scatter(tgts, preds, alpha=0.6)
        lims = [min(tgts.min(), preds.min()), max(tgts.max(), preds.max())]
        plt.plot(lims, lims, 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Scatter Original')
        plt.grid(True)
        
        # Smoothed predictions scatter
        plt.subplot(1, 2, 2)
        plt.scatter(tgts, preds_smooth, alpha=0.6)
        lims = [min(tgts.min(), preds_smooth.min()), max(tgts.max(), preds_smooth.max())]
        plt.plot(lims, lims, 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted(Smooth)')
        plt.title('Scatter Smooth')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(out_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

# ===============================================================
# Dataset & DataProcessor
# ===============================================================
class BatteryDataset(Dataset):
    """PyTorch Dataset for battery time series data"""
    
    def __init__(self, df, seq_len):
        # Extract features: Voltage, Current, Temperature
        feats = df[['Voltage[V]', 'Current[A]', 'Temperature[°C]']].values
        self.X = torch.tensor(feats, dtype=torch.float32)
        # Target: State of Health (SOH)
        self.y = torch.tensor(df['SOH_ZHU'].values, dtype=torch.float32)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.X) - self.seq_len
    
    def __getitem__(self, idx):
        # Return sequence of length seq_len and next SOH value
        return (self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len])

class DataProcessor:
    """Handles data loading, preprocessing, and scaling"""
    
    def __init__(self, data_dir, resample='10min', config=None):
        self.data_dir = Path(data_dir)
        self.resample = resample
        self.scaler = RobustScaler()  # Robust to outliers
        self.config = config
    
    def load_cell_data(self):
        """Load all battery cell data files"""
        files = sorted(self.data_dir.glob('*.parquet'), key=lambda x: int(x.stem.split('_')[-1]))
        return {fp.stem.split('_')[-1]: fp for fp in files}
    
    def process_file(self, fp):
        """Process single battery cell file"""
        # Load relevant columns
        df = pd.read_parquet(fp)[['Testtime[s]', 'Voltage[V]', 'Current[A]', 'Temperature[°C]', 'SOH_ZHU']]
        df = df.dropna().reset_index(drop=True)
        
        # Round timestamps and create datetime index
        df['Testtime[s]'] = df['Testtime[s]'].round().astype(int)
        df['Datetime'] = pd.date_range('2023-02-02', periods=len(df), freq='s')
        
        # Resample to reduce data size and smooth noise
        df = df.set_index('Datetime').resample(self.resample).mean().reset_index()
        df['cell_id'] = fp.stem.split('_')[-1]
        
        return df
    
    def prepare_joint_data(self, cfg):
        """Prepare data for joint training (baseline)"""
        info = self.load_cell_data()
        
        def build(ids): 
            return pd.concat([self.process_file(info[c]) for c in ids], ignore_index=True) if ids else pd.DataFrame()
        
        df_train = build(cfg['train_ids'])
        df_val = build(cfg['val_ids'])
        df_test = self.process_file(info[cfg['test_id']])
        
        logger.info("Joint training - Train IDs: %s, size: %d", 
                   cfg['train_ids'], len(df_train))
        logger.info("Joint training - Val IDs: %s, size: %d", 
                   cfg['val_ids'], len(df_val))
        logger.info("Joint training - Test ID: %s, size: %d", 
                   cfg['test_id'], len(df_test))
        
        # Fit scaler on training data only
        feat_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        self.scaler.fit(df_train[feat_cols])
        logger.info("  (Scaler) Scaler centers: %s", self.scaler.center_)
        logger.info("  (Scaler) Scaler scales: %s", self.scaler.scale_)

        
        def scale(df):
            df2 = df.copy()
            if not df2.empty:
                df2[feat_cols] = self.scaler.transform(df2[feat_cols])
            return df2
        
        return {'train': scale(df_train), 'val': scale(df_val), 'test': scale(df_test)}
    
    def prepare_incremental_data(self, cfg):
        """Prepare data for incremental learning"""
        info = self.load_cell_data()
        
        def build(ids): 
            return pd.concat([self.process_file(info[c]) for c in ids], ignore_index=True) if ids else pd.DataFrame()
        
        # Build datasets for each task
        df0t = build(cfg['task0_train_ids']); df0v_full = build(cfg['task0_val_ids'])
        df1t = build(cfg['task1_train_ids']); df1v_full = build(cfg['task1_val_ids'])
        df2t = build(cfg['task2_train_ids']); df2v_full = build(cfg['task2_val_ids'])
        df_test = self.process_file(info[cfg['test_id']])

        def split_val_test(df_full, split_ratio=0.7):
            n = len(df_full)
            split_idx = int(n * split_ratio)
            df_val  = df_full.iloc[:split_idx].reset_index(drop=True)
            df_test = df_full.iloc[split_idx:].reset_index(drop=True)
            return df_val, df_test
        
        # Split validation data into train/val for each task
        df0v, df0test = split_val_test(df0v_full)
        df1v, df1test = split_val_test(df1v_full)
        df2v, df2test = split_val_test(df2v_full)
        
        dfs_train = [df0t, df1t, df2t]
        dfs_val   = [df0v, df1v, df2v]
        dfs_test  = [df0test, df1test, df2test]

        for i in range(3):
            logger.info(
                "Incremental training - Task %d Train IDs: %s, size: %d",
                i, cfg[f'task{i}_train_ids'], len(dfs_train[i])
            )
            logger.info(
                "Incremental training - Task %d Val IDs: %s, size: %d",
                i, cfg[f'task{i}_val_ids'],   len(dfs_val[i])
            )
            logger.info(
                "Incremental training - Test Task %d size: %d",
                i, len(dfs_test[i])
            )
        
        
        # Fit scaler on first task training data only
        feat_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        self.scaler.fit(dfs_train[0][feat_cols])
        logger.info("  (Scaler) Scaler centers: %s", self.scaler.center_)
        logger.info("  (Scaler) Scaler scales: %s", self.scaler.scale_)

        
        def scale(df):
            df2 = df.copy()
            if not df2.empty:
                df2[feat_cols] = self.scaler.transform(df2[feat_cols])
            return df2
        
        return {
            'task0_train': scale(df0t), 'task0_val': scale(df0v),
            'task1_train': scale(df1t), 'task1_val': scale(df1v),
            'task2_train': scale(df2t), 'task2_val': scale(df2v),
            'test_full': scale(df_test),
            'test_task0': scale(df0test), 'test_task1': scale(df1test), 'test_task2': scale(df2test)
        }

# ===============================================================
# Model & Adapter
# ===============================================================
class AdapterModule(nn.Module):
    def __init__(self, hidden_size, bottle_size, dropout=0.3):
        super().__init__()
        # 1) 下采样投影
        self.down = nn.Linear(hidden_size, bottle_size)
        self.norm1 = nn.LayerNorm(bottle_size)
        # 2) 上采样投影
        self.up   = nn.Linear(bottle_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        # 3) 最后归一化
        self.norm2 = nn.LayerNorm(hidden_size)
        self.act   = nn.GELU()

    def forward(self, x):
        # x: (B, T, hidden)
        # —————————— Bottleneck path ——————————
        # a) 降维 + 归一化
        z = self.norm1(self.down(x))
        # b) 激活 + Dropout
        z = self.act(z)
        z = self.dropout(z)
        # c) 升维
        z = self.up(z)
        z = self.dropout(z)
        # —————————— 残差 + 最后归一化 ——————————
        out = x + z
        return self.norm2(out)


class SOHLSTM(nn.Module):
    """LSTM model for State of Health (SOH) prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), 
            nn.Mish(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        # LSTM forward pass
        out, _ = self.lstm(x)
        # Use only the last time step output
        return self.fc(out[:, -1, :]).squeeze(-1)

class LSTMAdapter(nn.Module):
    """LSTM model with adapter for incremental learning"""
    
    def __init__(self, base_model, adapter_size=32, freeze_backbone=True):
        super().__init__()
        
        # Store base model components
        self.lstm = base_model.lstm
        self.fc = base_model.fc
        
        # Freeze backbone if specified
        if freeze_backbone:
            self.freeze_backbone()
        
        # Add adapter after LSTM
        hidden_size = self.lstm.hidden_size
        self.adapter = AdapterModule(hidden_size, adapter_size)
        
        logger.info("Adapter initialized with size %d", adapter_size)
    
    def freeze_backbone(self):
        """Freeze LSTM and original FC layers"""
        for param in self.lstm.parameters():
            param.requires_grad = False
        # for param in self.fc.parameters():
        #     param.requires_grad = False
        logger.info("Backbone LSTM frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for joint training comparison"""
        for param in self.lstm.parameters():
            param.requires_grad = True
        for param in self.fc.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")
    
    def get_trainable_parameters(self):
        """Get only trainable parameters (adapter)"""
        return list(self.adapter.parameters())
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply adapter to LSTM output
        adapted_out = self.adapter(lstm_out)
        
        # Use last time step for prediction
        last_step = adapted_out[:, -1, :]
        
        # Prediction using original FC
        return self.fc(last_step).squeeze(-1)

# ===============================================================
# Trainer
# ===============================================================
class Trainer:
    """Main training class supporting both pure fine-tuning and adapter fine-tuning"""
    
    def __init__(self, model, device, config, task_dir=None):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.task_dir = Path(task_dir) if task_dir else None
        if self.task_dir: 
            self.task_dir.mkdir(parents=True, exist_ok=True)
    
    def train_task(self, train_loader, val_loader, task_id):
        """
        Train model on a single task.
        Automatically handles adapter vs pure fine-tuning based on model type.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            task_id: Current task identifier
        """
        # Setup optimizer based on model type
        if isinstance(self.model, LSTMAdapter):
            # Adapter mode: only train adapter parameters
            trainable_params = self.model.get_trainable_parameters()
            param_count = sum(p.numel() for p in trainable_params)
            logger.info("Training adapter with %d parameters", param_count)
            opt = torch.optim.Adam(trainable_params, lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        else:
            # Pure fine-tuning: train all parameters
            param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info("Training full model with %d parameters", param_count)
            opt = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)
        
        # Early stopping variables
        best_val = float('inf')
        no_imp = 0
        best_state = None
        
        # Training history tracking
        history = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'lr', 'time']}
        
        # Training loop
        for epoch in tqdm.tqdm(range(self.config.EPOCHS), desc=f"Task{task_id}"):
            start = time.time()
            self.model.train()
            
            # Training phase
            tot_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                
                # Forward pass
                yp = self.model(x)
                
                # Task-specific loss (MSE for regression)
                loss = F.mse_loss(yp, y)
                
                # Backward pass and parameter update
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                opt.step()
                
                tot_loss += loss.item() * x.size(0)
            
            # Calculate epoch average
            train_loss = tot_loss / len(train_loader.dataset)
            
            # Record training history
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            
            lr_cur = opt.param_groups[0]['lr']
            history['lr'].append(lr_cur)
            history['time'].append(time.time() - start)
            
            # Validation evaluation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    val_loss += F.mse_loss(self.model(x.to(self.device)), 
                                         y.to(self.device)).item() * x.size(0)
            
            val_loss = val_loss / len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            sched.step(val_loss)
            
            # Logging
            logger.info("Epoch %d train=%.4e val=%.4e lr=%.2e time=%.2fs",
                       epoch, train_loss, val_loss, lr_cur, history['time'][-1])
            
            # Early stopping check
            if val_loss < best_val:
                best_val = val_loss
                no_imp = 0
                best_state = copy.deepcopy(self.model.state_dict())
                
                # Save best model checkpoint
                if self.task_dir: 
                    torch.save({'model_state': best_state}, 
                              self.task_dir / f"task{task_id}_best.pt")
            else:
                no_imp += 1
                if no_imp >= self.config.PATIENCE:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
        
        # Restore best model
        if best_state: 
            self.model.load_state_dict(best_state)
        
        return history
    
    def evaluate(self, loader, alpha=0.1, log=True):
        """
        Evaluate model performance on a dataset.
        
        Args:
            loader: Data loader for evaluation
            alpha: Smoothing factor for exponential smoothing
            log: Whether to log results
            
        Returns:
            predictions, targets, metrics dictionary
        """
        self.model.eval()
        preds = []
        tgts = []
        
        with torch.no_grad():
            for x, y in loader:
                preds.append(self.model(x.to(self.device)).cpu().numpy())
                tgts.append(y.numpy())
        
        preds = np.concatenate(preds)
        tgts = np.concatenate(tgts)
        
        # Calculate metrics
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(tgts, preds)),
            'MAE': mean_absolute_error(tgts, preds),
            'R2': r2_score(tgts, preds)
        }
        
        # Smoothed predictions metrics
        preds_smooth = pd.Series(preds).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        metrics['RMSE_smooth'] = np.sqrt(mean_squared_error(tgts, preds_smooth))
        metrics['MAE_smooth'] = mean_absolute_error(tgts, preds_smooth)
        metrics['R2_smooth'] = r2_score(tgts, preds_smooth)
        
        if log: 
            logger.info("Eval RMSE %.4e MAE %.4e R2 %.4f", 
                       metrics['RMSE'], metrics['MAE'], metrics['R2'])
        
        return preds, tgts, metrics

# ===============================================================
# Utilities
# ===============================================================
def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def create_dataloaders(datasets, seq_len, batch_size):
    """Create PyTorch DataLoaders from processed datasets"""
    loaders = {}
    for k, df in datasets.items():
        if not df.empty and any(x in k for x in ['train', 'val', 'test']):
            ds = BatteryDataset(df, seq_len)
            loaders[k] = DataLoader(ds, batch_size=batch_size, shuffle=('train' in k))
    return loaders

def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    log_path = log_dir / 'train.log'
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) 
              for h in logger.handlers):
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(fh)
    
    return logger

# ===============================================================
# Training Pipelines
# ===============================================================
def joint_training(config):
    """
    Joint training baseline: train on all tasks simultaneously.
    """
    logger.info("==== Joint Training (Baseline) ====")
    
    # Setup directories
    joint_dir = config.BASE_DIR / "joint_training"
    ckpt = joint_dir / 'checkpoints'
    res = joint_dir / 'results'
    ckpt.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    dp = DataProcessor(config.DATA_DIR, config.RESAMPLE, config)
    data = dp.prepare_joint_data(config.joint_datasets)
    loaders = create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    
    # Initialize model and trainer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    trainer = Trainer(model, device, config, ckpt)
    
    # Train model
    history = trainer.train_task(loaders['train'], loaders['val'], 0)
    
    # Save training history and create visualizations
    pd.DataFrame(history).to_csv(ckpt / 'training_history.csv', index=False)
    Visualizer.plot_losses(history, res)
    
    # Evaluate on test set
    preds, tgts, metrics = trainer.evaluate(loaders['test'], alpha=config.ALPHA)
    Visualizer.plot_predictions(preds, tgts, metrics, res, alpha=config.ALPHA)
    Visualizer.plot_prediction_scatter(preds, tgts, res, alpha=config.ALPHA)
    
    # Save test metrics
    pd.DataFrame([metrics]).to_csv(res / 'test_metrics.csv', index=False)
    
    logger.info("==== Joint Training Complete ====")

def incremental_training(config):
    """
    Incremental training supporting both pure fine-tuning and adapter fine-tuning.
    """
    method_name = "Adapter Fine-tuning" if config.USE_ADAPTER else "Pure Fine-tuning"
    logger.info("==== Incremental Training (%s) ====", method_name)
    
    # Setup directories
    inc_dir = config.BASE_DIR / "incremental_training"
    inc_dir.mkdir(parents=True, exist_ok=True)
    
    # Get number of tasks from config
    num_tasks = config.NUM_TASKS
    logger.info("Number of tasks: %d", num_tasks)
    
    # Prepare incremental learning data
    dp = DataProcessor(config.DATA_DIR, config.RESAMPLE, config)
    data = dp.prepare_incremental_data(config.incremental_datasets)
    loaders = create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    
    # Initialize base model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    logger.info("Initialized base LSTM model")
    
    # Sequential task training
    for task_idx in range(num_tasks):
        task_name = f"task{task_idx}"
        
        logger.info("--- %s ---", task_name)
        
        # Setup task directory
        task_dir = inc_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Set task-specific random seed for reproducibility
        set_seed(config.SEED + task_idx)

        if task_idx == 0:
            task0_checkpoint_path = Path("task0_best.pt")  # 修改为您的实际路径
            
            if task0_checkpoint_path.exists():
                logger.info("Loading pre-trained Task 0 model from: %s", task0_checkpoint_path)
                checkpoint = torch.load(task0_checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state'])
                
                torch.save({'model_state': model.state_dict()},
                    task_dir / 'task0_best.pt')
                logger.info("Successfully loaded Task 0 model, skipping Task 0 training")
                
                # 跳过Task 0的训练，直接continue到下一个任务
                continue
        
        # Convert to adapter mode after Task 0 (if enabled)
        if config.USE_ADAPTER and task_idx == 1:
            logger.info("Converting to adapter mode after Task 0...")
            model = LSTMAdapter(
                model, 
                adapter_size=config.ADAPTER_SIZE,
                freeze_backbone=config.FREEZE_BACKBONE
            ).to(device)
            logger.info("Converted to adapter with size %d, backbone frozen: %s", 
                       config.ADAPTER_SIZE, config.FREEZE_BACKBONE)

        trainer = Trainer(model, device, config, task_dir)
        
        # Train on current task
        history = trainer.train_task(
            loaders[f"{task_name}_train"], 
            loaders[f"{task_name}_val"], 
            task_idx
        )
        
        # Save training history and visualizations
        pd.DataFrame(history).to_csv(task_dir / 'training_history.csv', index=False)
        Visualizer.plot_losses(history, task_dir)
        
        logger.info("Task %d completed.", task_idx)
    
    logger.info("==== Incremental Training Complete ====")
    
    # Comprehensive evaluation phase
    return evaluate_incremental_learning(config, inc_dir, num_tasks, loaders, device)

def evaluate_incremental_learning(config, inc_dir, num_tasks, loaders, device):
    """
    Comprehensive evaluation of incremental learning performance.
    """
    logger.info("==== Starting Comprehensive Evaluation ====")
    
    # Create evaluation directory
    eval_dir = inc_dir / 'metrics'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Performance matrix R[i][j] = performance of model after task i on test set of task j
    R_matrix = np.zeros((num_tasks, num_tasks))
    metrics_summary = []
    
    # Evaluate each trained model on all test sets
    for trained_task_idx in range(num_tasks):
        logger.info("Evaluating model trained after task %d...", trained_task_idx)
        
        # Load model checkpoint from this training stage
        checkpoint_path = inc_dir / f"task{trained_task_idx}" / f"task{trained_task_idx}_best.pt"
        if not checkpoint_path.exists():
            logger.error("Checkpoint not found: %s", checkpoint_path)
            continue
        
        # Create appropriate model based on config and task
        if config.USE_ADAPTER and trained_task_idx >= 1:
            # For tasks 1+, need to recreate the adapter structure
            base_model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
            eval_model = LSTMAdapter(
                base_model,
                adapter_size=config.ADAPTER_SIZE,
                freeze_backbone=config.FREEZE_BACKBONE
            ).to(device)
        else:
            # For task 0 or non-adapter mode, use base model
            eval_model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
        
        # Load trained weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        eval_model.load_state_dict(checkpoint['model_state'])
        eval_model.eval()
        
        # Create temporary trainer for evaluation
        eval_trainer = Trainer(eval_model, device, config, None)
        
        # Evaluate on full test set (complete battery degradation curve)
        full_preds, full_targets, full_metrics = eval_trainer.evaluate(
            loaders['test_full'], alpha=config.ALPHA, log=False
        )
        
        # Save full test set predictions plot
        Visualizer.plot_predictions(
            full_preds, full_targets, full_metrics,
            inc_dir / f"task{trained_task_idx}", alpha=config.ALPHA
        )
        
        # Record full test performance
        metrics_summary.append({
            "trained_after_task": f"task{trained_task_idx}",
            "evaluated_on_task": "full_test",
            "trained_task_idx": trained_task_idx,
            "eval_task_idx": -1,  # -1 indicates full test set
            "MAE": full_metrics['MAE'],
            "MAE_smooth": full_metrics['MAE_smooth'],
            "RMSE": full_metrics['RMSE'],
            "RMSE_smooth": full_metrics['RMSE_smooth'],
            "R2": full_metrics['R2'],
            "R2_smooth": full_metrics['R2_smooth'],
            "R_value": -full_metrics['MAE']  # Negative MAE for maximization
        })
        
        # Evaluate on each task-specific test set
        for eval_task_idx in range(num_tasks):
            test_loader_key = f'test_task{eval_task_idx}'
            test_loader = loaders[test_loader_key]
            
            # Get predictions and metrics for this task
            _, _, task_metrics = eval_trainer.evaluate(test_loader, alpha=config.ALPHA, log=False)
            
            # Store performance in R matrix
            R_matrix[trained_task_idx][eval_task_idx] = -task_metrics['MAE']
            
            # Record detailed metrics
            metrics_summary.append({
                "trained_after_task": f"task{trained_task_idx}",
                "evaluated_on_task": f"test_task{eval_task_idx}",
                "trained_task_idx": trained_task_idx,
                "eval_task_idx": eval_task_idx,
                "MAE": task_metrics['MAE'],
                "MAE_smooth": task_metrics['MAE_smooth'],
                "RMSE": task_metrics['RMSE'],
                "RMSE_smooth": task_metrics['RMSE_smooth'],
                "R2": task_metrics['R2'],
                "R2_smooth": task_metrics['R2_smooth'],
                "R_value": R_matrix[trained_task_idx][eval_task_idx]
            })
            
            logger.info("  Task %d -> Test Task %d: MAE=%.4e, R2=%.4f", 
                       trained_task_idx, eval_task_idx, 
                       task_metrics['MAE'], task_metrics['R2'])
    
    # ===============================================================
    # Calculate Continual Learning Metrics
    # ===============================================================
    logger.info("==== Computing Continual Learning Metrics ====")
    
    # Compute baseline performance for Forward Transfer calculation
    logger.info("Computing random initialization baselines...")
    torch.manual_seed(config.SEED + 999)  # Different seed for baseline
    
    baseline_model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    baseline_trainer = Trainer(baseline_model, device, config, None)
    
    baseline_performance = np.zeros(num_tasks)
    for j in range(num_tasks):
        test_loader = loaders[f'test_task{j}']
        _, _, baseline_metrics = baseline_trainer.evaluate(test_loader, alpha=config.ALPHA, log=False)
        baseline_performance[j] = -baseline_metrics['MAE']
        logger.info("  Baseline Task %d: R=%.4f", j, baseline_performance[j])
    
    # Calculate BWT (Backward Transfer)
    # BWT measures how much old task performance degrades after learning new tasks
    if num_tasks > 1:
        bwt_scores = []
        for i in range(num_tasks - 1):  # Tasks 0 to T-2
            final_perf = R_matrix[num_tasks - 1, i]  # Performance after all tasks
            when_learned_perf = R_matrix[i, i]       # Performance when task was learned
            bwt_scores.append(final_perf - when_learned_perf)
        BWT = np.mean(bwt_scores)
    else:
        BWT = 0.0
    
    # Calculate FWT (Forward Transfer)  
    # FWT measures how much learning previous tasks helps with new tasks
    if num_tasks > 1:
        fwt_scores = []
        for i in range(1, num_tasks):  # Tasks 1 to T-1
            when_learned_perf = R_matrix[i - 1, i]  # Performance on task i after learning task i-1
            baseline_perf = baseline_performance[i]  # Random initialization performance
            fwt_scores.append(when_learned_perf - baseline_perf)
        FWT = np.mean(fwt_scores)
    else:
        FWT = 0.0
    
    # Calculate ACC (Average Accuracy)
    # ACC measures overall performance: average final performance across all tasks
    ACC = np.mean(R_matrix[num_tasks - 1, :])
    
    # Compile continual learning metrics
    continual_learning_metrics = {
        "BWT": BWT,  # Backward Transfer (negative = forgetting)
        "FWT": FWT,  # Forward Transfer (positive = beneficial transfer)
        "ACC": ACC,  # Average final accuracy
        "num_tasks": num_tasks
    }
    
    # Log results
    logger.info("==== Continual Learning Results ====")
    logger.info("BWT (Backward Transfer): %.4f %s", BWT, 
               "(less negative = less forgetting)" if BWT < 0 else "(positive = backward gain)")
    logger.info("FWT (Forward Transfer): %.4f %s", FWT,
               "(positive = beneficial transfer)" if FWT > 0 else "(negative = interference)")
    logger.info("ACC (Average Accuracy): %.4f", ACC)
    
    # Print R matrix for detailed inspection
    logger.info("==== Performance Matrix R[i][j] ====")
    logger.info("Rows: trained after task i, Columns: evaluated on task j")
    header = "       " + " ".join([f"Task{j:2d}" for j in range(num_tasks)])
    logger.info(header)
    for i in range(num_tasks):
        row_values = " ".join([f"{R_matrix[i,j]:7.4f}" for j in range(num_tasks)])
        logger.info("Task%2d: %s", i, row_values)
    
    # ===============================================================
    # Save All Results
    # ===============================================================
    
    # Save detailed evaluation metrics
    summary_df = pd.DataFrame(metrics_summary)
    summary_df.to_csv(eval_dir / 'detailed_evaluation_results.csv', index=False)
    
    # Save continual learning metrics summary
    cl_metrics_df = pd.DataFrame([continual_learning_metrics])
    cl_metrics_df.to_csv(eval_dir / 'continual_learning_metrics.csv', index=False)
    
    # Save performance matrix
    r_matrix_df = pd.DataFrame(
        R_matrix, 
        index=[f"after_task{i}" for i in range(num_tasks)],
        columns=[f"eval_task{j}" for j in range(num_tasks)]
    )
    r_matrix_df.to_csv(eval_dir / 'R_matrix.csv')
    
    # Save baseline performance for reference
    baseline_df = pd.DataFrame({
        'task': [f'task{i}' for i in range(num_tasks)],
        'baseline_performance': baseline_performance
    })
    baseline_df.to_csv(eval_dir / 'baseline_performance.csv', index=False)
    
    logger.info("==== Evaluation Complete ====")
    logger.info("All results saved to: %s", eval_dir)
    
    return continual_learning_metrics, R_matrix

# ===============================================================
# Main Pipeline
# ===============================================================
def main():
    """Main execution pipeline"""
    # Load configuration
    config = Config()
    
    # Set random seed BEFORE creating data splits
    set_seed(config.SEED)
    
    # Setup directories and logging
    config.BASE_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(config.BASE_DIR)
    
    # Save configuration for reproducibility
    config.save(config.BASE_DIR / 'config.json')
    
    # Log experiment setup
    logger.info("==== Experiment Setup ====")
    logger.info("Mode: %s", config.MODE)
    logger.info("Method: %s", config.Info["method"])
    logger.info("Number of tasks: %d", config.NUM_TASKS)
    logger.info("Random seed: %d", config.SEED)
    if config.USE_ADAPTER:
        logger.info("Adapter size: %d", config.ADAPTER_SIZE)
        logger.info("Freeze backbone: %s", config.FREEZE_BACKBONE)
    logger.info("Base directory: %s", config.BASE_DIR)
    
    # Run appropriate training pipeline
    if config.MODE == 'joint':
        joint_training(config)
    elif config.MODE == 'incremental':
        cl_metrics, r_matrix = incremental_training(config)
        
        # Log final summary
        logger.info("==== Final Summary ====")
        logger.info("Continual Learning Metrics:")
        for metric, value in cl_metrics.items():
            logger.info("  %s: %.4f", metric, value)
    else:
        logger.error("Unknown mode: %s. Use 'joint' or 'incremental'", config.MODE)
        return
    
    logger.info("==== Experiment Complete ====")

if __name__ == '__main__':
    main()