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
# LoRA Implementation
# ===============================================================
class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer implementation"""
    
    def __init__(self, original_layer, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Get dimensions from original layer
        if isinstance(original_layer, nn.Linear):
            in_features = original_layer.in_features
            out_features = original_layer.out_features
        elif isinstance(original_layer, nn.LSTM):
            # For LSTM, we'll adapt the input-to-hidden weights
            in_features = original_layer.input_size
            out_features = original_layer.hidden_size * 4  # 4 gates
        else:
            raise ValueError(f"Unsupported layer type: {type(original_layer)}")
        
        # LoRA matrices: W = W0 + BA where B is (out, rank) and A is (rank, in)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Get original output
        if isinstance(self.original_layer, nn.Linear):
            original_output = self.original_layer(x)
            # Add LoRA adaptation: alpha/rank * B @ A @ x
            lora_output = (self.alpha / self.rank) * F.linear(x, self.lora_B @ self.lora_A)
            return original_output + self.dropout(lora_output)
        else:
            # For LSTM, this is more complex - we'll handle it differently
            return self.original_layer(x)

class LoRALinear(nn.Module):
    """LoRA adapted Linear layer"""
    
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        original_output = self.linear(x)
        lora_output = (self.alpha / self.rank) * F.linear(x, self.lora_B @ self.lora_A)
        return original_output + self.dropout(lora_output)

class LoRALSTM(nn.Module):
    """LoRA adapted LSTM layer"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, rank=16, alpha=32, 
                 dropout=0.1, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rank = rank
        self.alpha = alpha
        self.batch_first = batch_first
        
        # Original LSTM (frozen)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=batch_first, dropout=dropout if num_layers > 1 else 0)
        for param in self.lstm.parameters():
            param.requires_grad = False
        
        # LoRA adaptations for each layer
        self.lora_layers = nn.ModuleList()
        for layer in range(num_layers):
            # For each LSTM layer, we have input-to-hidden and hidden-to-hidden weights
            layer_input_size = input_size if layer == 0 else hidden_size
            
            # LoRA for input-to-hidden (4 gates: i, f, g, o)
            lora_ih = nn.ModuleDict({
                'lora_A_ih': nn.Parameter(torch.randn(rank, layer_input_size) * 0.01),
                'lora_B_ih': nn.Parameter(torch.zeros(4 * hidden_size, rank))
            })
            
            # LoRA for hidden-to-hidden (4 gates: i, f, g, o)
            lora_hh = nn.ModuleDict({
                'lora_A_hh': nn.Parameter(torch.randn(rank, hidden_size) * 0.01),
                'lora_B_hh': nn.Parameter(torch.zeros(4 * hidden_size, rank))
            })
            
            self.lora_layers.append(nn.ModuleDict({
                'ih': lora_ih,
                'hh': lora_hh
            }))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        # Get original LSTM output
        lstm_out, lstm_hidden = self.lstm(x, hidden)
        
        # For simplicity, we'll add LoRA adaptation as a residual connection
        # In a full implementation, you'd need to manually implement LSTM forward pass
        # with LoRA adaptations integrated into each gate computation
        
        # Apply a simple LoRA-like adaptation to the output
        batch_size, seq_len, _ = lstm_out.shape
        
        # Reshape for adaptation
        lstm_out_flat = lstm_out.reshape(-1, self.hidden_size)
        
        # Apply first layer LoRA adaptation as an example
        if len(self.lora_layers) > 0:
            lora_layer = self.lora_layers[0]['hh']  # Use hidden-to-hidden for output adaptation
            lora_adaptation = (self.alpha / self.rank) * F.linear(
                lstm_out_flat, 
                lora_layer['lora_B_hh'][:self.hidden_size] @ lora_layer['lora_A_hh']
            )
            lstm_out_flat = lstm_out_flat + self.dropout(lora_adaptation)
        
        # Reshape back
        lstm_out = lstm_out_flat.reshape(batch_size, seq_len, self.hidden_size)
        
        return lstm_out, lstm_hidden

# ===============================================================
# Configuration Class (Updated with LoRA parameters)
# ===============================================================
class Config:
    """Configuration class for continual learning experiments with LoRA support"""
    def __init__(self, **kwargs):
        # Training mode: 'joint' for baseline, 'incremental' for continual learning
        self.MODE = 'incremental'  # Changed default to incremental for LoRA
        
        # Directory structure
        self.BASE_DIR = Path.cwd() / "lora_experiments"
        self.DATA_DIR = Path('../01_Datenaufbereitung/Output/Calculated/')
        
        # Model hyperparameters
        self.SEQUENCE_LENGTH = 720  # Input sequence length for LSTM
        self.HIDDEN_SIZE = 128      # LSTM hidden state size
        self.NUM_LAYERS = 2         # Number of LSTM layers
        self.DROPOUT = 0.3          # Dropout rate
        
        # LoRA hyperparameters
        self.USE_LORA = True        # Enable/disable LoRA
        self.LORA_RANK = 16         # LoRA rank (lower = more compression)
        self.LORA_ALPHA = 32        # LoRA scaling factor
        self.LORA_DROPOUT = 0.1     # LoRA-specific dropout
        self.LORA_TARGET_MODULES = ['lstm', 'fc']  # Which modules to adapt with LoRA
        
        # Training hyperparameters
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-3   # Higher LR for LoRA fine-tuning
        self.EPOCHS = 200
        self.PATIENCE = 20          # Early stopping patience
        self.WEIGHT_DECAY = 1e-6
        
        # Data preprocessing
        self.SCALER = "RobustScaler"
        self.RESAMPLE = '10min'     # Time series resampling frequency
        self.ALPHA = 0.1            # Smoothing factor for predictions
        
        # Continual Learning parameters
        self.NUM_TASKS = 3          # Number of incremental tasks

        # Random seed for reproducibility
        self.SEED = 42
        
        # Dataset splits for joint training (baseline)
        self.joint_datasets = {
            'train_ids': ['03', '05', '07', '09', '11', '15', '21', '23', '25', '27', '29'],
            'val_ids': ['01', '19', '13'],
            'test_id': '17'
        }
        
        # Dataset splits for incremental learning
        self.incremental_datasets = self._create_incremental_splits()
        
        # Experiment metadata
        self.Info = {
            "method": self.MODE,
            "resample": self.RESAMPLE,
            "scaler": "RobustScaler - fit on base train",
            "smooth_alpha": self.ALPHA,
            "num_tasks": self.NUM_TASKS,
            "use_lora": self.USE_LORA,
            "lora_rank": self.LORA_RANK,
            "lora_alpha": self.LORA_ALPHA
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
        """Create incremental learning splits with mixed Task0 and random Task1&2"""
        random.seed(self.SEED)
        
        normal_cells = ['03', '05', '07', '27']
        fast_cells = ['21', '23', '25']
        faster_cells = ['09', '11', '15', '29']
        
        task0_normal = random.sample(normal_cells, 3)
        task0_fast = random.sample(fast_cells, 1)
        task0_faster = random.sample(faster_cells, 1)
        task0_train_ids = task0_normal + task0_fast + task0_faster
        
        task1_train_ids = ([c for c in fast_cells if c not in task0_fast] +
                            [c for c in normal_cells if c not in task0_normal])
        
        task2_train_ids = [c for c in faster_cells if c not in task0_faster]
        
        logger.info("=== Data Split Strategy ===")
        logger.info("Task 0 (Mixed): %s", task0_train_ids)
        logger.info("Task 1 (Random): %s", task1_train_ids)
        logger.info("Task 2 (Random): %s", task2_train_ids)
        
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
# Visualization Functions (Unchanged)
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
        
        plt.subplot(1, 2, 1)
        plt.scatter(tgts, preds, alpha=0.6)
        lims = [min(tgts.min(), preds.min()), max(tgts.max(), preds.max())]
        plt.plot(lims, lims, 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Scatter Original')
        plt.grid(True)
        
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
# Dataset & DataProcessor (Unchanged)
# ===============================================================
class BatteryDataset(Dataset):
    """PyTorch Dataset for battery time series data"""
    
    def __init__(self, df, seq_len):
        feats = df[['Voltage[V]', 'Current[A]', 'Temperature[°C]']].values
        self.X = torch.tensor(feats, dtype=torch.float32)
        self.y = torch.tensor(df['SOH_ZHU'].values, dtype=torch.float32)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.X) - self.seq_len
    
    def __getitem__(self, idx):
        return (self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len])

class DataProcessor:
    """Handles data loading, preprocessing, and scaling"""
    
    def __init__(self, data_dir, resample='10min', config=None):
        self.data_dir = Path(data_dir)
        self.resample = resample
        self.scaler = RobustScaler()
        self.config = config
    
    def load_cell_data(self):
        """Load all battery cell data files"""
        files = sorted(self.data_dir.glob('*.parquet'), key=lambda x: int(x.stem.split('_')[-1]))
        return {fp.stem.split('_')[-1]: fp for fp in files}
    
    def process_file(self, fp):
        """Process single battery cell file"""
        df = pd.read_parquet(fp)[['Testtime[s]', 'Voltage[V]', 'Current[A]', 'Temperature[°C]', 'SOH_ZHU']]
        df = df.dropna().reset_index(drop=True)
        
        df['Testtime[s]'] = df['Testtime[s]'].round().astype(int)
        df['Datetime'] = pd.date_range('2023-02-02', periods=len(df), freq='s')
        
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
        
        logger.info("Joint training - Train IDs: %s, size: %d", cfg['train_ids'], len(df_train))
        logger.info("Joint training - Val IDs: %s, size: %d", cfg['val_ids'], len(df_val))
        logger.info("Joint training - Test ID: %s, size: %d", cfg['test_id'], len(df_test))
        
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
        
        df0v, df0test = split_val_test(df0v_full)
        df1v, df1test = split_val_test(df1v_full)
        df2v, df2test = split_val_test(df2v_full)
        
        dfs_train = [df0t, df1t, df2t]
        dfs_val   = [df0v, df1v, df2v]
        dfs_test  = [df0test, df1test, df2test]

        for i in range(3):
            logger.info("Incremental training - Task %d Train IDs: %s, size: %d",
                        i, cfg[f'task{i}_train_ids'], len(dfs_train[i]))
            logger.info("Incremental training - Task %d Val IDs: %s, size: %d",
                        i, cfg[f'task{i}_val_ids'], len(dfs_val[i]))
            logger.info("Incremental training - Test Task %d size: %d", i, len(dfs_test[i]))
        
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
# LoRA-Enhanced Model
# ===============================================================
class LoRASOHLSTM(nn.Module):
    """LoRA-enhanced LSTM model for State of Health (SOH) prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, config=None):
        super().__init__()
        self.config = config
        self.use_lora = config.USE_LORA if config else False
        
        if self.use_lora and 'lstm' in config.LORA_TARGET_MODULES:
            # Use LoRA-adapted LSTM
            self.lstm = LoRALSTM(
                input_size, hidden_size, num_layers,
                rank=config.LORA_RANK,
                alpha=config.LORA_ALPHA,
                dropout=config.LORA_DROPOUT,
                batch_first=True
            )
        else:
            # Use standard LSTM
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Adaptive fusion layer
        if self.use_lora and 'fc' in config.LORA_TARGET_MODULES:
            self.adaptive_fusion = LoRALinear(
                hidden_size, 1,
                rank=config.LORA_RANK,
                alpha=config.LORA_ALPHA,
                dropout=config.LORA_DROPOUT
            )
            
            self.fc = nn.Sequential(
                LoRALinear(hidden_size, hidden_size // 2,
                          rank=config.LORA_RANK,
                          alpha=config.LORA_ALPHA,
                          dropout=config.LORA_DROPOUT),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                LoRALinear(hidden_size // 2, 1,
                          rank=config.LORA_RANK,
                          alpha=config.LORA_ALPHA,
                          dropout=config.LORA_DROPOUT)
            )
        else:
            self.adaptive_fusion = nn.Linear(hidden_size, 1)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2), 
                nn.LeakyReLU(), 
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        avg_step = out.mean(dim=1)
    
        adjustment = torch.tanh(self.adaptive_fusion(last_step))
        combined = avg_step + adjustment * (last_step - avg_step)
        
        return self.fc(combined).squeeze(-1)
    
    def get_lora_parameters(self):
        """Get only LoRA parameters for optimization"""
        lora_params = []
        for name, module in self.named_modules():
            if isinstance(module, (LoRALinear, LoRALSTM)):
                for param_name, param in module.named_parameters():
                    if 'lora_' in param_name:
                        lora_params.append(param)
        return lora_params
    
    def get_trainable_parameters(self):
        """Get all trainable parameters"""
        if self.use_lora:
            return self.get_lora_parameters()
        else:
            return self.parameters()
    
    def print_trainable_parameters(self):
        """Print number of trainable parameters"""
        if self.use_lora:
            lora_params = sum(p.numel() for p in self.get_lora_parameters())
            total_params = sum(p.numel() for p in self.parameters())
            logger.info(f"LoRA trainable parameters: {lora_params:,}")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable percentage: {100 * lora_params / total_params:.2f}%")
        else:
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Total parameters: {total_params:,}")

# Backward compatibility
class SOHLSTM(LoRASOHLSTM):
    """Backward compatibility wrapper"""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        # Create a dummy config for non-LoRA usage
        class DummyConfig:
            USE_LORA = False
        
        super().__init__(input_size, hidden_size, num_layers, dropout, DummyConfig())

# ===============================================================
# LoRA-Enhanced Trainer
# ===============================================================
class LoRATrainer:
    """Enhanced trainer with LoRA support"""
    
    def __init__(self, model, device, config, task_dir=None):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.task_dir = Path(task_dir) if task_dir else None
        if self.task_dir: 
            self.task_dir.mkdir(parents=True, exist_ok=True)
        
        # Print model info
        self.model.print_trainable_parameters()
    
    def train_task(self, train_loader, val_loader, task_id):
        """Train model on a single task using LoRA fine-tuning"""
        
        # Setup optimizer - only optimize LoRA parameters if using LoRA
        if self.config.USE_LORA:
            trainable_params = self.model.get_trainable_parameters()
            logger.info(f"Task {task_id}: Optimizing LoRA parameters only")
        else:
            trainable_params = self.model.parameters()
            logger.info(f"Task {task_id}: Optimizing all parameters")
        
        opt = torch.optim.Adam(trainable_params, 
                              lr=self.config.LEARNING_RATE,
                              weight_decay=self.config.WEIGHT_DECAY)
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
                nn.utils.clip_grad_norm_(trainable_params, 1)
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
                if self.config.USE_LORA:
                    # Save only LoRA parameters
                    lora_state = {}
                    for name, param in self.model.named_parameters():
                        if 'lora_' in name and param.requires_grad:
                            lora_state[name] = param.data.clone()
                    best_state = lora_state
                else:
                    best_state = copy.deepcopy(self.model.state_dict())
                
                # Save best model checkpoint
                if self.task_dir: 
                    if self.config.USE_LORA:
                        torch.save({'lora_state': best_state}, 
                                  self.task_dir / f"task{task_id}_lora_best.pt")
                    else:
                        torch.save({'model_state': best_state}, 
                                  self.task_dir / f"task{task_id}_best.pt")
            else:
                no_imp += 1
                if no_imp >= self.config.PATIENCE:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
        
        # Restore best model
        if best_state:
            if self.config.USE_LORA:
                # Load only LoRA parameters
                model_dict = self.model.state_dict()
                model_dict.update(best_state)
                self.model.load_state_dict(model_dict)
            else:
                self.model.load_state_dict(best_state)
        
        return history
    
    def evaluate(self, loader, alpha=0.1, log=True):
        """Evaluate model performance on a dataset"""
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
    
    def save_lora_checkpoint(self, path, task_id):
        """Save LoRA parameters checkpoint"""
        if self.config.USE_LORA:
            lora_state = {}
            for name, param in self.model.named_parameters():
                if 'lora_' in name:
                    lora_state[name] = param.data.clone()
            
            checkpoint = {
                'lora_state': lora_state,
                'task_id': task_id,
                'config': self.config.__dict__
            }
            torch.save(checkpoint, path)
            logger.info(f"LoRA checkpoint saved: {path}")
        else:
            # Save full model for non-LoRA case
            torch.save({'model_state': self.model.state_dict(), 'task_id': task_id}, path)
    
    def load_lora_checkpoint(self, path):
        """Load LoRA parameters checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'lora_state' in checkpoint and self.config.USE_LORA:
            # Load LoRA parameters
            model_dict = self.model.state_dict()
            model_dict.update(checkpoint['lora_state'])
            self.model.load_state_dict(model_dict)
            logger.info(f"LoRA checkpoint loaded: {path}")
        elif 'model_state' in checkpoint:
            # Load full model
            self.model.load_state_dict(checkpoint['model_state'])
            logger.info(f"Model checkpoint loaded: {path}")

# Backward compatibility
class Trainer(LoRATrainer):
    """Backward compatibility wrapper"""
    pass

# ===============================================================
# Utilities (Updated for LoRA)
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
# Training Pipelines (Updated for LoRA)
# ===============================================================
def joint_training(config):
    """Joint training baseline with optional LoRA support"""
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
    model = LoRASOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT, config)
    trainer = LoRATrainer(model, device, config, ckpt)
    
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
    """LoRA-enhanced incremental training"""
    logger.info("==== LoRA-Enhanced Incremental Training ====")
    
    # Setup directories
    inc_dir = config.BASE_DIR / "incremental_training"
    inc_dir.mkdir(parents=True, exist_ok=True)
    
    # Get number of tasks from config
    num_tasks = config.NUM_TASKS
    logger.info("Number of tasks: %d", num_tasks)
    logger.info("Using LoRA: %s", config.USE_LORA)
    if config.USE_LORA:
        logger.info("LoRA rank: %d, alpha: %d", config.LORA_RANK, config.LORA_ALPHA)
        logger.info("LoRA target modules: %s", config.LORA_TARGET_MODULES)
    
    # Prepare incremental learning data
    dp = DataProcessor(config.DATA_DIR, config.RESAMPLE, config)
    data = dp.prepare_incremental_data(config.incremental_datasets)
    loaders = create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    
    # Initialize model and trainer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LoRASOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT, config)
    trainer = LoRATrainer(model, device, config, inc_dir)
    
    # Sequential task training
    for task_idx in range(num_tasks):
        task_name = f"task{task_idx}"
        
        logger.info("--- %s ---", task_name)
        
        # Setup task directory
        task_dir = inc_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        trainer.task_dir = task_dir
        
        # Set task-specific random seed for reproducibility
        set_seed(config.SEED + task_idx)
        
        # For LoRA: Reset LoRA parameters for new task (optional - can be configurable)
        if config.USE_LORA and task_idx > 0:
            # Option 1: Keep accumulated LoRA adaptations (continual learning)
            # Option 2: Reset LoRA parameters (task-specific adaptation)
            # Here we keep accumulated adaptations for continual learning
            logger.info("Task %d: Continuing with accumulated LoRA adaptations", task_idx)
        
        # Train on current task
        history = trainer.train_task(
            loaders[f"{task_name}_train"], 
            loaders[f"{task_name}_val"], 
            task_idx
        )
        
        # Save training history and visualizations
        pd.DataFrame(history).to_csv(task_dir / 'training_history.csv', index=False)
        Visualizer.plot_losses(history, task_dir)
        
        # Save LoRA checkpoint
        if config.USE_LORA:
            trainer.save_lora_checkpoint(task_dir / f"task{task_idx}_lora_checkpoint.pt", task_idx)
        
        logger.info("Task %d completed.", task_idx)
    
    logger.info("==== LoRA-Enhanced Incremental Training Complete ====")
    
    # Comprehensive evaluation phase
    return evaluate_incremental_learning(config, inc_dir, num_tasks, loaders, device)

def evaluate_incremental_learning(config, inc_dir, num_tasks, loaders, device):
    """Comprehensive evaluation of LoRA-enhanced incremental learning performance"""
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
        if config.USE_LORA:
            checkpoint_path = inc_dir / f"task{trained_task_idx}" / f"task{trained_task_idx}_lora_checkpoint.pt"
        else:
            checkpoint_path = inc_dir / f"task{trained_task_idx}" / f"task{trained_task_idx}_best.pt"
        
        if not checkpoint_path.exists():
            logger.error("Checkpoint not found: %s", checkpoint_path)
            continue
        
        # Create fresh model and load trained weights
        eval_model = LoRASOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT, config)
        eval_model = eval_model.to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if config.USE_LORA and 'lora_state' in checkpoint:
            # Load LoRA parameters
            model_dict = eval_model.state_dict()
            model_dict.update(checkpoint['lora_state'])
            eval_model.load_state_dict(model_dict)
        elif 'model_state' in checkpoint:
            # Load full model
            eval_model.load_state_dict(checkpoint['model_state'])
        
        eval_model.eval()
        
        # Create temporary trainer for evaluation
        eval_trainer = LoRATrainer(eval_model, device, config, None)
        
        # Evaluate on full test set (complete battery degradation curve)
        full_preds, full_targets, full_metrics = eval_trainer.evaluate(
            loaders['test_full'], alpha=config.ALPHA, log=False
        )
        logger.info("Full test set evaluation: MAE=%.4e, R2=%.4f", 
                    full_metrics['MAE'], full_metrics['R2'])
        
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
    baseline_model = LoRASOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT, config)
    baseline_model = baseline_model.to(device)
    baseline_trainer = LoRATrainer(baseline_model, device, config, None)
    
    baseline_performance = np.zeros(num_tasks)
    for j in range(num_tasks):
        test_loader = loaders[f'test_task{j}']
        _, _, baseline_metrics = baseline_trainer.evaluate(test_loader, alpha=config.ALPHA, log=False)
        baseline_performance[j] = -baseline_metrics['MAE']
        logger.info("  Baseline Task %d: R=%.4f", j, baseline_performance[j])
    
    # Calculate BWT (Backward Transfer)
    if num_tasks > 1:
        bwt_scores = []
        for i in range(num_tasks - 1):
            final_perf = R_matrix[num_tasks - 1, i]
            when_learned_perf = R_matrix[i, i]
            bwt_scores.append(final_perf - when_learned_perf)
        BWT = np.mean(bwt_scores)
    else:
        BWT = 0.0
    
    # Calculate FWT (Forward Transfer)
    if num_tasks > 1:
        fwt_scores = []
        for i in range(1, num_tasks):
            when_learned_perf = R_matrix[i - 1, i]
            baseline_perf = baseline_performance[i]
            fwt_scores.append(when_learned_perf - baseline_perf)
        FWT = np.mean(fwt_scores)
    else:
        FWT = 0.0
    
    # Calculate ACC (Average Accuracy)
    ACC = np.mean(R_matrix[num_tasks - 1, :])
    
    # Compile continual learning metrics
    continual_learning_metrics = {
        "BWT": BWT,
        "FWT": FWT,
        "ACC": ACC,
        "num_tasks": num_tasks,
        "use_lora": config.USE_LORA,
        "lora_rank": config.LORA_RANK if config.USE_LORA else None,
        "lora_alpha": config.LORA_ALPHA if config.USE_LORA else None
    }
    
    # Log results
    logger.info("==== Continual Learning Results ====")
    logger.info("BWT (Backward Transfer): %.4f %s", BWT, 
               "(less negative = less forgetting)" if BWT < 0 else "(positive = backward gain)")
    logger.info("FWT (Forward Transfer): %.4f %s", FWT,
               "(positive = beneficial transfer)" if FWT > 0 else "(negative = interference)")
    logger.info("ACC (Average Accuracy): %.4f", ACC)
    
    if config.USE_LORA:
        logger.info("LoRA Configuration - Rank: %d, Alpha: %d", config.LORA_RANK, config.LORA_ALPHA)
    
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
# Main Pipeline (Updated)
# ===============================================================
def main():
    """Main execution pipeline with LoRA support"""
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
    logger.info("Number of tasks: %d", config.NUM_TASKS)
    logger.info("Random seed: %d", config.SEED)
    logger.info("Base directory: %s", config.BASE_DIR)
    logger.info("Using LoRA: %s", config.USE_LORA)
    
    if config.USE_LORA:
        logger.info("LoRA Configuration:")
        logger.info("  Rank: %d", config.LORA_RANK)
        logger.info("  Alpha: %d", config.LORA_ALPHA)
        logger.info("  Dropout: %.2f", config.LORA_DROPOUT)
        logger.info("  Target Modules: %s", config.LORA_TARGET_MODULES)
    
    # Run appropriate training pipeline
    if config.MODE == 'joint':
        joint_training(config)
    elif config.MODE == 'incremental':
        cl_metrics, r_matrix = incremental_training(config)
        
        # Log final summary
        logger.info("==== Final Summary ====")
        logger.info("Continual Learning Metrics:")
        for metric, value in cl_metrics.items():
            if isinstance(value, (int, float)):
                logger.info("  %s: %.4f", metric, value)
            else:
                logger.info("  %s: %s", metric, value)
    else:
        logger.error("Unknown mode: %s. Use 'joint' or 'incremental'", config.MODE)
        return
    
    logger.info("==== Experiment Complete ====")

if __name__ == '__main__':
    main()