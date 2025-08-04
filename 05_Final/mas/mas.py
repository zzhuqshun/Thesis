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

from ewc.ewc import (Visualizer, DataProcessor, SOHLSTM, 
                    set_seed, create_dataloaders, setup_logging, 
                    joint_training, evaluate_incremental_learning)

logger = logging.getLogger(__name__)

# ===============================================================
# Configuration Class
# ===============================================================
class Config:
    """Configuration class for continual learning experiments"""
    def __init__(self, **kwargs):
        # Training mode: 'joint' for baseline, 'incremental' for continual learning
        self.MODE = 'incremental'  
        
        # Directory structure
        self.BASE_DIR = Path.cwd() /"mas" / "strategies" / "fine-tuning"
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
        self.LWF_ALPHAS = [0.0, 0.0, 0.0]    # Learning without Forgetting weights
        self.MAS_LAMBDAS = [0.0, 0.0, 0.0]    # MAS regularization weights
        
        # Random seed for reproducibility
        self.SEED = 42
        
        # Dataset splits for joint training (baseline)
        self.joint_datasets = {
            'train_ids': ['03', '05', '07', '09', '11', '15', '21', '23', '25', '27', '29'],
            'val_ids': ['01', '19', '13'],
            'test_id': '17'
        }
        
        # Dataset splits for incremental learning
        # Each task focuses on different degradation types
        # Each validation will be split into val/test(7:3 ratio)
        # Task 0: '03', '05', '07', '27'; '01'
        # Task 1: '21', '23', '25'; '19'
        # Task 2: '09', '11', '15', '29'; '13'
        # Test set: '17' (common for all tasks)

        self.incremental_datasets = {
            'task0_train_ids': ['03', '05', '07', '27'],
            'task0_val_ids': ['01'],
            'task1_train_ids': ['21', '23', '25'],
            'task1_val_ids': ['19'],
            'task2_train_ids': ['09', '11', '15', '29'],
            'task2_val_ids': ['13'],
            'test_id': '17'
        }
        
        # Experiment metadata
        self.Info = {
            "method": "MAS",  # Memory Aware Synapses
            "resample": self.RESAMPLE,
            "scaler": "RobustScaler - fit on base train",
            "smooth_alpha": self.ALPHA,
            "lwf_alphas": self.LWF_ALPHAS,
            "mas_lambdas": self.MAS_LAMBDAS,
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

class MAS:
    """
    Memory Aware Synapses (MAS) for continual learning.

    MAS estimates parameter importance based on the magnitude of gradients
    of the squared L2‐norm of the model outputs, then applies regularization
    to prevent important parameters from changing too much in subsequent tasks.

    Reference: Aljundi et al. "Memory Aware Synapses: Learning what (not) to forget" (2018)
    """

    def __init__(self, model, dataloader, device, mas_lambda=1.0):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.mas_lambda = mas_lambda

        # Store a snapshot of parameters after the previous task
        self.params = {
            name: param.clone().detach()
            for name, param in model.named_parameters() if param.requires_grad
        }

        # Compute importance weights (Ω) for the current task
        self.importance = self._compute_importance()

    def _compute_importance(self):
        """
        Compute parameter importance Ω based on gradients of the squared L2‐norm
        of the outputs. For each batch, we do:
            loss = ||F(x)||_2^2
        and accumulate |∂ loss / ∂ θ_i| over all batches, then average.
        """
        was_training = self.model.training
        self.model.train()

        # Initialize Ω with zeros
        importance = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.model.named_parameters() if param.requires_grad
        }

        # Accumulate absolute gradients over the dataset
        for x, _ in self.dataloader:
            x = x.to(self.device)

            # Zero gradients (faster than .zero_grad())
            self.model.zero_grad(set_to_none=True)

            # Forward: compute squared L2‐norm of outputs
            output = self.model(x)
            loss_imp = output.pow(2).sum()

            # Backward: gradients w.r.t. every parameter
            loss_imp.backward()

            # Accumulate |grad| into Ω
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        importance[name] += param.grad.abs()

        # Average over number of batches
        num_batches = float(len(self.dataloader))
        for name in importance:
            importance[name] /= num_batches

        # Restore original mode
        if not was_training:
            self.model.eval()

        return importance

    def penalty(self, model):
        """
        Compute the MAS regularization term:
            L_reg = λ * Σ_i Ω_i * (θ_i - θ_i^*)^2
        where θ_i^* are the stored parameters from the previous task.
        """
        reg_loss = 0.0
        for name, param in model.named_parameters():
            if name in self.importance and param.requires_grad:
                delta = param - self.params[name]
                reg_loss += (self.importance[name] * delta.pow(2)).sum()

        return self.mas_lambda * reg_loss


# ===============================================================
# Trainer
# ===============================================================
class Trainer:
    """Main training class with support for continual learning"""
    
    def __init__(self, model, device, config, task_dir=None):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.mas_tasks = []     # List of MAS regularizers from previous tasks
        self.old_model = None   # Previous model for knowledge distillation
        self.task_dir = Path(task_dir) if task_dir else None
        if self.task_dir: 
            self.task_dir.mkdir(parents=True, exist_ok=True)
    
    def train_task(self, train_loader, val_loader, task_id, alpha_lwf=0.0):
        """
        Train model on a single task with continual learning regularization.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            task_id: Current task identifier
            alpha_lwf: Learning without Forgetting weight (knowledge distillation)
        """
        # Setup optimizer and scheduler
        opt = torch.optim.Adam(self.model.parameters(), 
                              lr=self.config.LEARNING_RATE,
                              weight_decay=self.config.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)
        
        # Early stopping variables
        best_val = float('inf')
        no_imp = 0
        best_state = None
        
        # Training history tracking
        history = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'lr', 'time', 
                                  'task_loss', 'kd_loss', 'mas_loss']}
        
        # Training loop
        for epoch in tqdm.tqdm(range(self.config.EPOCHS), desc=f"Task{task_id}"):
            start = time.time()
            self.model.train()
            
            # Loss components tracking
            tot_loss = 0
            sum_task = sum_kd = sum_mas = 0
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                
                # Forward pass
                yp = self.model(x)
                
                # Task-specific loss (MSE for regression)
                task_loss = F.mse_loss(yp, y)
                
                # Knowledge distillation loss (Learning without Forgetting)
                kd_loss = torch.zeros((), device=self.device)
                if alpha_lwf > 0 and self.old_model is not None:
                    with torch.no_grad():
                        old_output = self.old_model(x)
                    kd_loss = F.mse_loss(yp, old_output)
                
                # MAS regularization loss
                mas_loss = torch.zeros((), device=self.device)
                if self.mas_tasks:
                    mas_loss = sum(mas_reg.penalty(self.model) for mas_reg in self.mas_tasks)
                
                # Total loss combination
                loss = task_loss + alpha_lwf * kd_loss + mas_loss
                
                # Backward pass and parameter update
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                opt.step()
                
                # Track loss components
                bs = x.size(0)
                sum_task += task_loss.item() * bs
                sum_kd += kd_loss.item() * bs
                sum_mas += mas_loss.item() * bs
                tot_loss += loss.item() * bs
            
            # Calculate epoch averages
            n = len(train_loader.dataset)
            train_loss = tot_loss / n
            
            if sum_task > 0:
                reg_ratio = sum_mas / sum_task * 100
                logger.info("Epoch %d: MAS loss is %.2f%% of task loss", epoch, reg_ratio)

            # Record training history
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['task_loss'].append(sum_task / n)
            history['kd_loss'].append(sum_kd / n)
            history['mas_loss'].append(sum_mas / n)
            
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
            logger.info("Epoch %d task=%.4e kd=%.4e mas=%.4e val=%.4e lr=%.2e time=%.2fs",
                       epoch, sum_task/n, sum_kd/n, sum_mas/n, val_loss, lr_cur, history['time'][-1])
            
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
    
    def consolidate(self, loader, task_id=None, mas_lambda=0.0):
        """
        Consolidate knowledge after task completion using MAS.
        
        Args:
            loader: Data loader for importance computation
            task_id: Task identifier
            mas_lambda: MAS regularization strength
        """
        # Create MAS regularizer for this task
        mas_reg = MAS(self.model, loader, self.device, mas_lambda)
        
        # 归一化
        all_imp = torch.cat([v.flatten() for v in mas_reg.importance.values()])
        scale = all_imp.mean().clamp(min=1e-12)   
        for name in mas_reg.importance:
            mas_reg.importance[name] /= scale
        
        logger.info("MAS importance normalized with scale %.4f", scale.item())
        self.mas_tasks.append(mas_reg)
        
        # Save model for knowledge distillation
        self.old_model = copy.deepcopy(self.model).to(self.device)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)
        
        logger.info("Task %s consolidated with MAS lambda=%.4f", task_id, mas_lambda)
    
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
# Training Pipelines
# ===============================================================
def incremental_training(config):
    """
    Incremental training with Memory Aware Synapses (MAS).
    Train on tasks sequentially while preventing catastrophic forgetting.
    """
    logger.info("==== Incremental Training with MAS ====")
    
    # Setup directories
    inc_dir = config.BASE_DIR
    inc_dir.mkdir(parents=True, exist_ok=True)
    
    # Get number of tasks from config
    num_tasks = config.NUM_TASKS
    logger.info("Number of tasks: %d", num_tasks)
    
    # Prepare incremental learning data
    dp = DataProcessor(config.DATA_DIR, config.RESAMPLE, config)
    data = dp.prepare_incremental_data(config.incremental_datasets)
    loaders = create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    
    # Initialize model and trainer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    trainer = Trainer(model, device, config, inc_dir)
    
    # Sequential task training
    for task_idx in range(num_tasks):
        current_lambda = config.MAS_LAMBDAS[task_idx]
        current_alpha = config.LWF_ALPHAS[task_idx]
        task_name = f"task{task_idx}"
        
        logger.info("--- %s (LWF α=%.4f, MAS λ=%.4f) ---", 
                   task_name, current_alpha, current_lambda)
        
        # Setup task directory
        task_dir = inc_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        trainer.task_dir = task_dir
        
        # Set task-specific random seed for reproducibility
        set_seed(config.SEED + task_idx)
        
        # Train on current task
        history = trainer.train_task(
            loaders[f"{task_name}_train"], 
            loaders[f"{task_name}_val"], 
            task_idx, 
            alpha_lwf=current_alpha
        )
        
        # Save training history and visualizations
        pd.DataFrame(history).to_csv(task_dir / 'training_history.csv', index=False)
        Visualizer.plot_losses(history, task_dir)
        
        # Consolidate: compute importance and save model for next task
        logger.info("Consolidating task %d with MAS lambda: %.4f", task_idx, current_lambda)
        trainer.consolidate(loaders[f"{task_name}_train"], task_id=task_idx, mas_lambda=current_lambda)
        logger.info("Task %d completed and consolidated.", task_idx)
    
    logger.info("==== Incremental Training Complete ====")
    
    # Comprehensive evaluation phase
    return evaluate_incremental_learning(config, inc_dir, num_tasks, loaders, device)


# ===============================================================
# Main Pipeline
# ===============================================================
def main():
    """Main execution pipeline"""
    # Load configuration
    config = Config()
    
    # Setup directories and logging
    config.BASE_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(config.BASE_DIR)
    
    # Save configuration for reproducibility
    config.save(config.BASE_DIR / 'config.json')
    
    # Set random seed for reproducibility
    set_seed(config.SEED)
    
    # Log experiment setup
    logger.info("==== Experiment Setup ====")
    logger.info("Mode: %s", config.MODE)
    logger.info("Number of tasks: %d", config.NUM_TASKS)
    logger.info("MAS lambdas: %s", config.MAS_LAMBDAS)
    logger.info("LWF alphas: %s", config.LWF_ALPHAS)
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