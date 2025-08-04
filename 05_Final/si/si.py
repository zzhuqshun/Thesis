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
        self.BASE_DIR = Path.cwd()/ "si" / "strategies" / "fine-tuning"
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
        self.SI_LAMBDAS = [0.0, 0.0, 0.0]    # Synaptic Intelligence regularization weights
        self.SI_EPSILON = 0.000831
        
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
            "method": "SI",  # Synaptic Intelligence
            "resample": self.RESAMPLE,
            "scaler": "RobustScaler - fit on base train",
            "smooth_alpha": self.ALPHA,
            "lwf_alphas": self.LWF_ALPHAS,
            "si_lambdas": self.SI_LAMBDAS,
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

class SI:
    """
    Synaptic Intelligence (SI) for continual learning.

    SI estimates parameter importance by accumulating 
    -∂θL * Δθ over each optimizer step, then at end of task:
        Ω_i += w_i / (Δθ_i_total^2 + ε)

    Reference: Zenke et al. "Continual Learning Through Synaptic Intelligence" (2017)
    """

    def __init__(self, model, si_lambda=1.0, epsilon=0.1):
        self.model = model
        self.si_lambda = si_lambda
        self.epsilon = epsilon

        # Ω: accumulated importance across tasks
        self.omega = {
            n: torch.zeros_like(p.data)
            for n, p in model.named_parameters() if p.requires_grad
        }

        # placeholders for task‐specific accumulators
        self._reset_task_buffers()

    def _reset_task_buffers(self):
        # w: path‐integral accumulator for this task
        self.w = {
            n: torch.zeros_like(p.data)
            for n, p in self.model.named_parameters() if p.requires_grad
        }
        # θ_old: parameter values at last update (or start of task)
        self.theta_old = {
            n: p.data.clone().detach()
            for n, p in self.model.named_parameters() if p.requires_grad
        }
        # θ_start: snapshot at beginning of task
        self.theta_start = {
            n: p.data.clone().detach()
            for n, p in self.model.named_parameters() if p.requires_grad
        }

    def begin_task(self):
        """Call once before training on a new task."""
        self._reset_task_buffers()

    def update_contributions(self):
        """
        Call AFTER each optimizer.step().
        Accumulate w_i += -grad_i * Δθ_i, where Δθ_i = θ_i_new - θ_i_old.
        """
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            # change this step
            delta = param.data - self.theta_old[name]
            # accumulate contribution
            self.w[name] += -param.grad.data * delta
            # update theta_old for next step
            self.theta_old[name] = param.data.clone().detach()

    def end_task(self):
        """
        Call once after finishing a task.
        Compute Ω_i += w_i / ( (θ_i_end - θ_i_start)^2 + ε ).
        """
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # total change over the task
            delta_total = param.data - self.theta_start[name]
            # update importance
            self.omega[name] += self.w[name] / (delta_total.pow(2) + self.epsilon)
        
        all_omega = torch.cat([v.flatten() for v in self.omega.values()])
        scale = all_omega.mean().clamp(min=1e-12)
        for name in self.omega:
            self.omega[name] /= scale
        logger.info("SI importance normalized with scale %.4f", scale.item())

    def penalty(self):
        """
        Compute SI regularization:
           loss_reg = λ * Σ_i Ω_i * (θ_i - θ_i_start)^2
        """
        loss_reg = 0.0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # deviation from task‐start
            delta = param.data - self.theta_start[name]
            loss_reg += (self.omega[name] * delta.pow(2)).sum()

        return self.si_lambda * loss_reg


# ===============================================================
# Trainer
# ===============================================================
class Trainer:
    """Main training class with support for continual learning"""
    
    def __init__(self, model, device, config, task_dir=None):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.si = None          # Synaptic Intelligence regularizer
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
                                  'task_loss', 'kd_loss', 'si_loss']}
        
        # Training loop
        for epoch in tqdm.tqdm(range(self.config.EPOCHS), desc=f"Task{task_id}"):
            start = time.time()
            self.model.train()
            
            # Loss components tracking
            tot_loss = 0
            sum_task = sum_kd = sum_si = 0
            
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
                
                # Synaptic Intelligence regularization loss
                si_loss = self.si.penalty() if self.si is not None else torch.zeros((), device=self.device)
                
                # Total loss combination
                loss = task_loss + alpha_lwf * kd_loss + si_loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                
                # Parameter update
                opt.step()
                
                if self.si is not None: 
                    self.si.update_contributions()
                
                # Track loss components
                bs = x.size(0)
                sum_task += task_loss.item() * bs
                sum_kd += kd_loss.item() * bs
                sum_si += si_loss.item() * bs
                tot_loss += loss.item() * bs
            
            # Calculate epoch averages
            n = len(train_loader.dataset)
            train_loss = tot_loss / n
            
            if sum_task > 0:
                reg_ratio = sum_si / sum_task * 100
                logger.info("Epoch %d: SI loss is %.2f%% of task loss", epoch, reg_ratio)

            
            # Record training history
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['task_loss'].append(sum_task / n)
            history['kd_loss'].append(sum_kd / n)
            history['si_loss'].append(sum_si / n)
            
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
            logger.info("Epoch %d task=%.4e kd=%.4e si=%.4e val=%.4e lr=%.2e time=%.2fs",
                       epoch, sum_task/n, sum_kd/n, sum_si/n, val_loss, lr_cur, history['time'][-1])
            
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
# Training Pipelines
# ===============================================================
def incremental_training(config):
    """
    Incremental training with Synaptic Intelligence.
    Train on tasks sequentially while preventing catastrophic forgetting.
    """
    logger.info("==== Incremental Training with SI ====")
    
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
    
    # Initialize Synaptic Intelligence with first task's lambda
    trainer.si = SI(model, si_lambda=config.SI_LAMBDAS[0], epsilon=config.SI_EPSILON)
    trainer.si.begin_task()
    
    # Sequential task training
    for task_idx in range(num_tasks):
        current_lambda = config.SI_LAMBDAS[task_idx]
        current_alpha = config.LWF_ALPHAS[task_idx]
        task_name = f"task{task_idx}"
        
        logger.info("--- %s (LWF α=%.4f, SI λ=%.4f, epsilon=%.4f) ---",
                    task_name, current_alpha, current_lambda, config.SI_EPSILON)
        
        # Update SI regularization strength for current task
        trainer.si.si_lambda = current_lambda
        
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
        
        # End current task: update importance weights
        trainer.si.end_task()
        logger.info("Task %d completed. SI importance weights updated.", task_idx)
        
        # Prepare for next task (if not the last task)
        if task_idx < num_tasks - 1:
            # Save current model for knowledge distillation
            trainer.old_model = copy.deepcopy(trainer.model).to(device)
            trainer.old_model.eval()
            for p in trainer.old_model.parameters():
                p.requires_grad_(False)
            
            # Reset SI state for next task
            trainer.si.begin_task()
            logger.info("Prepared for next task. Old model saved for knowledge distillation.")
    
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
    logger.info("SI lambdas: %s", config.SI_LAMBDAS)
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
