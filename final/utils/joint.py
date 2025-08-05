import logging
from pathlib import Path
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from utils.config import Config
from utils.data import DataProcessor, create_dataloaders
from utils.base import SOHLSTM
from utils.evaluate import evaluate, plot_losses, plot_predictions, plot_prediction_scatter
from utils.utils import print_model_summary

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer for joint training (pure fine-tuning)"""
    def __init__(self, model: nn.Module, device: torch.device, config: Config, task_dir: Path):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.task_dir = task_dir
        if self.task_dir:
            self.task_dir.mkdir(parents=True, exist_ok=True)

    def train_task(self, train_loader: DataLoader, val_loader: DataLoader, task_id: int):
        """
        Train model on a single joint task (all data at once).
        Returns training history dict.
        """
        # Optimizer and scheduler
        opt = torch.optim.Adam(self.model.parameters(),
                               lr=self.config.LEARNING_RATE,
                               weight_decay=self.config.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)

        best_val = float('inf')
        no_imp = 0
        best_state = None
        history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': [], 'time': []}

        for epoch in range(self.config.EPOCHS):
            start = time.time()
            self.model.train()
            tot_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                yp = self.model(x)
                loss = F.mse_loss(yp, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                opt.step()
                tot_loss += loss.item() * x.size(0)
            train_loss = tot_loss / len(train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    val_loss += F.mse_loss(self.model(x), y).item() * x.size(0)
            val_loss /= len(val_loader.dataset)

            lr_cur = opt.param_groups[0]['lr']
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(lr_cur)
            history['time'].append(time.time() - start)

            logger.info("Epoch %d train=%.4e val=%.4e lr=%.2e time=%.2fs",
                        epoch, train_loss, val_loss, lr_cur, history['time'][-1])

            # Scheduler step and early stopping
            sched.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                no_imp = 0
                best_state = copy.deepcopy(self.model.state_dict())
                # Save checkpoint
                torch.save({'model_state': best_state}, self.task_dir / f"task{task_id}_best.pt")
            else:
                no_imp += 1
                if no_imp >= self.config.PATIENCE:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Restore best
        if best_state:
            self.model.load_state_dict(best_state)
        return history


def training(config: Config):
    """Run joint training over all data as baseline."""
    logger.info("==== Starting Joint Training ====")
    base_dir = Path(config.BASE_DIR) 
    ckpt_dir = base_dir / 'checkpoints'
    res_dir = base_dir / 'results'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Data prep
    dp = DataProcessor(config.DATA_DIR, config.RESAMPLE, config)
    datasets = dp.prepare_joint_data(config.joint_datasets)
    loaders = create_dataloaders(datasets, config.SEQUENCE_LENGTH, config.BATCH_SIZE)

    # Model and Trainer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    trainer = Trainer(model, device, config, ckpt_dir)
    print_model_summary(model)
    # Train
    history = trainer.train_task(loaders['train'], loaders['val'], task_id=0)
    pd.DataFrame(history).to_csv(ckpt_dir / 'training_history.csv', index=False)
    plot_losses(history, res_dir)

    # Evaluate
    preds, tgts, metrics = evaluate(model, loaders['test'], alpha=config.ALPHA)
    logger.info("Test metrics -> RMSE: %.4e, MAE: %.4e, R2: %.4f", 
                metrics['RMSE'], metrics['MAE'], metrics['R2'])
    plot_predictions(preds, tgts, metrics, res_dir, alpha=config.ALPHA)
    plot_prediction_scatter(preds, tgts, res_dir, alpha=config.ALPHA)
    pd.DataFrame([metrics]).to_csv(res_dir / 'test_metrics.csv', index=False)

    logger.info("==== Joint Training Complete ====")
