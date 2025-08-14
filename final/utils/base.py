import logging
import copy
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.config import Config

logger = logging.getLogger(__name__)

class SOHLSTM(nn.Module):
    """LSTM model for State of Health (SOH) prediction"""
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, *, return_hidden: bool = False, detach_hidden: bool = False):
        if hasattr(self.lstm, "flatten_parameters"):
            self.lstm.flatten_parameters()
        out, _ = self.lstm(x)          # [B, T, H]
        h_last = out[:, -1, :]         # [B, H]
        y = self.fc(h_last).squeeze(-1)

        if return_hidden:
            if detach_hidden:
                h_last = h_last.detach()
            return y, h_last
        return y


class Trainer:
    """Joint training"""
    def __init__(self, model: nn.Module, device: torch.device, config: Config, task_dir: Path):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.task_dir = task_dir
        if self.task_dir:
            self.task_dir.mkdir(parents=True, exist_ok=True)

    def train_task(self, train_loader: DataLoader, val_loader: DataLoader, task_id: int):
        """
        Training on the whole data sets
        """

        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5
        )

        best_val = float('inf')
        no_imp = 0
        best_state = None

        history = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'lr', 'time']}

        for epoch in range(self.config.EPOCHS):
            start = time.time()
            
            # Training
            self.model.train()
            tot_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad(set_to_none=True)
                yp = self.model(x)
                loss = F.mse_loss(yp, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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

            plateau_sched.step(val_loss)

            lr_cur = opt.param_groups[0]['lr']
            elapsed = time.time() - start
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(lr_cur)
            history['time'].append(elapsed)

            logger.info("Epoch %d train=%.4e val=%.4e lr=%.2e time=%.2fs",
                        epoch, train_loss, val_loss, lr_cur, elapsed)

            if val_loss < best_val - 1e-12:
                best_val = val_loss
                no_imp = 0
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save({'model_state': best_state}, self.task_dir / f"task{task_id}_best.pt")
            else:
                no_imp += 1

            if no_imp >= self.config.PATIENCE:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history
