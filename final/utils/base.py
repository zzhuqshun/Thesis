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
        # self.lstm1 = nn.LSTM(input_size,  hidden_size, batch_first=True)
        # self.dropout_between = nn.Dropout(dropout)
        # self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), 
            nn.LeakyReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        # # LSTM forward pass
        # h1, _ = self.lstm1(x)
        # h1 = self.dropout_between(h1)
        # out, _ = self.lstm2(h1)
        out, _ = self.lstm(x)
        # Use only the last time step output
        return self.fc(out[:, -1, :]).squeeze(-1)
    
# ===========================
# Adapter & Adapter-augmented LSTM
# ===========================
class Adapter(nn.Module):
    """Bottleneck adapter with residual gate."""
    def __init__(self, hidden_size: int, bottleneck: int = 32, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)

    def forward(self, x, scale: float = 1.0):
        # Residual + external scale
        z = self.up(self.drop(self.act(self.down(x))))
        return x + scale * z

class SOHLSTMAdapter(nn.Module):
    """Two-layer LSTM backbone + LN + Adapter + MLP head (many-to-one)."""
    def __init__(self, input_size: int, hidden_size: int, dropout: float,
                 adapter_bottleneck: int = 32, adapter_dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(hidden_size)
        self.adapter = Adapter(hidden_size, adapter_bottleneck, dropout=adapter_dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def encode_last(self, x):
        # Expose encoder last state for KL computation
        out, _ = self.lstm(x)
        # h_last = self.ln(out[:, -1, :])
        h_last = out[:, -1, :]
        return h_last

    def forward(self, x, adapter_scale: float = 1.0):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        if adapter_scale == 0:
            return self.head(h_last).squeeze(-1)
         
        h_norm = self.ln(h_last)
        h_adapt = self.adapter(h_norm, scale=adapter_scale)
        return self.head(h_adapt).squeeze(-1)


class Trainer:
    """Trainer for joint training (pure fine-tuning)"""
    def __init__(self, model: nn.Module, device: torch.device, config: Config, task_dir: Path):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.task_dir = task_dir
        if self.task_dir:
            self.task_dir.mkdir(parents=True, exist_ok=True)

    def train_task(self, train_loader, val_loader, task_id):
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
        
        history = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'lr', 'time']}
        
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
            
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)

            lr_cur = opt.param_groups[0]['lr']
            history['lr'].append(lr_cur)
            history['time'].append(time.time() - start)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    val_loss += F.mse_loss(self.model(x), y).item() * x.size(0)
            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)

            # Scheduler step and early stopping
            sched.step(val_loss)
            
            logger.info("Epoch %d train=%.4e val=%.4e lr=%.2e time=%.2fs",
                epoch, train_loss, val_loss, lr_cur, history['time'][-1])

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