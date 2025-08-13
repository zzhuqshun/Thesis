import logging
import copy
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn  # <-- NEW
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
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

class Trainer:
    """Trainer for joint training (pure fine-tuning) with optional SWA."""
    def __init__(self, model: nn.Module, device: torch.device, config: Config, task_dir: Path):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.task_dir = task_dir
        if self.task_dir:
            self.task_dir.mkdir(parents=True, exist_ok=True)

        # SWA config（可在 Config 里覆盖）
        self.use_swa = getattr(config, "USE_SWA", True)
        self.swa_trigger = getattr(config, "SWA_TRIGGER", "lr")  # 'lr' 或 'plateau'
        self.swa_lr = getattr(config, "SWA_LR", max(config.LEARNING_RATE * 0.5, 1e-5))
        self.swa_lr_threshold = getattr(config, "SWA_LR_THRESHOLD", 5e-5)
        self.swa_plateau_patience = getattr(config, "SWA_PLATEAU_PATIENCE", 5)

    @torch.no_grad()
    def _evaluate_loss(self, model, val_loader):
        model.eval()
        loss_sum = 0.0
        n = 0
        for x, y in val_loader:
            x, y = x.to(self.device), y.to(self.device)
            yp = model(x)
            loss_sum += F.mse_loss(yp, y, reduction="sum").item()
            n += y.numel()
        return loss_sum / max(1, n)  # MSE

    def train_task(self, train_loader: DataLoader, val_loader: DataLoader, task_id: int):
        """
        Train on all data at once (joint). Returns training history dict.
        """
        # Optimizer & schedulers
        opt = torch.optim.Adam(self.model.parameters(),
                               lr=self.config.LEARNING_RATE,
                               weight_decay=self.config.WEIGHT_DECAY)
        # 验证集不提升时降 LR
        plateu_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5
        )

        # SWA wrappers（先准备好，是否启用由触发条件决定）
        swa_model = AveragedModel(self.model)
        swa_sched = SWALR(opt, swa_lr=self.swa_lr)
        swa_enabled = False

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
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tot_loss += loss.item() * x.size(0)

            train_loss = tot_loss / len(train_loader.dataset)

            # 验证
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    val_loss += F.mse_loss(self.model(x), y).item() * x.size(0)
            val_loss /= len(val_loader.dataset)

            # 记录
            lr_cur = opt.param_groups[0]['lr']
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(lr_cur)
            history['time'].append(time.time() - start)

            logger.info("Epoch %d train=%.4e val=%.4e lr=%.2e time=%.2fs",
                        epoch, train_loss, val_loss, lr_cur, history['time'][-1])

            # 更新 best
            if val_loss < best_val - 1e-12:
                best_val = val_loss
                no_imp = 0
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save({'model_state': best_state}, self.task_dir / f"task{task_id}_best.pt")
            else:
                no_imp += 1

            # 触发 SWA？
            if self.use_swa and not swa_enabled:
                if self.swa_trigger == "lr":
                    if lr_cur <= self.swa_lr_threshold:
                        swa_enabled = True
                        logger.info(f"[SWA] Enabled at epoch %d (lr %.2e ≤ %.2e)",
                                    epoch, lr_cur, self.swa_lr_threshold)
                elif self.swa_trigger == "plateau":
                    if no_imp >= self.swa_plateau_patience:
                        swa_enabled = True
                        logger.info(f"[SWA] Enabled at epoch %d (no improvement for %d epochs)",
                                    epoch, self.swa_plateau_patience)

            # 调度器：SWA 开始后用 SWALR，否则用 ReduceLROnPlateau
            if swa_enabled:
                swa_model.update_parameters(self.model)
                swa_sched.step()
            else:
                plateu_sched.step(val_loss)

            # 早停（注意：SWA 后也允许早停）
            if no_imp >= self.config.PATIENCE:
                logger.info("Early stopping at epoch %d", epoch)
                break

        # 训练结束：若启用 SWA，做一次 BN 更新并评估 SWA
        swa_ckpt = self.task_dir / f"task{task_id}_swa.pt"
        if swa_enabled:
            try:
                update_bn(train_loader, swa_model)  # 无 BN 也安全
            except Exception as e:
                logger.warning("[SWA] update_bn skipped: %s", str(e))

            # 评估 SWA 模型在验证集上的 MSE
            val_loss_swa = self._evaluate_loss(swa_model, val_loader)
            torch.save({'model_state': swa_model.state_dict()}, swa_ckpt)
            logger.info("[SWA] val_loss=%.4e (best non-SWA=%.4e)", val_loss_swa, best_val)

            # 若 SWA 更好，则覆盖 best，并把模型权重切到 SWA
            if val_loss_swa <= best_val:
                best_val = val_loss_swa
                best_state = copy.deepcopy(swa_model.state_dict())
                torch.save({'model_state': best_state}, self.task_dir / f"task{task_id}_best.pt")
                self.model.load_state_dict(best_state)
                logger.info("[SWA] SWA model is better -> overwrite best checkpoint.")
            else:
                # 否则回到非 SWA 的 best
                if best_state:
                    self.model.load_state_dict(best_state)
        else:
            # 没启用 SWA：恢复 best
            if best_state:
                self.model.load_state_dict(best_state)

        return history
