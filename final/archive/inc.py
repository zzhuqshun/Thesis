# inc.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.base import SOHLSTM
from utils.si import SITrainer  # <- use SI instead of EWC


# ===========================
# Distribution shift (KL)
# ===========================
def compute_data_stats(loader: DataLoader):
    """
    Compute per-dimension mean/variance from a DataLoader of (x, y).
    x shape: (batch, seq_len, feat_dim) -> flatten over time.
    """
    xs = []
    for x, _ in loader:
        b, s, f = x.shape
        xs.append(x.reshape(-1, f).cpu().numpy())
    arr = np.concatenate(xs, axis=0)
    mu = arr.mean(axis=0)
    var = arr.var(axis=0) + 1e-8
    return mu, var


def kl_divergence_gaussian(mu0, var0, mu1, var1):
    """
    KL between factorized Gaussians:
      KL = 0.5 * sum( log(var1/var0) + (var0 + (mu0-mu1)^2)/var1 - 1 )
    """
    term1 = np.log(var1 / var0)
    term2 = (var0 + (mu0 - mu1) ** 2) / var1
    kl = 0.5 * np.sum(term1 + term2 - 1.0)
    return kl


def compute_kl_between_loaders(old_loader: DataLoader, new_loader: DataLoader):
    """
    Compute KL(old || new) across the 3 features C/V/T with Gaussian approx.
    """
    mu0, var0 = compute_data_stats(old_loader)
    mu1, var1 = compute_data_stats(new_loader)
    return kl_divergence_gaussian(mu0, var0, mu1, var1)


# ===========================
# Adapter-enhanced LSTM
# ===========================
class Adapter(nn.Module):
    """Simple bottleneck adapter with external gate (scale)."""
    def __init__(self, hidden_size: int, bottleneck: int = 32, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)

    def forward(self, x, scale: float = 1.0):
        # Residual with external scale
        z = self.up(self.drop(self.act(self.down(x))))
        return x + scale * z


class SOHLSTMAdapter(nn.Module):
    """
    LSTM backbone + LayerNorm + Adapter + MLP head (many-to-one).
    Adds an optional external adapter_scale for gating.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 dropout: float,
                 adapter_bottleneck: int = 32,
                 adapter_dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers=2,
                            batch_first=True,
                            dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.adapter = Adapter(hidden_size, adapter_bottleneck, dropout=adapter_dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, adapter_scale: float = 1.0):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        h_last = self.layer_norm(h_last)
        h_adapt = self.adapter(h_last, scale=adapter_scale)
        return self.fc(h_adapt).squeeze(-1)


# ===========================
# SI + KD + KL-driven scheduler
# ===========================
class RegularizedTrainer(SITrainer):
    """
    Wrap SITrainer to:
      - compute KL(old||new) between tasks
      - use KL to scale lambda_si (regularization strength)
      - gate adapter strength by KL
    """
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 config,
                 task_dir=None,
                 base_lambda: float = 1.0,
                 ref_loader: DataLoader = None,
                 # KL -> lambda mapping
                 k_scale: float = 1.0,
                 lam_max: float = None,
                 # KL -> adapter gate mapping (sigmoid)
                 adapter_alpha_max: float = 1.0,
                 gate_k: float = 5.0,
                 gate_tau: float = 0.1,
                 si_epsilon: float = 1e-3):
        super().__init__(model, device, config, task_dir, si_epsilon=si_epsilon)
        self.base_lambda = base_lambda
        self.ref_loader = ref_loader  # previous task's train loader

        self.k_scale = k_scale
        self.lam_max = lam_max

        self.adapter_alpha_max = adapter_alpha_max
        self.gate_k = gate_k
        self.gate_tau = gate_tau

    def _lambda_from_kl(self, kl_value: float) -> float:
        lam = self.base_lambda * (1.0 + self.k_scale * max(0.0, kl_value))
        if self.lam_max is not None:
            lam = min(lam, self.lam_max)
        return lam

    def _adapter_scale_from_kl(self, kl_value: float) -> float:
        # sigmoid gate: alpha_max * sigma(k*(kl - tau))
        s = 1.0 / (1.0 + np.exp(-self.gate_k * (kl_value - self.gate_tau)))
        return float(self.adapter_alpha_max * s)

    def train_task(self,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   task_id: int,
                   alpha_lwf: float = 0.0):
        """
        Train a task with KL-driven SI lambda and adapter gate.
        """
        # Compute KL if reference loader available
        if self.ref_loader is not None:
            kl = compute_kl_between_loaders(self.ref_loader, train_loader)
        else:
            kl = 0.0

        lambda_si = self._lambda_from_kl(kl)
        adapter_scale = self._adapter_scale_from_kl(kl)

        # Log the dynamic settings
        import logging as _log
        _log.getLogger(__name__).info(
            "Task %d: KL=%.6f -> lambda_si=%.6f, adapter_scale=%.4f",
            task_id, kl, lambda_si, adapter_scale
        )

        # Call base trainer
        history = super().train_task(
            train_loader, val_loader, task_id,
            alpha_lwf=alpha_lwf,
            lambda_si=lambda_si,
            adapter_scale=adapter_scale
        )

        # Consolidate SI & update KD teacher
        self.consolidate(task_id=task_id)

        # Update reference loader to current train for the next task
        self.ref_loader = train_loader

        return history
