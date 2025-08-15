# utils/si.py
# SI (Synaptic Intelligence) + KD (LwF) with KL-driven scheduling.
# Baseline-aligned trainer: Adam + ReduceLROnPlateau + EarlyStop + strict ValBest restore.
# When SI/KD are disabled, this equals pure fine-tuning (no extra overhead).

import logging
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ================================================================
# SI core
# ================================================================
class SI:
    """
    Synaptic Intelligence (Zenke et al., 2017) - online importance accumulation.

    Usage:
      - begin_task(model) at task start
      - During training:
          update_w(model) BEFORE optimizer.step()
          on_param_update(model) RIGHT AFTER optimizer.step()
      - consolidate(model) at task end to build Omega and theta_star
      - penalty(model) returns SI regularizer value for current model
    """

    def __init__(self, model: nn.Module, device: torch.device, epsilon: float = 1e-3):
        self.device = device
        self.epsilon = float(epsilon) if epsilon is not None else 1e-3
        if self.epsilon <= 0.0:
            self.epsilon = 1e-3  # avoid zero denom

        self.w: Dict[str, torch.Tensor] = {}
        self.theta_task_start: Dict[str, torch.Tensor] = {}
        self.omega: Dict[str, torch.Tensor] = {}
        self.theta_star: Dict[str, torch.Tensor] = {}

        for n, p in model.named_parameters():
            if p.requires_grad:
                z = torch.zeros_like(p, device=self.device)
                self.w[n] = z.clone()
                self.theta_task_start[n] = p.detach().clone()
                self.omega[n] = z.clone()
                self.theta_star[n] = p.detach().clone()

    @torch.no_grad()
    def begin_task(self, model: nn.Module):
        """Reset online accumulators at task start."""
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.w[n].zero_()
            self.theta_task_start[n] = p.detach().clone()

    @torch.no_grad()
    def update_w(self, model):
        """Call BEFORE optimizer.step(). Snapshot grads AND pre-step params."""
        self._grads = {}
        self._theta_pre_step = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    self._grads[n] = p.grad.detach().clone()
                self._theta_pre_step[n] = p.detach().clone()

    @torch.no_grad()
    def on_param_update(self, model):
        """Call RIGHT AFTER optimizer.step(). Accumulate path integral from this step."""
        if not hasattr(self, "_grads"):
            return
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n in self._grads and n in self._theta_pre_step:
                delta_step = p.detach() - self._theta_pre_step[n]
                self.w[n].add_(-self._grads[n] * delta_step)

        del self._grads
        del self._theta_pre_step

    @torch.no_grad()
    def consolidate(self, model: nn.Module):
        """
        Convert online w to Omega and store theta_star at end of task:
          Omega += relu(w) / ( (theta - theta_task_start)^2 + eps )
        """
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            delta = p.detach() - self.theta_task_start[n]
            denom = delta.pow(2).add_(self.epsilon)
            self.omega[n].add_(torch.relu(self.w[n]) / denom)
            self.theta_star[n] = p.detach().clone()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """L_SI = sum_i Omega_i * (theta_i - theta_i^*)^2"""
        loss = torch.zeros((), device=self.device)
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n in self.omega:
                loss = loss + (self.omega[n] * (p - self.theta_star[n]).pow(2)).sum()
        return loss


# ================================================================
# Regularization & switches
# ================================================================
@dataclass
class RegConfig:
    # KD/SI proportions scheduled by KL in [floor, max]
    kd_feat_weight: float = 0.0
    p_kd_max: float = 0.40
    p_si_max: float = 0.25
    kd_delta: float = 1e-2
    kl_tau: float = 50.0
    scale_clip: float = 5.0
    warmup_epochs: int = 5
    p_kd_floor: float = 0.05
    p_si_floor: float = 0.02

    # KL smoothing & reference merging
    kl_ema_alpha: float = 0.90   # EMA for kl_norm
    kl_ref_alpha: float = 0.80   # EMA merge factor for (mu_ref, var_ref)
    kl_sigma_floor: float = 1e-4 # variance floor for numerical stability


@dataclass
class Switches:
    """Centralized feature switches."""
    use_kd: bool
    use_si: bool
    use_feat_kd: bool
    use_kl: bool


# ================================================================
# Feature stats & KL helper
# ================================================================
class FeatureStatTracker:
    """Online mean/var tracker for features with Welford's algorithm."""
    def __init__(self, feat_dim: int, device: torch.device):
        self.device = device
        self.n = 0
        self.mean = torch.zeros(feat_dim, device=device)
        self.M2 = torch.zeros(feat_dim, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        # x: [B, D]
        if x.ndim != 2:
            x = x.view(x.size(0), -1)
        b = x.size(0)
        self.n += b
        delta = x.mean(dim=0) - self.mean
        self.mean += delta * (b / max(self.n, 1))
        batch_var = x.var(dim=0, unbiased=False)
        self.M2 += batch_var * b + delta.pow(2) * (self.n - b) * b / max(self.n, 1)

    @torch.no_grad()
    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        var = self.M2 / max(self.n, 1)
        var = torch.clamp(var, min=1e-6)
        return self.mean.clone(), var.clone()


def _sym_kl_diag(mu_p, var_p, mu_q, var_q) -> torch.Tensor:
    """Symmetric KL between two diagonal Gaussians N(mu, var). Returns scalar."""
    var_p = torch.clamp(var_p, min=1e-6)
    var_q = torch.clamp(var_q, min=1e-6)
    term_pq = var_p / var_q + (mu_q - mu_p).pow(2) / var_q - 1.0 + torch.log(var_q / var_p)
    term_qp = var_q / var_p + (mu_p - mu_q).pow(2) / var_p - 1.0 + torch.log(var_p / var_q)
    kl = 0.5 * (term_pq.sum() + term_qp.sum())
    return torch.relu(kl)  # non-negative safeguard


# ================================================================
# SITrainer (baseline-aligned, no EMA)
# ================================================================
class SITrainer:
    """
    Baseline-aligned incremental trainer:
      - Adam + ReduceLROnPlateau scheduler on val_mse (factor=0.5, patience=5)
      - Early stopping with strict ValBest restore
      - Optional SI/KD with KL-driven scheduling
      - No EMA (removed to avoid ambiguity and ensure comparability)

    Turn off SI/KD to get pure fine-tuning identical in spirit to base trainer.
    """

    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-6,
                 grad_clip: float = 1.0,
                 si_epsilon: float = 1e-3,
                 reg_cfg: Optional[RegConfig] = None,
                 # LR scheduler (ReduceLROnPlateau) aligned with your base
                 lr_factor: float = 0.5,
                 lr_patience: int = 5,
                 # Early stopping
                 early_stop_patience: int = 20):
        self.model = model.to(device)
        self.device = device
        self.base_lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        self.reg_cfg = reg_cfg or RegConfig()

        # Centralized switches
        use_kd = (self.reg_cfg.p_kd_max > 0.0 or self.reg_cfg.p_kd_floor > 0.0)
        use_si = (self.reg_cfg.p_si_max > 0.0 or self.reg_cfg.p_si_floor > 0.0)
        use_feat_kd = use_kd and (self.reg_cfg.kd_feat_weight > 0.0)
        use_kl = use_kd or use_si
        self.sw = Switches(use_kd=use_kd, use_si=use_si, use_feat_kd=use_feat_kd, use_kl=use_kl)

        # SI state
        self.si = SI(self.model, device=self.device, epsilon=si_epsilon) if self.sw.use_si else None

        # Teacher / KL reference
        self.teacher: Optional[nn.Module] = None
        self.kl_ref: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Optimizer & LR scheduler (mirror your base snippet)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=lr_factor, patience=lr_patience
        )

        # Early stopping
        self.early_stop_patience = int(early_stop_patience)

        # KL EMA accumulator
        self._kl_ema: Optional[float] = None

    # ------------------------ Loss pieces ------------------------
    def _task_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_pred, y_true)

    def _kd_raw_loss(self, y_student: torch.Tensor, y_teacher: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(y_student, y_teacher, beta=self.reg_cfg.kd_delta)

    def _feat_kd_loss_from(self, h_s: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(h_s, h_t.detach(), dim=1).mean()

    # ------------------------ Helpers ------------------------
    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _log_epoch(self, epoch: int, avg_task: float, kd_pct: float, si_pct: float,
                   avg_kl: float, val_mse: Optional[float], lr: float):
        """Compact epoch log aligned with base trainer style."""
        if val_mse is None:
            logger.info("Epoch %d: task=%.4e kd%%=%.2f%% si%%=%.2f%% kl=%.3f lr=%.5e",
                        epoch, avg_task, kd_pct * 100.0, si_pct * 100.0, avg_kl, lr)
        else:
            logger.info("Epoch %d: task=%.4e kd%%=%.2f%% si%%=%.2f%% kl=%.3f val_mse=%.4e lr=%.5e",
                epoch, avg_task, kd_pct * 100.0, si_pct * 100.0, avg_kl, val_mse, lr)

    # ------------------------ Train one task ------------------------
    def train_one_task(self,
                       train_loader: DataLoader,
                       val_loader: Optional[DataLoader],
                       epochs: int,
                       task_id: int) -> Dict[str, float]:
        # Mode summary
        mode = ("SI+KD" if (self.sw.use_si and self.sw.use_kd) else
                "KD-only" if (self.sw.use_kd and not self.sw.use_si) else
                "SI-only" if (self.sw.use_si and not self.sw.use_kd) else
                "pure-finetune")
        logger.info("=== Train task %d in mode: %s ===", task_id, mode)

        # Reset KL EMA
        self._kl_ema = None

        feat_tracker = None  # lazy init after we see the first teacher hidden
        
        best_val = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        no_improve = 0
        patience = self.early_stop_patience

        history = {"epoch": [], "train_task": [], "kd_pct": [], "si_pct": [], "kl": [], "val_mse": [], "lr": []}

        # SI begin
        if self.sw.use_si:
            self.si.begin_task(self.model)

        for epoch in range(epochs):
            self.model.train()
            epoch_task = epoch_kd_pct = epoch_si_pct = epoch_kl = 0.0
            n_batches = 0

            # Proportion warmup for KD/SI
            warmup = min(1.0, (epoch + 1) / max(1, self.reg_cfg.warmup_epochs)) if (self.sw.use_kd or self.sw.use_si) else 0.0

            for x, y in train_loader:
                x = x.to(self.device); y = y.to(self.device)

                # Whether hidden states are needed
                need_s_hidden = (self.teacher is not None) and self.sw.use_feat_kd
                need_t_hidden = (self.teacher is not None) and (self.sw.use_feat_kd or self.sw.use_kl)

                # Student forward
                if need_s_hidden:
                    y_pred, h_s = self.model(x, return_hidden=True)
                else:
                    y_pred = self.model(x)
                    h_s = None

                # Teacher forward (for KD / KL)
                if (self.teacher is not None) and (self.sw.use_kd or self.sw.use_kl):
                    with torch.no_grad():
                        if need_t_hidden:
                            out = self.teacher(x, return_hidden=True)
                            if self.sw.use_kd:
                                y_teacher, h_t = out
                            else:
                                _, h_t = out
                                y_teacher = None
                        else:
                            y_teacher = self.teacher(x) if self.sw.use_kd else None
                            h_t = None
                else:
                    y_teacher, h_t = None, None

                # Loss: task
                L_task = self._task_loss(y_pred, y)

                # Loss: KD
                if (y_teacher is not None) and self.sw.use_kd:
                    L_kd_raw = self._kd_raw_loss(y_pred, y_teacher)
                    if self.sw.use_feat_kd and (h_t is not None) and (h_s is not None):
                        L_kd_raw = L_kd_raw + self.reg_cfg.kd_feat_weight * self._feat_kd_loss_from(h_s, h_t)
                else:
                    L_kd_raw = torch.zeros_like(L_task)

                # Loss: SI
                if self.sw.use_si:
                    L_si_raw = self.si.penalty(self.model)
                else:
                    L_si_raw = torch.zeros_like(L_task)

                # KL scheduling
                if self.sw.use_kl and (h_t is not None):
                    kl_norm = self._kl_norm_from_features(h_t)
                else:
                    kl_norm = 0.0

                # Schedule proportions in [floor, max] with warmup
                if self.sw.use_kd:
                    p_kd = (self.reg_cfg.p_kd_floor + kl_norm * (self.reg_cfg.p_kd_max - self.reg_cfg.p_kd_floor)) * warmup
                else:
                    p_kd = 0.0
                if self.sw.use_si:
                    p_si = (self.reg_cfg.p_si_floor + (1.0 - kl_norm) * (self.reg_cfg.p_si_max - self.reg_cfg.p_si_floor)) * warmup
                else:
                    p_si = 0.0

                # Self-balancing coefficients c_kd / c_si
                eps = 1e-12
                c_kd = ((p_kd * L_task.detach()) / (L_kd_raw.detach() + eps)
                        if (self.sw.use_kd and (y_teacher is not None)) else torch.tensor(0.0, device=self.device))
                c_si = ((p_si * L_task.detach()) / (L_si_raw.detach() + eps)
                        if (self.sw.use_si and (L_si_raw.detach().item() > 0)) else torch.tensor(0.0, device=self.device))
                c_kd = torch.clamp(c_kd, max=self.reg_cfg.scale_clip)
                c_si = torch.clamp(c_si, max=self.reg_cfg.scale_clip)

                # Backprop
                L_total = L_task + c_kd * L_kd_raw + c_si * L_si_raw
                self.optimizer.zero_grad(set_to_none=True)
                L_total.backward()
                if self.sw.use_si:
                    self.si.update_w(self.model)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                if self.sw.use_si:
                    self.si.on_param_update(self.model)

                # KL feature stats
                if h_t is not None and self.sw.use_kl:
                    ht = h_t.detach()
                    if feat_tracker is None:
                        d = ht.view(ht.size(0), -1).size(1)
                        feat_tracker = FeatureStatTracker(d, self.device)
                    feat_tracker.update(ht)

                # Bookkeeping
                with torch.no_grad():
                    kd_contrib = (c_kd * L_kd_raw).item()
                    si_contrib = (c_si * L_si_raw).item()
                    epoch_task += L_task.item()
                    denom = L_task.item() + 1e-12
                    epoch_kd_pct += (kd_contrib / denom) if self.sw.use_kd else 0.0
                    epoch_si_pct += (si_contrib / denom) if self.sw.use_si else 0.0
                    epoch_kl += kl_norm if self.sw.use_kl else 0.0
                    n_batches += 1

            # -------- Validation & LR scheduling --------
            val_mse = None
            if val_loader is not None:
                mse_sum, n_elems = 0.0, 0
                self.model.eval()
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx = vx.to(self.device); vy = vy.to(self.device)
                        vpred = self.model(vx)  # no hidden on validation
                        mse_sum += F.mse_loss(vpred, vy, reduction="sum").item()
                        n_elems += vy.numel()
                self.model.train()
                val_mse = (mse_sum / n_elems) if n_elems > 0 else float("inf")

            # Step LR scheduler and log LR
            if val_mse is not None:
                self.scheduler.step(val_mse)

            kd_pct = epoch_kd_pct / max(1, n_batches)
            si_pct = epoch_si_pct / max(1, n_batches)
            avg_kl = epoch_kl / max(1, n_batches)
            avg_task = epoch_task / max(1, n_batches)
            cur_lr = self._current_lr()
            self._log_epoch(epoch, avg_task, kd_pct, si_pct, avg_kl, val_mse, cur_lr)

            history["epoch"].append(epoch)
            history["train_task"].append(avg_task)
            history["kd_pct"].append(kd_pct * 100.0)
            history["si_pct"].append(si_pct * 100.0)
            history["kl"].append(avg_kl)
            history["val_mse"].append(val_mse if val_mse is not None else float("inf"))
            history["lr"].append(cur_lr)

            # -------- Early stopping with strict ValBest --------
            if val_mse is not None:
                improved = (val_mse < best_val - 1e-6)
                if improved:
                    best_val = val_mse
                    best_state = copy.deepcopy(self.model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # -------- Strictly restore ValBest (baseline-aligned) --------
        self.model.load_state_dict(best_state)
        logger.info("Task %d finished. Restored ValBest weights (val_mse=%.6e).", task_id, best_val)

        # -------- End of task: consolidate SI / update teacher & KL ref --------
        if self.sw.use_si:
            self.si.consolidate(self.model)

        if (self.sw.use_kd or self.sw.use_kl):
            self._update_teacher()  # teacher = copy of current (ValBest) model
            if feat_tracker is not None:
                self._update_kl_ref_from_tracker(feat_tracker)
        else:
            self.teacher = None
            self.kl_ref = None

        return {"best_val": best_val if val_loader is not None else float("nan"),
                "history": history}

    # ------------------------ Teacher / KL maintenance ------------------------
    def _update_teacher(self):
        self.teacher = copy.deepcopy(self.model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _update_kl_ref_from_tracker(self, tracker: Optional[FeatureStatTracker]):
        if (tracker is None) or (getattr(tracker, "n", 0) == 0):
            return
        mu_new, var_new = tracker.finalize()
        sigma_floor = float(self.reg_cfg.kl_sigma_floor)
        if self.kl_ref is None:
            self.kl_ref = (mu_new, torch.clamp(var_new + sigma_floor, min=1e-6))
            return
        mu_ref, var_ref = self.kl_ref
        alpha = float(self.reg_cfg.kl_ref_alpha)
        mu  = alpha * mu_ref + (1.0 - alpha) * mu_new
        var = alpha * var_ref + (1.0 - alpha) * var_new + sigma_floor
        self.kl_ref = (mu, torch.clamp(var, min=1e-6))

    @torch.no_grad()
    def prime_kl_ref(self, loader, max_batches: int = 64):
        """Use teacher on task0 data to initialize (mu_ref, var_ref)."""
        if (self.teacher is None) or (not self.sw.use_kl):
            return
        tracker = None
        for i, (x, _) in enumerate(loader):
            x = x.to(self.device)
            _, h = self.teacher(x, return_hidden=True)
            if tracker is None:
                tracker = FeatureStatTracker(h.size(1), self.device)
            tracker.update(h)
            if (i + 1) >= max_batches:
                break
        if tracker is not None:
            self.kl_ref = tracker.finalize()

    @torch.no_grad()
    def _kl_norm_from_features(self, h: Optional[torch.Tensor]) -> float:
        """
        Input h is teacher hidden; flatten to [B, D], compute batch (mu,var),
        then symmetric KL to (mu_ref,var_ref) -> normalize to [0,1] with EMA.
        """
        if (h is None) or (h.numel() == 0):
            return 0.0

        if h.dim() > 2:
            h = h.view(h.size(0), -1)
        elif h.dim() == 1:
            h = h.view(1, -1)

        mu_cur = h.mean(dim=0)
        var_cur = torch.clamp(h.var(dim=0, unbiased=False), min=1e-6)

        if self.kl_ref is None:
            self.kl_ref = (mu_cur.detach().clone(), var_cur.detach().clone())
            self._kl_ema = 0.0
            return 0.0

        mu_ref, var_ref = self.kl_ref
        kl_value = _sym_kl_diag(mu_cur, var_cur, mu_ref, var_ref).item()

        s = max(self.reg_cfg.kl_tau, 1e-6)
        raw = float(kl_value / (kl_value + s))   # map to [0,1]

        a = float(self.reg_cfg.kl_ema_alpha)
        self._kl_ema = raw if (self._kl_ema is None) else (a * self._kl_ema + (1.0 - a) * raw)
        return float(self._kl_ema)
