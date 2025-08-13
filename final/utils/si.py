# utils/si.py
# Synaptic Intelligence (SI) with numerically-stable epsilon and a simple penalty() API.
# SITrainer with KL-driven scheduling and auto-balancing for KD/SI.
# - KD uses Smooth L1 (Huber) on outputs.
# - KL is computed on teacher's last LSTM hidden state vs a cross-task reference (diagonal Gaussians).
# - Coefficients c_kd/c_si are computed so that each regularizer contributes a target fraction of L_task.
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
logger = logging.getLogger(__name__)

class SI:
    """
    Synaptic Intelligence (Zenke et al., 2017) - online importance accumulation.

    Usage:
      - Call `begin_task(model)` at the start of a task.
      - During training, call `update_w(model)` BEFORE optimizer.step() to snapshot gradients,
        and call `on_param_update(model)` RIGHT AFTER optimizer.step() to accumulate path integral.
      - After finishing a task, call `consolidate(model)` to build/update Omega and theta_star.
      - During later tasks, add `penalty(model)` into the total loss (weighted outside).
    """

    def __init__(self, model: nn.Module, device: torch.device, epsilon: float = 1e-3):
        self.device = device
        self.epsilon = float(epsilon) if epsilon is not None else 1e-3
        if self.epsilon <= 0.0:
            # Never allow epsilon=0, it breaks the denominator
            self.epsilon = 1e-3

        self.w: Dict[str, torch.Tensor] = {}
        self.theta_task_start: Dict[str, torch.Tensor] = {}
        self.omega: Dict[str, torch.Tensor] = {}
        self.theta_star: Dict[str, torch.Tensor] = {}
        self.theta_prev: Dict[str, torch.Tensor] = {}

        for n, p in model.named_parameters():
            if p.requires_grad:
                z = torch.zeros_like(p, device=self.device)
                self.w[n] = z.clone()
                self.theta_task_start[n] = p.detach().clone()
                self.omega[n] = z.clone()
                self.theta_star[n] = p.detach().clone()
                self.theta_prev[n] = p.detach().clone()

    @torch.no_grad()
    def begin_task(self, model: nn.Module):
        """Reset online accumulators at task start."""
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.w[n].zero_()
            self.theta_task_start[n] = p.detach().clone()
            self.theta_prev[n] = p.detach().clone()

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
        """Call RIGHT AFTER optimizer.step(). Use per-step delta."""
        if not hasattr(self, "_grads"): return
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if n in self._grads and n in self._theta_pre_step:
                delta_step = p.detach() - self._theta_pre_step[n]   # ← 本步Δθ
                self.w[n].add_(-self._grads[n] * delta_step)
                self.theta_prev[n] = p.detach().clone()
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
        """
        L_SI = sum_i Omega_i * (theta_i - theta_i^*)^2
        """
        loss = torch.zeros((), device=self.device)
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n in self.omega:
                loss = loss + (self.omega[n] * (p - self.theta_star[n]).pow(2)).sum()
        return loss

@dataclass
class RegConfig:
    kd_feat_weight: float = 0.0 
    # Max proportions of L_task allocated to KD/SI (before KL & warmup)
    p_kd_max: float = 0.40
    p_si_max: float = 0.25
    # Huber delta for KD
    kd_delta: float = 1e-2
    # KL normalization temperature (bigger -> smaller kl_norm)
    kl_tau: float = 50
    # Clip the scaling factor c_* to avoid numeric spikes
    scale_clip: float = 5.0
    # Warmup epochs for p_* (linearly ramp from 0 to max)
    warmup_epochs: int = 5
    p_kd_floor: float = 0.05   # ← 新增：就算 KL 高也给点粮
    p_si_floor: float = 0.02   # ← 新增

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
        # For per-dim var, combine batch var + mean shift
        batch_var = x.var(dim=0, unbiased=False)
        self.M2 += batch_var * b + delta.pow(2) * (self.n - b) * b / max(self.n, 1)

    @torch.no_grad()
    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        var = self.M2 / max(self.n, 1)
        var = torch.clamp(var, min=1e-6)
        return self.mean.clone(), var.clone()

def _sym_kl_diag(mu_p, var_p, mu_q, var_q) -> torch.Tensor:
    """Symmetric KL between two diagonal Gaussians N(mu, var). Returns scalar."""
    # Dkl(p||q) = 0.5 * sum( var_p/var_q + (mu_q - mu_p)^2/var_q - 1 + log(var_q/var_p) )
    # Sym: D(p||q)+D(q||p)
    var_p = torch.clamp(var_p, min=1e-6)
    var_q = torch.clamp(var_q, min=1e-6)
    term_pq = var_p / var_q + (mu_q - mu_p).pow(2) / var_q - 1.0 + torch.log(var_q / var_p)
    term_qp = var_q / var_p + (mu_p - mu_q).pow(2) / var_p - 1.0 + torch.log(var_p / var_q)
    kl = 0.5 * (term_pq.sum() + term_qp.sum())
    return torch.relu(kl)  # non-negative safeguard

class SITrainer:
    """
    Incremental trainer with SI regularization and KD (LwF-style), driven by KL scheduling.
    Assumes model has attributes: model.lstm and model.fc, and forward returns prediction.
    """

    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-6,
                 grad_clip: float = 1.0,
                 si_epsilon: float = 1e-3,
                 reg_cfg: Optional[RegConfig] = None,
                 val_metric: str = "mse" ):
        self.model = model.to(device)
        self.device = device
        self.base_lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        self.si = SI(self.model, device=self.device, epsilon=si_epsilon)
        self.teacher: Optional[nn.Module] = None

        self.reg_cfg = reg_cfg or RegConfig()

        # Cross-task KL reference (mu_ref, var_ref) on teacher hidden features
        self.kl_ref: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Optimizer with layer-wise LR (LSTM first layer smaller)
        # If you want exact layerwise control, split param groups accordingly.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        
        self.ema_decay: Optional[float] = 0.99   # or set from config
        self.ema_start_epoch: Optional[int] = None  # will set later per task
        self._ema_state: Optional[Dict[str, torch.Tensor]] = None

        self.val_metric = val_metric.lower().strip()
        if self.val_metric not in ("mse", "mae"):
            self.val_metric = "mse"  # default
    @torch.no_grad()
    def _ema_reset(self):
        self._ema_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    @torch.no_grad()
    def _ema_update(self):
        if self._ema_state is None:
            self._ema_reset()
            return
        d = self.ema_decay if self.ema_decay is not None else 0.99
        msd = self.model.state_dict()
        for k, v in msd.items():
            self._ema_state[k].mul_(d).add_(v.detach(), alpha=(1.0 - d))

    # ------------------------ Feature helpers ------------------------
    @torch.no_grad()
    def _last_hidden(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Extract last hidden state from LSTM outputs.
        Assumes model.lstm returns (out, (hn, cn)), where out is [B, T, H].
        """
        out, _ = model.lstm(x)
        h_last = out[:, -1, :]  # [B, H]
        return h_last
    
    @torch.no_grad()
    def _kl_norm_from_batch(self, x: torch.Tensor, feat_dim: int) -> float:
        """
        Compute kl_norm in [0,1]: compare teacher hidden features on batch vs cross-task reference.
        If reference is missing (e.g., first incremental task), bootstrap it from current batch (kl=0).
        """
        if self.teacher is None:
            return 0.0
        h = self._last_hidden(self.teacher, x)  # [B, H]
        mu_cur = h.mean(dim=0)
        var_cur = torch.clamp(h.var(dim=0, unbiased=False), min=1e-6)

        if self.kl_ref is None:
            # bootstrap: set reference so KL=0 initially
            self.kl_ref = (mu_cur.detach().clone(), var_cur.detach().clone())
            return 0.0

        mu_ref, var_ref = self.kl_ref
        kl = _sym_kl_diag(mu_cur, var_cur, mu_ref, var_ref)
        kl_value = kl.item()
        s = max(self.reg_cfg.kl_tau, 1e-6)
        kl_norm = float(kl_value / (kl_value + s))
        return float(kl_norm)

    # ------------------------ Loss pieces ------------------------
    def _task_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Keep your original supervised loss; MSE is common here.
        return F.mse_loss(y_pred, y_true)

    def _kd_raw_loss(self, y_student: torch.Tensor, y_teacher: torch.Tensor) -> torch.Tensor:
        # Huber (Smooth L1) KD for regression
        return F.smooth_l1_loss(y_student, y_teacher, beta=self.reg_cfg.kd_delta)

    # ------------------------ Train one task ------------------------
    def train_one_task(self,
                       train_loader: DataLoader,
                       val_loader: Optional[DataLoader],
                       epochs: int,
                       task_id: int) -> Dict[str, float]:
        """
        Train model for a single task with SI+KD (KL-scheduled, auto-balanced).
        Returns dict of last-epoch logs (for convenience).
        """
        # Prepare SI for this task
        self.si.begin_task(self.model)
        # reset EMA at task start
        self._ema_reset()

        # Hidden feature dimension for trackers
        with torch.no_grad():
            sample_x, _ = next(iter(train_loader))
            sample_x = sample_x.to(self.device)
            feat_dim = self._last_hidden(self.model, sample_x).size(1)

        # Track current task teacher features to update kl_ref after consolidate
        feat_tracker = FeatureStatTracker(feat_dim=feat_dim, device=self.device)

        best_val = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        no_improve = 0
        patience = 20  # keep your original patience if needed

        history = {
            "epoch": [],
            "train_task": [],
            "kd_pct": [],
            "si_pct": [],
            "kl": [],
            "val_mae": [],   
            "val_mse": [],   
        }

        for epoch in range(epochs):
            self.model.train()
            epoch_task, epoch_kd_contrib, epoch_si_contrib = 0.0, 0.0, 0.0
            epoch_kd_pct, epoch_si_pct, epoch_kl = 0.0, 0.0, 0.0
            n_batches = 0

            # Linear warmup
            warmup = min(1.0, (epoch + 1) / max(1, self.reg_cfg.warmup_epochs))

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward - student
                y_pred = self.model(x)
                L_task = self._task_loss(y_pred, y)

                # KD raw (if teacher exists)
                if self.teacher is not None:
                    with torch.no_grad():
                        y_teacher = self.teacher(x)
                    L_kd_raw = self._kd_raw_loss(y_pred, y_teacher)

                    # + feature KD (counts into KD bucket)
                    if self.reg_cfg.kd_feat_weight > 0.0:
                        L_kd_feat = self._feat_kd_loss(x)
                        L_kd_raw = L_kd_raw + self.reg_cfg.kd_feat_weight * L_kd_feat
                else:
                    L_kd_raw = torch.zeros_like(L_task)

                # SI raw penalty
                L_si_raw = self.si.penalty(self.model)

                # KL scheduling (based on teacher hidden features vs reference)
                kl_norm = self._kl_norm_from_batch(x, feat_dim=feat_dim)
                # Target proportions this step
                p_kd = (self.reg_cfg.p_kd_floor 
                        + kl_norm * (self.reg_cfg.p_kd_max - self.reg_cfg.p_kd_floor)) * warmup
                p_si = (self.reg_cfg.p_si_floor 
                        + (1.0 - kl_norm) * (self.reg_cfg.p_si_max - self.reg_cfg.p_si_floor)) * warmup

                # Auto-balance coefficients so each reg contributes ~ p_* * L_task
                eps = 1e-12
                c_kd = (p_kd * L_task.detach()) / (L_kd_raw.detach() + eps) if self.teacher is not None else torch.tensor(0.0, device=self.device)
                c_si = (p_si * L_task.detach()) / (L_si_raw.detach() + eps) if L_si_raw.detach().item() > 0 else torch.tensor(0.0, device=self.device)

                # Clip scaling to avoid numeric explosion when raw term is tiny
                c_kd = torch.clamp(c_kd, max=self.reg_cfg.scale_clip)
                c_si = torch.clamp(c_si, max=self.reg_cfg.scale_clip)

                # Total loss
                L_total = L_task + c_kd * L_kd_raw + c_si * L_si_raw

                # Backward
                self.optimizer.zero_grad(set_to_none=True)
                # ---- SI needs grads BEFORE step (for w update) ----
                L_total.backward()
                # (Record grads for SI path integral)
                self.si.update_w(self.model)

                # Step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                # Accumulate SI after the params have changed
                self.si.on_param_update(self.model)
                
                # Always update EMA every batch (full-run EMA)
                if self.ema_decay is not None:
                    self._ema_update()



                # Track feature stats on TEACHER (reference lives in teacher space)
                if self.teacher is not None:
                    with torch.no_grad():
                        h_teacher = self._last_hidden(self.teacher, x)  # [B, H]
                    feat_tracker.update(h_teacher)


                # Logs
                with torch.no_grad():
                    kd_contrib = (c_kd * L_kd_raw).item()
                    si_contrib = (c_si * L_si_raw).item()
                    epoch_task += L_task.item()
                    epoch_kd_contrib += kd_contrib
                    epoch_si_contrib += si_contrib
                    denom = L_task.item() + 1e-12
                    epoch_kd_pct += kd_contrib / denom
                    epoch_si_pct += si_contrib / denom
                    epoch_kl += kl_norm
                    n_batches += 1

            # ---- Validation ----
            val_mae, val_mse = float("nan"), float("nan")
            if val_loader is not None:
                v_mae, v_mse = self.evaluate_metrics(self.model, val_loader)
                val_mae, val_mse = float(v_mae), float(v_mse)
                # choose the monitored value
                val_monitored = val_mse if self.val_metric == "mse" else val_mae
            else:
                val_monitored = None

            # Print epoch summary
            kd_pct = epoch_kd_pct / max(1, n_batches)
            si_pct = epoch_si_pct / max(1, n_batches)
            avg_kl = epoch_kl / max(1, n_batches)
            avg_task = epoch_task / max(1, n_batches)

            if val_loader is not None:
                logger.info(
                    "Epoch %d: task=%.4e kd%%=%.2f%% si%%=%.2f%% kl=%.3f val_mae=%.4e val_mse=%.4e",
                    epoch, avg_task, kd_pct * 100.0, si_pct * 100.0, avg_kl, val_mae, val_mse
                )
            else:
                logger.info(
                    "Epoch %d: task=%.4e kd%%=%.2f%% si%%=%.2f%% kl=%.3f",
                    epoch, avg_task, kd_pct * 100.0, si_pct * 100.0, avg_kl
                )

            # Record history (keep both columns)
            history["epoch"].append(epoch)
            history["train_task"].append(avg_task)
            history["kd_pct"].append(kd_pct * 100.0)
            history["si_pct"].append(si_pct * 100.0)
            history["kl"].append(avg_kl)
            history["val_mae"].append(val_mae)
            history["val_mse"].append(val_mse)

            # Early stopping on chosen metric
            if val_loader is not None:
                if val_monitored < best_val - 1e-6:
                    best_val = val_monitored
                    best_state = copy.deepcopy(self.model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        chosen_is_ema = False
        # Restore best (val-based) and compare with EMA on the same metric
        if val_loader is not None:
            self.model.load_state_dict(best_state)
            chosen_state = copy.deepcopy(best_state)
            chosen_val = best_val

            if self._ema_state is not None:
                cur_state = copy.deepcopy(self.model.state_dict())
                self.model.load_state_dict(self._ema_state)
                ema_mae, ema_mse = self.evaluate_metrics(self.model, val_loader)
                ema_val = ema_mse if self.val_metric == "mse" else ema_mae
                if ema_val < chosen_val - 1e-6:
                    chosen_state = copy.deepcopy(self._ema_state)
                    chosen_val = ema_val
                    chosen_is_ema = True
                self.model.load_state_dict(cur_state)

            self.model.load_state_dict(chosen_state)
            logger.info("Task %d picked %s weights for consolidation (val_%s=%.6e).",
            task_id, "EMA" if chosen_is_ema else "ValBest", self.val_metric, chosen_val)
            
        # Consolidate SI and update teacher + KL reference
        self.si.consolidate(self.model)
        self._update_teacher()
        self._update_kl_ref_from_tracker(feat_tracker)
        
        return {
            "best_val": best_val if val_loader is not None else float("nan"),
            "history": history
        }

    @torch.no_grad()
    def evaluate_metrics(self, model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
        """Return (mae, mse) averaged per-element over the loader."""
        model.eval()
        mae_sum, mse_sum, n = 0.0, 0.0, 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = model(x)
            # sum reduction to compute dataset average
            mae_sum += F.l1_loss(pred, y, reduction="sum").item()
            mse_sum += F.mse_loss(pred, y, reduction="sum").item()
            n += y.numel()
        if n == 0:
            return float("inf"), float("inf")
        return mae_sum / n, mse_sum / n


    # ------------------------ Teacher / KL ref maintenance ------------------------
    def _update_teacher(self):
        self.teacher = copy.deepcopy(self.model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
    
    @torch.no_grad()
    def _update_kl_ref_from_tracker(self, tracker: FeatureStatTracker):
        if getattr(tracker, "n", 0) == 0:
            return
        mu_new, var_new = tracker.finalize()
        sigma_floor = 1e-4  # widen the ref a bit

        if self.kl_ref is None:
            self.kl_ref = (mu_new, torch.clamp(var_new + sigma_floor, min=1e-6))
            return

        mu_ref, var_ref = self.kl_ref
        alpha = 0.8  # put more weight on historical ref
        mu  = alpha * mu_ref + (1.0 - alpha) * mu_new
        var = alpha * var_ref + (1.0 - alpha) * var_new + sigma_floor
        self.kl_ref = (mu, torch.clamp(var, min=1e-6))

    @torch.no_grad()
    def prime_kl_ref(self, loader, max_batches: int = 64):
        """Use teacher on task0 data to initialize (mu_ref, var_ref)."""
        if self.teacher is None:
            return
        tracker = None
        n = 0
        for x, _ in loader:
            x = x.to(self.device)
            h = self._last_hidden(self.teacher, x)  # [B, H]
            if tracker is None:
                tracker = FeatureStatTracker(h.size(1), self.device)
            tracker.update(h)
            n += 1
            if n >= max_batches:
                break
        if tracker is not None:
            self.kl_ref = tracker.finalize()
    def _feat_kd_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Cosine feature KD on last hidden; returns scalar."""
        with torch.no_grad():
            h_t = self._last_hidden(self.teacher, x)  # [B,H]
        h_s = self._last_hidden(self.model, x)
        # cosine distance = 1 - cos
        return 1.0 - F.cosine_similarity(h_s, h_t, dim=1).mean()

