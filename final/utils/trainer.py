# utils/trainer.py
import copy
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ================================================================
# SI (Synaptic Intelligence) - self-contained implementation
# ================================================================
class SI:
    """
    Synaptic Intelligence (Zenke et al., 2017)
    Tracks parameter importance online and builds an omega penalty
    after each task to protect important parameters.
    """
    def __init__(self, model: nn.Module, device: torch.device, epsilon: float = 1e-3):
        self.device = device
        self.epsilon = epsilon
        self.w: Dict[str, torch.Tensor] = {}
        self.theta_task_start: Dict[str, torch.Tensor] = {}
        self.omega: Dict[str, torch.Tensor] = {}
        self.theta_star: Dict[str, torch.Tensor] = {}

        for n, p in model.named_parameters():
            if p.requires_grad:
                z = torch.zeros_like(p, device=self.device)
                self.w[n] = z.clone()
                self.theta_task_start[n] = p.detach().clone()

    def begin_task(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.w[n] = torch.zeros_like(p, device=self.device)
                self.theta_task_start[n] = p.detach().clone()

    @torch.no_grad()
    def accumulate_step(self, model: nn.Module,
                        grads: Dict[str, torch.Tensor],
                        old_params: Dict[str, torch.Tensor]):
        # w += -g * Δθ
        for n, p in model.named_parameters():
            if not p.requires_grad or n not in grads:
                continue
            delta = p.data - old_params[n]
            self.w[n] += (-grads[n]) * delta

    @torch.no_grad()
    def consolidate_task(self, model: nn.Module):
        # omega += relu(w) / ((θ_T - θ_0)^2 + eps)
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            theta_now = p.detach()
            theta_start = self.theta_task_start[n]
            denom = (theta_now - theta_start).pow(2) + self.epsilon
            omega_add = torch.relu(self.w[n]) / denom
            self.omega[n] = self.omega.get(n, torch.zeros_like(omega_add)) + omega_add
            self.theta_star[n] = theta_now.clone()

        # reset w for next task
        for n in list(self.w.keys()):
            self.w[n].zero_()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        loss = torch.zeros((), device=self.device)
        for n, p in model.named_parameters():
            if p.requires_grad and (n in self.omega) and (n in self.theta_star):
                loss = loss + (self.omega[n] * (p - self.theta_star[n]) ** 2).sum()
        return loss


# ================================================================
# KL utilities on hidden states
# ================================================================
@torch.no_grad()
def _collect_hidden_stats(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Collect mean/var of the last hidden representation (encoder output).
    - If model has 'encode_last(x)', use it.
    - Otherwise, try to locate LSTM output and take the last step.
    """
    hs = []
    for x, _ in loader:
        x = x.to(device)
        if hasattr(model, "encode_last"):
            h = model.encode_last(x)
        else:
            # Fallback: try a common attribute name 'lstm'
            if hasattr(model, "lstm"):
                out = model.lstm(x)[0]  # (B, S, H)
                h = out[:, -1, :]
            else:
                # As a last resort, run forward and take a feature-like tensor if available
                y = model(x)
                # If forward returns scalar, KL won't be meaningful; return zeros
                if y.ndim == 1:
                    h = torch.zeros((x.size(0), 16), device=device)  # dummy
                else:
                    h = y
        hs.append(h.detach().float().cpu().numpy())

    arr = np.concatenate(hs, axis=0)  # (N, H)
    mu = arr.mean(axis=0)
    var = arr.var(axis=0) + 1e-8
    return mu, var


def _kl_gaussian(mu0, var0, mu1, var1) -> float:
    """
    KL( N(mu0, var0) || N(mu1, var1) ) with diagonal covariances.
    """
    term1 = np.log(var1 / var0)
    term2 = (var0 + (mu0 - mu1) ** 2) / var1
    kl = 0.5 * float(np.sum(term1 + term2 - 1.0))
    return max(0.0, kl)


def compute_kl_hidden(model_ref: nn.Module, model_new: nn.Module,
                      loader_ref: DataLoader, loader_new: DataLoader,
                      device: torch.device) -> float:
    """
    Compute hidden-state Gaussian KL between a frozen reference encoder
    and the current encoder using two loaders.
    """
    mu0, var0 = _collect_hidden_stats(model_ref, loader_ref, device)
    mu1, var1 = _collect_hidden_stats(model_new, loader_new, device)
    return _kl_gaussian(mu0, var0, mu1, var1)


# ================================================================
# Unified Trainer: KD + SI + KL-driven Adapter scheduling
# ================================================================
class Trainer:
    """
    A unified trainer that supports:
      - Task loss (MSE)
      - LwF-style KD loss against a frozen teacher (alpha)
      - SI regularization loss (lambda)
      - Adapter scale scheduling driven by KL divergence on hidden states

    Usage patterns:
      1) Dynamic mode (default):
         - alpha = alpha_max * exp(-alpha_c * KL)
         - lambda = lam_base * exp(-lam_c * KL), floored by lam_min
         - adapter_scale = adapt_max * sigmoid(adapt_k * (KL - adapt_tau))
      2) Fixed mode:
         - Set use_dynamic=False and provide fixed_alpha/fixed_lambda/fixed_adapter.

    Notes:
      - If your model forward(x, adapter_scale) is available, we pass adapter_scale.
        Otherwise we call forward(x) and ignore adapter scaling.
      - The reference snapshot and loader are updated after each task.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config,
        task_dir: Optional[Path] = None,
        # SI
        si_epsilon: float = 1e-3,
        # Dynamic scheduling switch
        use_dynamic: bool = True,
        # KD schedule: alpha = alpha_max * exp(-alpha_c * KL)
        alpha_max: float = 1.0,
        alpha_c: float = 2.0,
        # SI schedule: lambda = lam_base * exp(-lam_c * KL)
        lam_base: float = 1.0,
        lam_c: float = 1.0,
        lam_min: float = 0.0,
        # Adapter schedule: scale = adapt_max * sigmoid(adapt_k * (KL - adapt_tau))
        adapt_max: float = 1.0,
        adapt_k: float = 5.0,
        adapt_tau: float = 0.1,
        # KL smoothing and clipping
        kl_smooth: float = 0.1,    # EMA factor for KL; 0 means no smoothing
        kl_clip: Optional[float] = None,  # e.g., 10.0
        # Fixed weights for baseline mode
        fixed_alpha: float = 0.0,
        fixed_lambda: float = 0.0,
        fixed_adapter: float = 0.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.task_dir = Path(task_dir) if task_dir else None
        if self.task_dir:
            self.task_dir.mkdir(parents=True, exist_ok=True)

        # SI
        self.si = SI(self.model, self.device, epsilon=si_epsilon)

        # KD teacher (frozen)
        self.teacher: Optional[nn.Module] = None

        # KL reference snapshot & loader
        self.ref_snapshot: Optional[nn.Module] = None
        self.ref_loader: Optional[DataLoader] = None

        # Dynamic settings
        self.use_dynamic = use_dynamic
        self.alpha_max = alpha_max
        self.alpha_c = alpha_c
        self.lam_base = lam_base
        self.lam_c = lam_c
        self.lam_min = lam_min
        self.adapt_max = adapt_max
        self.adapt_k = adapt_k
        self.adapt_tau = adapt_tau
        self.kl_smooth = kl_smooth
        self.kl_clip = kl_clip
        self.kl_ema: Optional[float] = None

        # Fixed weights
        self.fixed_alpha = fixed_alpha
        self.fixed_lambda = fixed_lambda
        self.fixed_adapter = fixed_adapter

    # ------------------------------
    # Scheduling helpers
    # ------------------------------
    def _alpha_from_kl(self, kl: float) -> float:
        # KD decreases with KL (exp decay)
        return float(self.alpha_max * np.exp(-self.alpha_c * max(0.0, kl)))

    def _lambda_from_kl(self, kl: float) -> float:
        # SI decreases with KL, floored by lam_min
        lam = self.lam_base * np.exp(-self.lam_c * max(0.0, kl))
        return float(max(self.lam_min, lam))

    def _adapter_from_kl(self, kl: float) -> float:
        # Adapter gate increases with KL (sigmoid)
        s = 1.0 / (1.0 + np.exp(-self.adapt_k * (kl - self.adapt_tau)))
        return float(self.adapt_max * s)

    def _forward_with_adapter(self, x: torch.Tensor, adapter_scale: Optional[float]):
        # Try to forward with adapter_scale; fallback to plain forward
        try:
            if adapter_scale is not None:
                return self.model(x, adapter_scale=adapter_scale)
        except TypeError:
            pass
        return self.model(x)

    def calibrate_from_kl(self, probe_loader: DataLoader,
                      quantiles=(0.5, 0.6, 0.75, 0.9),
                      target_kd_ratio=0.3, target_si_ratio=0.2,
                      kd_loss_est=None, si_pen_est=None, task_loss_est=None):
        """Auto-set alpha_c, lam_c, adapt_k, adapt_tau; optionally alpha_max, lam_base."""
        # 1) collect several KL samples
        kls = []
        for _ in range(5):
            kl_now = compute_kl_hidden(
                self.ref_snapshot if self.ref_snapshot is not None else self.model,
                self.model, probe_loader, probe_loader, self.device
            )
            kls.append(kl_now)
        import numpy as np, math, logging
        KL50, KL60, KL75, KL90 = [float(np.quantile(kls, q)) for q in quantiles]

        # 2) make KD/SI half at KL50
        if KL50 > 1e-8:
            self.alpha_c = math.log(2.0) / KL50
            self.lam_c   = math.log(2.0) / KL50

        # 3) optional: set alpha_max/lam_base by desired loss shares at KL≈0
        if (kd_loss_est and task_loss_est and kd_loss_est > 0):
            self.alpha_max = float(target_kd_ratio * task_loss_est / kd_loss_est)
        if (si_pen_est and task_loss_est and si_pen_est > 0):
            self.lam_base = float(target_si_ratio * task_loss_est / si_pen_est)

        # 4) adapter gate: center near KL75, slope from KL60->KL90
        self.adapt_tau = KL75
        if KL90 > KL60:
            self.adapt_k = float(math.log(9.0) / (KL90 - KL60))

        logging.getLogger(__name__).info(
            "[Auto-Calibrate] KL50=%.4f KL60=%.4f KL75=%.4f KL90=%.4f | "
            "alpha_max=%.3f alpha_c=%.3f | lam_base=%.3f lam_c=%.3f | adapt_k=%.2f adapt_tau=%.3f",
            KL50, KL60, KL75, KL90, self.alpha_max, self.alpha_c, self.lam_base, self.lam_c,
            self.adapt_k, self.adapt_tau
        )
    
    # ------------------------------
    # KL computation (with EMA/clipping)
    # ------------------------------
    def _compute_kl(self, train_loader: DataLoader) -> float:
        # First task: if no reference, set ref=cur and KL=0
        if self.ref_snapshot is None:
            self.ref_snapshot = copy.deepcopy(self.model).to(self.device).eval()
            for p in self.ref_snapshot.parameters():
                p.requires_grad_(False)
            self.ref_loader = train_loader
            self.kl_ema = 0.0
            return 0.0

        try:
            kl_now = compute_kl_hidden(
                self.ref_snapshot, self.model,
                self.ref_loader or train_loader, train_loader,
                self.device
            )
        except Exception as e:
            logger.warning("KL(hidden) failed (%s). Fallback to KL=0.", str(e))
            kl_now = 0.0

        # EMA smoothing if enabled
        if self.kl_smooth and self.kl_smooth > 0.0:
            if self.kl_ema is None:
                self.kl_ema = kl_now
            else:
                beta = float(np.clip(self.kl_smooth, 0.0, 1.0))
                self.kl_ema = (1 - beta) * self.kl_ema + beta * kl_now
            kl_used = self.kl_ema
        else:
            kl_used = kl_now

        # Optional clipping
        if self.kl_clip is not None:
            kl_used = float(np.clip(kl_used, 0.0, self.kl_clip))

        return float(kl_used)

    # ------------------------------
    # Public API
    # ------------------------------
    def train_task(self,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   task_id: int):
        """
        Train one task with dynamic or fixed schedules.
        - Saves best checkpoint to self.task_dir / f"task{task_id}_best.pt"
        - Returns history dict
        """
        if self.use_dynamic and task_id == 0 and getattr(self.config, "AUTO_CALIBRATE", False):
            self.calibrate_from_kl(train_loader)
        # Prepare optimizer/scheduler
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5
        )

        # SI start
        self.si.begin_task(self.model)

        best_val = float('inf')
        best_state = None
        no_imp = 0

        hist = {k: [] for k in
                ['epoch', 'time', 'lr', 'train_loss', 'val_loss',
                 'task_loss', 'kd_loss', 'si_loss',
                 'KL', 'alpha', 'lambda', 'adapter']}

        # Compute KL and derive schedules (or fixed)
        kl = self._compute_kl(train_loader)
        if self.use_dynamic:
            alpha = self._alpha_from_kl(kl)
            lam = self._lambda_from_kl(kl)
            adapter_scale = self._adapter_from_kl(kl)
        else:
            alpha = float(self.fixed_alpha)
            lam = float(self.fixed_lambda)
            adapter_scale = float(self.fixed_adapter)

        logger.info("Task %d | KL=%.6f -> alpha=%.4f, lambda=%.4f, adapter=%.3f",
                    task_id, kl, alpha, lam, adapter_scale)

        # Train epochs
        for ep in range(self.config.EPOCHS):
            t0 = time.time()
            self.model.train()

            total, task_sum, kd_sum, si_sum = 0.0, 0.0, 0.0, 0.0
            n_samples = 0

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                opt.zero_grad(set_to_none=True)

                # Forward with adapter scaling
                yp = self._forward_with_adapter(x, adapter_scale=adapter_scale)
                task_loss = F.mse_loss(yp, y)

                # KD against teacher if exists
                kd_loss = torch.zeros((), device=self.device)
                if alpha > 0.0 and self.teacher is not None:
                    with torch.no_grad():
                        try:
                            y_old = self.teacher(x, adapter_scale=None)
                        except TypeError:
                            y_old = self.teacher(x)
                    kd_loss = F.mse_loss(yp, y_old)

                # SI penalty
                si_pen = self.si.penalty(self.model)

                loss = task_loss + alpha * kd_loss + lam * si_pen
                loss.backward()

                # Collect grads for SI
                grads, old = {}, {}
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        old[n] = p.data.detach().clone()
                        if p.grad is not None:
                            grads[n] = p.grad.detach().clone()

                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                # SI accumulate
                self.si.accumulate_step(self.model, grads, old)

                bs = x.size(0)
                n_samples += bs
                total += loss.item() * bs
                task_sum += task_loss.item() * bs
                kd_sum += kd_loss.item() * bs
                si_sum += si_pen.item() * bs

            train_loss = total / max(1, n_samples)
            tl = task_sum / max(1, n_samples)
            kloss = kd_sum / max(1, n_samples)
            sil = si_sum / max(1, n_samples)

            # Validation
            self.model.eval()
            v = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    yp = self._forward_with_adapter(x, adapter_scale=adapter_scale)
                    v += F.mse_loss(yp, y).item() * x.size(0)
            val_loss = v / len(val_loader.dataset)

            # Log & history
            lr = opt.param_groups[0]['lr']
            dt = time.time() - t0
            sched.step(val_loss)

            hist['epoch'].append(ep)
            hist['time'].append(dt)
            hist['lr'].append(lr)
            hist['train_loss'].append(train_loss)
            hist['val_loss'].append(val_loss)
            hist['task_loss'].append(tl)
            hist['kd_loss'].append(kloss)
            hist['si_loss'].append(sil)
            hist['KL'].append(kl)
            hist['alpha'].append(alpha)
            hist['lambda'].append(lam)
            hist['adapter'].append(adapter_scale)

            logger.info(
                "Ep %03d | train=%.4e val=%.4e | task=%.4e kd=%.4e si=%.4e | KL=%.4f a=%.3f l=%.3f ad=%.3f | lr=%.2e | %.1fs",
                ep, train_loss, val_loss, tl, kloss, sil, kl, alpha, lam, adapter_scale, lr, dt
            )

            # Early stopping
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                no_imp = 0
                if self.task_dir:
                    ckpt = {'model_state': best_state}
                    torch.save(ckpt, self.task_dir / f"task{task_id}_best.pt")
            else:
                no_imp += 1
                if no_imp >= self.config.PATIENCE:
                    logger.info("Early stopping at epoch %d", ep)
                    break

        # Restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Consolidate: build teacher & update KL reference
        self._consolidate_after_task(train_loader, task_id)

        return hist

    # ------------------------------
    # After-task consolidation
    # ------------------------------
    def _consolidate_after_task(self, train_loader: DataLoader, task_id: int):
        # Finalize SI for this task
        self.si.consolidate_task(self.model)

        # Update KD teacher
        self.teacher = copy.deepcopy(self.model).to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Update KL reference snapshot & loader
        self.ref_snapshot = copy.deepcopy(self.model).to(self.device).eval()
        for p in self.ref_snapshot.parameters():
            p.requires_grad_(False)
        self.ref_loader = train_loader

        logger.info("Task %d consolidated: teacher and KL reference updated.", task_id)
