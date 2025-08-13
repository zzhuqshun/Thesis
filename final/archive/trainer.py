# utils/trainer.py
import copy
import time
import math
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ================================================================
# SI (Synaptic Intelligence) - adapter-free implementation
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

        # Online accumulators and references
        self.w: Dict[str, torch.Tensor] = {}
        self.prev_params: Dict[str, torch.Tensor] = {}

        # Consolidated importance and anchors
        self.omega: Dict[str, torch.Tensor] = {}
        self.theta_star: Dict[str, torch.Tensor] = {}

        for n, p in model.named_parameters():
            if p.requires_grad:
                z = torch.zeros_like(p, device=self.device)
                self.w[n] = z.clone()
                self.prev_params[n] = p.data.detach().clone()
                self.omega[n] = z.clone()
                self.theta_star[n] = p.data.detach().clone()

    def begin_task(self, model: nn.Module):
        """Reset online accumulators at the start of a task."""
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.w[n].zero_()
                self.prev_params[n] = p.data.detach().clone()

    def accumulate_importance(self, model: nn.Module, grad_snapshot: Dict[str, torch.Tensor]):
        """
        Accumulate path integral importance:
        w[n] += - g[n] * (theta_t - theta_{t-1})
        Call AFTER optimizer.step() with gradients snapshot taken BEFORE step.
        """
        for n, p in model.named_parameters():
            if (not p.requires_grad) or (n not in grad_snapshot):
                continue
            delta = p.data.detach() - self.prev_params[n]
            self.w[n] += (-grad_snapshot[n]) * delta
            self.prev_params[n] = p.data.detach().clone()

    def end_task(self, model: nn.Module):
        """Convert online w to omega and set theta_star to current params."""
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            delta = (p.data.detach() - self.theta_star[n]) ** 2
            denom = delta + self.epsilon
            self.omega[n] += self.w[n] / denom
            self.theta_star[n] = p.data.detach().clone()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute quadratic SI penalty from consolidated omega/theta_star."""
        loss = 0.0
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            loss = loss + (self.omega[n] * (p - self.theta_star[n]) ** 2).sum()
        return loss


# ================================================================
# KL helpers (diagonal Gaussian on per-channel statistics)
# ================================================================
def batch_mean_var(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel mean/var for input batch.
    Accepts (B, T, C) or (B, C, T). We flatten B and T and keep C.
    """
    if x.dim() == 3:
        # Heuristic: more likely (B, T, C) for time series
        if x.shape[1] > x.shape[2]:
            feat = x.reshape(-1, x.shape[2])       # (B*T, C)
        else:
            feat = x.transpose(1, 2).reshape(-1, x.shape[1])  # (B*T, C)
    elif x.dim() == 2:
        feat = x  # (B, C)
    else:
        feat = x.view(x.size(0), -1)

    mu = feat.mean(dim=0)
    var = feat.var(dim=0, unbiased=False)
    return mu, var


def gaussian_diag_kl(mu_q: torch.Tensor, var_q: torch.Tensor,
                     mu_p: torch.Tensor, var_p: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """
    KL( Q || P ) for diagonal Gaussians; all inputs are 1D tensors with shape (C,).
    """
    var_q = torch.clamp(var_q, min=eps)
    var_p = torch.clamp(var_p, min=eps)
    # Standard closed form:
    # KL = 0.5 * sum( log(var_p/var_q) + (var_q + (mu_q - mu_p)^2)/var_p - 1 )
    log_ratio = torch.log(var_p / var_q)
    sq_diff = (mu_q - mu_p) ** 2
    kl = 0.5 * (log_ratio + (var_q + sq_diff) / var_p - 1.0)
    return torch.clamp(kl.mean(), min=0.0)


# ================================================================
# Trainer (NO adapter; one-shot dataset-level KL -> fixed alpha/lambda)
# with one-shot auto-calibration of alpha_max and lam_base
# ================================================================
class Trainer:
    """
    Adapter-free trainer that supports:
      - Task loss (L1)
      - KD loss against a frozen teacher (MSE on predictions)
      - SI regularization
      - One-shot dataset-level KL (previous task vs current task) to compute
        fixed KD weight (alpha) and SI weight (lambda) per task.
      - One-shot auto-calibration for starting magnitudes of KD/SI via target ratios.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config,
        task_dir: Optional[Path] = None,
        si_epsilon: float = 1e-3,
        # ---- one-shot KL mapping (defaults good for many cases) ----
        use_dynamic: bool = False,       # kept for API compatibility (unused)
        alpha_max: float = 1.0,          # KD upper bound when KL ~ 0
        alpha_c: float = 3.2,            # decay coefficient for KD
        lam_base: float = 1.0,           # SI upper/base when KL ~ 0
        lam_c: float = 2.4,              # decay coefficient for SI
        lam_min: float = 0.0,            # min clamp for SI
        kl_smooth: float = 0.1,          # kept for logging compatibility (unused)
        kl_clip: Optional[float] = None, # kept for API compatibility (unused)
        fixed_alpha: float = 0.0,        # used for first task if no ref
        fixed_lambda: float = 0.0,       # used for first task if no ref
        # ---- auto-calibration (optional but recommended) ----
        calibrate: bool = True,          # turn on one-shot auto-calibration
        calib_batches: int = 2,          # how many batches to estimate unit losses
        target_kd_ratio: float = 0.10,   # KD aims ~10% of task loss when KL≈0
        target_si_ratio: float = 0.10,   # SI aims ~10% of task loss when KL≈0
        alpha_cap: float = 5.0,          # cap to avoid exploding alpha_max
        lam_cap: float = 100.0,          # cap to avoid exploding lam_base
        kd_floor: float = 1e-8,          # if KD unit loss < floor, disable KD
        si_floor: float = 1e-8,          # if SI unit loss < floor, disable SI
    ):
        self.model = model
        self.device = device
        self.cfg = config
        self.task_dir = task_dir

        # SI across tasks
        self.si = SI(model, device, epsilon=si_epsilon)

        # KD teacher (frozen); becomes current model after each task
        self.teacher: Optional[nn.Module] = None

        # One-shot KL -> fixed weights
        self.alpha_max = alpha_max
        self.alpha_c = alpha_c
        self.lam_base = lam_base
        self.lam_c = lam_c
        self.lam_min = lam_min

        # For first task (no ref), fall back to fixed values (often 0, 0)
        self.fixed_alpha_fallback = float(fixed_alpha)
        self.fixed_lambda_fallback = float(fixed_lambda)

        # Reference distribution from previous task (inputs' per-channel stats)
        self.ref_mu: Optional[torch.Tensor] = None
        self.ref_var: Optional[torch.Tensor] = None

        # For logging
        self.kl_once: float = 0.0
        self.alpha_star: float = self.fixed_alpha_fallback
        self.lambda_star: float = self.fixed_lambda_fallback

        # Auto-calibration opts
        self.calibrate = calibrate
        self.calib_batches = calib_batches
        self.target_kd_ratio = target_kd_ratio
        self.target_si_ratio = target_si_ratio
        self.alpha_cap = alpha_cap
        self.lam_cap = lam_cap
        self.kd_floor = kd_floor
        self.si_floor = si_floor

    # ----------------------------
    # Public API per task
    # ----------------------------
    def train_task(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        task_id: int,
    ) -> List[Dict]:
        """
        Train for one task with early stopping; returns per-epoch history.
        Per-task KD/SI weights are computed ONCE from dataset-level KL.
        """
        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.LEARNING_RATE,
            weight_decay=self.cfg.WEIGHT_DECAY
        )
        scheduler = None  # keep simple and stable

        # SI bookkeeping
        self.si.begin_task(self.model)

        # --- One-shot: compute dataset-level KL (current train vs previous ref) ---
        if (self.ref_mu is not None) and (self.ref_var is not None):
            new_mu, new_var = self._dataset_mean_var(train_loader)
            new_mu = new_mu.to(self.device)
            new_var = new_var.to(self.device)
            kl_t = gaussian_diag_kl(new_mu, new_var, self.ref_mu.to(self.device), self.ref_var.to(self.device))
            self.kl_once = float(kl_t.item())

            # ---- One-shot auto-calibration of alpha_max / lam_base (optional) ----
            if self.calibrate:
                with torch.no_grad():
                    # Estimate mean losses over a few small batches
                    task_vals, kd_vals, si_vals = [], [], []
                    seen = 0
                    for xb_c, yb_c in train_loader:
                        xb_c = xb_c.to(self.device, non_blocking=True)
                        yb_c = yb_c.to(self.device, non_blocking=True)
                        # unit task loss (mean)
                        pred_c = self.model(xb_c)
                        l_task = F.l1_loss(pred_c, yb_c).item()
                        task_vals.append(l_task)

                        # unit KD loss (mean) with alpha=1
                        if self.teacher is not None:
                            t_out_c = self.teacher(xb_c)
                            l_kd = F.mse_loss(pred_c, t_out_c).item()
                        else:
                            l_kd = 0.0
                        kd_vals.append(l_kd)

                        # unit SI loss with lambda=1 (current omega/theta_star)
                        l_si = self.si.penalty(self.model).item()
                        si_vals.append(l_si)

                        seen += 1
                        if seen >= self.calib_batches:
                            break

                    L_task = float(np.mean(task_vals)) if task_vals else 0.0
                    L_kd1  = float(np.mean(kd_vals))   if kd_vals   else 0.0
                    L_si1  = float(np.mean(si_vals))   if si_vals   else 0.0

                # Derive alpha_max and lam_base s.t. they contribute ~target ratio at KL≈0
                if (self.teacher is None) or (L_kd1 < self.kd_floor):
                    alpha0 = 0.0
                    kd_msg = f"KD disabled (teacher missing or kd_unit={L_kd1:.3e} < floor)"
                else:
                    alpha0 = self.target_kd_ratio * L_task / max(L_kd1, self.kd_floor)
                    alpha0 = float(min(self.alpha_cap, max(0.0, alpha0)))
                    kd_msg = f"alpha_max(auto)={alpha0:.4f} from L_task={L_task:.4e}, L_kd1={L_kd1:.4e}"

                if L_si1 < self.si_floor:
                    lam0 = 0.0
                    si_msg = f"SI disabled (si_unit={L_si1:.3e} < floor)"
                else:
                    lam0 = self.target_si_ratio * L_task / max(L_si1, self.si_floor)
                    lam0 = float(min(self.lam_cap, max(0.0, lam0)))
                    si_msg = f"lam_base(auto)={lam0:.4f} from L_task={L_task:.4e}, L_si1={L_si1:.4e}"

                # Override starting maxima with auto-calibrated ones
                self.alpha_max = alpha0
                self.lam_base = lam0
                logger.info("[AutoCalib] %s | %s", kd_msg, si_msg)

            # Map KL -> fixed alpha/lambda (monotonic decay)
            self.alpha_star = self.alpha_max * math.exp(-self.alpha_c * self.kl_once)
            self.lambda_star = max(self.lam_min, self.lam_base * math.exp(-self.lam_c * self.kl_once))
            logger.info("[One-shot KL] KL=%.6f => alpha=%.4f, lambda=%.4f",
                        self.kl_once, self.alpha_star, self.lambda_star)

            # Cache for next task right away (use this task's stats as new ref)
            self.cur_ref_mu = new_mu.detach().clone()
            self.cur_ref_var = new_var.detach().clone()
        else:
            # First task: no ref; keep fallback (often 0 and 0 for pure fit)
            self.kl_once = 0.0
            self.alpha_star = self.fixed_alpha_fallback
            self.lambda_star = self.fixed_lambda_fallback
            logger.info("[One-shot KL] No previous ref => alpha=%.4f, lambda=%.4f",
                        self.alpha_star, self.lambda_star)
            self.cur_ref_mu = None
            self.cur_ref_var = None

        best_val = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        wait = 0

        history: List[Dict] = []
        start_time = time.time()

        for epoch in range(self.cfg.EPOCHS):
            self.model.train()
            train_task_sum = 0.0
            kd_loss_sum = 0.0
            si_loss_sum = 0.0
            n_train = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                # Task loss (L1)
                pred = self.model(xb)
                loss_task = F.l1_loss(pred, yb)

                # KD loss (if teacher exists)
                loss_kd = torch.tensor(0.0, device=self.device)
                if self.teacher is not None and self.alpha_star > 0.0:
                    with torch.no_grad():
                        t_out = self.teacher(xb)
                    loss_kd = F.mse_loss(pred, t_out)

                # SI penalty
                loss_si = torch.tensor(0.0, device=self.device)
                if self.lambda_star > 0.0:
                    loss_si = self.si.penalty(self.model)

                # Total loss with fixed per-task weights
                loss = loss_task + self.alpha_star * loss_kd + self.lambda_star * loss_si

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Snapshot gradients for SI path integral
                grad_snapshot = {n: p.grad.detach().clone()
                                 for n, p in self.model.named_parameters()
                                 if p.requires_grad and p.grad is not None}

                optimizer.step()
                self.si.accumulate_importance(self.model, grad_snapshot)

                # Bookkeeping
                bs = yb.numel()
                train_task_sum += loss_task.item() * bs
                kd_loss_sum += loss_kd.item() * bs
                si_loss_sum += loss_si.item() * bs
                n_train += bs

            # Epoch metrics
            train_mae = train_task_sum / max(1, n_train)
            kd_mae = kd_loss_sum / max(1, n_train)
            si_val = si_loss_sum / max(1, n_train)

            # Validation (pure task loss)
            val_mae = self._evaluate_mae(val_loader)

            # Early stopping
            improved = val_mae < best_val - 1e-7
            if improved:
                best_val = val_mae
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1

            if scheduler is not None:
                scheduler.step(val_mae)

            history.append({
                "epoch": epoch + 1,
                "train_mae": float(train_mae),
                "val_mae": float(val_mae),
                "kd": float(kd_mae),
                "si": float(si_val),
                "kl": float(self.kl_once),
                "alpha": float(self.alpha_star),
                "lambda": float(self.lambda_star),
            })

            logger.info(
                "Task %d | Epoch %d/%d | train_mae=%.6f val_mae=%.6f | kd=%.6f si=%.6f | KL=%.6f alpha=%.4f lambda=%.4f",
                task_id, epoch + 1, self.cfg.EPOCHS, train_mae, val_mae, kd_mae, si_val,
                self.kl_once, self.alpha_star, self.lambda_star
            )

            # Save best checkpoint
            if improved and self.task_dir is not None:
                ckpt = self.task_dir / "best.pt"
                torch.save({"state_dict": best_state}, ckpt)

            if wait >= self.cfg.PATIENCE:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs).", epoch + 1, self.cfg.PATIENCE)
                break

        # Load best
        self.model.load_state_dict(best_state)

        # Finalize SI for this task (consolidate omega, set anchors)
        self.si.end_task(self.model)

        # Prepare teacher for next task
        self.teacher = copy.deepcopy(self.model).to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # Update reference distribution for NEXT task
        if self.cur_ref_mu is not None and self.cur_ref_var is not None:
            self.ref_mu = self.cur_ref_mu.detach().clone()
            self.ref_var = self.cur_ref_var.detach().clone()
        else:
            # If first task, derive ref from the just-finished train set to enable KL next task
            try:
                m, v = self._dataset_mean_var(train_loader)
                self.ref_mu = m.detach().clone()
                self.ref_var = v.detach().clone()
            except Exception:
                pass

        elapsed = time.time() - start_time
        logger.info("Task %d finished in %.1fs. Best val_mae=%.6f.", task_id, elapsed, best_val)
        return history

    # ----------------------------
    # Internals
    # ----------------------------
    @torch.no_grad()
    def _evaluate_mae(self, loader: DataLoader) -> float:
        """Compute MAE on a loader (pure task loss)."""
        self.model.eval()
        mae_sum, n = 0.0, 0
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            pred = self.model(xb)
            mae_sum += F.l1_loss(pred, yb, reduction="sum").item()
            n += yb.numel()
        return mae_sum / max(1, n)

    @torch.no_grad()
    def _dataset_mean_var(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate per-channel mean/var over the whole dataset loader.
        Weight by number of samples contributing (B*T per batch for sequences).
        """
        sum_mu = None
        sum_ex2 = None
        n_samples = 0
        for xb, _ in loader:
            xb = xb.to(self.device, non_blocking=True)
            mu_b, var_b = batch_mean_var(xb)
            # weight by number of contributing elements (B*T for sequences)
            bsz = xb.shape[0] * (xb.shape[1] if xb.dim() == 3 else 1)
            if sum_mu is None:
                sum_mu = mu_b * bsz
                sum_ex2 = (var_b + mu_b ** 2) * bsz
            else:
                sum_mu += mu_b * bsz
                sum_ex2 += (var_b + mu_b ** 2) * bsz
            n_samples += bsz

        mu = sum_mu / max(1, n_samples)
        ex2 = sum_ex2 / max(1, n_samples)
        var = torch.clamp(ex2 - mu ** 2, min=1e-8)
        return mu, var
