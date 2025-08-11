# si.py
# Synaptic Intelligence (SI) regularizer and a trainer with KD support.
# Author: Jason + Assistant
# Note: Comments are in English per your preference.

import copy
import time
from pathlib import Path
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SI:
    """
    Synaptic Intelligence (SI) regularizer.

    Core idea:
      - Accumulate per-parameter contribution during training:
          w_i += - g_i(t) * Δθ_i(t)
        where g_i is gradient (before the optimizer step) and Δθ_i is the
        parameter change in the current step (after the optimizer step).
      - After finishing a task, convert accumulated w to Omega:
          Ω_i += relu(w_i) / ( (θ_i^T - θ_i^0)^2 + eps )
        and store theta_star (parameters at the end of the task).
      - During the next tasks, add penalty:
          L_SI = sum_i Ω_i * (θ_i - θ_i^*)^2
    """

    def __init__(self, model: nn.Module, device: torch.device, epsilon: float = 1e-3):
        self.device = device
        self.epsilon = epsilon

        # Accumulators (reset every task)
        self.w: Dict[str, torch.Tensor] = {}
        self.theta_task_start: Dict[str, torch.Tensor] = {}

        # Cumulative importance and reference params across tasks
        self.omega: Dict[str, torch.Tensor] = {}
        self.theta_star: Dict[str, torch.Tensor] = {}

        # Initialize structures based on the model
        for n, p in model.named_parameters():
            if p.requires_grad:
                z = torch.zeros_like(p, device=self.device)
                self.w[n] = z.clone()
                self.theta_task_start[n] = p.detach().clone()

    def begin_task(self, model: nn.Module):
        """Reset per-task accumulators at the beginning of a task."""
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.w[n] = torch.zeros_like(p, device=self.device)
                self.theta_task_start[n] = p.detach().clone()

    @torch.no_grad()
    def accumulate_step(self,
                        model: nn.Module,
                        grads: Dict[str, torch.Tensor],
                        old_params: Dict[str, torch.Tensor]):
        """
        Accumulate path integral after an optimizer step.

        Args:
            model: current model after optimizer.step()
            grads: gradients captured BEFORE optimizer.step()
            old_params: parameter tensors captured BEFORE optimizer.step()
        """
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n not in grads:
                # No grad -> skip (e.g., frozen layer)
                continue
            delta = p.data - old_params[n]
            # w_i += - g_i * Δθ_i
            self.w[n] += (-grads[n]) * delta

    @torch.no_grad()
    def consolidate_task(self, model: nn.Module):
        """
        Convert accumulated w to Omega for the finished task and update theta_star.
        """
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue

            theta_now = p.detach()
            theta_start = self.theta_task_start[n]
            delta_total = theta_now - theta_start
            denom = delta_total.pow(2) + self.epsilon

            # Omega increment from this task
            w = self.w[n]
            omega_add = torch.relu(w) / denom

            if n in self.omega:
                self.omega[n] = self.omega[n] + omega_add
            else:
                self.omega[n] = omega_add.clone()

            # Update theta_star to the parameters at end of this task
            self.theta_star[n] = theta_now.clone()

        # Clear per-task accumulators
        for n in list(self.w.keys()):
            self.w[n].zero_()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute SI penalty using cumulative Ω and θ*."""
        loss = torch.zeros((), device=self.device)
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if (n in self.omega) and (n in self.theta_star):
                loss = loss + (self.omega[n] * (p - self.theta_star[n]) ** 2).sum()
        return loss


class SITrainer:
    """
    Trainer that supports:
      - Task loss (MSE)
      - KD loss (MSE to old model outputs)
      - SI penalty (with task-dependent lambda)
      - Early stopping & LR scheduling
      - Checkpointing best validation model per task
    """

    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 config,
                 task_dir: Optional[Path] = None,
                 si_epsilon: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.task_dir = Path(task_dir) if task_dir else None
        if self.task_dir:
            self.task_dir.mkdir(parents=True, exist_ok=True)

        self.si = SI(self.model, self.device, epsilon=si_epsilon)
        self.old_model: Optional[nn.Module] = None  # for KD

    def _forward_with_adapter_scale(self, x, adapter_scale: Optional[float] = None):
        """
        Call model forward. If the model supports adapter_scale, pass it.
        Otherwise fallback to plain forward(x).
        """
        if adapter_scale is None:
            return self.model(x)

        # Try passing adapter_scale; fallback if signature doesn't accept it.
        try:
            return self.model(x, adapter_scale=adapter_scale)
        except TypeError:
            return self.model(x)

    def train_task(self,
                   train_loader,
                   val_loader,
                   task_id: int,
                   alpha_lwf: float = 0.0,
                   lambda_si: float = 0.0,
                   adapter_scale: Optional[float] = None):
        """
        Train on a single task with SI + KD.

        Args:
            train_loader: dataloader of current task
            val_loader:   validation dataloader of current task
            task_id:      current task index
            alpha_lwf:    KD weight
            lambda_si:    SI penalty weight for this task
            adapter_scale: external gate for adapter (e.g., driven by KL)
        """
        # Prepare optimizer/scheduler
        opt = torch.optim.Adam(self.model.parameters(),
                               lr=self.config.LEARNING_RATE,
                               weight_decay=self.config.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)

        # Early stop
        best_val = float('inf')
        no_imp = 0
        best_state = None

        # History
        history = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'lr', 'time',
                                   'task_loss', 'kd_loss', 'si_loss']}

        # SI: mark task start
        self.si.begin_task(self.model)

        for epoch in range(self.config.EPOCHS):
            t0 = time.time()
            self.model.train()

            tot_loss = 0.0
            sum_task = sum_kd = sum_si = 0.0

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                opt.zero_grad(set_to_none=True)

                # Forward with optional adapter gate
                yp = self._forward_with_adapter_scale(x, adapter_scale)

                # Task loss (regression)
                task_loss = F.mse_loss(yp, y)

                # KD loss (output-level distillation)
                kd_loss = torch.zeros((), device=self.device)
                if alpha_lwf > 0.0 and self.old_model is not None:
                    with torch.no_grad():
                        old_out = self._forward_with_adapter_scale(x, adapter_scale=None)
                        # Old model should be teacher; call it explicitly
                        try:
                            old_out = self.old_model(x, adapter_scale=None)
                        except TypeError:
                            old_out = self.old_model(x)
                    kd_loss = F.mse_loss(yp, old_out)

                # SI penalty (uses cumulative omega & theta_star)
                si_pen = self.si.penalty(self.model)

                # Total loss
                loss = task_loss + alpha_lwf * kd_loss + lambda_si * si_pen
                loss.backward()

                # Capture grads and pre-step params for SI accumulation
                grads = {}
                old_params = {}
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        old_params[n] = p.data.detach().clone()
                        if p.grad is not None:
                            grads[n] = p.grad.detach().clone()

                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                # SI path integral accumulation
                self.si.accumulate_step(self.model, grads, old_params)

                # Stats
                bs = x.size(0)
                tot_loss += loss.item() * bs
                sum_task += task_loss.item() * bs
                sum_kd += kd_loss.item() * bs
                sum_si += si_pen.item() * bs

            n = len(train_loader.dataset)
            train_loss = tot_loss / max(1, n)

            # Log ratio for diagnostics
            if sum_task > 0:
                reg_ratio = (lambda_si * (sum_si / max(1, n))) / (sum_task / max(1, n)) * 100.0
                logger.info("Epoch %d: SI penalty is %.2f%% of task loss", epoch, reg_ratio)

            # History
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['task_loss'].append(sum_task / max(1, n))
            history['kd_loss'].append(sum_kd / max(1, n))
            history['si_loss'].append(sum_si / max(1, n))
            lr_cur = opt.param_groups[0]['lr']
            history['lr'].append(lr_cur)
            history['time'].append(time.time() - t0)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    yp = self._forward_with_adapter_scale(x, adapter_scale)
                    val_loss += F.mse_loss(yp, y).item() * x.size(0)
            val_loss = val_loss / len(val_loader.dataset)
            history['val_loss'].append(val_loss)

            sched.step(val_loss)

            logger.info(
                "Epoch %d task=%.4e kd=%.4e si=%.4e val=%.4e lr=%.2e time=%.2fs",
                epoch, history['task_loss'][-1], history['kd_loss'][-1],
                history['si_loss'][-1], val_loss, lr_cur, history['time'][-1]
            )

            # Early stopping
            if val_loss < best_val:
                best_val = val_loss
                no_imp = 0
                best_state = copy.deepcopy(self.model.state_dict())
                if self.task_dir:
                    torch.save({'model_state': best_state},
                               self.task_dir / f"task{task_id}_best.pt")
            else:
                no_imp += 1
                if no_imp >= self.config.PATIENCE:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Restore best checkpoint
        if best_state:
            self.model.load_state_dict(best_state)

        return history

    def consolidate(self, task_id: Optional[int] = None):
        """
        Finalize the finished task:
          - consolidate SI importance
          - snapshot old_model for next task's KD
        """
        # Consolidate SI (compute Omega, update theta_star)
        self.si.consolidate_task(self.model)

        # Save teacher for KD
        self.old_model = copy.deepcopy(self.model).to(self.device)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)

        logger.info("Task %s consolidated (SI). KD teacher updated.", str(task_id))
