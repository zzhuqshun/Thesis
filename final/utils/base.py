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

    def forward(self, x, return_hidden: bool = False, detach_hidden: bool = False):
        # LSTM forward
        out, _ = self.lstm(x)          # [B, T, H]
        h_last = out[:, -1, :]         # [B, H]
        y = self.fc(h_last).squeeze(-1)

        if return_hidden:
            h = h_last.detach() if detach_hidden else h_last
            return y, h
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

# ------------------------------ SI core ------------------------------
class SI:
    """
    Synaptic Intelligence (Zenke et al., 2017) - online importance accumulation.

    Usage per task:
      - begin_task(model): call at task start
      - For each optimizer step:
          pre_opt_step(model): snapshot params & grads BEFORE optimizer.step()
          post_opt_step(model): update path integral w AFTER optimizer.step()
      - consolidate(model): call at task end -> build Omega and theta_star
      - penalty(model): during forward, add SI regularizer
    """
    def __init__(self, model: nn.Module, device: torch.device, epsilon: float = 1e-3):
        self.device = device
        self.epsilon = float(epsilon)

        # Online accumulators & references
        self.w = {}                  # path integral accumulator
        self.theta_start = {}        # params at task start
        self.omega = {}              # parameter importance
        self.theta_star = {}         # consolidated reference (end of prev task)

        # buffers for one-step update
        self._grads = {}
        self._theta_old = {}

        for n, p in model.named_parameters():
            if p.requires_grad:
                z = torch.zeros_like(p, device=self.device)
                self.w[n] = z.clone()
                self._grads[n] = z.clone()
                self._theta_old[n] = z.clone()

    @torch.no_grad()
    def begin_task(self, model: nn.Module):
        """Mark task start."""
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.theta_start[n] = p.detach().clone()

    @torch.no_grad()
    def pre_opt_step(self, model: nn.Module):
        """Snapshot params & grads BEFORE optimizer.step()."""
        for n, p in model.named_parameters():
            if p.requires_grad:
                self._theta_old[n] = p.detach().clone()
                if p.grad is None:
                    self._grads[n].zero_()
                else:
                    self._grads[n] = p.grad.detach().clone()

    @torch.no_grad()
    def post_opt_step(self, model: nn.Module):
        """Accumulate path integral AFTER optimizer.step()."""
        for n, p in model.named_parameters():
            if p.requires_grad:
                delta = p.detach() - self._theta_old[n]
                # w += - g * delta
                self.w[n].add_(- self._grads[n] * delta)

    @torch.no_grad()
    def consolidate(self, model: nn.Module):
        """Build/update Omega and set theta_star at task end."""
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n not in self.omega:
                self.omega[n] = torch.zeros_like(p, device=self.device)

            # Δθ across this task
            delta = p.detach() - self.theta_start[n]
            denom = (delta.pow(2)).add_(self.epsilon)
            contrib = torch.relu(self.w[n]) / denom
            self.omega[n].add_(contrib)

            # reset path integral and update star
            self.w[n].zero_()
            self.theta_star[n] = p.detach().clone()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Quadratic penalty Σ Ω_i (θ_i - θ_i^*)^2 ."""
        if not self.omega or not self.theta_star:
            return torch.zeros((), device=self.device)

        loss = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.omega:
                delta = p - self.theta_star[n]
                loss = loss + (self.omega[n] * delta.pow(2)).sum()
        return loss

# ------------------------------ Incremental Trainer (SI + KD + KL) ------------------------------
class IncTrainer:
    """
    Incremental training with SI + KD (LwF-style) and KL-driven scheduling.

    Total loss:
        L = L_sup + λ_SI(KL) * L_SI + λ_KD(KL) * L_KD
    where scheduling uses: s = KL / (KL + TAU) in [0, 1), and
        λ_SI = SI_FLOOR + s * (SI_MAX - SI_FLOOR)
        λ_KD = KD_FLOOR + s * (KD_MAX - KD_FLOOR)

    - KL can be computed on input features ('input'), hidden representation ('hidden'), or their sum ('both').
    - KD uses a frozen teacher (model at the end of previous task). For regression we use MSE by default.
    """

    def __init__(self, model: nn.Module, device: torch.device, config: Config, inc_dir: Path):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.inc_dir = Path(inc_dir)
        self.inc_dir.mkdir(parents=True, exist_ok=True)

        # SI regularizer
        self.si = SI(self.model, device, epsilon=self.config.SI_EPSILON)

        # Teacher model & reference statistics from previous task
        self.prev_model = None
        self.prev_stats = None  # {'x': {'mu': Tensor[3], 'var': Tensor[3]}, 'h': {'mu': Tensor[H], 'var': Tensor[H]}}

    # --------- helpers: KL & stats ----------
    @staticmethod
    def _diag_gauss_kl(mu1: torch.Tensor, var1: torch.Tensor,
                       mu0: torch.Tensor, var0: torch.Tensor, eps: float) -> torch.Tensor:
        """KL(N1||N0) for diagonal Gaussians. Returns scalar (sum over dims)."""
        var1 = var1.clamp_min(eps)
        var0 = var0.clamp_min(eps)
        log_ratio = torch.log(var0) - torch.log(var1)
        frac = (var1 + (mu1 - mu0).pow(2)) / var0
        kl = 0.5 * (log_ratio + frac - 1.0).sum()
        return kl

    def _batch_stats_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, 3] -> compute stats over B*T for each feature dim."""
        b, t, d = x.shape
        flat = x.reshape(b * t, d)
        mu = flat.mean(dim=0)
        var = flat.var(dim=0, unbiased=False)
        return mu, var

    @torch.no_grad()
    def _batch_hidden(self, x: torch.Tensor, use_teacher: bool = True) -> torch.Tensor:
        """Return [B, H] hidden from teacher (default) to isolate domain shift."""
        m = self.prev_model if (use_teacher and self.prev_model is not None) else self.model
        m.eval()
        _, h = m(x, return_hidden=True, detach_hidden=True)
        return h

    def _batch_stats_hidden(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = h.mean(dim=0)
        var = h.var(dim=0, unbiased=False)
        return mu, var

    def _compute_kl(self, x: torch.Tensor) -> float:
        """Compute KL according to KL_MODE with prev_stats."""
        if (not self.config.USE_KL) or (self.prev_stats is None):
            return 0.0

        eps = self.config.KL_EPS
        s = 0.0

        if self.config.KL_MODE in ('input', 'both'):
            mu1, var1 = self._batch_stats_input(x)                        # current batch
            mu0, var0 = self.prev_stats['x']['mu'], self.prev_stats['x']['var']
            kl_x = self._diag_gauss_kl(mu1.to(self.device), var1.to(self.device),
                                       mu0.to(self.device), var0.to(self.device), eps)
            # normalize by dim to keep scales comparable
            s += (kl_x / mu1.numel()).item()

        if self.config.KL_MODE in ('hidden', 'both')    :
            with torch.no_grad():
                h = self._batch_hidden(x, use_teacher=True)
            mu1, var1 = self._batch_stats_hidden(h)                       # current batch hidden
            mu0, var0 = self.prev_stats['h']['mu'], self.prev_stats['h']['var']
            kl_h = self._diag_gauss_kl(mu1.to(self.device), var1.to(self.device),
                                       mu0.to(self.device), var0.to(self.device), eps)
            s += (kl_h / mu1.numel()).item()

        return float(max(0.0, s))

    def _sched(self, kl_value: float) -> float:
        """Map KL -> [0,1): s = KL / (KL + TAU)."""
        tau = max(1e-12, float(self.config.TAU))
        return float(kl_value / (kl_value + tau))

    def _kd_loss(self, student_y: torch.Tensor, teacher_y: torch.Tensor) -> torch.Tensor:
        typ = self.config.KD_LOSS.lower()
        if typ == 'l1':
            return F.l1_loss(student_y, teacher_y)
        elif typ == 'smoothl1':
            return F.smooth_l1_loss(student_y, teacher_y)
        # default 'mse'
        return F.mse_loss(student_y, teacher_y)

    # --------- training per task ----------
    def train_task(self, train_loader: DataLoader, val_loader: DataLoader, task_id: int):
        """
        Train on one task with SI + KD and KL-driven scheduling.
        Saves best checkpoint under inc_dir/task{task_id}/task{task_id}_best.pt
        """
        task_dir = self.inc_dir / f"task{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)

        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

        # SI begin for this task
        if self.config.USE_SI:
            self.si.begin_task(self.model)

        best_val = float('inf')
        best_state = None
        no_imp = 0

        # history fields (minimal but informative)
        hist = {k: [] for k in [
            'epoch', 'train_loss', 'train_sup','train_kd', 'train_si', 'val_loss',
            'kl', 'lam_kd', 'lam_si', 'kd_ratio', 'si_ratio',
            'lr', 'time'
        ]}

        for epoch in range(self.config.EPOCHS):
            t0 = time.time()
            self.model.train()

            # epoch accumulators (sample-weighted)
            tot_sup = tot_kd = tot_si = tot_kl = 0.0
            tot_lam_kd = tot_lam_si = 0.0
            nb = 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # KL-driven schedule (compute only if needed)
                do_sched = (task_id > 0) and self.config.USE_KL and (self.config.USE_SI or self.config.USE_KD)
                kl_val = self._compute_kl(x) if do_sched else 0.0
                s = (kl_val / (kl_val + self.config.TAU)) if kl_val > 0 else 0.0

                # adaptive lambdas (0 on task0 or when toggles are off)
                lam_kd = 0.0
                lam_si = 0.0
                if task_id > 0 and self.config.USE_KD:
                    lam_kd = self.config.KD_FLOOR + (1-s) * (self.config.KD_MAX - self.config.KD_FLOOR)
                    if getattr(self.config, "KD_WARMUP_EPOCHS", 0) > 0:
                        warm = min(1.0, epoch / float(self.config.KD_WARMUP_EPOCHS))
                        lam_kd *= warm
                if task_id > 0 and self.config.USE_SI:
                    lam_si = self.config.SI_FLOOR + s * (self.config.SI_MAX - self.config.SI_FLOOR)
                    if getattr(self.config, "SI_WARMUP_EPOCHS", 0) > 0:
                        warm = min(1.0, epoch / float(self.config.SI_WARMUP_EPOCHS))
                        lam_si *= warm

                # supervised forward
                opt.zero_grad(set_to_none=True)
                y_pred = self.model(x)
                loss_sup = F.mse_loss(y_pred, y)

                # KD loss (teacher prediction)
                loss_kd = torch.zeros((), device=self.device)
                if task_id > 0 and self.config.USE_KD and (self.prev_model is not None):
                    with torch.no_grad():
                        y_t = self.prev_model(x)
                    loss_kd = self._kd_loss(y_pred, y_t)

                # SI penalty
                loss_si = torch.zeros((), device=self.device)
                if task_id > 0 and self.config.USE_SI:
                    loss_si = self.si.penalty(self.model)
                    
                # total loss
                loss = loss_sup + lam_kd * loss_kd + lam_si * loss_si
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
                
                if self.config.USE_SI:
                    self.si.pre_opt_step(self.model)
                
                opt.step()

                # SI post-step
                if self.config.USE_SI:
                    self.si.post_opt_step(self.model)

                # accumulate (sample-weighted)
                bs = x.size(0)
                tot_sup += loss_sup.item() * bs
                tot_kd  += (lam_kd * loss_kd).item() * bs
                tot_si  += (lam_si * loss_si).item() * bs
                tot_kl  += kl_val * bs
                tot_lam_kd += lam_kd * bs
                tot_lam_si += lam_si * bs
                nb += bs

            # epoch metrics
            train_loss = (tot_sup + tot_kd + tot_si) / max(1, nb)

            # validation (pure supervised objective)
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    val_loss += F.mse_loss(self.model(x), y).item() * x.size(0)
            val_loss /= len(val_loader.dataset)

            plateau.step(val_loss)

            lr_cur = opt.param_groups[0]['lr']
            elapsed = time.time() - t0

            # epoch-level averages
            avg_kl = tot_kl / max(1, nb)
            lam_kd_avg = tot_lam_kd / max(1, nb)
            lam_si_avg = tot_lam_si / max(1, nb)

            # loss shares (portion of total training loss from KD/SI)
            total_acc = tot_sup + tot_kd + tot_si
            if total_acc > 0:
                kd_ratio = tot_kd / total_acc
                si_ratio = tot_si / total_acc
            else:
                kd_ratio = si_ratio = 0.0

            # record history
            hist['epoch'].append(epoch)
            hist['train_loss'].append(train_loss)
            hist['train_sup'].append(tot_sup / nb)
            hist['train_kd'].append(tot_kd / nb)
            hist['train_si'].append(tot_si / nb)
            hist['val_loss'].append(val_loss)
            hist['kl'].append(avg_kl)
            hist['lam_kd'].append(lam_kd_avg)
            hist['lam_si'].append(lam_si_avg)
            hist['kd_ratio'].append(kd_ratio)
            hist['si_ratio'].append(si_ratio)
            hist['lr'].append(lr_cur)
            hist['time'].append(elapsed)

            # concise logging
            logger.info(
                "[Task %d][Epoch %d] train=%.4e val=%.4e | KL=%.3f | Lambda: KD=%.4f SI=%.4f | Ratio: KD=%.2f%% SI=%.2f%% | lr=%.2e time=%.2fs",
                task_id, epoch, train_loss, val_loss, avg_kl, lam_kd_avg, lam_si_avg,
                kd_ratio*100, si_ratio*100, lr_cur, elapsed
            )

            # save best
            if val_loss < best_val - 1e-12:
                best_val = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                no_imp = 0
            else:
                no_imp += 1

            if no_imp >= self.config.PATIENCE:
                logger.info("[Task %d] Early stopping at epoch %d", task_id, epoch)
                break

        # load best
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # consolidate SI (build/update omega) and prepare teacher & reference stats for NEXT task
        if self.config.USE_SI:
            self.si.consolidate(self.model)

        # after-task finalize -> build teacher & stats for next task
        self._after_task_finalize(train_loader)

        # if task_id == 0:
        #     ckpt_path = task_dir / f"task{task_id}_best.pt"

        #     try:
        #         ckpt = torch.load(ckpt_path, map_location="cpu")
        #     except FileNotFoundError:
        #         ckpt = {"model_state": copy.deepcopy(self.model.state_dict())}

        #     si_state = None
        #     if self.config.USE_SI and (self.si.omega or self.si.theta_star):
        #         si_state = {
        #             "omega": {n: t.detach().cpu() for n, t in self.si.omega.items()},
        #             "theta_star": {n: t.detach().cpu() for n, t in self.si.theta_star.items()},
        #         }

        #     ckpt.update({
        #         "si_state": si_state
        #     })
        #     torch.save(ckpt, ckpt_path)
        #     logger.info("[Task %d] Augmented checkpoint saved with SI & KL stats at %s",
        #                 task_id, ckpt_path)
            
        return hist


    @torch.no_grad()
    def _after_task_finalize(self, train_loader: DataLoader):
        """Freeze current model as teacher and compute reference stats on inputs/hidden for KL of the next task."""
        # teacher
        self.prev_model = copy.deepcopy(self.model).to(self.device)
        self.prev_model.eval()
        for p in self.prev_model.parameters():
            p.requires_grad_(False)

        # stats
        need_x = self.config.KL_MODE in ('input', 'both')
        need_h = self.config.KL_MODE in ('hidden', 'both')

        mu_x = var_x = mu_h = var_h = None
        n_x = n_h = 0

        for x, _ in train_loader:
            x = x.to(self.device)
            if need_x:
                bx, tx, dx = x.shape
                xf = x.reshape(bx*tx, dx).float()
                if mu_x is None:
                    mu_x = xf.mean(dim=0)
                    var_x = xf.var(dim=0, unbiased=False)
                    n_x = bx * tx
                else:
                    # running merge of two batches
                    m2 = xf.mean(dim=0)
                    v2 = xf.var(dim=0, unbiased=False)
                    n2 = bx * tx
                    # merge means/vars (parallel algorithm)
                    delta = m2 - mu_x
                    tot = n_x + n2
                    new_mu = mu_x + delta * (n2 / tot)
                    new_var = (n_x*var_x + n2*v2 + delta.pow(2) * (n_x*n2)/tot) / tot
                    mu_x, var_x, n_x = new_mu, new_var, tot

            if need_h:
                y, h = self.prev_model(x, return_hidden=True, detach_hidden=True)
                if mu_h is None:
                    mu_h = h.mean(dim=0)
                    var_h = h.var(dim=0, unbiased=False)
                    n_h = h.size(0)
                else:
                    m2 = h.mean(dim=0)
                    v2 = h.var(dim=0, unbiased=False)
                    n2 = h.size(0)
                    delta = m2 - mu_h
                    tot = n_h + n2
                    new_mu = mu_h + delta * (n2 / tot)
                    new_var = (n_h*var_h + n2*v2 + delta.pow(2) * (n_h*n2)/tot) / tot
                    mu_h, var_h, n_h = new_mu, new_var, tot

        stats = {}
        if need_x:
            stats.setdefault('x', {})['mu'] = mu_x if mu_x is not None else torch.zeros(3, device=self.device)
            stats['x']['var'] = (var_x if var_x is not None else torch.ones(3, device=self.device))
        if need_h:
            # infer hidden size from model if empty
            if mu_h is None:
                # try to guess hidden size from model FC input
                hdim = getattr(self.model.lstm, 'hidden_size', 128)
                mu_h = torch.zeros(hdim, device=self.device)
                var_h = torch.ones(hdim, device=self.device)
            stats.setdefault('h', {})['mu'] = mu_h
            stats['h']['var'] = var_h

        self.prev_stats = stats

    # ---------- convenience: run all tasks ----------
    def run_incremental(self, loaders: dict):
        """
        loaders must contain keys: 'task{i}_train', 'task{i}_val' for i in [0..NUM_TASKS-1]
        Returns: list of per-task histories.
        """
        histories = []
        for t in range(self.config.NUM_TASKS):
            tr_key = f"task{t}_train"
            va_key = f"task{t}_val"
            if tr_key not in loaders or va_key not in loaders:
                raise KeyError(f"Missing dataloaders for task {t}: '{tr_key}' / '{va_key}'")

            logger.info("==== Incremental Training (SI+KD+KL) Task %d ====", t)
            hist = self.train_task(loaders[tr_key], loaders[va_key], task_id=t)
            histories.append(hist)
        return histories
