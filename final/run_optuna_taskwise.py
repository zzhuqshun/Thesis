# si_kd_optuna.py
import os
import copy
import logging
from pathlib import Path

import numpy as np
import optuna
import torch

from utils.config import Config
from utils.utils import set_seed, setup_logging
from utils.data import DataProcessor, create_dataloaders
from utils.base import SOHLSTM, IncTrainer
from utils.evaluate import evaluate  # returns preds, targets, metrics (incl. MAE)

# -----------------------------
# Utilities
# -----------------------------
def make_loaders_once(cfg: Config):
    """Build frames and dataloaders ONCE for all trials to keep splits & scaling fixed."""
    dp = DataProcessor(cfg.DATA_DIR, cfg.RESAMPLE, cfg)
    frames = dp.prepare_incremental_data(cfg.incremental_datasets)  # fixed splits
    loaders = create_dataloaders(frames, cfg.SEQUENCE_LENGTH, cfg.BATCH_SIZE)
    return frames, loaders

def eval_mae(model, loader, alpha=0.1):
    """Return MAE on a loader using provided evaluate() helper."""
    _, _, m = evaluate(model, loader, alpha=alpha, log=False)
    return float(m["MAE"])

# -----------------------------
# Objective
# -----------------------------
def build_objective(
    base_cfg: Config,
    frames: dict,
    device: torch.device,
    out_root: Path,
):
    """
    Create objective function closure so that Optuna trials re-use fixed splits & scaling.
    - We create dataloaders per trial from fixed 'frames' (cheap).
    - We re-init model & trainer per trial (fresh weights).
    """
    def objective(trial: optuna.Trial) -> float:
        # ---- search space ----
        # Per-task lambdas (log-scale). You can narrow ranges if you already have priors.
        si_t1 = trial.suggest_float("SI_LAMBDA_T1", 1e-5, 1e-2, log=True)
        kd_t1 = trial.suggest_float("KD_LAMBDA_T1", 1e-5, 1e-2, log=True)
        si_t2 = trial.suggest_float("SI_LAMBDA_T2", 1e-5, 1e-2, log=True)
        kd_t2 = trial.suggest_float("KD_LAMBDA_T2", 1e-5, 1e-2, log=True)
        si_eps = trial.suggest_float("SI_EPSILON",   1e-3, 3e-1, log=True)

        # ---- per-trial config (copy base; keep incremental_splits fixed) ----
        cfg = copy.deepcopy(base_cfg)
        cfg.SI_EPSILON = float(si_eps)
        # SI/KD base lambdas are set per-task before calling trainer.train_task()
        # We keep your KL scheduling and warmup unchanged; only the base scales differ per task.

        # trial-specific dir (keeps checkpoints history per trial, optional)
        trial_dir = out_root / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        cfg.BASE_DIR = trial_dir  # IncTrainer will write under this
        # (Optional) quiet logging per trial:
        # setup_logging(trial_dir, level=logging.WARNING)

        # ---- dataloaders (rebuild cheaply from cached 'frames') ----
        loaders = create_dataloaders(frames, cfg.SEQUENCE_LENGTH, cfg.BATCH_SIZE)

        # ---- model & trainer ----
        set_seed(cfg.SEED + trial.number)  # stable but unique
        model = SOHLSTM(3, cfg.HIDDEN_SIZE, cfg.DROPOUT).to(device)
        trainer = IncTrainer(model=model, device=device, config=cfg, inc_dir=cfg.BASE_DIR)

        # ============================
        # Task 0 (no SI/KD)
        # ============================
        cfg.SI_LAMBDA = 0.0
        cfg.KD_LAMBDA = 0.0
        set_seed(cfg.SEED + 0)
        trainer.train_task(loaders["task0_train"], loaders["task0_val"], task_id=0)

        # ============================
        # Task 1 (use per-task λ)
        # ============================
        cfg.SI_LAMBDA = float(si_t1)
        cfg.KD_LAMBDA = float(kd_t1)
        set_seed(cfg.SEED + 1)
        trainer.train_task(loaders["task1_train"], loaders["task1_val"], task_id=1)

        # intermediate metric for pruning (task1 val MAE after finishing task1)
        mae_t1 = eval_mae(trainer.model, loaders["task1_val"], alpha=cfg.ALPHA)
        trial.report(mae_t1, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # ============================
        # Task 2 (use per-task λ)
        # ============================
        cfg.SI_LAMBDA = float(si_t2)
        cfg.KD_LAMBDA = float(kd_t2)
        set_seed(cfg.SEED + 2)
        trainer.train_task(loaders["task2_train"], loaders["task2_val"], task_id=2)

        # ============================
        # Final objective (avg MAE on all tasks' val after Task2)
        # ============================
        mae_t0 = eval_mae(trainer.model, loaders["task0_val"], alpha=cfg.ALPHA)
        mae_t1_final = eval_mae(trainer.model, loaders["task1_val"], alpha=cfg.ALPHA)
        mae_t2 = eval_mae(trainer.model, loaders["task2_val"], alpha=cfg.ALPHA)

        obj = float(np.mean([mae_t0, mae_t1_final, mae_t2]))
        # Report final for dashboards
        trial.set_user_attr("MAE_t0", mae_t0)
        trial.set_user_attr("MAE_t1", mae_t1_final)
        trial.set_user_attr("MAE_t2", mae_t2)
        trial.set_user_attr("MAE_t1_mid", mae_t1)  # after finishing task1

        return obj

    return objective

# -----------------------------
# Main
# -----------------------------
def main():
    # ---- Base config (fixed splits, toggles) ----
    cfg = Config()
    cfg.MODE = "incremental"

    # switches (keep as you currently use them)
    cfg.USE_SI = True
    cfg.USE_KD = True
    cfg.USE_KL = False

    # Root output for the study
    out_root = Path.cwd() / "optuna_si_kd_taskwise"
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root)

    # Fix splits & scaling once for the whole study (very important for fair comparison)
    set_seed(cfg.SEED)
    frames, _ = make_loaders_once(cfg)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.getLogger().info("Using device: %s", device)

    # Build objective
    objective = build_objective(cfg, frames, device, out_root)

    # Study
    sampler = optuna.samplers.TPESampler(seed=cfg.SEED)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=4)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner, study_name="si_kd_taskwise")
    study.optimize(objective, n_trials=50, gc_after_trial=True)

    # Save results
    best = study.best_trial
    logging.getLogger().info("Best value (Avg MAE over val0/1/2 after task2): %.6f", best.value)
    logging.getLogger().info("Best params: %s", best.params)
    # Persist best params
    with open(out_root / "best_params.txt", "w", encoding="utf-8") as f:
        print("Best value:", best.value, file=f)
        print("Params:", best.params, file=f)
        print("User attrs:", best.user_attrs, file=f)

    # Also dump a CSV of all trials
    try:
        import pandas as pd
        df = study.trials_dataframe(attrs=("number","value","state","params","user_attrs"))
        df.to_csv(out_root / "trials.csv", index=False)
    except Exception as e:
        logging.getLogger().warning("Failed to write trials.csv: %s", e)

if __name__ == "__main__":
    main()
