#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Light‑weight Optuna search for incremental SOH‑LSTM
===================================================
• Hyper‑parameters: λ₀, λ₁ (EWC) and α₁, α₂ (LWF)
• Objective: mean MAE across three test splits
• Base task is trained once; its weights + Fisher are cached
• The script prints a one‑line “★ Best …” summary at the end
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.metrics import mean_absolute_error  # MAE

from lstm import (  # all heavy‑lifting lives in lstm.py
    Config,
    DataProcessor,
    create_dataloaders,
    SOHLSTM,
    Trainer,
    EWC,
    set_seed,
)

# --------------------------------------------------
# Paths / device and a base (fixed) configuration
# --------------------------------------------------
DATA_DIR = Path("../../01_Datenaufbereitung/Output/Calculated")
OUT_DIR  = Path("incL_search-no-pruner")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BASE_CFG: dict = dict(
    SEQUENCE_LENGTH=720,
    HIDDEN_SIZE=128,
    NUM_LAYERS=2,
    DROPOUT=0.3,
    BATCH_SIZE=32,
    LEARNING_RATE=1e-4,
    WEIGHT_DECAY=1e-6,
    EPOCHS=200,
    PATIENCE=20,
    SEED=42,
    SCALER="RobustScaler",
    RESAMPLE="10min",
    MODE="incremental",
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# --------------------------------------------------
# Helper utilities
# --------------------------------------------------
def make_loaders(cfg: Config, cache=None):
    """Return dataloaders; optionally re‑use a cached dataframe dict."""
    if cache is None:
        cache = DataProcessor(DATA_DIR, cfg).prepare_data()
    return create_dataloaders(cache, cfg), cache


def mae_on_loader(model: torch.nn.Module, loader) -> float:
    """Compute MAE for an entire loader."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(DEVICE)).cpu().numpy())
            targets.append(y.numpy())
    return mean_absolute_error(np.concatenate(targets), np.concatenate(preds))


# --------------------------------------------------
# 1) Train the base task once and cache checkpoint
# --------------------------------------------------
BASE_PTH = OUT_DIR / "base_ckpt.pth"


def train_base_once(cfg: Config, cache) -> None:
    if BASE_PTH.exists():
        log.info("[BASE] checkpoint already present → %s", BASE_PTH)
        return

    loaders, _ = make_loaders(cfg, cache)
    trainer = Trainer(
        SOHLSTM(len(cfg.FEATURES_COLS), cfg.HIDDEN_SIZE, cfg.NUM_LAYERS, cfg.DROPOUT),
        DEVICE,
        cfg,
    )

    history = trainer.train_task(
        train_loader=loaders["base_train"],
        val_loader=loaders.get("base_val") or loaders["base_train"],
        task_id=0,
        apply_ewc=False,
    )
    best_val = min(history["val_loss"])
    log.info("[BASE] best validation loss: %.4e", best_val)

    trainer.consolidate(loaders["base_train"], lam=1.0)  # λ=1 for caching
    ewc = trainer.ewc_tasks[0]
    torch.save(
        {
            "state": trainer.model.state_dict(),
            "params": {k: v.cpu() for k, v in ewc.params.items()},
            "fisher": {k: v.cpu() for k, v in ewc.fisher.items()},
        },
        BASE_PTH,
    )
    log.info("[BASE] checkpoint saved → %s", BASE_PTH)


# --------------------------------------------------
# 2) Optuna objective
# --------------------------------------------------
def objective(trial: optuna.Trial, cfg_dict: dict, cache) -> float:
    # Sample hyper‑parameters
    lam0 = trial.suggest_float("lam0", 1e1, 1e4, log=True)
    lam1 = trial.suggest_float("lam1", 1e1, 1e4, log=True)
    a1   = trial.suggest_float("alpha1", 0.0, 2.0)
    a2   = trial.suggest_float("alpha2", 0.0, 2.0)

    # Re‑seed for reproducibility
    set_seed(cfg_dict["SEED"] + trial.number)
    cfg = Config(**cfg_dict)

    # Build loaders and a fresh trainer
    loaders, _ = make_loaders(cfg, cache)
    trainer = Trainer(
        SOHLSTM(len(cfg.FEATURES_COLS), cfg.HIDDEN_SIZE, cfg.NUM_LAYERS, cfg.DROPOUT),
        DEVICE,
        cfg,
    )

    # Load cached base weights + Fisher
    chk = torch.load(BASE_PTH, map_location=DEVICE)
    trainer.model.load_state_dict(chk["state"])

    ewc0 = EWC.__new__(EWC)
    ewc0.device = DEVICE
    ewc0.lam = lam0
    ewc0.params = {k: t.to(DEVICE) for k, t in chk["params"].items()}
    ewc0.fisher = {k: t.to(DEVICE) for k, t in chk["fisher"].items()}
    trainer.ewc_tasks = [ewc0]

    trainer.old_model = copy.deepcopy(trainer.model).to(DEVICE)
    trainer.old_model.eval()
    for p in trainer.old_model.parameters():
        p.requires_grad_(False)

    # ---- update‑1 ---------------------------------------------------
    trainer.train_task(
        loaders["update1_train"],
        loaders.get("update1_val") or loaders["update1_train"],
        task_id=1,
        apply_ewc=True,
        alpha_lwf=a1,
    )
    # trial.report(mae_on_loader(trainer.model, loaders["test_update1"]), step=0)
    # if trial.should_prune():
    #     raise optuna.TrialPruned()

    # ---- update‑2 ---------------------------------------------------
    trainer.consolidate(loaders["update1_train"], lam=lam1)
    trainer.train_task(
        loaders["update2_train"],
        loaders.get("update2_val") or loaders["update2_train"],
        task_id=2,
        apply_ewc=True,
        alpha_lwf=a2,
    )

    # ---- final objective -------------------------------------------
    mae_avg = np.mean(
        [mae_on_loader(trainer.model, loaders[k])
         for k in ("test_base", "test_update1", "test_update2")]
    )
    return mae_avg


# --------------------------------------------------
# 3) Entry point
# --------------------------------------------------
if __name__ == "__main__":
    base_cfg = Config(**BASE_CFG)
    data_cache = DataProcessor(DATA_DIR, base_cfg).prepare_data()
    log.info("[DATA] splits: %s", {k: len(v) for k, v in data_cache.items()})

    train_base_once(base_cfg, data_cache)

    study = optuna.create_study(
        direction="minimize",
        # pruner=optuna.pruners.MedianPruner(n_warmup_steps=0, n_min_trials=10),
    )
    study.optimize(lambda t: objective(t, BASE_CFG, data_cache), n_trials=50,
                   show_progress_bar=True)

    best = study.best_trial
    print(
        f"\n★ Best average MAE = {best.value:.4f}\n"
        f"★ Best params      = {best.params}"
    )
