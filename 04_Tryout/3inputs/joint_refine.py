# -*- coding: utf-8 -*-
"""
Two-stage hyperparameter optimization with real-time best-trial reporting.
Stage 1: Find best optimizer/scheduler combination (50 trials)
Stage 2: Fine-tune learning rate and smoothing with best combo (50 trials)
"""
import os
import json
from pathlib import Path
from functools import partial

import optuna
import torch
from sklearn.metrics import mean_absolute_error

from model import (
    Config,
    DataProcessor,
    SOHLSTM,
    create_dataloaders,
    set_seed,
)

out_folder = Path("optuna_joint")
summary_path = out_folder / 'best_params.json'
if not summary_path.exists():
    raise FileNotFoundError(f"找不到 {summary_path}")
with open(summary_path, 'r') as f:
    summary = json.load(f)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Training runner: returns the best smoothed validation MAE
def run_training(
    cfg: Config,
    optimizer_name: str,
    scheduler_name: str,
    scheduler_params: dict,
    loaders: dict[str, torch.utils.data.DataLoader],
    trial: optuna.Trial | None = None,
) -> float:
    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize model
    model = SOHLSTM(3, cfg.HIDDEN_SIZE, cfg.NUM_LAYERS, cfg.DROPOUT).to(device)
    
    # Initialize optimizer
    optimizer = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "RMSprop": torch.optim.RMSprop,
    }[optimizer_name](model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    # Initialize learning rate scheduler
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", **scheduler_params
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    best_val = float('inf')
    no_improve_count = 0

    for epoch in range(1, cfg.EPOCHS + 1):
        # Training phase
        model.train()
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = torch.nn.functional.mse_loss(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation phase
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in loaders['val']:
                x, y = x.to(device), y.to(device)
                preds.append(model(x).cpu().numpy())
                trues.append(y.cpu().numpy())
        import numpy as _np
        import pandas as _pd
        preds_arr = _np.concatenate(preds)
        trues_arr = _np.concatenate(trues)
        # Apply exponential smoothing to predictions
        smooth_preds = _pd.Series(preds_arr).ewm(alpha=cfg.ALPHA, adjust=False).mean().to_numpy()
        val_mae = mean_absolute_error(trues_arr, smooth_preds)

        # Scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_mae)
        else:
            scheduler.step()

        # Report intermediate results to Optuna for pruning
        if trial:
            trial.report(val_mae, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Early stopping based on patience
        if val_mae < best_val:
            best_val = val_mae
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= cfg.PATIENCE:
            break

    return best_val

# =========================================
# # Stage 1: (暂时注释掉，保留以备后用)
# =========================================
# def stage1_objective(trial: optuna.Trial, loaders):
#     # Suggest optimizer and scheduler
#     optimizer_name = trial.suggest_categorical(
#         "optimizer", ["Adam", "SGD", "RMSprop"]
#     )
#     scheduler_name = trial.suggest_categorical(
#         "scheduler", ["StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"]
#     )
#     scheduler_params = {}
#     if scheduler_name == "StepLR":
#         scheduler_params['step_size'] = trial.suggest_int("step_size", 10, 30)
#         scheduler_params['gamma'] = trial.suggest_float("gamma", 0.3, 0.7)
#     elif scheduler_name == "ReduceLROnPlateau":
#         scheduler_params['factor'] = trial.suggest_float("factor", 0.3, 0.7)
#
#     cfg = Config()
#     cfg.MODE = "joint"
#     cfg.LEARNING_RATE = 1e-4
#     cfg.ALPHA = 0.1
#     cfg.PATIENCE = 20
#     cfg.LWF_ALPHAS = [0.0, 0.0, 0.0]
#     cfg.EWC_LAMBDAS = [0.0, 0.0, 0.0]
#     set_seed(cfg.SEED + trial.number)
#
#     return run_training(
#         cfg, optimizer_name, scheduler_name, scheduler_params, loaders, trial
#     )

# =========================================
# Stage 2: Fine-tuning learning rate and smoothing
# =========================================
def stage2_objective(
    trial: optuna.Trial,
    best_optimizer: str,
    best_scheduler: str,
    loaders,
):
    # Suggest learning rate and smoothing factor
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    alpha = trial.suggest_float("alpha", 0.01, 0.15)
    scheduler_params = {'factor': summary['best_params'].get('factor', 0.5)}

    cfg = Config()
    cfg.MODE = "joint"
    cfg.LEARNING_RATE = lr
    cfg.ALPHA = alpha
    cfg.PATIENCE = 20
    cfg.LWF_ALPHAS = [0.0, 0.0, 0.0]
    cfg.EWC_LAMBDAS = [0.0, 0.0, 0.0]
    set_seed(cfg.SEED + trial.number + 1000)

    return run_training(
        cfg,
        best_optimizer,
        best_scheduler,
        scheduler_params,
        loaders,
        trial,
    )

# Main execution
if __name__ == '__main__':
    # Prepare shared data loaders
    cfg0 = Config()
    dp = DataProcessor(
        data_dir=cfg0.DATA_DIR,
        resample=cfg0.RESAMPLE,
        config=cfg0,
    )
    datasets = dp.prepare_joint_data(cfg0.joint_datasets)
    shared_loaders = create_dataloaders(
        datasets, cfg0.SEQUENCE_LENGTH, cfg0.BATCH_SIZE
    )

    job_id = os.getenv('JOB_ID') or os.getenv('SLURM_JOB_ID')
    out_folder = f"optuna_joint_{job_id}" if job_id else "optuna_joint"
    Path(out_folder).mkdir(exist_ok=True)

    # =========================================
    # # Stage 1 运行（已注释）
    # =========================================
    # print("=" * 60)
    # print("Stage 1: Finding best optimizer/scheduler combo")
    # print("=" * 60)
    # study1 = optuna.create_study(
    #     direction='minimize',
    #     study_name='joint_opt_stage1',
    # )
    # study1.optimize(
    #     lambda t: stage1_objective(t, shared_loaders),
    #     n_trials=50,
    #     show_progress_bar=True,
    # )
    # best_opt = study1.best_params['optimizer']
    # best_sch = study1.best_params['scheduler']
    # print(
    #     f"Stage1 Complete: optimizer={best_opt}, scheduler={best_sch}, MAE_smooth={study1.best_value:.6f}"
    # )

    # =========================================
    # Stage 2 运行
    # =========================================
    print(" " + "=" * 60)
    print("Stage 2: Fine-tuning learning rate and smoothing")
    print("=" * 60)
    study2 = optuna.create_study(
        direction='minimize',
        study_name='joint_opt_stage2_refined'
    )
    study2.optimize(
        partial(stage2_objective, best_optimizer=summary['best_params']['optimizer'], best_scheduler=summary['best_params']['scheduler'], loaders=shared_loaders),
        n_trials=50,
        show_progress_bar=True,
    )
    print(
        f"Stage2 Complete: lr={study2.best_params['lr']:.6e}, "
        f"alpha={study2.best_params['alpha']:.4f}, MAE_smooth={study2.best_value:.6f}"
    )

    # Save summary to JSON (更新refined)
    summary['stage2_refined'] = {
        'lr': study2.best_params['lr'],
        'alpha': study2.best_params['alpha'],
        'scheduler_params': {'factor': summary['best_params']['factor']},
        'mae_smooth': float(study2.best_value),
    }
    with open(Path(out_folder) / 'best_params_refined.json', 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Refined parameters saved to {out_folder}/best_params_refined.json")
