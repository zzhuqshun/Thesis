# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
from pathlib import Path
import copy

import numpy as np
import optuna
import torch
from sklearn.metrics import mean_absolute_error

# 新版框架
from model import (
    Config,
    DataProcessor,
    SOHLSTM,
    Trainer,
    EWC,
    create_dataloaders,
    set_seed,
)

# 使用已有的 Task0 模型目录（需确保已经训练并存储于 models 文件夹）
MODELS_DIR = Path.cwd() / "models"
TASK0_CKPT_DIR = MODELS_DIR 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# task0 配置模板
TASK0_CONFIG = {
    "SEQUENCE_LENGTH": 720,
    "HIDDEN_SIZE": 128,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.3,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-6,
    "EPOCHS": 200,
    "PATIENCE": 20,
    "SEED": 42,
    "SCALER": "RobustScaler",
    "RESAMPLE": "10min",
    # placeholders，下面 objective 会填充
    "LWF_ALPHAS": [0.0, 0.0, 0.0],
    "EWC_LAMBDAS": [0.0, 0.0, 0.0],
}


def calc_mae(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    ys_p, ys_t = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).cpu().numpy()
            ys_p.append(pred)
            ys_t.append(y.numpy())
    ys_p = np.concatenate(ys_p)
    ys_t = np.concatenate(ys_t)
    return mean_absolute_error(ys_t, ys_p)


def objective(trial: optuna.Trial, base_cfg: dict, cached_data):
    # 1) 采样
    lam0 = trial.suggest_float("lambda0", 1e1, 1e4, log=True)
    lam1 = trial.suggest_float("lambda1", 1e1, 1e4, log=True)
    lam2 = 0
    alpha1 = trial.suggest_float("alpha1", 0.0, 2.0)
    alpha2 = trial.suggest_float("alpha2", 0.0, 2.0)

    # 2) 构造 Config
    cfg_dict = base_cfg.copy()
    cfg_dict["LWF_ALPHAS"] = [0.0, alpha1, alpha2]
    cfg_dict["EWC_LAMBDAS"] = [lam0, lam1, 0.0]
    cfg = Config(**cfg_dict)
    set_seed(cfg.SEED + trial.number)

    # 3) 加载数据与模型
    loaders = create_dataloaders(cached_data, cfg.SEQUENCE_LENGTH, cfg.BATCH_SIZE)
    model = SOHLSTM(3, cfg.HIDDEN_SIZE, cfg.NUM_LAYERS, cfg.DROPOUT).to(DEVICE)
    trainer = Trainer(model, DEVICE, cfg)

    # 4) 载入 Task0 模型和 EWC 数据
    ckpt = TASK0_CKPT_DIR / "task0_best.pt"
    state = torch.load(ckpt, map_location=DEVICE)
    print("Loaded keys from task0 checkpoint:", state.keys())
    print("Number of EWC entries:", len(state.get('ewc_tasks', [])))
    trainer.model.load_state_dict(state['model_state'])
    # 复用保存的 EWC fisher & params
    trainer.ewc_tasks = []
    for e_data in state.get('ewc_tasks', []):
        e = EWC.__new__(EWC)
        e.model = trainer.model
        e.device = DEVICE
        e.params = {n: p.to(DEVICE) for n, p in e_data['params'].items()}
        e.fisher = {n: f.to(DEVICE) for n, f in e_data['fisher'].items()}
        e.lam_max = e_data.get('lam',lam0)
        e.lam = e.lam_max
        trainer.ewc_tasks.append(e)
    # 设置 old_model 以支持 LwF
    trainer.old_model = copy.deepcopy(trainer.model).to(DEVICE)
    trainer.old_model.eval()
    for p in trainer.old_model.parameters():
        p.requires_grad_(False)

    # 5) Task1 更新（启用 EWC 和 LwF）
    trainer.train_task(
        train_loader=loaders['task1_train'],
        val_loader=loaders['task1_val'],
        task_id=1,
        apply_ewc=True,
        alpha_lwf=cfg.LWF_ALPHAS[1],
    )
    trainer.consolidate(loaders['task1_train'], task_id=1, lam=lam1)

    # 6) Task2 更新
    trainer.train_task(
        train_loader=loaders['task2_train'],
        val_loader=loaders['task2_val'],
        task_id=2,
        apply_ewc=True,
        alpha_lwf=cfg.LWF_ALPHAS[2],
    )
    trainer.consolidate(loaders['task2_train'], task_id=2, lam =lam2)
    # 7) 评估 MAE
    maes = [
        calc_mae(trainer.model, loaders[f'task{i}_val'], DEVICE)
        for i in range(3)
    ]
    
    avg = float(np.mean(maes))
    trial.set_user_attr('mae_values', maes)
    trial.set_user_attr('avg_mae', avg)
    return avg


def main():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Loading Task0 checkpoint from: {TASK0_CKPT_DIR}")

    # 1) 准备数据
    base_cfg = TASK0_CONFIG
    dp = DataProcessor(
        data_dir=Config().DATA_DIR,
        resample=Config().RESAMPLE,
        config=Config(),
    )
    cached = dp.prepare_incremental_data(Config().incremental_datasets)

    # 2) Optuna 调参
    study = optuna.create_study(direction='minimize', study_name='ewc_lambda_opt')
    study.optimize(lambda t: objective(t, base_cfg, cached), n_trials=50)

    # 3) 保存结果
    # Determine output folder, appending JOB_ID if available
    job_id = os.getenv('JOB_ID') or os.getenv('SLURM_JOB_ID')
    rdir = Path(f"optuna_results_{job_id}" if job_id else "optuna_results")
    rdir.mkdir(exist_ok=True)
    
    study.trials_dataframe(attrs=('number','state','values','params','user_attrs')).to_csv(rdir/'all_trials.csv', index=False)
    best = study.best_trial
    summary = {
        'best_avg_mae': float(best.value),
        'best_params': best.params,
        'mae_values': best.user_attrs['mae_values'],
        'trial_number': best.number,
    }
    with open(rdir/'best_params.json','w') as f:
        json.dump(summary, f, indent=4)

    print("\n" + "="*40)
    print(f"Best MAE = {best.value:.4f}  (trial {best.number})")
    for k, v in best.params.items():
        print(f"  {k} = {v:.4f}")
    print("="*40)

if __name__ == '__main__':
    main()
