# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import tempfile
from pathlib import Path

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

TMP_ROOT = Path("optuna_search-no-pruner")
TMP_ROOT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# task0 配置：注意新版用 LWF_ALPHAS、EWC_LAMBDAS 列表
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

def train_task0_once(cfg: Config, cached_data, ckpt_dir: Path):
    ckpt_path = ckpt_dir / "task0_best.pt"
    if ckpt_path.exists():
        print(f"[INFO] task0 exists → {ckpt_path}")
        return

    print("[INFO] Training task0 …")
    loaders = create_dataloaders(cached_data, cfg.SEQUENCE_LENGTH, cfg.BATCH_SIZE)
    model = SOHLSTM(3, cfg.HIDDEN_SIZE, cfg.NUM_LAYERS, cfg.DROPOUT).to(DEVICE)
    trainer = Trainer(model, DEVICE, cfg, checkpoint_dir=ckpt_dir)
    # task0，不用 EWC，也不做 LWF
    history = trainer.train_task(
        train_loader=loaders["task0_train"],
        val_loader=loaders["task0_val"],
        task_id=0,
        apply_ewc=False,
        alpha_lwf=0.0,
    )
    print(f"[INFO] task0 done, best loss: {min(history['val_loss']):.4e}")
    # consolidate（λ=1.0 仅缓存 fisher）
    trainer.consolidate(loaders["task0_train"], task_id=0, lam=1.0)
    print(f"[INFO] task0 done → {ckpt_path}")

def objective(trial: optuna.Trial, task0_cfg: dict, cached_data, task0_ckpt_dir: Path):
    # 1) 采样
    lam0 = trial.suggest_float("lambda0", 1e1, 1e4, log=True)
    lam1 = trial.suggest_float("lambda1", 1e1, 1e4, log=True)
    alpha1 = trial.suggest_float("alpha1", 0.0, 2.0)
    alpha2 = trial.suggest_float("alpha2", 0.0, 2.0)

    # 2) 构造新版参数列表
    cfg_dict = task0_cfg.copy()
    cfg_dict["LWF_ALPHAS"] = [0.0, alpha1, alpha2]
    cfg_dict["EWC_LAMBDAS"] = [lam0, lam1, 0.0]
    cfg = Config(**cfg_dict)
    set_seed(cfg.SEED + trial.number)

    # 3) DataLoader & 模型
    loaders = create_dataloaders(cached_data, cfg.SEQUENCE_LENGTH, cfg.BATCH_SIZE)
    model = SOHLSTM(3, cfg.HIDDEN_SIZE, cfg.NUM_LAYERS, cfg.DROPOUT).to(DEVICE)

    # 4) 临时目录 checkpoint
    with tempfile.TemporaryDirectory() as td:
        trainer = Trainer(model, DEVICE, cfg, checkpoint_dir=td)
        # 5) 加载 task0
        ckpt = task0_ckpt_dir / "task0_best.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=DEVICE)
            trainer.model.load_state_dict(state["model_state"])
            # 重建 EWC：用保存的 params & fisher
            trainer.ewc_tasks = []
            for e_data in state.get("ewc_tasks", []):
                e = EWC.__new__(EWC)
                e.model = trainer.model
                e.device = DEVICE
                e.params = {n: p.to(DEVICE) for n, p in e_data["params"].items()}
                e.fisher = {n: f.to(DEVICE) for n, f in e_data["fisher"].items()}
                e.lam = lam0
                trainer.ewc_tasks.append(e)

        # 6) 更新 1
        trainer.train_task(
            train_loader=loaders["task1_train"],
            val_loader=loaders["task1_val"],
            task_id=1,
            apply_ewc=True,
            alpha_lwf=cfg.LWF_ALPHAS[1],
        )
        trainer.consolidate(loaders["task1_train"], task_id=1, lam=lam1)

        # 7) 更新 2
        trainer.train_task(
            train_loader=loaders["task2_train"],
            val_loader=loaders["task2_val"],
            task_id=2,
            apply_ewc=True,
            alpha_lwf=cfg.LWF_ALPHAS[2],
        )

        # 8) 测 MAE
        maes = [
            calc_mae(trainer.model, loaders[k], DEVICE)
            for k in ("task0_train", "task1_train", "task2_train")
        ]
        avg = float(np.mean(maes))
        trial.set_user_attr("mae_values", maes)
        trial.set_user_attr("avg_mae", avg)
        return avg

def main():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Workspace: {TMP_ROOT.resolve()}")

    # 1) 准备数据
    cfg0 = Config(**TASK0_CONFIG)
    print("[INFO] Loading data …")
    dp = DataProcessor(
        data_dir=cfg0.DATA_DIR,
        resample=cfg0.RESAMPLE,
        config=cfg0,
    )
    cached = dp.prepare_incremental_data(cfg0.incremental_datasets)
    print("[INFO] Data cached.")

    # 2) task0
    task0_ckpt_dir = TMP_ROOT / "task0_checkpoints"
    task0_ckpt_dir.mkdir(exist_ok=True)
    train_task0_once(cfg0, cached, task0_ckpt_dir)

    # 3) Optuna
    study = optuna.create_study(direction="minimize", study_name="ewc_lambda_opt")
    study.optimize(
        lambda t: objective(t, TASK0_CONFIG, cached, task0_ckpt_dir),
        n_trials=50,
        show_progress_bar=True,
    )

    # 4) 保存结果
    rdir = TMP_ROOT / "results"; rdir.mkdir(exist_ok=True)
    study.trials_dataframe(
        attrs=("number","state","values","params","user_attrs")
    ).to_csv(rdir/"all_trials.csv", index=False)
    best = study.best_trial
    summary = {
        "best_avg_mae": float(best.value),
        "best_params": best.params,
        "mae_values": best.user_attrs["mae_values"],
        "trial_number": best.number,
    }
    with open(rdir/"best_params.json","w") as f:
        json.dump(summary, f, indent=4)

    # 控制台输出
    print("\n" + "="*40)
    print(f"Best MAE = {best.value:.4f}  (trial {best.number})")
    for k,v in best.params.items():
        print(f"  {k} = {v:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
