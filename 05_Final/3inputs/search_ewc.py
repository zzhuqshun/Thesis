# -*- coding: utf-8 -*-
"""Optuna 超参数搜索（以 MAE 为目标，增量 SOH‑LSTM）
--------------------------------------------------------------------
* 仅搜索 EWC λ₀、λ₁ α₁、α₂。
* 目标函数：三个测试集 MAE 的算术平均（越小越好）。
* base 任务仅训练一次并缓存；后续 trial 直接加载。

假设新版 `ewc.py` 中已实现：
    Config, DataProcessor, SOHLSTM, Trainer, EWC,
    create_dataloaders, set_seed, get_predictions
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.metrics import mean_absolute_error

# ---- 新框架 -------------------------------------------------------
from ewc import (
    Config,
    DataProcessor,
    SOHLSTM,
    Trainer,
    EWC,
    create_dataloaders,
    set_seed,
    get_predictions,
)

# ------------------------------------------------------------------
# 常量 / 路径
# ------------------------------------------------------------------
TMP_ROOT = Path("optuna_search-no-pruner")
TMP_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# baseline 配置（除搜索超参外全部固定）
BASE_CONFIG: dict = {
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
    "LWF_ALPHA0": 0.0,
    "LWF_ALPHA1": 0.0,  # 待搜索
    "LWF_ALPHA2": 0.0,  # 待搜索
    "EWC_LAMBDA0": 0.0,  # 待搜索
    "EWC_LAMBDA1": 0.0,  # 待搜索
}

# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def calc_mae(model: torch.nn.Module, loader, device: torch.device) -> float:
    """在 loader 上计算 MAE"""
    preds, tgts = get_predictions(model, loader, device)
    return mean_absolute_error(tgts, preds)


def train_base_once(cfg: Config, cached_data, ckpt_dir: Path) -> None:
    """如果不存在，就训练并缓存 base 任务模型权重和 Fisher"""
    ckpt_path = ckpt_dir / "task0_best.pt"
    if ckpt_path.exists():
        print(f"[INFO] Base checkpoint already exists → {ckpt_path}")
        return

    print("[INFO] Training base task …")
    loaders = create_dataloaders(cached_data, cfg.SEQUENCE_LENGTH, cfg.BATCH_SIZE)

    model = SOHLSTM(3, cfg.HIDDEN_SIZE, cfg.NUM_LAYERS, cfg.DROPOUT).to(DEVICE)
    trainer = Trainer(model, DEVICE, cfg, checkpoint_dir=str(ckpt_dir))

    trainer.train_task(
        train_loader=loaders["base_train"],
        val_loader=loaders["base_val"],
        task_id=0,
        apply_ewc=False,
        alpha_lwf=0.0,
        resume=False,
    )

    # consolidate 一次，λ 固定为 1.0 只用于缓存
    trainer.consolidate(loaders["base_train"], task_id=0, lam=1.0)
    print(f"[INFO] Base task finished → {ckpt_path}")


# ------------------------------------------------------------------
# Optuna 目标函数
# ------------------------------------------------------------------

def objective(trial: optuna.Trial, base_cfg: dict, cached_data, base_ckpt_dir: Path) -> float:
    try:
        # 1) 采样超参
        lam0 = trial.suggest_float("lambda0", 1e1, 1e4, log=True)
        lam1 = trial.suggest_float("lambda1", 1e1, 1e4, log=True)
        alpha1 = trial.suggest_float("alpha1", 0.0, 2.0)
        alpha2 = trial.suggest_float("alpha2", 0.0, 2.0)

        # 2) 配置 + 随机种子
        cfg_dict = base_cfg.copy()
        cfg_dict.update(
            {
                "EWC_LAMBDA0": lam0,
                "EWC_LAMBDA1": lam1,
                "LWF_ALPHA1": alpha1,
                "LWF_ALPHA2": alpha2,
            }
        )
        cfg = Config(**cfg_dict)
        set_seed(cfg.SEED + trial.number)

        # 3) 数据、模型、Trainer
        loaders = create_dataloaders(cached_data, cfg.SEQUENCE_LENGTH, cfg.BATCH_SIZE)
        model = SOHLSTM(3, cfg.HIDDEN_SIZE, cfg.NUM_LAYERS, cfg.DROPOUT).to(DEVICE)

        # 4) 为每个 trial 用临时目录保存 checkpoint（可 debug）
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(model, DEVICE, cfg, checkpoint_dir=tmp_dir)

            # 5) 加载 base 任务权重 + Fisher
            base_ckpt = base_ckpt_dir / "task0_best.pt"
            if base_ckpt.exists():
                state = torch.load(base_ckpt, map_location=DEVICE)
                trainer.model.load_state_dict(state["model_state"])

                # 重建 EWC 对象（沿用 consolidate 时缓存的 λ=1.0）
                trainer.ewc_tasks = []
                for data in state.get("ewc_tasks", []):
                    e = EWC.__new__(EWC)
                    e.model = trainer.model
                    e.device = DEVICE
                    e.params = {k: v.to(DEVICE) for k, v in data["params"].items()}
                    e.fisher = {k: v.to(DEVICE) for k, v in data["fisher"].items()}
                    e.lam = lam0  # 对 base 任务应用新的 λ₀
                    trainer.ewc_tasks.append(e)

            # 6) update‑1 训练
            trainer.train_task(
                train_loader=loaders["update1_train"],
                val_loader=loaders["update1_val"],
                task_id=1,
                apply_ewc=True,
                alpha_lwf=cfg.LWF_ALPHA1,
                resume=False,
            )

            # consolidate（带 λ₁）
            trainer.consolidate(loaders["update1_train"], task_id=1, lam=lam1)

            # 7) update‑2 训练
            trainer.train_task(
                train_loader=loaders["update2_train"],
                val_loader=loaders["update2_val"],
                task_id=2,
                apply_ewc=True,
                alpha_lwf=cfg.LWF_ALPHA2,
                resume=False,
            )

            # 8) 评估三个任务 MAE
            maes = [
                calc_mae(trainer.model, loaders[key], DEVICE)
                for key in ("test_base", "test_update1", "test_update2")
            ]
            avg_mae = float(np.mean(maes))

            # 记录到 user_attr 便于事后分析
            trial.set_user_attr("mae_values", [float(m) for m in maes])
            trial.set_user_attr("avg_mae", avg_mae)
            return avg_mae  # **minimize** MAE

    except Exception as err:
        # 任何错误 → 返回正无穷，等价于“最差表现”
        print(f"[ERROR] Trial {trial.number} failed: {err}")
        return float("inf")


# ------------------------------------------------------------------
# 主程序入口
# ------------------------------------------------------------------

def main() -> None:
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Workspace: {TMP_ROOT.resolve()}")

    # 1) 数据准备（一次性缓存到内存）
    cfg = Config(**BASE_CONFIG)
    print("[INFO] Loading & caching data …")
    dp = DataProcessor(
        data_dir="../../01_Datenaufbereitung/Output/Calculated/",
        resample=cfg.RESAMPLE,
        config=cfg,
        base_train_ids=["03", "05", "07", "27"],
        base_val_ids=["01"],
        update1_train_ids=["21", "23", "25"],
        update1_val_ids=["19"],
        update2_train_ids=["09", "11", "15", "29"],
        update2_val_ids=["13"],
    )
    cached_data = dp.prepare_data()
    print("[INFO] Data cached.")

    # 2) 训练 base 任务
    base_ckpt_dir = TMP_ROOT / "base_checkpoints"
    base_ckpt_dir.mkdir(exist_ok=True)
    train_base_once(cfg, cached_data, base_ckpt_dir)

    # 3) 创建 Optuna Study（不启用剪枝）
    study = optuna.create_study(
        direction="minimize",
        study_name="ewc_lambda_optimization",
    )

    # 4) 运行搜索
    study.optimize(
        lambda t: objective(t, BASE_CONFIG, cached_data, base_ckpt_dir),
        n_trials=50,
        show_progress_bar=True,
    )

    # 5) 输出 / 保存结果
    results_dir = TMP_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    study.trials_dataframe(
        attrs=("number", "state", "values", "params", "user_attrs")
    ).to_csv(results_dir / "all_trials.csv", index=False)

    best = study.best_trial
    summary = {
        "best_avg_mae": float(best.value),
        "best_params": {k: float(v) for k, v in best.params.items()},
        "mae_values": best.user_attrs.get("mae_values", []),
        "trial_number": best.number,
    }
    with open(results_dir / "best_params.json", "w") as fh:
        json.dump(summary, fh, indent=4)

    # 控制台摘要
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print(f"Best average MAE = {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v:.4f}")
    if "mae_values" in best.user_attrs:
        base_mae, u1_mae, u2_mae = best.user_attrs["mae_values"]
        print("Detailed MAE:")
        print(f"  Base    : {base_mae:.4f}")
        print(f"  Update1 : {u1_mae:.4f}")
        print(f"  Update2 : {u2_mae:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
