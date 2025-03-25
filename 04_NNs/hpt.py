import json
import optuna
import torch
import numpy as np
import random
from pathlib import Path



# ===== 导入你原先的脚本 =====
from soh_lstm import (
    set_seed,
    load_data,
    scale_data,
    BatteryDataset,
    SOHLSTM,
    train_and_validate_model,
    # evaluate_model,
    device,  # soh_lstm.py 中定义的 device
    save_dir  # 用于保存模型等输出的基础目录
)

from torch.utils.data import DataLoader

def objective(trial):
    """
    Optuna 的目标函数，在每个 trial（试验）中被调用一次。
    会根据每个 trial 采样一组超参数，训练模型，并把验证集上的最佳损失返回给 Optuna。
    """

    # 1. 设置随机种子，保证结果可重复
    set_seed(42)

    # 2. 为本 trial 采样一组超参数（你可以根据需要自由调整搜索空间）
    seq_length = trial.suggest_int("SEQUENCE_LENGTH", 144, 1008, step=144)
    hidden_size = trial.suggest_categorical("HIDDEN_SIZE", [32, 64, 128])
    num_layers = trial.suggest_int("NUM_LAYERS", 2, 5)
    dropout = trial.suggest_float("DROPOUT", 0.0, 0.5, step=0.1)
    learning_rate = trial.suggest_categorical("LEARNING_RATE", [1e-4, 1e-3])
    weight_decay = trial.suggest_categorical("WEIGHT_DECAY", [0.0, 1e-6, 1e-5, 1e-4])
    batch_size = trial.suggest_categorical("BATCH_SIZE", [32, 64, 128])

    # 3. 定义超参数字典
    hyperparams = {
        "SEQUENCE_LENGTH": seq_length,
        "HIDDEN_SIZE": hidden_size,
        "NUM_LAYERS": num_layers,
        "DROPOUT": dropout,
        "LEARNING_RATE": learning_rate,
        "WEIGHT_DECAY": weight_decay,
        "BATCH_SIZE": batch_size,
        "EPOCHS": 100,
        "PATIENCE": 10
    }

    # 4. 数据加载与预处理
    data_dir = Path("../01_Datenaufbereitung/Output/Calculated/")
    df_train, df_val, df_test = load_data(data_dir)

    df_train_scaled, df_val_scaled, df_test_scaled = scale_data(
        df_train, df_val, df_test, scaler_type='standard'
    )

    train_dataset = BatteryDataset(df_train_scaled, hyperparams["SEQUENCE_LENGTH"])
    val_dataset   = BatteryDataset(df_val_scaled,   hyperparams["SEQUENCE_LENGTH"])

    train_loader = DataLoader(train_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=hyperparams["BATCH_SIZE"], shuffle=False)

    # 5. 针对本 trial，创建一个**独立的**文件夹，用来保存 best/last 模型、训练历史文件
    trial_save_dir = save_dir / f"trial_{trial.number}"
    trial_save_dir.mkdir(exist_ok=True, parents=True)

    save_path = {
        'best':    trial_save_dir / 'best_soh_model.pth',
        'last':    trial_save_dir / 'last_soh_model.pth',
        'history': trial_save_dir / 'train_history.parquet'
    }

    # 6. 定义并训练模型，获取历史和最优验证 loss
    model = SOHLSTM(
        input_size=3,
        hidden_size=hyperparams["HIDDEN_SIZE"],
        num_layers=hyperparams["NUM_LAYERS"],
        dropout=hyperparams["DROPOUT"]
    ).to(device)

    history, best_val_loss = train_and_validate_model(
        model,
        train_loader,
        val_loader,
        save_path
    )
    
    # 7. 保存超参数和 best_val_loss 到 trial 文件夹
    hyperparam_path = trial_save_dir / "hyperparams.json"
    with open(hyperparam_path, "w") as f:
        # 建议把需要的信息放进一个 dict，一并保存
        hyperparams["best_val_loss"] = best_val_loss
        json.dump(hyperparams, f, indent=4)
    # 7. 把 best_val_loss 返回给 Optuna
    return best_val_loss


def main():
    """
    运行 Optuna 超参数搜索，并输出搜索到的最佳结果。
    """
    # direction="minimize" 表示我们想要最小化验证损失
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, timeout=None)

    print("\n============================")
    print("      Search Finished!      ")
    print("============================")
    print(f"Best trial ID: {study.best_trial.number}")
    print(f"Best trial value (Val. Loss): {study.best_trial.value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("============================")

    # 你也可以把 best_params 保存到 json，用于后续正式训练
    with open("best_hyperparams.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)


if __name__ == "__main__":
    main()
