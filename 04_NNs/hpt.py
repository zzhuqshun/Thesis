import json
import optuna
import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
from datetime import datetime

# ===== 导入你原先的脚本 =====
from soh_lstm import (
    set_seed,
    # load_data,  # 我们将替换成下面定义的consistent_load_data
    scale_data,
    BatteryDataset,
    SOHLSTM,
    train_and_validate_model,
    # evaluate_model,
    device  # soh_lstm.py 中定义的 device
)

from torch.utils.data import DataLoader

# 创建保存优化结果的全局目录
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
optuna_dir = Path(__file__).parent / f"models/optuna_with_dataset_split"
optuna_dir.mkdir(exist_ok=True, parents=True)

# 重新实现load_data函数，确保数据集划分一致性
def consistent_load_data(data_dir: Path, resample='10min', split_file_path='dataset_split.json'):
    """
    Load training, validation and test data with persistent dataset split.
    Saves the split on first run, then reuses the same split on subsequent runs.
    
    Args:
        data_dir: Directory containing the parquet files
        resample: Time interval for resampling data
        split_file_path: Path to save/load the dataset split configuration
    """
    # Get all parquet files and sort them by the numeric part of the filename
    parquet_files = sorted(
        [f for f in data_dir.glob('*.parquet') if f.is_file()],
        key=lambda x: int(x.stem.split('_')[-1])  # Sort by the number after the last underscore
    )
    
    # Convert Path objects to strings for JSON serialization
    parquet_files_str = [str(f) for f in parquet_files]
    
    # Check if split file exists (from a previous run)
    split_file = Path(split_file_path)
    if split_file.exists():
        # Load the existing split configuration
        print(f"Loading existing dataset split from {split_file_path}")
        with open(split_file, 'r') as f:
            split_config = json.load(f)
            
        # Convert string paths back to Path objects
        test_file = Path(split_config['test_file'])
        train_files = [Path(f) for f in split_config['train_files']]
        val_files = [Path(f) for f in split_config['val_files']]
        
        print(f"Using predefined split")
    else:
        # First run - create a new split
        print(f"Creating new dataset split (will be saved to {split_file_path})")
        
        # Set seed to ensure the split is reproducible
        random.seed(42)
        
        # Randomly select one file as the test set
        test_file = random.choice(parquet_files)
        
        # Remaining files for training and validation
        train_val_files = [f for f in parquet_files if f != test_file]
        
        # Randomly select 1/5 of the files for validation
        val_files = random.sample(train_val_files, len(train_val_files) // 5)
        
        # The remaining files are for training
        train_files = [f for f in train_val_files if f not in val_files]
        
        # Save the split configuration for future runs
        split_config = {
            'test_file': str(test_file),
            'train_files': [str(f) for f in train_files],
            'val_files': [str(f) for f in val_files]
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(split_file) if os.path.dirname(split_file) else '.', exist_ok=True)
        
        with open(split_file, 'w') as f:
            json.dump(split_config, f, indent=4)
        
        print(f"Created and saved new dataset split")

    # Print filenames for each dataset
    print(f"Training files: {[f.name for f in train_files]}")
    print(f"Validation files: {[f.name for f in val_files]}")
    print(f"Testing file: {test_file.name}")

    # Process the files
    def process_file(file_path: Path):
        """Internal function to read and process each parquet file."""
        df = pd.read_parquet(file_path)
        
        # Keep only needed columns to reduce memory usage
        columns_to_keep = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 
                           'Temperature[°C]', 'SOC_ZHU', 'SOH_ZHU']

        df_processed = df[columns_to_keep].copy()
        df_processed.dropna(inplace=True)
        # Process time column into integers and generate corresponding Datetime column
        df_processed['Testtime[s]'] = df_processed['Testtime[s]'].round().astype(int)
        start_date = pd.Timestamp("2023-02-02")
        df_processed['Datetime'] = pd.date_range(
            start=start_date,
            periods=len(df_processed),
            freq='s'
        )
        
        # Sample data every 10 minutes to reduce data size
        df_sampled = df_processed.resample(resample, on='Datetime').mean().reset_index(drop=False)
        
        df_sampled["cell_id"] = file_path.stem.split('_')[1]
        
        return df_sampled, file_path.name

    # Process training, validation, and test files
    test_data = [process_file(test_file)]
    val_data = [process_file(f) for f in val_files]
    train_data = [process_file(f) for f in train_files]

    # Combine data
    df_train = pd.concat([t[0] for t in train_data], ignore_index=True)
    df_val = pd.concat([v[0] for v in val_data], ignore_index=True)
    df_test = test_data[0][0]

    print(f"\nTraining dataframe shape: {df_train.shape}")
    print(f"Validation dataframe shape: {df_val.shape}")
    print(f"Testing dataframe shape: {df_test.shape}\n")

    return df_train, df_val, df_test

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

    # 4. 数据加载与预处理 - 使用一致的数据划分
    data_dir = Path("../01_Datenaufbereitung/Output/Calculated/")
    df_train, df_val, df_test = consistent_load_data(data_dir, split_file_path='optuna_dataset_split.json')

    df_train_scaled, df_val_scaled, df_test_scaled = scale_data(
        df_train, df_val, df_test, scaler_type='standard'
    )

    train_dataset = BatteryDataset(df_train_scaled, hyperparams["SEQUENCE_LENGTH"])
    val_dataset   = BatteryDataset(df_val_scaled,   hyperparams["SEQUENCE_LENGTH"])

    train_loader = DataLoader(train_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=hyperparams["BATCH_SIZE"], shuffle=False)

    # 5. 针对本 trial，创建一个**独立的**文件夹，用来保存 best/last 模型、训练历史文件
    trial_save_dir = optuna_dir / f"trial_{trial.number}"
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
    
    # 8. 将训练中的最佳epoch数作为附加信息存储到trial中
    if isinstance(history, dict) and 'val_loss' in history:
        best_epoch = np.argmin(history['val_loss']) + 1
        trial.set_user_attr('best_epoch', int(best_epoch))
    
    # 9. 把 best_val_loss 返回给 Optuna
    return best_val_loss


def main():
    """
    运行 Optuna 超参数搜索，并输出搜索到的最佳结果。
    """
    # 创建study并设置方向为最小化 - 使用固定随机种子
    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=50, timeout=None)

    print("\n============================")
    print("      Search Finished!      ")
    print("============================")
    print(f"Best trial ID: {study.best_trial.number}")
    print(f"Best trial value (Val. Loss): {study.best_trial.value:.4e}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("============================")

    # 保存最佳超参数
    best_params_path = optuna_dir / "best_hyperparams.json"
    with open(best_params_path, "w") as f:
        # 添加额外信息到最佳参数
        best_params = study.best_trial.params.copy()
        best_params["best_val_loss"] = study.best_trial.value
        best_params["trial_number"] = study.best_trial.number
        if 'best_epoch' in study.best_trial.user_attrs:
            best_params["best_epoch"] = study.best_trial.user_attrs['best_epoch']
        json.dump(best_params, f, indent=4)

    # 保存完整的优化历史
    trials_df = study.trials_dataframe()
    trials_df.to_csv(optuna_dir / "optuna_history.csv", index=False)
    
    # 保存完整study对象(包含所有trials信息)
    with open(optuna_dir / "study.pkl", "wb") as f:
        pickle.dump(study, f)
    
    # 绘制优化过程的可视化图
    try:
        create_optuna_visualizations(study, optuna_dir)
    except ImportError:
        print("无法创建可视化，可能需要安装plotly和kaleido包")


def create_optuna_visualizations(study, save_dir):
    """创建Optuna优化过程的可视化图"""
    try:
        import plotly
        
        # 1. 优化历史
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(str(save_dir / "optimization_history.png"))
        
        # 2. 参数重要性
        try:
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(str(save_dir / "param_importances.png"))
        except:
            print("无法生成参数重要性图")
        
        # 3. 超参数间的交互效应
        try:
            fig = optuna.visualization.plot_contour(study)
            fig.write_image(str(save_dir / "contour.png"))
        except:
            print("无法生成超参数轮廓图")
        
        # 4. 平行坐标图
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(str(save_dir / "parallel_coordinate.png"))
        
        # 5. 切片图
        fig = optuna.visualization.plot_slice(study)
        fig.write_image(str(save_dir / "slice.png"))
    
    except Exception as e:
        print(f"创建可视化时发生错误: {e}")
        print("尝试保存简单的可视化...")
        
        # 简单的优化历史图
        plt.figure(figsize=(10, 6))
        trials = study.trials
        values = [t.value for t in trials if t.value is not None]
        best_values = np.minimum.accumulate(values)
        plt.plot(values, 'o-', label='Trial values')
        plt.plot(best_values, 'r-', label='Best value')
        plt.xlabel('Trial number')
        plt.ylabel('Value')
        plt.title('Optimization History')
        plt.legend()
        plt.grid(True)
        plt.savefig(str(save_dir / "simple_history.png"))
        plt.close()


if __name__ == "__main__":
    main()