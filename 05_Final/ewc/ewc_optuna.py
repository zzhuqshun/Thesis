import optuna
import torch
import numpy as np
import copy
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# 导入你的现有模块
from ewc.ewc import (
    Config, set_seed, DataProcessor, SOHLSTM, Trainer, 
    create_dataloaders, EWC
)

# 配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TASK0_CONFIG = {
    "MODE": "incremental",
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
    "ALPHA": 0.1,
    "NUM_TASKS": 3,
    "LWF_ALPHAS": [0.0, 0.0, 0.0],  # 关闭LWF
    "EWC_LAMBDAS": [0.0, 0.0, 0.0],  # 会被optuna覆盖
}

# 缓存数据和loaders，避免重复加载
cached_loaders = None

def calc_mae(model, loader, device):
    """计算MAE"""
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

def objective(trial):
    """Optuna目标函数"""
    global cached_loaders
    
    # 采样EWC参数
    lam0 = trial.suggest_float("lambda0", 1e-2, 30, log=True)
    lam1 = lam0
    lam2 = lam0  # 最后任务不需要EWC
    
    # 构造配置
    cfg_dict = TASK0_CONFIG.copy()
    cfg_dict["EWC_LAMBDAS"] = [lam0, lam1, lam2]
    cfg = Config(**cfg_dict)
    print("Inncremental_datasets: ", cfg.incremental_datasets)
    set_seed(cfg.SEED + trial.number)
    
    # 加载数据（只在第一次加载）
    if cached_loaders is None:
        print("Loading data and creating dataloaders...")
        dp = DataProcessor(cfg.DATA_DIR, cfg.RESAMPLE, cfg)
        data = dp.prepare_incremental_data(cfg.incremental_datasets)

        cached_loaders = create_dataloaders(data, cfg.SEQUENCE_LENGTH, cfg.BATCH_SIZE)
        print("Data loaded!")
    
    # 初始化模型和trainer
    model = SOHLSTM(3, cfg.HIDDEN_SIZE, cfg.NUM_LAYERS, cfg.DROPOUT).to(DEVICE)
    trainer = Trainer(model, DEVICE, cfg, task_dir=None)
    
    # ========= Task0 =========
    history0 = trainer.train_task(
        train_loader=cached_loaders['task0_train'],
        val_loader=cached_loaders['task0_val'],
        task_id=0,
        alpha_lwf=cfg.LWF_ALPHAS[0]
    )
    trainer.consolidate(cached_loaders['task0_train'], task_id=0, ewc_lambda=lam0)

    # 取最后一个 epoch 的 ewc_loss / task_loss
    ratio0 = history0['ewc_loss'][-1] / (history0['task_loss'][-1] + 1e-12)
    print(f"[Trial {trial.number}] Task0  ratio = {ratio0:.2%}  (λ={lam0:.3g})")

    # ========= Task1 =========
    history1 = trainer.train_task(
        train_loader=cached_loaders['task1_train'],
        val_loader=cached_loaders['task1_val'],
        task_id=1,
        alpha_lwf=cfg.LWF_ALPHAS[1]
    )
    trainer.consolidate(cached_loaders['task1_train'], task_id=1, ewc_lambda=lam1)

    ratio1 = history1['ewc_loss'][-1] / (history1['task_loss'][-1] + 1e-12)
    print(f"[Trial {trial.number}] Task1  ratio = {ratio1:.2%}  (λ={lam1:.3g})")

    # ========= Task2 =========
    history2 = trainer.train_task(
        train_loader=cached_loaders['task2_train'],
        val_loader=cached_loaders['task2_val'],
        task_id=2,
        alpha_lwf=cfg.LWF_ALPHAS[2]
    )
    # Task2 如果 λ=0 就不用 consolidate；保留也无妨
    trainer.consolidate(cached_loaders['task2_train'], task_id=2, ewc_lambda=lam2)

    ratio2 = history2['ewc_loss'][-1] / (history2['task_loss'][-1] + 1e-12)
    print(f"[Trial {trial.number}] Task2  ratio = {ratio2:.2%}  (λ={lam2:.3g})")

    # 计算所有任务的MAE
    maes = []
    for task_idx in range(cfg.NUM_TASKS):
        val_loader = cached_loaders[f'task{task_idx}_val']
        mae = calc_mae(trainer.model, val_loader, DEVICE)
        maes.append(mae)
    
    avg_mae = float(np.mean(maes))
    
    # 打印结果
    print(f"Trial {trial.number:2d}: λ0={lam0:.4f}, λ1={lam1:.4f} → "
          f"MAEs=[{maes[0]:.4f}, {maes[1]:.4f}, {maes[2]:.4f}], Avg={avg_mae:.4f}")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return avg_mae

def optimize_ewc(n_trials=50):
    """执行EWC参数优化"""
    print(f"Starting EWC optimization with {n_trials} trials")
    print(f"Device: {DEVICE}")
    # 创建study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        study_name='ewc_lambda_optimization'
    )
    
    # 开始优化
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)
    
    # 输出最佳结果
    best_trial = study.best_trial
    print("\n" + "="*50)
    print("EWC OPTIMIZATION COMPLETED!")
    print("="*50)
    print(f"Best Validation MAE: {best_trial.value:.6f}")
    print("Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value:.6f}")
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print("="*50)
    
    return study, best_trial

if __name__ == "__main__":   
    results_dir = Path.cwd() / "ewc" /"optuna_newsplit"
    results_dir.mkdir(exist_ok=True)
    # 运行优化
    study, best_trial = optimize_ewc(n_trials=30)
    
    df_all = study.trials_dataframe(attrs=("number", "values", "params", "user_attrs"))
    df_all.to_csv(results_dir / "all_trials.csv", index=False)
    
    
    print(f"\nFinal Best MAE: {best_trial.value:.6f}")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v:.6f}")