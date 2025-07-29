import optuna
import torch
import numpy as np
import copy
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# 导入你的现有模块
from ewc import (
    Config, set_seed, DataProcessor, SOHLSTM, Trainer, 
    create_dataloaders, EWC
)

# 配置
TASK0_CKPT_DIR = Path.cwd() / "strategies/fine-tuning/incremental/task0"
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
    lam0 = trial.suggest_float("lambda0", 1.0, 1e4, log=True)
    lam1 = trial.suggest_float("lambda1", 1.0, 1e4, log=True)
    lam2 = 0.0  # 最后任务不需要EWC
    
    # 构造配置
    cfg_dict = TASK0_CONFIG.copy()
    cfg_dict["EWC_LAMBDAS"] = [lam0, lam1, lam2]
    cfg = Config(**cfg_dict)
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
    
    # 载入Task0预训练模型
    ckpt_path = TASK0_CKPT_DIR / "task0_best.pt"
    state = torch.load(ckpt_path, map_location=DEVICE)
    
    # 处理EWC任务数据
    if 'ewc_tasks' not in state or len(state.get('ewc_tasks', [])) == 0:
        # 需要为Task0做consolidation
        trainer.model.load_state_dict(state['model_state'])
        trainer.consolidate(
            loader=cached_loaders['task0_train'],
            task_id=0,
            ewc_lambda=lam0
        )
        # 保存更新的checkpoint
        serialized = []
        for reg in trainer.ewc_tasks:
            serialized.append({
                'params': {n: p.cpu() for n, p in reg.params.items()},
                'fisher': {n: f.cpu() for n, f in reg.fisher.items()},
                'ewc_lambda': reg.ewc_lambda,
            })
        torch.save({
            'model_state': trainer.model.state_dict(),
            'ewc_tasks': serialized
        }, ckpt_path)
        print(f"Trial {trial.number}: Updated checkpoint with Task0 EWC info")
    else:
        trainer.model.load_state_dict(state['model_state'])
        # 重建EWC正则化器
        trainer.ewc_tasks = []
        ewc_task_data = state['ewc_tasks']
        ewc_reg = EWC.__new__(EWC)
        ewc_reg.model = trainer.model
        ewc_reg.device = DEVICE
        ewc_reg.params = {n: p.to(DEVICE) for n, p in ewc_task_data[0]['params'].items()}
        ewc_reg.fisher = {n: f.to(DEVICE) for n, f in ewc_task_data[0]['fisher'].items()}
        ewc_reg.ewc_lambda = lam0  # 使用当前试验的lambda
        trainer.ewc_tasks.append(ewc_reg)
    
    # 设置old_model（虽然LWF关闭了，但保持结构完整）
    trainer.old_model = copy.deepcopy(trainer.model).to(DEVICE)
    trainer.old_model.eval()
    for p in trainer.old_model.parameters():
        p.requires_grad_(False)
    
    # Task1训练
    trainer.train_task(
        train_loader=cached_loaders['task1_train'],
        val_loader=cached_loaders['task1_val'],
        task_id=1,
        alpha_lwf=0.0
    )
    trainer.consolidate(cached_loaders['task1_train'], task_id=1, ewc_lambda=lam1)
    
    # Task2训练
    trainer.train_task(
        train_loader=cached_loaders['task2_train'],
        val_loader=cached_loaders['task2_val'],
        task_id=2,
        alpha_lwf=0.0
    )
    trainer.consolidate(cached_loaders['task2_train'], task_id=2, ewc_lambda=lam2)
    
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
    print(f"Task0 checkpoint: {TASK0_CKPT_DIR}")
    
    # 创建study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
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
    # 检查Task0模型文件
    ckpt_path = TASK0_CKPT_DIR / "task0_best.pt"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        exit(1)
    
    # 运行优化
    study, best_trial = optimize_ewc(n_trials=50)
    
    results_dir = Path.cwd() / "optuna_results"
    results_dir.mkdir(exist_ok=True)
    df_all = study.trials_dataframe(attrs=("number", "values", "params", "user_attrs"))
    df_all.to_csv(results_dir / "all_trials.csv", index=False)
    
    
    print(f"\nFinal Best MAE: {best_trial.value:.6f}")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v:.6f}")