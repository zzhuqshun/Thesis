import optuna
import torch
import numpy as np
import copy
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# 导入你的现有模块
from si.si import (
    Config, set_seed, DataProcessor, SOHLSTM, Trainer, 
    create_dataloaders, SI
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
    "SI_LAMBDAS": [0.0, 0.0, 0.0],  # 会被optuna覆盖
    "SI_EPSILON": 1e-3,  # 会被optuna覆盖
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
    """Optuna目标函数 - SI从头开始训练所有任务"""
    global cached_loaders
    
    # 采样SI参数 - 基于原论文的范围
    lam0 = trial.suggest_float("lambda0", 1e-2, 10, log=True)
    lam1 = lam0
    lam2 = lam0
    epsilon = trial.suggest_float("epsilon", 1e-5, 1e-2, log=True)

    
    # 构造配置
    cfg_dict = TASK0_CONFIG.copy()
    cfg_dict["SI_LAMBDAS"] = [lam0, lam1, lam2]
    cfg_dict["SI_EPSILON"] = epsilon
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
    
    # 初始化模型和trainer（SI从头开始训练）
    model = SOHLSTM(3, cfg.HIDDEN_SIZE, cfg.NUM_LAYERS, cfg.DROPOUT).to(DEVICE)
    trainer = Trainer(model, DEVICE, cfg, task_dir=None)
    
    # 初始化SI - 注意SI的特殊初始化方式
    trainer.si = SI(model, si_lambda=cfg.SI_LAMBDAS[0], epsilon=cfg.SI_EPSILON)
    trainer.si.begin_task()  # 开始第一个任务

    # 逐任务训练（SI的特有逻辑）
    for task_idx in range(cfg.NUM_TASKS):
        current_lambda = cfg.SI_LAMBDAS[task_idx]
        current_alpha = cfg.LWF_ALPHAS[task_idx]  # 已关闭LWF
        print(f"--- Task {task_idx} (LWF α={current_alpha:.4f}, SI λ={current_lambda:.4f}, ε={epsilon:.6f}) ---")
        # 更新SI参数
        trainer.si.si_lambda = current_lambda
        
        # 获取当前任务的数据加载器
        train_loader = cached_loaders[f'task{task_idx}_train']
        val_loader = cached_loaders[f'task{task_idx}_val']
        
        # 训练当前任务
        trainer.train_task(train_loader, val_loader, task_idx, alpha_lwf=current_alpha)
        
        # SI任务结束处理
        trainer.si.end_task()
        
        # 为下一个任务准备（除了最后一个任务）
        if task_idx < cfg.NUM_TASKS - 1:
            # 保存当前模型用于知识蒸馏（虽然LWF已关闭）
            trainer.old_model = copy.deepcopy(trainer.model).to(DEVICE)
            trainer.old_model.eval()
            for p in trainer.old_model.parameters():
                p.requires_grad_(False)
            
            # 重置SI状态，开始下一个任务
            trainer.si.begin_task()
    
    # 计算所有任务的MAE
    maes = []
    for task_idx in range(cfg.NUM_TASKS):
        val_loader = cached_loaders[f'task{task_idx}_val']
        mae = calc_mae(trainer.model, val_loader, DEVICE)
        maes.append(mae)
    
    avg_mae = float(np.mean(maes))
    
    # 打印结果
    print(f"Trial {trial.number:2d}: λ=[{lam0:.4f},{lam1:.4f},{lam2:.4f}], ε={epsilon:.6f} → "
          f"MAEs=[{maes[0]:.4f},{maes[1]:.4f},{maes[2]:.4f}], Avg={avg_mae:.4f}")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return avg_mae

def optimize_si(n_trials=50):
    """执行SI参数优化"""
    print(f"Starting SI optimization with {n_trials} trials")
    print(f"Device: {DEVICE}")
    print("Note: SI trains from scratch (no pretrained Task0 model)")
    
    # 创建study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name='si_lambda_optimization'
    )
    
    # 开始优化
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)
    
    # 输出最佳结果
    best_trial = study.best_trial
    print("\n" + "="*50)
    print("SI OPTIMIZATION COMPLETED!")
    print("="*50)
    print(f"Best Validation MAE: {best_trial.value:.6f}")
    print("Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value:.6f}")
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print("="*50)
    
    return study, best_trial

if __name__ == "__main__":
    # 运行优化（SI不需要预训练的Task0模型）
    results_dir = Path.cwd() / "si" / "optuna_new_split"
    results_dir.mkdir(exist_ok=True)
    study, best_trial = optimize_si(n_trials=30)
    
    df_all = study.trials_dataframe(attrs=("number", "values", "params", "user_attrs"))
    df_all.to_csv(results_dir / "all_trials.csv", index=False)
    
    print(f"\nFinal Best MAE: {best_trial.value:.6f}")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v:.6f}")