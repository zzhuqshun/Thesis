import optuna
import torch
import numpy as np
import logging
from pathlib import Path
import json
import copy
from datetime import datetime

# 导入你的现有模块
from si import (
    Config, set_seed, DataProcessor, SOHLSTM, Trainer, 
    create_dataloaders, SI
)

class FastSIOptimizer:
    """快速SI超参数优化器 - 最小化版本"""
    
    def __init__(self, base_config, n_trials=30):
        self.base_config = base_config
        self.n_trials = n_trials
        self.study = None
        
        # 简单日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def objective(self, trial):
        """Optuna目标函数 - 直接在内存中完成训练和评估"""
        try:
            # 创建试验配置
            trial_config = copy.deepcopy(self.base_config)
            
            # 只优化SI的4个核心参数
            lambda_0 = trial.suggest_float('lambda_task_0', 0.0, 10.0, log=True)
            lambda_1 = trial.suggest_float('lambda_task_1', 0.0, 10.0, log=True)
            lambda_2 = trial.suggest_float('lambda_task_2', 0.0, 10.0, log=True)
            epsilon = trial.suggest_float('epsilon', 0.001, 1.0, log=True)
            
            # 应用SI参数
            trial_config.SI_LAMBDAS = [lambda_0, lambda_1, lambda_2]
            trial_config.SI_EPSILON = epsilon
            
            # 设置随机种子
            set_seed(trial_config.SEED + trial.number)
            
            # 直接在内存中训练和评估
            val_mae = self.train_and_evaluate_in_memory(trial_config)
            
            # 简单日志 - 显示使用的参数值
            self.logger.info(
                f"Trial {trial.number:2d}: "
                f"SI_LAMBDAS=[{lambda_0:.4f}, {lambda_1:.4f}, {lambda_2:.4f}], "
                f"SI_EPSILON={epsilon:.6f} → Val_MAE={val_mae:.6f}"
            )
            
            return -val_mae  # 负号是因为要最小化MAE，但Optuna要最大化
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            return -1000.0
    
    def train_and_evaluate_in_memory(self, config):
        """
        在内存中完成完整的增量学习训练和validation评估
        注意：这里不需要BASE_DIR，因为不保存任何文件
        """
        # 准备数据
        dp = DataProcessor(config.DATA_DIR, config.RESAMPLE, config)
        data = dp.prepare_incremental_data(config.incremental_datasets)
        loaders = create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
        
        # 初始化设备和模型
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
        
        # 创建trainer，task_dir=None表示不保存文件
        trainer = Trainer(model, device, config, task_dir=None)
        
        # 初始化SI
        trainer.si = SI(model, si_lambda=config.SI_LAMBDAS[0], epsilon=config.SI_EPSILON)
        
        # 调试：验证epsilon是否正确传递
        if hasattr(trainer.si, 'epsilon'):
            assert abs(trainer.si.epsilon - config.SI_EPSILON) < 1e-8, f"Epsilon not passed correctly: {trainer.si.epsilon} vs {config.SI_EPSILON}"
        
        trainer.si.begin_task()
        
        # 逐任务训练
        for task_idx in range(config.NUM_TASKS):
            current_lambda = config.SI_LAMBDAS[task_idx]
            current_alpha = config.LWF_ALPHAS[task_idx]
            
            # 更新SI参数
            trainer.si.si_lambda = current_lambda
            
            # 获取当前任务的数据加载器
            train_loader = loaders[f'task{task_idx}_train']
            val_loader = loaders[f'task{task_idx}_val']
            
            # 训练当前任务（在内存中，不保存模型）
            _ = trainer.train_task(train_loader, val_loader, task_idx, alpha_lwf=current_alpha)
            
            # 结束当前任务
            trainer.si.end_task()
            
            # 为下一个任务准备
            if task_idx < config.NUM_TASKS - 1:
                # 保存当前模型用于知识蒸馏（在内存中）
                trainer.old_model = copy.deepcopy(trainer.model).to(device)
                trainer.old_model.eval()
                for p in trainer.old_model.parameters():
                    p.requires_grad_(False)
                
                # 重置SI状态
                trainer.si.begin_task()
        
        # 训练完成后，用最终模型在所有validation sets上评估
        val_maes = []
        for task_idx in range(config.NUM_TASKS):
            val_loader = loaders[f'task{task_idx}_val']
            _, _, metrics = trainer.evaluate(val_loader, alpha=config.ALPHA, log=False)
            val_maes.append(metrics['MAE'])
        
        # 返回平均validation MAE
        return np.mean(val_maes) if val_maes else 1000.0
    
    
    # 删除不再需要的注释，因为现在直接在内存中训练
    
    def optimize(self):
        """执行快速优化"""
        self.logger.info(f"Starting fast SI optimization with {self.n_trials} trials")
        self.logger.info("Using validation-based MAE scoring for fair hyperparameter search")
        
        # 创建研究对象
        self.study = optuna.create_study(
            direction='maximize',  # 最大化负MAE（即最小化MAE）
            sampler=optuna.samplers.TPESampler(seed=42)
            # 第一次搜索不使用pruner，确保所有trial完整运行，便于分析
        )
        
        # 开始优化
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        # 输出最佳结果
        best_trial = self.study.best_trial
        best_mae = -best_trial.value  # 转换回正的MAE值
        
        # 创建最佳配置（之前漏了这一步）
        best_config = self.create_best_config()
        
        print("\n" + "="*60)
        print("SI OPTIMIZATION COMPLETED!")
        print("="*60)
        print(f"Best Validation MAE: {best_mae:.6f}")
        print(f"Best Parameters:")
        print(f"  SI_LAMBDAS: {best_config.SI_LAMBDAS}")
        print(f"  SI_EPSILON: {best_config.SI_EPSILON}")
        print("="*60)
        
        # 保存最佳配置
        self.save_best_config(best_config, best_mae)
        
        return self.study, best_config
    
    def create_best_config(self):
        """创建最佳配置"""
        best_params = self.study.best_trial.params
        best_config = copy.deepcopy(self.base_config)
        
        best_config.SI_LAMBDAS = [
            best_params['lambda_task_0'],
            best_params['lambda_task_1'], 
            best_params['lambda_task_2']
        ]
        best_config.SI_EPSILON = best_params['epsilon']
        
        return best_config
    
    def save_best_config(self, best_config, best_mae):
        """保存最佳配置"""
        results = {
            'best_si_lambdas': best_config.SI_LAMBDAS,
            'best_si_epsilon': best_config.SI_EPSILON,
            'best_validation_mae': best_mae,
            'optimization_method': 'validation_based_mae',
            'optimization_date': datetime.now().isoformat(),
            'n_trials': self.n_trials,
            'optuna_best_value': self.study.best_trial.value  # 这是负的MAE值
        }
        
        Path("optuna_results").mkdir(exist_ok=True)
        with open('optuna_results/best_si_config.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Best configuration saved to: optuna_results/best_si_config.json")

# 使用示例
def optimize_si_fast(base_config, n_trials=30):
    """快速优化SI参数的主函数"""
    
    optimizer = FastSIOptimizer(base_config, n_trials)
    study, best_config = optimizer.optimize()
    
    return best_config

# 直接运行示例
if __name__ == "__main__":
    # 创建基础配置
    base_config = Config()
    
    # 快速优化
    best_config = optimize_si_fast(base_config, n_trials=30)
    
    print(f"\nOptimization complete! Best config saved to: optuna_results/best_si_config.json")