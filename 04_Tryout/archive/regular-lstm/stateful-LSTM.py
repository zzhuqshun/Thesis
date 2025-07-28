from __future__ import annotations
import json
import os
import time
import random
import tempfile
import copy
from pathlib import Path
from datetime import datetime
import logging
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

# ===============================================================
# Configuration Class
# ===============================================================
class Config:
    def __init__(self, **kwargs):
        self.SEQUENCE_LENGTH =720
        self.HIDDEN_SIZE = 128
        self.NUM_LAYERS = 2
        self.DROPOUT = 0.3
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-4
        self.EPOCHS = 200
        self.PATIENCE = 20
        self.WEIGHT_DECAY = 1e-6
        self.SCALER = "RobustScaler"
        self.SEED = 42
        self.RESAMPLE = '10min'
        
        self.LWF_ALPHA0 = 0.0  # No LWF for task0
        self.LWF_ALPHA1 = 1.9161084252463925
        self.LWF_ALPHA2 = 0.5711627077804184
                
        self.EWC_LAMBDA0 = 0.0  
        self.EWC_LAMBDA1 = 7780.1555769014285
        self.EWC_LAMBDA2 = 141.35935551752303 # Default value for lambda2, can be adjusted later
        self.Info = {
            "description": "Incremental learning",
            "resample": self.RESAMPLE ,
            "training data": "['03', '05', '07', '09', '11', '15', '21', '23', '25', '27', '29']",
            "validation data": "['01','13','19']",
            "test data": "['17']",
            "base dataset": "['03', '05', '07', '27'], ['01']",
            "update1 dataset": "['21', '23', '25'], ['19']",
            "update2 dataset": "['09', '11', '15', '29'], ['13']",
            "test dataset": "['17']",
            "scaler": "RobustScaler - fit on base train",
            "lambda-types": "None for all tasks",
            "lwf_alpha0": self.LWF_ALPHA0,
            "lwf_alpha1": self.LWF_ALPHA1,
            "lwf_alpha2": self.LWF_ALPHA2,
            "lambda0": self.EWC_LAMBDA0,
            "lambda1": self.EWC_LAMBDA1,
            "lambda2": self.EWC_LAMBDA2,
        }
        for k,v in kwargs.items(): setattr(self, k, v)
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

# ===============================================================
# Main Pipeline
# ===============================================================
def main(joint_training: bool = True):
    # config and logging
    config   = Config()
    base_dir = Path(__file__).parent / 'model' / 'tryouts'/ 'joint_4features' 
    base_dir.mkdir(parents=True, exist_ok=True)
    # single log file for both phases
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = base_dir / 'train.log'
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str(log_path)
               for h in logger.handlers):
        log_f = logging.FileHandler(log_path, encoding='utf-8')
        log_f.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(log_f)

    config.save(base_dir / 'config.json')
    set_seed(config.SEED)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if joint_training:
        # ------------------- Regular LSTM Training -------------------
        # ---- regular dirs ----
        reg_dir       = base_dir / 'regular'
        reg_ckpt_dir  = reg_dir / 'checkpoints'
        reg_results   = reg_dir / 'results'
        reg_ckpt_dir.mkdir(parents=True, exist_ok=True)
        reg_results.mkdir(parents=True, exist_ok=True)
        logger.info("==== Regular LSTM Training Phase ====")
        lstm_train_ids =  ['03', '05', '07', '09', '11', '15', '21', '23', '25', '27', '29']
        lstm_val_ids   = ['01','19','13']
        dp_lstm = DataProcessor(
            data_dir='../01_Datenaufbereitung/Output/Calculated/',
            resample=config.RESAMPLE,
            config=config,
            base_train_ids=lstm_train_ids,
            base_val_ids=lstm_val_ids,
            update1_train_ids=[], update1_val_ids=[],
            update2_train_ids=[], update2_val_ids=[]
        )
        data_lstm    = dp_lstm.prepare_data()
        loaders_lstm = create_dataloaders(data_lstm, config.SEQUENCE_LENGTH, config.BATCH_SIZE, group_by_cell=False)
        model_lstm   = SOHLSTM(4, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
        trainer_lstm = Trainer(model_lstm, device, config, reg_ckpt_dir)
        history_lstm = trainer_lstm.train_task(
            train_loader=loaders_lstm['base_train'],
            val_loader=loaders_lstm['base_val'],
            task_id=0,
            apply_ewc=False,
            resume=True
        )
        # save losses & predictions under regular/results
        plot_losses(history_lstm, reg_results)
        best_ckpt = reg_ckpt_dir / "task0_best.pt"
        if best_ckpt.exists():
            trainer_lstm.evaluate_checkpoint(
                ckpt_path=best_ckpt,
                loader=loaders_lstm['test_full'],
                df=data_lstm['test_full'],
                seq_len=config.SEQUENCE_LENGTH,
                out_dir=reg_results,
                tag="Joint training best model predictions"
            )
    
    else:
        logger.info("==== Skipping joint LSTM Training Phase ====")
        # ------------------- Incremental EWC Training -------------------
        logger.info("==== Incremental EWC Training Phase ====")
        # ---- incremental dir ----
        inc_dir = base_dir / 'incremental'
        inc_dir.mkdir(parents=True, exist_ok=True)

        dp_inc = DataProcessor(
            data_dir='../01_Datenaufbereitung/Output/Calculated/',
            resample=config.RESAMPLE,
            config=config,
            base_train_ids=['03', '05', '07', '27'],
            base_val_ids=['01'],
            update1_train_ids=['21', '23', '25'],
            update1_val_ids=['19'],
            update2_train_ids=['09', '11', '15', '29'],
            update2_val_ids=['13']
        )
        data_inc = dp_inc.prepare_data()
        loaders  = create_dataloaders(data_inc, config.SEQUENCE_LENGTH, config.BATCH_SIZE)

        model   = SOHLSTM(4, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
        trainer = Trainer(model, device, config, checkpoint_dir=str(inc_dir))

        tasks = [
            ('task0', 'base_train',    'base_val',    'test_base',      False,  config.EWC_LAMBDA0, config.LWF_ALPHA0), # Save lambda0 for future tasks
            ('task1', 'update1_train', 'update1_val', 'test_update1',   True,   config.EWC_LAMBDA1, config.LWF_ALPHA1), # EWC on task 0; Save lambda1 for future tasks
            ('task2', 'update2_train', 'update2_val', 'test_update2',   True,   config.EWC_LAMBDA2, config.LWF_ALPHA2) # EWC on task0,1;
        ]

        baseline_metrics: dict[str, dict] = {}
        best_mae   = {}      
        curr_mae   = {}      
        tilde_mae  = {}     
        metric_hist = []     
        delta_hist  = []     

        
        for i, (name, train_key, val_key, test_key, use_ewc, lam, alpha) in enumerate(tasks):
            tr_loader   = loaders.get(train_key)
            val_loader  = loaders.get(val_key)
            test_loader = loaders.get(test_key)
            full_loader = loaders.get('test_full')
            full_df     = data_inc.get('test_full')
            
            if i > 0:
                prev_name, *_ = tasks[i-1]
                best_ckpt = inc_dir / prev_name / 'checkpoints' / f"{prev_name}_best.pt"
                if best_ckpt.exists():
                    logger.info("[%s] Loading best checkpoint from previous task %s...", name, prev_name)
                    state = torch.load(best_ckpt, map_location=device)
                    trainer.model.load_state_dict(state['model_state'])
                    trainer.ewc_tasks = []
                    for data in state.get('ewc_tasks', []):
                        e = EWC.__new__(EWC)
                        e.params = {n: p.to(device) for n, p in data['params'].items()}
                        e.fisher = {n: f.to(device) for n, f in data['fisher'].items()}
                        e.lam = data.get('lam', 0.0)
                        trainer.ewc_tasks.append(e)
                else:
                    logger.warning("[%s] Previous best checkpoint not found, skipping load", name)


            ckpt_dir    = inc_dir / name / 'checkpoints'
            results_dir = inc_dir / name / 'results'
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            results_dir.mkdir(parents=True, exist_ok=True) 
            trainer.checkpoint_dir = ckpt_dir

            best_ckpt = ckpt_dir / f"{name}_best.pt"
            last_ckpt   = ckpt_dir / f"{name}_last.pt"
            trained_f   = ckpt_dir / f"{name}.trained"
            consol_f    = ckpt_dir / f"{name}.consolidated"
            
            # ---------------------- Pre-FWT baseline on the incoming task ----------------
            if i > 0 and test_loader:
                metrics_pre = trainer.evaluate_checkpoint(
                    ckpt_path = None,       
                    loader    = test_loader,
                    df        = data_inc.get(test_key),
                    seq_len   = config.SEQUENCE_LENGTH,
                    out_dir   = None,             
                    tag       = f"{name} Pre-FWT baseline",
                    print_r2  = False
                )
                tilde_mae[name] = metrics_pre['MAE']     
            
            
            # ---------------------- Training phase ----------------
            if tr_loader and val_loader and not trained_f.exists():
                logger.info("[%s] Training...", name)
                if trainer.ewc_tasks:                     
                    lam_map = {f"Task {idx}": float(f"{e.lam:.4e}")          
                        for idx, e in enumerate(trainer.ewc_tasks)}
                    logger.info("[%s] EWC active: %s", name, json.dumps(lam_map))
                else:
                    logger.info("[%s] EWC active: None (no previous tasks)", name)

                logger.info("[%s] This task will be stored with λ = %s%s",
                            name, lam, "" if use_ewc else " (not used)")
                logger.info("[%s] Training with alpha LWF = %.2f", name, alpha)
                history = trainer.train_task(
                    tr_loader, val_loader, task_id=i, 
                    apply_ewc=use_ewc, 
                    alpha_lwf=alpha,
                    resume=last_ckpt.exists())
                pd.DataFrame(history).to_csv(ckpt_dir / f"{name}_history.csv", index=False)
                # Save last checkpoint
                plot_losses(history, results_dir / 'losses')
                trained_f.write_text(datetime.now().isoformat())
                logger.info("[%s] Training completed.", name)
            else:
                logger.info("[%s] Skipping training (already done or no data).", name)

            # ---------------- Consolidation phase ----------------
            if tr_loader and not consol_f.exists():
                logger.info("[%s] Consolidating EWC...", name)
                lam = lam if lam is not None else 0.0
                trainer.consolidate(tr_loader, task_id=i, lam = lam)
                consol_f.write_text(datetime.now().isoformat())
                logger.info("[%s] Consolidation done.", name)
            else:
                logger.info("[%s] No EWC consolidation needed or already done.", name)
        

            # ---------------- Baseline testing on own task ----------------
            if best_ckpt.exists() and test_loader:
                logger.info("[%s] Baseline evaluation on own task %s ...", name, name)
                metrics_own = trainer.evaluate_checkpoint(
                    ckpt_path=best_ckpt,
                    loader=test_loader,
                    df=data_inc.get(test_key),
                    seq_len=config.SEQUENCE_LENGTH,
                    out_dir=results_dir / 'baseline' / name / 'test',
                    tag=f"{name} Baseline on {name}",
                    print_r2=False
                )
                baseline_metrics[name] = metrics_own
                best_mae.setdefault(name, metrics_own['MAE'])
                curr_mae[name] = metrics_own['MAE']
            logger.info("[%s] Baseline testing completed.", name)
            
            # ---------------- Backward testing on test subsets ----------------
            if i > 0:
                for j in range(i):
                    prev_name, _, _, prev_test_key, _, _, _ = tasks[j]
                    prev_loader = loaders.get(prev_test_key)
                    prev_df     = data_inc.get(prev_test_key)
                    if best_ckpt.exists() and prev_loader:
                        #  Backward testing on previous task 
                        logger.info("[%s] Backward testing on previous task %s...", name, prev_name)
                        metrics_prev = trainer.evaluate_checkpoint(
                            ckpt_path=best_ckpt,
                            loader=prev_loader,
                            df=prev_df,
                            seq_len=config.SEQUENCE_LENGTH,
                            out_dir=inc_dir / name / 'results' / 'backward' / prev_name,
                            tag=f"{name} BACKWARD on {prev_name}",
                            print_r2=False
                        )
                        curr_mae[prev_name] = metrics_prev['MAE']

            # ---------------- ACC / BWT / FWT /  ----------------
            old_tasks = [t for t in best_mae if t != name]

            for t in old_tasks:
                delta = curr_mae[t] - best_mae[t]           
                delta_hist.append({'stage': name, 'task': t, 'ΔMAE': delta})
                logger.info("[%s] ΔMAE on %s: %+.4e", name, t, delta)

            # ---- ACC (Average Accuracy)  -------------------------------
            ACC = - np.mean(list(curr_mae.values()))         
            logger.info("[%s] ACC (-MAE): %.4e", name, ACC)

            # Backward-transfer (positive ⇒ forgetting, negative ⇒ backward boost)
            if old_tasks:
                BWT = np.mean([curr_mae[t] - best_mae[t] for t in old_tasks])
                logger.info("[%s] BWT: %+.4e", name, BWT)
            else:
                BWT = np.nan

            # Forward-transfer on the new task (positive ⇒ prior knowledge helped)
            if name in tilde_mae:
                FWT = tilde_mae[name] - curr_mae[name]        
                logger.info("[%s] FWT: %+.4e", name, FWT)
            else:
                FWT = np.nan

            metric_hist.append({'task': name, 'ACC': ACC, 'BWT': BWT, 'FWT': FWT})
                            
            # ---------------- Evaluate on full test set ----------------
            if best_ckpt.exists():
                logger.info("[%s] Evaluating BEST checkpoint...", name)
                trainer.evaluate_checkpoint(
                    ckpt_path=best_ckpt,
                    loader=full_loader,
                    df=full_df,
                    seq_len=config.SEQUENCE_LENGTH,
                    out_dir=results_dir / "forward" / "test_full",
                    tag=f"{name} Evaluation on full test set"
                )
            logger.info("[%s] Evaluation completed.", name)

        # ---------------- Save final metrics and delta MAE history ----------------
        df_m = pd.DataFrame(metric_hist)
        df_m.to_csv(inc_dir / "continual_metrics.csv", index=False)
        logger.info("Saved ACC/BWT/FWT history to %s", inc_dir / "continual_metrics.csv")

        # ---------------- Plot transfer curves ----------------
        plt.figure(figsize=(6,4))
        plt.plot(df_m['task'], df_m['BWT'], marker='o', label='BWT')
        plt.plot(df_m['task'], df_m['FWT'], marker='s', label='FWT')
        plt.ylabel('MAE difference'); plt.grid(True); plt.legend()
        plt.savefig(inc_dir / "transfer_curves.png"); plt.close()

        # ---------------- Save delta MAE history ----------------
        if delta_hist:
            pd.DataFrame(delta_hist).to_csv(inc_dir / "delta_MAE_history.csv", index=False)


        logger.info("==== All tasks completed ====")
# ===============================================================
# Utilities
# ===============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# get raw predictions and targets for plotting
def get_predictions(model, loader, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for x,y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x).cpu().numpy()
            preds.append(out)
            tgts.append(y.cpu().numpy().ravel())
    return np.concatenate(preds), np.concatenate(tgts)

# plotting functions
def plot_losses(history, out_dir):
    df = pd.DataFrame(history)
    plt.figure(figsize=(8,5))
    plt.semilogy(df['epoch'], df['train_loss'], label='train')
    plt.semilogy(df['epoch'], df['val_loss'],   label='val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir/'loss_curves.png'); plt.close()

def plot_predictions(preds, tgts, metrics, df, seq_len, out_dir):
    dates = df['Datetime'].iloc[seq_len:].values
    plt.figure(figsize=(10,6))
    plt.plot(dates, tgts, label='actual')
    plt.plot(dates, preds, label='predicted', alpha=0.7)
    plt.xlabel('Time'); plt.ylabel('SOH'); plt.legend(); plt.grid(True)
    title = f"Predictions and actual SOH\n RMSE: {metrics['RMSE']:.4e}, MAE: {metrics['MAE']:.4e}, R2: {metrics['R2']:.4f}"
    plt.title(title)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir/'prediction_timeseries.png'); plt.close()

def plot_prediction_scatter(preds, tgts, out_dir):
    plt.figure(figsize=(6,6))
    plt.scatter(tgts, preds, alpha=0.6)
    lims = [min(tgts.min(), preds.min()), max(tgts.max(), preds.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel('Actual SOH'); plt.ylabel('Predicted SOH'); plt.grid(True)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir/'prediction_scatter.png'); plt.close()

# ===============================================================
# Dataset & DataProcessor
# ===============================================================
class BatteryDataset(Dataset):
    def __init__(self, df, seq_len, group_by_cell=True):
        """
        Args:
            df: DataFrame containing battery data
            seq_len: Sequence length for windowing
            group_by_cell: If True, create windows within each cell (no cross-cell sequences)
                          If False, create windows across the entire dataset (may cross cells)
        """
        self.seq_len = seq_len
        self.group_by_cell = group_by_cell
        self.samples = []
        self.cell_stats = {}
        
        if group_by_cell:
            # 原有逻辑：按 cell_id 分组处理
            self._create_grouped_samples(df)
        else:
            # 新逻辑：不分组，可能跨cell创建窗口
            self._create_ungrouped_samples(df)
        
        # 打印详细统计
        self._print_statistics(df)
    
    def _create_grouped_samples(self, df):
        """按cell分组创建样本（原有逻辑）"""
        for cell_id, group in df.groupby('cell_id'):
            # 重置索引以确保连续性
            group = group.reset_index(drop=True)
            
            # 提取特征和标签
            feats = group[['Voltage[V]', 'Current[A]', 'Temperature[°C]', 'EFC']].values
            targets = group['SOH_ZHU'].values
            
            # 记录统计信息
            original_length = len(feats)
            usable_samples = max(0, original_length - self.seq_len)
            
            self.cell_stats[cell_id] = {
                'original_length': original_length,
                'usable_samples': usable_samples,
                'dropped_samples': self.seq_len if original_length > self.seq_len else original_length
            }
            
            # 为当前电池单元创建样本
            if original_length > self.seq_len:
                for i in range(usable_samples):
                    x = torch.tensor(feats[i:i+self.seq_len], dtype=torch.float32)
                    y = torch.tensor(targets[i+self.seq_len], dtype=torch.float32)
                    self.samples.append((x, y, cell_id))
    
    def _create_ungrouped_samples(self, df):
        """不分组创建样本（可能跨cell）"""
        # 直接在整个数据集上创建窗口
        feats = df[['Voltage[V]', 'Current[A]', 'Temperature[°C]', 'EFC']].values
        targets = df['SOH_ZHU'].values
        cell_ids = df['cell_id'].values
        
        total_length = len(feats)
        usable_samples = max(0, total_length - self.seq_len)
        
        # 创建样本（可能跨cell边界）
        cross_cell_count = 0
        for i in range(usable_samples):
            x = torch.tensor(feats[i:i+self.seq_len], dtype=torch.float32)
            y = torch.tensor(targets[i+self.seq_len], dtype=torch.float32)
            
            # 检查是否跨cell（用于统计）
            window_cells = set(cell_ids[i:i+self.seq_len+1])  # 包括target的cell
            is_cross_cell = len(window_cells) > 1
            if is_cross_cell:
                cross_cell_count += 1
            
            # 记录主要的cell_id（窗口中最多的那个）
            window_cell_ids = cell_ids[i:i+self.seq_len]
            from collections import Counter
            main_cell = Counter(window_cell_ids).most_common(1)[0][0]
            
            self.samples.append((x, y, main_cell))
        
        # 统计信息（对于ungrouped模式）
        self.cell_stats['all_cells'] = {
            'original_length': total_length,
            'usable_samples': usable_samples,
            'cross_cell_samples': cross_cell_count,
            'cross_cell_ratio': cross_cell_count / usable_samples if usable_samples > 0 else 0,
            'dropped_samples': self.seq_len
        }
    
    def _print_statistics(self, df):
        total_samples = len(self.samples)
        total_cells = df['cell_id'].nunique()
        
        logger.info("\n=== Dataset Statistics ===")
        logger.info("Grouping mode: %s", "By Cell" if self.group_by_cell else "Ungrouped (Cross-Cell)")
        logger.info("Total cells: %d", total_cells)
        logger.info("Total samples created: %d", total_samples)
        logger.info("Sequence length: %d", self.seq_len)
        
        if self.group_by_cell:
            # 原有的per-cell统计
            logger.info("\nPer-cell breakdown:")
            for cell_id, stats in self.cell_stats.items():
                logger.info(
                    "  Cell %s: %d points → %d samples (predict from index %d)",
                    cell_id, stats['original_length'], stats['usable_samples'], self.seq_len
                )
            
            total_original = sum(stats['original_length'] for stats in self.cell_stats.values())
            efficiency = (total_samples * 100) / total_original if total_original > 0 else 0
            
            logger.info("\nOverall efficiency: %.1f%% (%d/%d points used)", efficiency, total_samples, total_original)
        else:
            # 跨cell统计
            stats = self.cell_stats['all_cells']
            logger.info("\nCross-cell analysis:")
            logger.info("  Total data points: %d", stats['original_length'])
            logger.info("  Cross-cell samples: %d (%.1f%%)", 
                       stats['cross_cell_samples'], stats['cross_cell_ratio'] * 100)
            logger.info("  Within-cell samples: %d (%.1f%%)", 
                       total_samples - stats['cross_cell_samples'],
                       (1 - stats['cross_cell_ratio']) * 100)
            
            efficiency = (total_samples * 100) / stats['original_length'] if stats['original_length'] > 0 else 0
            logger.info("  Overall efficiency: %.1f%% (%d/%d points used)", 
                       efficiency, total_samples, stats['original_length'])
        
        logger.info("Cell IDs: %s", sorted(df['cell_id'].unique()))
        logger.info("=" * 30)
    
    def get_cross_cell_analysis(self):
        """获取跨cell样本的详细分析（仅在ungrouped模式下有用）"""
        if self.group_by_cell:
            return "Analysis only available in ungrouped mode"
        
        return self.cell_stats['all_cells']
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 返回 (x, y) 或 (x, y, cell_id) 根据需要
        return self.samples[idx][:2]  # 只返回 x, y


# 使用示例
def create_dataloaders(datasets, seq_len, batch_size, group_by_cell=False):
    """
    创建数据加载器
    
    Args:
        datasets: 数据集字典
        seq_len: 序列长度
        batch_size: 批次大小
        group_by_cell: 是否按cell分组创建窗口
    """
    loaders = {}
    
    for key, df in datasets.items():
        if not df.empty and ('train' in key or 'val' in key or 'test' in key):
            logger.info("\nCreating dataset for %s...", key)
            ds = BatteryDataset(df, seq_len, group_by_cell=group_by_cell)
            shuffle = 'train' in key
            loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
            loaders[key] = loader
            logger.info("DataLoader for %s: %d samples, %d batches (shuffle=%s)",
                        key, len(ds), len(loader), shuffle)
    
    return loaders



class DataProcessor:
    def __init__(self, data_dir, resample='10min', config = None,
                 base_train_ids=None, base_val_ids=None,
                 update1_train_ids=None, update1_val_ids=None,
                 update2_train_ids=None, update2_val_ids=None,
                 test_cell_id='17',
                 feature_cols=['Voltage[V]', 'Current[A]', 'Temperature[°C]', 'EFC']):
        self.data_dir = Path(data_dir)
        self.config = config or Config()
        self.resample = resample
        self.scaler = RobustScaler()
        self.feature_cols = feature_cols
        # manual splits
        self.base_train_ids    = base_train_ids or []
        self.base_val_ids      = base_val_ids or []
        self.update1_train_ids = update1_train_ids or []
        self.update1_val_ids   = update1_val_ids or []
        self.update2_train_ids = update2_train_ids or []
        self.update2_val_ids   = update2_val_ids or []
        self.test_cell_id      = test_cell_id

    def load_cell_data(self):
        files = sorted(self.data_dir.glob('*.parquet'),
                       key=lambda x: int(x.stem.split('_')[-1]))
        info = {fp.stem.split('_')[-1]: fp for fp in files}
        if self.test_cell_id not in info:
            raise ValueError(f"Test cell {self.test_cell_id} not found")
        test_fp = info.pop(self.test_cell_id)
        return info, test_fp

    def process_file(self, fp):
        df = pd.read_parquet(fp)[[
            'Testtime[s]','Voltage[V]','Current[A]','Temperature[°C]','EFC','SOH_ZHU']]
        df = df.dropna().reset_index(drop=True)
        df['Testtime[s]'] = df['Testtime[s]'].round().astype(int)
        df['Datetime'] = pd.date_range('2023-02-02', periods=len(df), freq='s')
        df = df.set_index('Datetime').resample(self.resample).mean().reset_index()
        df['cell_id'] = fp.stem.split('_')[-1]
        return df

    def prepare_data(self):
        info_map, test_fp = self.load_cell_data()
        def build(ids):
            if not ids: 
                return pd.DataFrame()
            dfs = [self.process_file(info_map[c]) for c in ids]
            return pd.concat(dfs, ignore_index=True)
        
        # build phases
        df_btr = build(self.base_train_ids)
        df_bval= build(self.base_val_ids)
        df_u1t= build(self.update1_train_ids)
        df_u1v= build(self.update1_val_ids)
        df_u2t= build(self.update2_train_ids)
        df_u2v= build(self.update2_val_ids)
        
        logger.info("Base train IDs: %s", self.base_train_ids)
        logger.info("Base val IDs: %s", self.base_val_ids)        
        logger.info("Update1 train IDs: %s", self.update1_train_ids)
        logger.info("Update1 val IDs: %s", self.update1_val_ids)
        logger.info("Update2 train IDs: %s", self.update2_train_ids)
        logger.info("Update2 val IDs: %s", self.update2_val_ids)
        
        # test splits
        df_test = self.process_file(test_fp)
        df_t_base    = df_test[df_test['SOH_ZHU']>=0.9].reset_index(drop=True)
        df_t_update1 = df_test[(df_test['SOH_ZHU']<0.9)&(df_test['SOH_ZHU']>=0.8)].reset_index(drop=True)
        df_t_update2 = df_test[df_test['SOH_ZHU']<0.8].reset_index(drop=True)
        
        logger.info("Test cell ID: %s", self.test_cell_id)
        
        
        def scale_df(df):
            if df.empty:
                return df
            df2 = df.copy()
            df2[self.feature_cols] = \
                self.scaler.transform(df2[self.feature_cols])
            return df2 
        
        # fit scaler on base_train
        # scale all datasets
        self.scaler = self.scaler.fit(df_btr[self.feature_cols])
        df_btr_scaled    = scale_df(df_btr)
        df_bval_scaled   = scale_df(df_bval)
        df_u1t_scaled    = scale_df(df_u1t)
        df_u1v_scaled    = scale_df(df_u1v)
        df_u2t_scaled    = scale_df(df_u2t)
        df_u2v_scaled    = scale_df(df_u2v)
        df_test_scaled     = scale_df(df_test)
        df_t_base_scaled   = scale_df(df_t_base)
        df_t_update1_scaled= scale_df(df_t_update1)
        df_t_update2_scaled= scale_df(df_t_update2)
        logger.info("[Scaler after fit] center_=%s", self.scaler.center_)
        logger.info("[Scaler after fit] scale_ =%s", self.scaler.scale_)
        logger.info("Feature columns: %s", self.feature_cols)
        logger.info("Resampling and scaling complete with %s", self.config.SCALER)

        return {
            'base_train':    df_btr_scaled,
            'base_val':      df_bval_scaled,
            'update1_train': df_u1t_scaled,
            'update1_val':   df_u1v_scaled,
            'update2_train': df_u2t_scaled,
            'update2_val':   df_u2v_scaled,
            'test_full':     df_test_scaled,
            'test_base':     df_t_base_scaled,
            'test_update1':  df_t_update1_scaled,
            'test_update2':  df_t_update2_scaled
        }

# ===============================================================
# Model & EWC
# ===============================================================
class SOHLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size//2,1)
        )
    def forward(self, x):
        out,_ = self.lstm(x)
        h = out[:,-1,:]
        return self.fc(h).squeeze(-1)

class EWC:
    def __init__(self, model, dataloader, device, lam):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.params = {n: p.clone().detach() for n,p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher()
        self.lam = lam

    def _compute_fisher(self):
        """
        Computes the Fisher Information Matrix for the model parameters
        using the provided dataloader.
        Returns a dictionary with parameter names as keys and Fisher
        information tensors as values.
        """
        # — 1. 拷贝模型并保持 train() —
        model_copy = copy.deepcopy(self.model).to(self.device)
        model_copy.train()                        # 允许 CuDNN backward

        # — 2. 关闭 Dropout 随机丢节点，但仍保持 train mode —
        for m in model_copy.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
            if isinstance(m, nn.LSTM):
                m.dropout = 0.0 

        # — 3. 初始化 Fisher 累积张量 —
        fisher = {n: torch.zeros_like(p, device=self.device)
                for n, p in model_copy.named_parameters() if p.requires_grad}

        n_processed = 0  # 真正参与计算的样本数（考虑 drop_last）

        for x, y in self.dataloader:
            x, y = x.to(self.device), y.to(self.device)

            model_copy.zero_grad(set_to_none=True)
            out = model_copy(x)
            loss = F.mse_loss(out, y)
            loss.backward()

            bs = x.size(0)
            n_processed += bs

            # 不需构建计算图，避免额外显存
            with torch.no_grad():
                for n, p in model_copy.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.pow(2) * bs   # 按 batch 大小加权

        # — 4. 归一化 —
        for n in fisher:
            fisher[n] /= float(n_processed)

        # — 5. 清理 —
        del model_copy
        torch.cuda.empty_cache()

        return fisher


    def penalty(self, model):
        loss=0
        for n,p in model.named_parameters():
            if p.requires_grad:
                loss += self.lam * ( self.fisher[n] * (p - self.params[n]).pow(2) ).sum()
        return loss

# ===============================================================
# Trainer
# ===============================================================
class Trainer:
    def __init__(self, model, device, config, checkpoint_dir):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.ewc_tasks = []
        self.old_model: nn.Module | None = None
        if checkpoint_dir is None:
            self.checkpoint_dir = None
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_task(self, train_loader, val_loader, task_id,
                   apply_ewc=True, alpha_lwf=0.0, resume=False):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.config.LEARNING_RATE,
                                     weight_decay=self.config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.5, patience=5)
        ckpt_last = self.checkpoint_dir/f"task{task_id}_last.pt"
        start_epoch = 0
        best_val = float('inf'); no_imp = 0
        # resume
        if resume and ckpt_last.exists():
            ck = torch.load(ckpt_last, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ck['model_state'])
            optimizer.load_state_dict(ck['optimizer_state'])
            scheduler.load_state_dict(ck['scheduler_state'])
            self.ewc_tasks = []
            # load ewc tasks
            for data in ck['ewc_tasks']:
                e = EWC.__new__(EWC)
                e.model = self.model
                e.device = self.device
                e.params = {n:p.to(self.device) for n,p in data['params'].items()}
                e.fisher = {n:f.to(self.device) for n,f in data['fisher'].items()}
                e.lam = data.get('lam', 0.0)
                self.ewc_tasks.append(e)
            start_epoch = ck['epoch'] + 1
            best_val = ck.get('best_val', best_val)
            no_imp    = ck.get('no_imp',    no_imp)

        history = {
            'epoch':[], 'train_loss':[], 'val_loss':[], 'lr':[], 'time':[],
            'task_loss':[], 'kd_loss':[], 'ewc_loss':[]
        }
                
        for epoch in tqdm.tqdm(range(start_epoch, self.config.EPOCHS), desc="Training"):
            epoch_start = time.time() 
            self.model.train()
            train_loss = 0
            sum_task, sum_kd, sum_ewc = 0., 0., 0.
            for x, y in train_loader:
                x,y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                
                y_pred = self.model(x)
                task_loss = F.mse_loss(y_pred, y)
                
                kd_loss = torch.zeros((), device=self.device)
                if alpha_lwf > 0 and self.old_model is not None:
                    with torch.no_grad():
                        y_old = self.old_model(x)
                    kd_loss = F.mse_loss(y_pred, y_old)
                
                ewc_loss = torch.zeros((), device=self.device)
                if apply_ewc and self.ewc_tasks:
                    ewc_loss = sum(t.penalty(self.model) for t in self.ewc_tasks)
                
                loss = task_loss + alpha_lwf * kd_loss + ewc_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
                sum_task += task_loss.item() * x.size(0)
                sum_kd   += kd_loss.item() * x.size(0)
                sum_ewc  += ewc_loss.item() * x.size(0)
                train_loss += loss.item() * x.size(0) 
            # average losses
            task_mean = sum_task / len(train_loader.dataset)
            kd_mean   = sum_kd   / len(train_loader.dataset)
            ewc_mean  = sum_ewc  / len(train_loader.dataset)
            train_loss /= len(train_loader.dataset)
            # validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x,y in val_loader:
                    x,y = x.to(self.device), y.to(self.device)
                    val_loss += F.mse_loss(self.model(x), y).item()*x.size(0)
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            epoch_time = time.time() - epoch_start
            history['epoch'].append(epoch+1)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['task_loss'].append(task_mean)
            history['kd_loss'].append(kd_mean)
            history['ewc_loss'].append(ewc_mean)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['time'].append(epoch_time)
            
            
            logger.info(
                "Epoch %03d | task %.4e | kd %.4e | ewc %.4e | val %.4e | lr %.2e | %.2fs",
                epoch+1, task_mean, kd_mean, ewc_mean, val_loss,
                optimizer.param_groups[0]['lr'], epoch_time
            )
        
            
            # checkpoints
            state = {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'ewc_tasks': [
                    {
                        'params':{n:p.clone().cpu() for n,p in e.params.items()},
                        'fisher':{n:f.clone().cpu() for n,f in e.fisher.items()},
                        'lam': e.lam
                        }
                    for e in self.ewc_tasks
                ],
                'best_val': best_val,
                'no_imp': no_imp
            }
            torch.save(state, ckpt_last)
            if val_loss < best_val:
                best_val = val_loss
                no_imp = 0
                best_path = self.checkpoint_dir/f"task{task_id}_best.pt"
                torch.save(state, best_path)
            else:
                no_imp += 1
                if no_imp >= self.config.PATIENCE:
                    logger.info("Early stopping at epoch %d", epoch+1)
                    break
        return history

    def consolidate(self, loader, task_id=None, lam=0.0):
        self.ewc_tasks.append(EWC(self.model, loader, self.device, lam))
        # embed into existing checkpoints
        path = self.checkpoint_dir / f"task{task_id}_best.pt"
        if path.exists():
            state = torch.load(path, map_location=self.device, weights_only=False)
            state['ewc_tasks'] = [
                {
                    'params':{n:p.cpu() for n,p in e.params.items()},
                    'fisher':{n:f.cpu() for n,f in e.fisher.items()},
                    'lam': e.lam
                }
                for e in self.ewc_tasks
            ]
            torch.save(state, path)
        self.old_model = copy.deepcopy(self.model).to(self.device)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)
                
    def evaluate_checkpoint(
        self,
        ckpt_path: Path,
        loader: DataLoader,
        df: pd.DataFrame,
        seq_len: int,
        out_dir: Path,
        tag: str = "",
        print_r2: bool = True
    ) -> dict:
        """
        Load the specified checkpoint, make predictions on the loader's data,
        compute RMSE/MAE/R2, save time series and scatter plots to out_dir, and log results.
        """
        if ckpt_path and Path(ckpt_path).exists():
            state = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state["model_state"])
        self.model.to(self.device).eval()

        preds, tgts = get_predictions(self.model, loader, self.device)

        metrics = {
            'RMSE': np.sqrt(mean_squared_error(tgts, preds)),
            'MAE':  mean_absolute_error(tgts, preds),
            'R2':   r2_score(tgts, preds)
        }
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

            plot_predictions(preds, tgts, metrics, df, seq_len, out_dir)
            plot_prediction_scatter(preds, tgts, out_dir)

        prefix = f"[{tag}]" if tag else ""
        if print_r2:
            logger.info(
                "%s RMSE: %.4e, MAE: %.4e, R2: %.4f",
                prefix, metrics['RMSE'], metrics['MAE'], metrics['R2']
            )
        else:   
            logger.info(
                "%s RMSE: %.4e, MAE: %.4e",
                prefix, metrics['RMSE'], metrics['MAE']
            )

        return metrics

if __name__ == '__main__':
    main()
