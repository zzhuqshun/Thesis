from __future__ import annotations
import json
import os
import time
import random
import tempfile
import copy
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import logging
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
        self.SEQUENCE_LENGTH =144
        self.HIDDEN_SIZE = 32
        self.NUM_LAYERS = 2
        self.DROPOUT = 0.0
        self.BATCH_SIZE = 1
        self.LEARNING_RATE = 1e-4
        self.EPOCHS = 200
        self.PATIENCE = 20
        self.WEIGHT_DECAY = 0.0
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
    base_dir = Path(__file__).parent / 'model' / 'Stateful-LSTM'/'many-to-one' 
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
        loaders_lstm = create_dataloaders(data_lstm, config.SEQUENCE_LENGTH)
        model_lstm   = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
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
        loaders  = create_dataloaders(data_inc, config.SEQUENCE_LENGTH)

        model   = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
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
        
def iterate_chunks(seq_len: int, win: int):
    """yield (start,end) pairs that partition range(seq_len) with window=win."""
    for s in range(0, seq_len, win):
        yield s, min(s + win, seq_len)

def get_predictions(model, loader, device, seq_len):

    model.eval()
    preds, tgts = [], []

    with torch.no_grad():
        for X_full, y_full in loader:          # 1 cell / batch
            X_full = X_full.squeeze(0).to(device)          # [T, F]
            y_full = y_full.squeeze(0).cpu().numpy()       # [T]
            hidden = None

            for s, e in iterate_chunks(len(X_full), seq_len):
                x = X_full[s:e].unsqueeze(0)               # [1, t_chunk, F]
                y_hat, hidden = model(x, hidden)   
                hidden = tuple(h.detach() for h in hidden)

                preds.append(y_hat.item())   
                tgts.append(y_full[e-1].item())

    return np.array(preds), np.array(tgts)




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
    if len(tgts) == len(df):
        dates = df['Datetime'].iloc[:len(tgts)]
    else:
        dates = [df['Datetime'].iloc[e-1]
                 for _, e in iterate_chunks(len(df), seq_len)]        
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

# ① 新增一个类，放在 BatteryDataset 旁边即可
class CellDataset(Dataset):
    """
    每个 __getitem__ 返回整段序列 (X_full, y_full)；
    X_full: [T, 3]   y_full: [T]
    """
    def __init__(self, df: pd.DataFrame):
        self.samples = []
        for _, g in df.groupby("cell_id", sort=False):
            X = torch.tensor(
                g[["Voltage[V]","Current[A]","Temperature[°C]"]].values,
                dtype=torch.float32
            )
            y = torch.tensor(g["SOH_ZHU"].values, dtype=torch.float32)
            self.samples.append((X, y))

    def __len__(self):  return len(self.samples)      # = #cells
    def __getitem__(self, idx):
        return self.samples[idx]                      # (X_full, y_full)

def create_dataloaders(datasets, seq_len):
    loaders = {}
    for key, df in datasets.items():
        if df.empty or not any(tag in key for tag in ("train","val","test")):
            continue

        # 每台电池 → 一条样本
        cell_ds = CellDataset(df)

        loader = DataLoader(
            cell_ds,
            batch_size=1,          # 一次一台电池
            shuffle=False,
            num_workers=2,         # 建议开多进程预取
            pin_memory=True,
        )
        loaders[key] = loader
    return loaders


class DataProcessor:
    def __init__(self, data_dir, resample='10min', config = None,
                 base_train_ids=None, base_val_ids=None,
                 update1_train_ids=None, update1_val_ids=None,
                 update2_train_ids=None, update2_val_ids=None,
                 test_cell_id='17'):
        self.data_dir = Path(data_dir)
        self.config = config or Config()
        self.resample = resample
        self.scaler = RobustScaler()

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
            'Testtime[s]','Voltage[V]','Current[A]','Temperature[°C]','SOH_ZHU']]
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
        logger.info("Base train size: %d", len(df_btr))
        logger.info("Base val IDs: %s", self.base_val_ids)        
        logger.info("Base val size: %d", len(df_bval))
        logger.info("Update1 train IDs: %s", self.update1_train_ids)
        logger.info("Update1 train size: %d", len(df_u1t))
        logger.info("Update1 val IDs: %s", self.update1_val_ids)
        logger.info("Update1 val size: %d", len(df_u1v))
        logger.info("Update2 train IDs: %s", self.update2_train_ids)
        logger.info("Update2 train size: %d", len(df_u2t))
        logger.info("Update2 val IDs: %s", self.update2_val_ids)
        logger.info("Update2 val size: %d", len(df_u2v))
        
        # test splits
        df_test = self.process_file(test_fp)
        df_t_base    = df_test[df_test['SOH_ZHU']>=0.9].reset_index(drop=True)
        df_t_update1 = df_test[(df_test['SOH_ZHU']<0.9)&(df_test['SOH_ZHU']>=0.8)].reset_index(drop=True)
        df_t_update2 = df_test[df_test['SOH_ZHU']<0.8].reset_index(drop=True)
        
        logger.info("Test cell ID: %s", self.test_cell_id)
        logger.info("Test size: %d", len(df_test))
        logger.info("Test base size: %d", len(df_t_base))
        logger.info("Test update1 size: %d", len(df_t_update1))
        logger.info("Test update2 size: %d", len(df_t_update2))
        
        
        def scale_df(df):
            if df.empty:
                return df
            df2 = df.copy()
            df2[['Voltage[V]','Current[A]','Temperature[°C]']] = \
                self.scaler.transform(df2[['Voltage[V]','Current[A]','Temperature[°C]']])
            return df2 
        
        # fit scaler on base_train
        # scale all datasets
        self.scaler = self.scaler.fit(df_btr[['Voltage[V]', 'Current[A]', 'Temperature[°C]']])
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
        
    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)        # [B, T, H]
        y_hat = self.fc(out[:, -1, :]).squeeze(-1)   # [B]
        
        return y_hat, hidden
    
class EWC(nn.Module):
    def __init__(self, model, dataloader, device, lam: float, seq_len: int):
        super().__init__()
        self.model      = model
        self.device     = device
        self.dataloader = dataloader
        self.lam        = lam
        self.seq_len    = seq_len

        self.params  = {n: p.clone().detach()
                        for n, p in model.named_parameters() if p.requires_grad}
        self.fisher  = self._compute_fisher()

    # -------- Fisher ----------
    def _compute_fisher(self):
        m = copy.deepcopy(self.model).to(self.device).train()
        for mod in m.modules():
            if isinstance(mod, (nn.Dropout, nn.LSTM)):
                mod.dropout = 0.0

        fisher = {n: torch.zeros_like(p, device=self.device)
                  for n, p in m.named_parameters() if p.requires_grad}

        n_samps = 0
        for X_full, y_full in self.dataloader:
            X_full = X_full.squeeze(0).to(self.device)
            y_full = y_full.squeeze(0).to(self.device)
            
            seq_len = min(self.seq_len, X_full.size(0))
            x = X_full[-seq_len:].unsqueeze(0)   # [1, seq_len, 3]
            y = y_full[-1].unsqueeze(0)          # [1]

            m.zero_grad(set_to_none=True)
            out, _ = m(x)
            F.mse_loss(out, y).backward()

            with torch.no_grad():
                for n, p in m.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.pow(2)
            n_samps += 1

        for n in fisher:
            fisher[n] /= float(n_samps)
        return fisher

    # -------- penalty ----------
    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += self.lam * (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss


# ===============================================================
# Trainer
# ===============================================================
class Trainer:
    def __init__(self, model: nn.Module, device, config, checkpoint_dir: Optional[Path]):
        self.model  = model.to(device)
        self.device = device
        self.config = config

        self.ewc_tasks: List["EWC"] = []
        self.old_model: Optional[nn.Module] = None

        if checkpoint_dir is None:
            self.checkpoint_dir = None
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------- #
    #                               训练一个任务                               #
    # ---------------------------------------------------------------------- #
    def train_task(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        task_id:      int,
        apply_ewc:    bool = True,
        alpha_lwf:    float = 0.0,
        resume:       bool = False,
    ):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        ckpt_last = self.checkpoint_dir / f"task{task_id}_last.pt"
        start_epoch, best_val, no_imp = 0, float("inf"), 0

        # --------- resume ---------
        if resume and ckpt_last.exists():
            ck = torch.load(ckpt_last, map_location=self.device)
            self.model.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["optimizer_state"])
            scheduler.load_state_dict(ck["scheduler_state"])

            # 还原 EWC 任务
            self.ewc_tasks = []
            for dat in ck["ewc_tasks"]:
                e = EWC.__new__(EWC)
                e.params  = {n: p.to(self.device) for n, p in dat["params"].items()}
                e.fisher  = {n: f.to(self.device) for n, f in dat["fisher"].items()}
                e.lam     = dat["lam"]
                e.seq_len = dat.get("seq_len", self.config.SEQUENCE_LENGTH)
                self.ewc_tasks.append(e)

            start_epoch = ck["epoch"] + 1
            best_val    = ck.get("best_val", best_val)
            no_imp      = ck.get("no_imp", no_imp)

        # --------- 日志 ---------
        hist: Dict[str, List] = {k: [] for k in [
            "epoch", "train_loss", "val_loss",
            "task_loss", "kd_loss", "ewc_loss",
            "lr", "time"
        ]}
        TBPTT = self.config.SEQUENCE_LENGTH

        # ==================  逐 epoch  ==================
        for epoch in range(start_epoch, self.config.EPOCHS):
            t0 = time.time()
            self.model.train()

            sum_task = sum_kd = sum_ewc = sum_loss = n_chunks = 0

            # ------------- TRAIN -------------
            for X_full, y_full in train_loader:
                X_full = X_full.squeeze(0).to(self.device)  # [T, F]
                y_full = y_full.squeeze(0).to(self.device)  # [T]
                hidden = None

                for s, e in iterate_chunks(len(X_full), TBPTT):
                    losses, hidden = self._tbptt_step(
                        X_full[s:e],            # x_chunk
                        y_full[s:e],            # y_chunk 
                        hidden,
                        optimizer,
                        apply_ewc,
                        alpha_lwf,
                    )
                    sum_task += losses["task"]
                    sum_kd   += losses["kd"]
                    sum_ewc  += losses["ewc"]
                    sum_loss += losses["total"]
                    n_chunks += 1

            if n_chunks == 0:
                continue  # should not happen

            task_mean  = sum_task / n_chunks
            kd_mean    = sum_kd   / n_chunks
            ewc_mean   = sum_ewc  / n_chunks
            train_loss = sum_loss / n_chunks

            # ------------- VAL -------------
            self.model.eval()
            val_loss = 0.0
            n_val = 0

            with torch.no_grad():
                for X_full, y_full in val_loader:
                    X_full = X_full.squeeze(0).to(self.device)
                    y_full = y_full.squeeze(0).to(self.device)
                    hidden = None

                    for s, e in iterate_chunks(len(X_full), TBPTT):
                        # many-to-one 预测
                        y_pred, hidden = self.model(
                            X_full[s:e].unsqueeze(0), hidden
                        )                              # [1, t_chunk]
                        hidden = tuple(h.detach() for h in hidden)

                        val_loss += F.mse_loss(
                            y_pred.squeeze(0),     # [t_chunk]
                            y_full[e-1]).item()
                        n_val += 1

            val_loss /= n_val
            scheduler.step(val_loss)

            # ------------- log & checkpoint -------------
            hist["epoch"].append(epoch + 1)
            hist["train_loss"].append(train_loss)
            hist["val_loss"].append(val_loss)
            hist["task_loss"].append(task_mean)
            hist["kd_loss"].append(kd_mean)
            hist["ewc_loss"].append(ewc_mean)
            hist["lr"].append(optimizer.param_groups[0]["lr"])
            hist["time"].append(time.time() - t0)

            logger.info(
                "Epoch %03d | task %.4e | kd %.4e | ewc %.4e | val %.4e",
                epoch + 1, task_mean, kd_mean, ewc_mean, val_loss
            )

            # ----- save -----
            state = dict(
                epoch=epoch,
                model_state=self.model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                ewc_tasks=[
                    {
                        "params":  {n: p.cpu() for n, p in e.params.items()},
                        "fisher":  {n: f.cpu() for n, f in e.fisher.items()},
                        "lam":     e.lam,
                        "seq_len": e.seq_len,
                    } for e in self.ewc_tasks
                ],
                best_val=best_val,
                no_imp=no_imp,
            )
            torch.save(state, ckpt_last)

            if val_loss < best_val:
                best_val, no_imp = val_loss, 0
                torch.save(state, self.checkpoint_dir / f"task{task_id}_best.pt")
            else:
                no_imp += 1
                if no_imp >= self.config.PATIENCE:
                    logger.info("Early stopping @ epoch %d", epoch + 1)
                    break

        return hist

    # ------------------------------------------------------------------ #
    #                     单个窗口的 TBPTT + 反传                         #
    # ------------------------------------------------------------------ #
    def _tbptt_step(
        self,
        x_chunk:   torch.Tensor,  # [t_chunk, F]
        y_chunk:   torch.Tensor,  # [t_chunk]
        hidden,
        optimizer,
        apply_ewc: bool,
        alpha_lwf: float,
    ) -> Tuple[Dict[str, float], Tuple[torch.Tensor, torch.Tensor]]:
        """
        执行一次截断反传，返回:
          • losses  dict(float)  —— task/kd/ewc/total
          • hidden  tuple        —— 已 detach，可传给下一窗口
        """
        optimizer.zero_grad()

        # ---- forward (many-to-one) ----
        y_hat, hidden = self.model(
            x_chunk.unsqueeze(0), hidden
        )                                # [1, t_chunk]

        # ---- task loss ----
        task_loss = F.mse_loss(
            y_hat.squeeze(0), y_chunk[-1])  

        # ---- LwF (KD) ----
        kd_loss = torch.zeros((), device=self.device)
        if alpha_lwf > 0 and self.old_model is not None:
            with torch.no_grad():
                y_old, _ = self.old_model(
                    x_chunk.unsqueeze(0), None
                )
            kd_loss = F.mse_loss(y_hat, y_old)

        # ---- EWC ----
        ewc_loss = torch.zeros((), device=self.device)
        if apply_ewc and self.ewc_tasks:
            ewc_loss = sum(e.penalty(self.model) for e in self.ewc_tasks)

        # ---- backward ----
        loss = task_loss + alpha_lwf * kd_loss + ewc_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        # detach hidden
        if hidden is not None:
            hidden = tuple(h.detach() for h in hidden)

        return (
            {
                "task":  task_loss.item(),
                "kd":    kd_loss.item() if torch.is_tensor(kd_loss) else kd_loss,
                "ewc":   ewc_loss.item() if torch.is_tensor(ewc_loss) else ewc_loss,
                "total": loss.item(),
            },
            hidden,
        )

    def consolidate(self, loader, task_id=None, lam: float = 0.0):
        # 新建 Fisher 并保存
        self.ewc_tasks.append(
            EWC(self.model, loader, self.device, lam, self.config.SEQUENCE_LENGTH)
        )

        # 将最新 EWC 信息嵌入 best ckpt，方便 resume
        path = self.checkpoint_dir / f"task{task_id}_best.pt"
        if path.exists():
            state = torch.load(path, map_location=self.device)
            state["ewc_tasks"] = [
                {
                    "params":  {n: p.cpu() for n, p in e.params.items()},
                    "fisher":  {n: f.cpu() for n, f in e.fisher.items()},
                    "lam":     e.lam,
                    "seq_len": e.seq_len,
                }
                for e in self.ewc_tasks
            ]
            torch.save(state, path)

        # 旧模型用于 LwF
        self.old_model = copy.deepcopy(self.model).to(self.device).eval()
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

        preds, tgts = get_predictions(self.model, loader, self.device, seq_len)

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
