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
        self.EWC_LAMBDA = None
        self.EWC_LAMBDA1 = 14.16
        self.EWC_LAMBDA2 = 4549.99
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
            "scaler": "RobustScaler-update after each task",
            "lambda-types": "None for all tasks",
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
def main(skip_regular=False):
    # config and logging
    config   = Config()
    base_dir = Path(__file__).parent / 'model' / 'Naive_Fine_Tuning'
    # ---- incremental dir ----
    inc_dir = base_dir / 'incremental'
    inc_dir.mkdir(parents=True, exist_ok=True)

    # single log file for both phases
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = base_dir / 'train.log'
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str(log_path)
               for h in logger.handlers):
        log_f = logging.FileHandler(log_path)
        log_f.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(log_f)

    config.save(base_dir / 'config.json')
    set_seed(config.SEED)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not skip_regular:
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
        loaders_lstm = create_dataloaders(data_lstm, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
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
        logger.info("==== Skipping Regular LSTM Training Phase ====")

    # ------------------- Incremental EWC Training -------------------
    logger.info("==== Incremental EWC Training Phase ====")
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

    model   = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    trainer = Trainer(model, device, config, checkpoint_dir=str(inc_dir))

    tasks = [
        ('task0', 'base_train',    'base_val',    'test_base',      False, config.EWC_LAMBDA), # no EWC for base task
        ('task1', 'update1_train', 'update1_val', 'test_update1',   False,  config.EWC_LAMBDA1), # EWC for update1
        ('task2', 'update2_train', 'update2_val', 'test_update2',   False,  config.EWC_LAMBDA2) # EWC for update2
    ]

    for i, (name, train_key, val_key, test_key, use_ewc, lam) in enumerate(tasks):
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
                state = torch.load(best_ckpt, map_location=device, weights_only=True)
                trainer.model.load_state_dict(state['model_state'])
                trainer.ewc_tasks = []
                for data in state.get('ewc_tasks', []):
                    e = EWC.__new__(EWC)
                    e.params = {n: p.to(device) for n, p in data['params'].items()}
                    e.fisher = {n: f.to(device) for n, f in data['fisher'].items()}
                    trainer.ewc_tasks.append(e)
            else:
                logger.warning("[%s] Previous best checkpoint not found, skipping load", name)


        ckpt_dir    = inc_dir / name / 'checkpoints'
        results_dir = inc_dir / name / 'results'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True) 
        trainer.checkpoint_dir = ckpt_dir

        trainer.config.EWC_LAMBDA = lam if use_ewc and lam is not None else 0

        best_ckpt = ckpt_dir / f"{name}_best.pt"
        last_ckpt   = ckpt_dir / f"{name}_last.pt"
        trained_f   = ckpt_dir / f"{name}.trained"
        consol_f    = ckpt_dir / f"{name}.consolidated"
        
        # ---------------------- Training phase ----------------
        if tr_loader and val_loader and not trained_f.exists():
            logger.info("[%s] Training...", name)
            history = trainer.train_task(tr_loader, val_loader, task_id=i, apply_ewc=use_ewc, resume=last_ckpt.exists())
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
            trainer.consolidate(tr_loader, task_id=i)
            consol_f.write_text(datetime.now().isoformat())
            logger.info("[%s] Consolidation done.", name)
        else:
            logger.info("[%s] No EWC consolidation needed or already done.", name)
        
        # ---------------- Backward testing on test subsets ----------------
        if i > 0:
            for j in range(i):
                prev_name, _, _, prev_test_key, _, _ = tasks[j]
                prev_loader = loaders.get(prev_test_key)
                prev_df     = data_inc.get(prev_test_key)
                if best_ckpt.exists() and prev_loader:
                    logger.info("[%s] Backward testing on previous task %s...", name, prev_name)
                    trainer.evaluate_checkpoint(
                        ckpt_path=best_ckpt,
                        loader=prev_loader,
                        df=prev_df,
                        seq_len=config.SEQUENCE_LENGTH,
                        out_dir=inc_dir / name / 'results' / 'backward' / prev_name,
                        tag=f"{name} BACKWARD on {prev_name}"
                    )

        logger.info("[%s] Backward on own task %s ...", name, name)
        # ---------------- Backward testing on own task ----------------
        if best_ckpt.exists() and test_loader:
            trainer.evaluate_checkpoint(
                ckpt_path=best_ckpt,
                loader=test_loader,
                df=data_inc.get(test_key),
                seq_len=config.SEQUENCE_LENGTH,
                out_dir=results_dir / 'backward' / name / 'test',
                tag=f"{name} BACKWARD on {name}"
            )
        logger.info("[%s] Backward testing completed.", name)
        # ---------------- Forward testing on full test set ----------------
        if best_ckpt.exists():
            logger.info("[%s] Evaluating BEST checkpoint...", name)
            trainer.evaluate_checkpoint(
                ckpt_path=best_ckpt,
                loader=full_loader,
                df=full_df,
                seq_len=config.SEQUENCE_LENGTH,
                out_dir=results_dir / "forward" / "test_full",
                tag=f"{name} FORWARD on test"
            )
        logger.info("[%s] Forward testing completed.", name)

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

# create PyTorch data loaders from dict of DataFrames
def create_dataloaders(datasets, seq_len, batch_size):
    loaders = {}
    for key, df in datasets.items():
        if not df.empty and ('train' in key or 'val' in key or 'test' in key):
            ds = BatteryDataset(df, seq_len)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=('train' in key))
            loaders[key] = loader
    return loaders

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
    def __init__(self, df, seq_len):
        feats = df[['Voltage[V]','Current[A]','Temperature[°C]']].values
        self.X = torch.tensor(feats, dtype=torch.float32)
        self.y = torch.tensor(df['SOH_ZHU'].values, dtype=torch.float32)
        self.seq_len = seq_len
    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, idx):
        return (
            self.X[idx:idx+self.seq_len],
            self.y[idx+self.seq_len]
        )

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
        # logger.info("[Scaler after fit] center_=%s", self.scaler.center_)
        # logger.info("[Scaler after fit] scale_ =%s", self.scaler.scale_)
        logger.info("Resampling and scaling complete with %s", self.config.SCALER)
        
        # # --------- 1) Base train: 初始化 all_seen, 更新 scaler ---------
        # all_seen = df_btr.copy()
        # if isinstance(self.scaler, RobustScaler):
        #     # RobustScaler 只能重新 fit
        #     self.scaler.fit(all_seen[['Voltage[V]', 'Current[A]', 'Temperature[°C]']])
        # else:
        #     # StandardScaler / MaxAbsScaler 支持 partial_fit
        #     self.scaler.partial_fit(all_seen[['Voltage[V]', 'Current[A]', 'Temperature[°C]']])
        # # 打印当前 scaler 属性
        # if hasattr(self.scaler, 'mean_'):
        #     logger.info("[Scaler after base train] mean=%s",  self.scaler.mean_)
        #     logger.info("[Scaler after base train] var =%s", self.scaler.var_)
        # else:
        #     logger.info("[Scaler after base train] center_=%s", self.scaler.center_)
        #     logger.info("[Scaler after base train] scale_ =%s", self.scaler.scale_)
        # df_btr_scaled     = scale_df(df_btr)
        # df_bval_scaled    = scale_df(df_bval)
        # df_t_base_scaled  = scale_df(df_t_base)

        # # --------- 2) Update1 train: 累加 all_seen，再更新 scaler ---------
        # if not df_u1t.empty:
        #     all_seen = pd.concat([all_seen, df_u1t], ignore_index=True)
        #     if isinstance(self.scaler, RobustScaler):
        #         self.scaler.fit(all_seen[['Voltage[V]', 'Current[A]', 'Temperature[°C]']])
        #     else:
        #         self.scaler.partial_fit(df_u1t[['Voltage[V]', 'Current[A]', 'Temperature[°C]']])
        # if hasattr(self.scaler, 'mean_'):
        #     logger.info("[Scaler after update1 train] mean=%s",  self.scaler.mean_)
        #     logger.info("[Scaler after update1 train] var =%s", self.scaler.var_)
        # else:
        #     logger.info("[Scaler after update1 train] center_=%s", self.scaler.center_)
        #     logger.info("[Scaler after update1 train] scale_ =%s", self.scaler.scale_)
        # df_u1t_scaled     = scale_df(df_u1t)
        # df_u1v_scaled     = scale_df(df_u1v)
        # df_t_update1_scaled = scale_df(df_t_update1)

        # # --------- 3) Update2 train: 再次累加 all_seen，并更新 scaler ---------
        # if not df_u2t.empty:
        #     all_seen = pd.concat([all_seen, df_u2t], ignore_index=True)
        #     if isinstance(self.scaler, RobustScaler):
        #         self.scaler.fit(all_seen[['Voltage[V]', 'Current[A]', 'Temperature[°C]']])
        #     else:
        #         self.scaler.partial_fit(df_u2t[['Voltage[V]', 'Current[A]', 'Temperature[°C]']])
        # if hasattr(self.scaler, 'mean_'):
        #     logger.info("[Scaler after update2 train] mean=%s",  self.scaler.mean_)
        #     logger.info("[Scaler after update2 train] var =%s", self.scaler.var_)
        # else:
        #     logger.info("[Scaler after update2 train] center_=%s", self.scaler.center_)
        #     logger.info("[Scaler after update2 train] scale_ =%s", self.scaler.scale_)
        # df_u2t_scaled     = scale_df(df_u2t)
        # df_u2v_scaled     = scale_df(df_u2v)
        # df_t_update2_scaled = scale_df(df_t_update2)

        # # --------- 4) Use the updated scaler to transform test ---------
        # df_test_scaled     = scale_df(df_test)
        # logger.info("Data scaling complete with %s", self.config.SCALER)
        
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
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.params = {n: p.clone().detach() for n,p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher()

    def _compute_fisher(self):
        orig_mode = self.model.training

        model_copy = copy.deepcopy(self.model).to(self.device)
        model_copy.train()  

        fisher = {n: torch.zeros_like(p) for n,p in model_copy.named_parameters() if p.requires_grad}
        total_samples = len(self.dataloader.dataset)

        for x, y in self.dataloader:
            model_copy.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            out = model_copy(x).squeeze()
            loss = F.mse_loss(out, y)
            loss.backward()

            batch_size = x.size(0)
            for n, p in model_copy.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone().pow(2) * batch_size

        for n in fisher:
            fisher[n] /= total_samples

        if orig_mode:
            self.model.train()
        else:
            self.model.eval()

        return fisher

    def penalty(self, model):
        loss=0
        for n,p in model.named_parameters():
            if p.requires_grad:
                loss += ( self.fisher[n] * (p - self.params[n]).pow(2) ).sum()
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
        if checkpoint_dir is None:
            self.checkpoint_dir = None
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_task(self, train_loader, val_loader, task_id,
                   apply_ewc=True, resume=False):
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
                e.model = self.model; e.device = self.device
                e.params = {n:p.to(self.device) for n,p in data['params'].items()}
                e.fisher = {n:f.to(self.device) for n,f in data['fisher'].items()}
                self.ewc_tasks.append(e)
            start_epoch = ck['epoch'] + 1
            best_val = ck.get('best_val', best_val)
            no_imp    = ck.get('no_imp',    no_imp)

        history = {'epoch':[], 'train_loss':[], 'val_loss':[], 'lr':[], 'time':[]}
        
        for epoch in tqdm.tqdm(range(start_epoch, self.config.EPOCHS), desc="Training"):
            epoch_start = time.time() 
            self.model.train()
            train_loss = 0
            for x,y in train_loader:
                x,y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = F.mse_loss(out, y)
                if apply_ewc and self.ewc_tasks:
                    ewc_penalty = sum(t.penalty(self.model) for t in self.ewc_tasks)
                    loss = loss + self.config.EWC_LAMBDA * ewc_penalty
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
                train_loss += loss.item()*x.size(0)
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
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['time'].append(epoch_time)
            
            logger.info("Epoch %d, Train Loss: %.4e, Val Loss: %.4e, LR: %.4e, Time: %.2fs",
                        epoch+1, train_loss, val_loss,
                        optimizer.param_groups[0]['lr'], epoch_time)
            
            # checkpoints
            state = {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'ewc_tasks': [
                    {'params':{n:p.clone().cpu() for n,p in e.params.items()},
                     'fisher':{n:f.clone().cpu() for n,f in e.fisher.items()}}
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

    def consolidate(self, loader, task_id=None):
        self.ewc_tasks.append(EWC(self.model, loader, self.device))
        # embed into existing checkpoints
        path = self.checkpoint_dir / f"task{task_id}_best.pt"
        if path.exists():
            state = torch.load(path, map_location=self.device, weights_only=False)
            state['ewc_tasks'] = [
                {'params':{n:p.cpu() for n,p in e.params.items()},
                'fisher':{n:f.cpu() for n,f in e.fisher.items()}}
                for e in self.ewc_tasks
            ]
            torch.save(state, path)
                
    def evaluate_checkpoint(
        self,
        ckpt_path: Path,
        loader: DataLoader,
        df: pd.DataFrame,
        seq_len: int,
        out_dir: Path,
        tag: str = ""
    ) -> dict:
        """
        Load the specified checkpoint, make predictions on the loader's data,
        compute RMSE/MAE/R2, save time series and scatter plots to out_dir, and log results.
        """
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state['model_state'])
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
        logger.info(
            "%s RMSE: %.4f, MAE: %.4f, R2: %.4f",
            prefix, metrics['RMSE'], metrics['MAE'], metrics['R2']
        )

        return metrics

if __name__ == '__main__':
    main()
