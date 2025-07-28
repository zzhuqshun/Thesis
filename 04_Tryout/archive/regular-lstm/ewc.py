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
import math

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
        self.SEQUENCE_LENGTH = 720  # 窗口大小
        self.HIDDEN_SIZE = 128
        self.NUM_LAYERS = 2
        self.DROPOUT = 0.3
        self.BATCH_SIZE = 1  # 每个batch一个cell的一个chunk
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
        self.EWC_LAMBDA2 = 141.35935551752303
        
        self.Info = {
            "description": "Incremental learning with seq-to-seq",
            "resample": self.RESAMPLE,
            "training data": "['03', '05', '07', '09', '11', '15', '21', '23', '25', '27', '29']",
            "validation data": "['01','13','19']",
            "test data": "['17']",
            "base dataset": "['03', '05', '07', '27'], ['01']",
            "update1 dataset": "['21', '23', '25'], ['19']",
            "update2 dataset": "['09', '11', '15', '29'], ['13']",
            "test dataset": "['17']",
            "scaler": "RobustScaler - fit on base train",
            "model": "Seq-to-Seq LSTM with non-overlapping windows"
        }
        for k,v in kwargs.items(): 
            setattr(self, k, v)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

# ===============================================================
# Dataset for Seq-to-Seq with Non-overlapping Windows
# ===============================================================
class CellSeq2SeqDataset(Dataset):
    """
    每个 cell 内按 sequence_length 划分 **非重叠窗口**。
    尾段不足 sequence_length 的部分直接忽略 —— 训练看不到它。
    为节省内存，只存索引而不提前切 tensor。
    """
    FEAT_COLS = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
    TARGET_COL = 'SOH_ZHU'

    def __init__(self, cell_data_dict, sequence_length: int):
        self.seq_len   = sequence_length
        self.cell_dfs  = cell_data_dict          # {cid: DataFrame}
        self.index_map = []                      # [(cid, start_idx), ...]

        for cid in sorted(cell_data_dict.keys()):
            df = cell_data_dict[cid]
            n_chunks = len(df) // sequence_length   # 整除；尾段丢弃
            for i in range(n_chunks):
                self.index_map.append((cid, i * sequence_length))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        cid, start = self.index_map[idx]
        df   = self.cell_dfs[cid]
        end  = start + self.seq_len

        x = torch.tensor(df[self.FEAT_COLS].values[start:end],
                         dtype=torch.float32)
        y = torch.tensor(df[self.TARGET_COL].values[start:end],
                         dtype=torch.float32)
        return x, y, cid 

# ===============================================================
# Data Processor - 按cell组织数据
# ===============================================================
class DataProcessor:
    def __init__(self, data_dir, resample='10min', config=None,
                 base_train_ids=None, base_val_ids=None,
                 update1_train_ids=None, update1_val_ids=None,
                 update2_train_ids=None, update2_val_ids=None,
                 test_cell_id='17'):
        self.data_dir = Path(data_dir)
        self.config = config or Config()
        self.resample = resample
        self.scaler = RobustScaler()

        self.base_train_ids = base_train_ids or []
        self.base_val_ids = base_val_ids or []
        self.update1_train_ids = update1_train_ids or []
        self.update1_val_ids = update1_val_ids or []
        self.update2_train_ids = update2_train_ids or []
        self.update2_val_ids = update2_val_ids or []
        self.test_cell_id = test_cell_id

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
        """返回按cell组织的数据字典"""
        info_map, test_fp = self.load_cell_data()
        
        def build_cell_dict(ids):
            """构建cell_id -> DataFrame的字典"""
            if not ids:
                return {}
            cell_dict = {}
            for cell_id in ids:
                if cell_id in info_map:
                    df = self.process_file(info_map[cell_id])
                    cell_dict[cell_id] = df
            return cell_dict
        
        # 构建各阶段的cell数据字典
        base_train_cells = build_cell_dict(self.base_train_ids)
        base_val_cells = build_cell_dict(self.base_val_ids)
        update1_train_cells = build_cell_dict(self.update1_train_ids)
        update1_val_cells = build_cell_dict(self.update1_val_ids)
        update2_train_cells = build_cell_dict(self.update2_train_ids)
        update2_val_cells = build_cell_dict(self.update2_val_ids)
        
        # 处理测试数据
        df_test = self.process_file(test_fp)
        test_full_cells = {self.test_cell_id: df_test}
        
        # 测试数据的子集
        df_t_base = df_test[df_test['SOH_ZHU'] >= 0.9].reset_index(drop=True)
        df_t_update1 = df_test[(df_test['SOH_ZHU'] < 0.9) & (df_test['SOH_ZHU'] >= 0.8)].reset_index(drop=True)
        df_t_update2 = df_test[df_test['SOH_ZHU'] < 0.8].reset_index(drop=True)
        
        test_base_cells = {self.test_cell_id: df_t_base} if len(df_t_base) > 0 else {}
        test_update1_cells = {self.test_cell_id: df_t_update1} if len(df_t_update1) > 0 else {}
        test_update2_cells = {self.test_cell_id: df_t_update2} if len(df_t_update2) > 0 else {}
        
        # Fit scaler on base train data
        all_base_train_data = []
        for df in base_train_cells.values():
            all_base_train_data.append(df[['Voltage[V]', 'Current[A]', 'Temperature[°C]']])
        if all_base_train_data:
            all_base_train_df = pd.concat(all_base_train_data)
            self.scaler.fit(all_base_train_df)
        
        # Scale all data
        def scale_cell_dict(cell_dict):
            scaled_dict = {}
            for cell_id, df in cell_dict.items():
                df_scaled = df.copy()
                if len(df_scaled) > 0:
                    df_scaled[['Voltage[V]','Current[A]','Temperature[°C]']] = \
                        self.scaler.transform(df_scaled[['Voltage[V]','Current[A]','Temperature[°C]']])
                scaled_dict[cell_id] = df_scaled
            return scaled_dict
        
        # 记录信息
        logger.info("Data loading summary:")
        logger.info(f"Base train cells: {list(base_train_cells.keys())}, total rows: {sum(len(df) for df in base_train_cells.values())}")
        logger.info(f"Base val cells: {list(base_val_cells.keys())}, total rows: {sum(len(df) for df in base_val_cells.values())}")
        logger.info(f"Update1 train cells: {list(update1_train_cells.keys())}, total rows: {sum(len(df) for df in update1_train_cells.values())}")
        logger.info(f"Update1 val cells: {list(update1_val_cells.keys())}, total rows: {sum(len(df) for df in update1_val_cells.values())}")
        logger.info(f"Update2 train cells: {list(update2_train_cells.keys())}, total rows: {sum(len(df) for df in update2_train_cells.values())}")
        logger.info(f"Update2 val cells: {list(update2_val_cells.keys())}, total rows: {sum(len(df) for df in update2_val_cells.values())}")
        logger.info(f"Test cell: {self.test_cell_id}, total rows: {len(df_test)}")
        
        return {
            'base_train': scale_cell_dict(base_train_cells),
            'base_val': scale_cell_dict(base_val_cells),
            'update1_train': scale_cell_dict(update1_train_cells),
            'update1_val': scale_cell_dict(update1_val_cells),
            'update2_train': scale_cell_dict(update2_train_cells),
            'update2_val': scale_cell_dict(update2_val_cells),
            'test_full': scale_cell_dict(test_full_cells),
            'test_base': scale_cell_dict(test_base_cells),
            'test_update1': scale_cell_dict(test_update1_cells),
            'test_update2': scale_cell_dict(test_update2_cells)
        }

# ===============================================================
# Seq-to-Seq Model
# ===============================================================
class SOHLSTMSeq2Seq(nn.Module):
    """Seq-to-Seq LSTM for SOH prediction"""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x, hidden=None):
        # x: (batch, seq_len, features)
        # Flatten parameters for cuDNN
        self.lstm.flatten_parameters()
        
        out, hidden_new = self.lstm(x, hidden)
        # out: (batch, seq_len, hidden_size)
        
        # Detach hidden states to prevent backprop through time across chunks
        if hidden_new is not None:
            hidden_new = (hidden_new[0].detach(), hidden_new[1].detach())
        
        # 对每个时间步应用全连接层
        batch, seq_len, _ = out.shape
        out_flat = out.contiguous().view(-1, self.hidden_size)
        predictions_flat = self.fc(out_flat)
        predictions = predictions_flat.view(batch, seq_len)
        
        return predictions, hidden_new
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)

# ===============================================================
# EWC for Seq-to-Seq
# ===============================================================
class EWC:
    def __init__(self, model, dataloader, device, lam):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher()
        self.lam = lam

    def _compute_fisher(self):
        """计算Fisher信息矩阵 - 适配seq-to-seq输出，传递hidden state"""
        model_copy = copy.deepcopy(self.model).to(self.device)
        model_copy.train()

        # 关闭 Dropout
        for m in model_copy.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
            if isinstance(m, nn.LSTM):
                m.dropout = 0.0 

        fisher = {n: torch.zeros_like(p, device=self.device)
                for n, p in model_copy.named_parameters() if p.requires_grad}

        n_processed = 0
        current_cell = None
        hidden = None

        for x, y, cell_id in self.dataloader:
            x, y = x.to(self.device), y.to(self.device)

            # 检查是否需要重置hidden state
            if current_cell != cell_id[0]:
                current_cell = cell_id[0]
                hidden = model_copy.init_hidden(x.size(0), self.device)

            model_copy.zero_grad(set_to_none=True)
            out, hidden = model_copy(x, hidden)  # out: (batch, seq_len)
            loss = F.mse_loss(out, y)
            loss.backward()

            bs = x.size(0) * x.size(1)  # batch_size * seq_len
            n_processed += bs

            with torch.no_grad():
                for n, p in model_copy.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.pow(2) * bs

        # 归一化
        for n in fisher:
            fisher[n] /= float(n_processed)

        del model_copy
        torch.cuda.empty_cache()
        return fisher

    def penalty(self, model):
        """计算EWC penalty"""
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.params:
                loss += self.lam * (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss

# ===============================================================
# Trainer for Seq-to-Seq
# ===============================================================
class Trainer:
    def __init__(self, model, device, config, checkpoint_dir):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.ewc_tasks = []
        self.old_model = None
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
        
        ckpt_last = self.checkpoint_dir / f"task{task_id}_last.pt"
        start_epoch = 0
        best_val = float('inf')
        no_imp = 0
        
        # Resume logic
        if resume and ckpt_last.exists():
            ck = torch.load(ckpt_last, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ck['model_state'])
            optimizer.load_state_dict(ck['optimizer_state'])
            scheduler.load_state_dict(ck['scheduler_state'])
            self.ewc_tasks = []
            for data in ck['ewc_tasks']:
                e = EWC.__new__(EWC)
                e.model = self.model
                e.device = self.device
                e.params = {n: p.to(self.device) for n, p in data['params'].items()}
                e.fisher = {n: f.to(self.device) for n, f in data['fisher'].items()}
                e.lam = data.get('lam', 0.0)
                self.ewc_tasks.append(e)
            start_epoch = ck['epoch'] + 1
            best_val = ck.get('best_val', best_val)
            no_imp = ck.get('no_imp', no_imp)

        history = {
            'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': [], 'time': [],
            'task_loss': [], 'kd_loss': [], 'ewc_loss': []
        }
                
        for epoch in tqdm.tqdm(range(start_epoch, self.config.EPOCHS), desc="Training"):
            epoch_start = time.time()
            self.model.train()
            train_loss = 0
            sum_task, sum_kd, sum_ewc = 0., 0., 0.
            n_samples = 0
            
            # Hidden state management - 在同一个cell的chunks之间传递
            current_cell = None
            hidden = None
            
            for x, y, cell_id in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # 检查是否需要重置hidden state（新的cell开始）
                if current_cell != cell_id[0]:  # cell_id是一个list/tuple
                    current_cell = cell_id[0]
                    hidden = self.model.init_hidden(x.size(0), self.device)
                
                optimizer.zero_grad()
                
                y_pred, hidden = self.model(x, hidden)
                task_loss = F.mse_loss(y_pred, y)
                
                # Knowledge distillation loss
                kd_loss = torch.zeros((), device=self.device)
                if alpha_lwf > 0 and self.old_model is not None:
                    with torch.no_grad():
                        # 为old model也初始化新的hidden
                        old_hidden = self.old_model.init_hidden(x.size(0), self.device)
                        y_old, _ = self.old_model(x, old_hidden)
                    kd_loss = F.mse_loss(y_pred, y_old)
                
                # EWC loss
                ewc_loss = torch.zeros((), device=self.device)
                if apply_ewc and self.ewc_tasks:
                    ewc_loss = sum(t.penalty(self.model) for t in self.ewc_tasks)
                
                loss = task_loss + alpha_lwf * kd_loss + ewc_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
                
                batch_samples = x.size(0) * x.size(1)
                sum_task += task_loss.item() * batch_samples
                sum_kd += kd_loss.item() * batch_samples
                sum_ewc += ewc_loss.item() * batch_samples
                train_loss += loss.item() * batch_samples
                n_samples += batch_samples
            
            # Average losses
            task_mean = sum_task / n_samples
            kd_mean = sum_kd / n_samples
            ewc_mean = sum_ewc / n_samples
            train_loss /= n_samples
            
            # Validation
            val_loss = self._validate_seq2seq(val_loader)
            scheduler.step(val_loss)
            
            epoch_time = time.time() - epoch_start
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['task_loss'].append(task_mean)
            history['kd_loss'].append(kd_mean)
            history['ewc_loss'].append(ewc_mean)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['time'].append(epoch_time)
            
            logger.info(
                "Epoch %03d | task %.4e | kd %.4e | ewc %.4e | val %.4e | lr %.2e | %.2fs",
                epoch + 1, task_mean, kd_mean, ewc_mean, val_loss,
                optimizer.param_groups[0]['lr'], epoch_time
            )
            
            # Checkpoints
            state = {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'ewc_tasks': [
                    {
                        'params': {n: p.clone().cpu() for n, p in e.params.items()},
                        'fisher': {n: f.clone().cpu() for n, f in e.fisher.items()},
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
                best_path = self.checkpoint_dir / f"task{task_id}_best.pt"
                torch.save(state, best_path)
            else:
                no_imp += 1
                if no_imp >= self.config.PATIENCE:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break
        return history

    def _validate_seq2seq(self, val_loader):
        """Seq-to-seq validation with hidden state management"""
        self.model.eval()
        val_loss = 0
        n_samples = 0
        current_cell = None
        hidden = None
        
        with torch.no_grad():
            for x, y, cell_id in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # 检查是否需要重置hidden state
                if current_cell != cell_id[0]:
                    current_cell = cell_id[0]
                    hidden = self.model.init_hidden(x.size(0), self.device)
                
                y_pred, hidden = self.model(x, hidden)
                val_loss += F.mse_loss(y_pred, y).item() * x.size(0) * x.size(1)
                n_samples += x.size(0) * x.size(1)
        
        return val_loss / n_samples

    def consolidate(self, loader, task_id=None, lam=0.0):
        """EWC consolidation"""
        self.ewc_tasks.append(EWC(self.model, loader, self.device, lam))
        path = self.checkpoint_dir / f"task{task_id}_best.pt"
        if path.exists():
            state = torch.load(path, map_location=self.device, weights_only=False)
            state['ewc_tasks'] = [
                {
                    'params': {n: p.cpu() for n, p in e.params.items()},
                    'fisher': {n: f.cpu() for n, f in e.fisher.items()},
                    'lam': e.lam
                }
                for e in self.ewc_tasks
            ]
            torch.save(state, path)
        self.old_model = copy.deepcopy(self.model).to(self.device)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)

    def evaluate_checkpoint(self, ckpt_path, loader, cell_data_dict, seq_len, out_dir, tag="", print_r2=True):
        """Evaluate using seq-to-seq approach on entire sequences"""
        if ckpt_path and Path(ckpt_path).exists():
            state = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state["model_state"])
        self.model.to(self.device).eval()

        # Get predictions for entire sequences
        all_preds = []
        all_targets = []
        
        for cell_id in sorted(cell_data_dict.keys()):
            df = cell_data_dict[cell_id]
            if len(df) < seq_len:
                continue
                
            # Evaluate entire sequence using chunking
            preds, targets = self._evaluate_cell_seq2seq(df, seq_len)
            all_preds.extend(preds)
            all_targets.extend(targets)
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(all_targets, all_preds)),
            'MAE': mean_absolute_error(all_targets, all_preds),
            'R2': r2_score(all_targets, all_preds)
        }
        
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            # Plot predictions
            plt.figure(figsize=(10, 6))
            n_plot = min(len(all_targets), 10000)  # Plot first 10k points
            plt.plot(range(n_plot), all_targets[:n_plot], label='Actual', alpha=0.7)
            plt.plot(range(n_plot), all_preds[:n_plot], label='Predicted', alpha=0.7)
            plt.xlabel('Time Step')
            plt.ylabel('SOH')
            plt.legend()
            plt.title(f"{tag}\nRMSE: {metrics['RMSE']:.4e}, MAE: {metrics['MAE']:.4e}, R2: {metrics['R2']:.4f}")
            plt.grid(True)
            plt.savefig(out_dir / 'predictions_seq2seq.png')
            plt.close()
            
            # Scatter plot
            plt.figure(figsize=(6, 6))
            plt.scatter(all_targets, all_preds, alpha=0.5, s=1)
            lims = [min(all_targets.min(), all_preds.min()), max(all_targets.max(), all_preds.max())]
            plt.plot(lims, lims, 'r--')
            plt.xlabel('Actual SOH')
            plt.ylabel('Predicted SOH')
            plt.grid(True)
            plt.savefig(out_dir / 'scatter_seq2seq.png')
            plt.close()

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

    def _evaluate_cell_seq2seq(self, df, seq_len):
        """
        推理一条完整 cell 曲线（many‑to‑many），尾段不足 seq_len 时不再 pad，
        直接把变长序列喂给 LSTM。
        """
        feats   = df[['Voltage[V]', 'Current[A]', 'Temperature[°C]']].values
        targets = df['SOH_ZHU'].values

        preds   = []
        hidden  = None
        total   = len(feats)

        with torch.no_grad():
            for start in range(0, total, seq_len):
                end      = min(start + seq_len, total)        # 尾段可能 < seq_len
                x_chunk  = torch.tensor(feats[start:end],
                                        dtype=torch.float32,
                                        device=self.device
                                    ).unsqueeze(0)         # (1, L, F)

                # 初始化或沿用隐藏态；同一 cell 内持续传递
                if hidden is None:
                    hidden = self.model.init_hidden(1, self.device)

                out, hidden = self.model(x_chunk, hidden)
                hidden      = tuple(h.detach() for h in hidden)

                preds.append(out.squeeze(0).cpu())            # (L,)

        preds   = torch.cat(preds).numpy()
        targets = targets[:len(preds)]
        return preds, targets

# ===============================================================
# Helper functions
# ===============================================================
def create_dataloaders_from_cells(cell_data_dict, seq_len, batch_size):
    """Create DataLoaders from cell-organized data"""
    loaders = {}
    for key, cells_dict in cell_data_dict.items():
        if cells_dict and ('train' in key or 'val' in key or 'test' in key):
            ds = CellSeq2SeqDataset(cells_dict, seq_len)
            if len(ds) > 0:
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
                loaders[key] = loader
    return loaders

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_losses(history, out_dir):
    df = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    plt.semilogy(df['epoch'], df['train_loss'], label='train')
    plt.semilogy(df['epoch'], df['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'loss_curves.png')
    plt.close()

# ===============================================================
# Main Pipeline
# ===============================================================
def main(joint_training: bool = True):
    # Config and logging
    config = Config()
    base_dir = Path(__file__).parent / 'model' / 'Stateful-LSTM-Seq2Seq' / 'joint'
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
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
        reg_dir = base_dir / 'regular'
        reg_ckpt_dir = reg_dir / 'checkpoints'
        reg_results = reg_dir / 'results'
        reg_ckpt_dir.mkdir(parents=True, exist_ok=True)
        reg_results.mkdir(parents=True, exist_ok=True)
        
        logger.info("==== Regular Seq2Seq LSTM Training Phase ====")
        lstm_train_ids = ['03', '05', '07', '09', '11', '15', '21', '23', '25', '27', '29']
        lstm_val_ids = ['01', '19', '13']
        
        dp_lstm = DataProcessor(
            data_dir='../01_Datenaufbereitung/Output/Calculated/',
            resample=config.RESAMPLE,
            config=config,
            base_train_ids=lstm_train_ids,
            base_val_ids=lstm_val_ids,
            update1_train_ids=[], update1_val_ids=[],
            update2_train_ids=[], update2_val_ids=[]
        )
        
        data_lstm = dp_lstm.prepare_data()
        loaders_lstm = create_dataloaders_from_cells(data_lstm, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
        
        model_lstm = SOHLSTMSeq2Seq(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
        trainer_lstm = Trainer(model_lstm, device, config, reg_ckpt_dir)
        
        history_lstm = trainer_lstm.train_task(
            train_loader=loaders_lstm['base_train'],
            val_loader=loaders_lstm['base_val'],
            task_id=0,
            apply_ewc=False,
            resume=True
        )
        
        # Save losses & predictions
        plot_losses(history_lstm, reg_results)
        best_ckpt = reg_ckpt_dir / "task0_best.pt"
        if best_ckpt.exists():
            trainer_lstm.evaluate_checkpoint(
                ckpt_path=best_ckpt,
                loader=loaders_lstm['test_full'],
                cell_data_dict=data_lstm['test_full'],
                seq_len=config.SEQUENCE_LENGTH,
                out_dir=reg_results,
                tag="Joint training best model predictions"
            )
    
    else:
        logger.info("==== Skipping joint LSTM Training Phase ====")
    
    # ------------------- Incremental EWC Training -------------------
    logger.info("==== Incremental EWC Training Phase ====")
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
    loaders = create_dataloaders_from_cells(data_inc, config.SEQUENCE_LENGTH, config.BATCH_SIZE)

    model = SOHLSTMSeq2Seq(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    trainer = Trainer(model, device, config, checkpoint_dir=str(inc_dir))

    tasks = [
        ('task0', 'base_train', 'base_val', 'test_base', False, config.EWC_LAMBDA0, config.LWF_ALPHA0),
        ('task1', 'update1_train', 'update1_val', 'test_update1', True, config.EWC_LAMBDA1, config.LWF_ALPHA1),
        ('task2', 'update2_train', 'update2_val', 'test_update2', True, config.EWC_LAMBDA2, config.LWF_ALPHA2)
    ]

    baseline_metrics = {}
    best_mae = {}
    curr_mae = {}
    tilde_mae = {}
    metric_hist = []
    delta_hist = []

    for i, (name, train_key, val_key, test_key, use_ewc, lam, alpha) in enumerate(tasks):
        tr_loader = loaders.get(train_key)
        val_loader = loaders.get(val_key)
        test_loader = loaders.get(test_key)
        full_loader = loaders.get('test_full')
        
        test_cells = data_inc.get(test_key, {})
        full_cells = data_inc.get('test_full', {})
        
        # Load previous checkpoint
        if i > 0:
            prev_name, *_ = tasks[i - 1]
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

        ckpt_dir = inc_dir / name / 'checkpoints'
        results_dir = inc_dir / name / 'results'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        trainer.checkpoint_dir = ckpt_dir

        best_ckpt = ckpt_dir / f"{name}_best.pt"
        last_ckpt = ckpt_dir / f"{name}_last.pt"
        trained_f = ckpt_dir / f"{name}.trained"
        consol_f = ckpt_dir / f"{name}.consolidated"
        
        # Pre-FWT baseline
        if i > 0 and test_loader and test_cells:
            metrics_pre = trainer.evaluate_checkpoint(
                ckpt_path=None,
                loader=test_loader,
                cell_data_dict=test_cells,
                seq_len=config.SEQUENCE_LENGTH,
                out_dir=None,
                tag=f"{name} Pre-FWT baseline",
                print_r2=False
            )
            tilde_mae[name] = metrics_pre['MAE']
        
        # Training phase
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
                resume=last_ckpt.exists()
            )
            
            pd.DataFrame(history).to_csv(ckpt_dir / f"{name}_history.csv", index=False)
            plot_losses(history, results_dir / 'losses')
            trained_f.write_text(datetime.now().isoformat())
            logger.info("[%s] Training completed.", name)
        
        # Consolidation phase
        if tr_loader and not consol_f.exists():
            logger.info("[%s] Consolidating EWC...", name)
            lam = lam if lam is not None else 0.0
            trainer.consolidate(tr_loader, task_id=i, lam=lam)
            consol_f.write_text(datetime.now().isoformat())
            logger.info("[%s] Consolidation done.", name)
        
        # Baseline testing on own task
        if best_ckpt.exists() and test_loader and test_cells:
            logger.info("[%s] Baseline evaluation on own task %s ...", name, name)
            metrics_own = trainer.evaluate_checkpoint(
                ckpt_path=best_ckpt,
                loader=test_loader,
                cell_data_dict=test_cells,
                seq_len=config.SEQUENCE_LENGTH,
                out_dir=results_dir / 'baseline' / name / 'test',
                tag=f"{name} Baseline on {name}",
                print_r2=False
            )
            baseline_metrics[name] = metrics_own
            best_mae.setdefault(name, metrics_own['MAE'])
            curr_mae[name] = metrics_own['MAE']
        
        # Backward testing on previous tasks
        if i > 0:
            for j in range(i):
                prev_name, _, _, prev_test_key, _, _, _ = tasks[j]
                prev_loader = loaders.get(prev_test_key)
                prev_cells = data_inc.get(prev_test_key, {})
                if best_ckpt.exists() and prev_loader and prev_cells:
                    logger.info("[%s] Backward testing on previous task %s...", name, prev_name)
                    metrics_prev = trainer.evaluate_checkpoint(
                        ckpt_path=best_ckpt,
                        loader=prev_loader,
                        cell_data_dict=prev_cells,
                        seq_len=config.SEQUENCE_LENGTH,
                        out_dir=inc_dir / name / 'results' / 'backward' / prev_name,
                        tag=f"{name} BACKWARD on {prev_name}",
                        print_r2=False
                    )
                    curr_mae[prev_name] = metrics_prev['MAE']
        
        # ACC / BWT / FWT
        old_tasks = [t for t in best_mae if t != name]
        
        for t in old_tasks:
            delta = curr_mae[t] - best_mae[t]
            delta_hist.append({'stage': name, 'task': t, 'ΔMAE': delta})
            logger.info("[%s] ΔMAE on %s: %+.4e", name, t, delta)
        
        # ACC (Average Accuracy)
        ACC = -np.mean(list(curr_mae.values()))
        logger.info("[%s] ACC (-MAE): %.4e", name, ACC)
        
        # Backward-transfer
        if old_tasks:
            BWT = np.mean([curr_mae[t] - best_mae[t] for t in old_tasks])
            logger.info("[%s] BWT: %+.4e", name, BWT)
        else:
            BWT = np.nan
        
        # Forward-transfer
        if name in tilde_mae:
            FWT = tilde_mae[name] - curr_mae[name]
            logger.info("[%s] FWT: %+.4e", name, FWT)
        else:
            FWT = np.nan
        
        metric_hist.append({'task': name, 'ACC': ACC, 'BWT': BWT, 'FWT': FWT})
        
        # Evaluate on full test set
        if best_ckpt.exists() and full_cells:
            logger.info("[%s] Evaluating BEST checkpoint on full test set...", name)
            trainer.evaluate_checkpoint(
                ckpt_path=best_ckpt,
                loader=full_loader,
                cell_data_dict=full_cells,
                seq_len=config.SEQUENCE_LENGTH,
                out_dir=results_dir / "forward" / "test_full",
                tag=f"{name} Evaluation on full test set"
            )
    
    # Save final metrics
    df_m = pd.DataFrame(metric_hist)
    df_m.to_csv(inc_dir / "continual_metrics.csv", index=False)
    logger.info("Saved ACC/BWT/FWT history to %s", inc_dir / "continual_metrics.csv")
    
    # Plot transfer curves
    plt.figure(figsize=(6, 4))
    plt.plot(df_m['task'], df_m['BWT'], marker='o', label='BWT')
    plt.plot(df_m['task'], df_m['FWT'], marker='s', label='FWT')
    plt.ylabel('MAE difference')
    plt.grid(True)
    plt.legend()
    plt.savefig(inc_dir / "transfer_curves.png")
    plt.close()
    
    # Save delta MAE history
    if delta_hist:
        pd.DataFrame(delta_hist).to_csv(inc_dir / "delta_MAE_history.csv", index=False)
    
    logger.info("==== All tasks completed ====")

if __name__ == '__main__':
    main()