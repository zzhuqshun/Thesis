import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# ===============================================================
# Configuration Class
# ===============================================================
class Config:
    def __init__(self, **kwargs):
        self.SEQUENCE_LENGTH = 864
        self.HIDDEN_SIZE = 256
        self.NUM_LAYERS = 2
        self.DROPOUT = 0.4
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-4
        self.EPOCHS = 100
        self.PATIENCE = 10
        self.WEIGHT_DECAY = 1e-6
        self.SEED = 42
        self.RESAMPLE = '10min'
        self.EWC_LAMBDA = 1000  # regularization strength
        for key, value in kwargs.items(): setattr(self, key, value)
    def save(self, path):
        with open(path, 'w') as f: json.dump(self.__dict__, f, indent=4)
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f: data = json.load(f)
        return cls(**data)

# ===============================================================
# Main
# ===============================================================
def main():
    save_dir = Path(__file__).parent/'models'/ "EWC" / "MinMax_6days"
    save_dir.mkdir(parents=True,exist_ok=True)
    
    ckpt_dir = save_dir/'checkpoints'
    
    # logging
    fh=logging.FileHandler(save_dir/'train.log')
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    # config, seed, device
    config=Config()
    config.save(save_dir/'config.json')
    set_seed(config.SEED)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # data
    dp=DataProcessor(Path('../01_Datenaufbereitung/Output/Calculated/'), resample=config.RESAMPLE, seed=config.SEED)
    data=dp.prepare_data()
    loaders=create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    # model & trainer
    model=SOHLSTM(3,config.HIDDEN_SIZE,config.NUM_LAYERS,config.DROPOUT).to(device)
    trainer = Trainer(model, device, config, checkpoint_dir=ckpt_dir)
    
    # phase1
    h1 = trainer.train_task(loaders['base_train'], loaders['base_val'], task_id=0, apply_ewc=False, resume=True)
    trainer.consolidate(loaders['base_train'])
    metrics1=trainer.evaluate(loaders['test_base'])
    logger.info(f"Task0 metrics: {metrics1}")
    
    # phase2
    h2=trainer.train_task(loaders['update1_train'], loaders['update1_val'], task_id=1, apply_ewc=True, resume=True)
    trainer.consolidate(loaders['update1_train'])
    metrics2=trainer.evaluate(loaders['test_update1'])
    logger.info(f"Task1 metrics: {metrics2}")
    
    # phase3
    h3=trainer.train_task(loaders['update2_train'],loaders['update2_val'], task_id=2, apply_ewc=True, resume=True)
    metrics3=trainer.evaluate(loaders['test_update2'])
    logger.info(f"Task2 metrics: {metrics3}")
    logger.info("EWC incremental learning complete!")


# ===============================================================
# Dataset
# ===============================================================
class BatteryDataset(Dataset):
    def __init__(self, df, sequence_length):
        self.seq_len = sequence_length
        feats = df[['Voltage[V]','Current[A]','Temperature[°C]']].values
        targets = df['SOH_ZHU'].values
        self.features = torch.tensor(feats, dtype=torch.float32)
        self.labels = torch.tensor(targets, dtype=torch.float32)
    def __len__(self): return len(self.features) - self.seq_len
    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]
        y = self.labels[idx+self.seq_len]
        return x, y


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

def create_dataloaders(datasets, sequence_length, batch_size):
    """Create data loaders"""
    loaders = {}
    
    # Create training and validation loaders
    for key, dataset in datasets.items():
        if not dataset.empty:
            if 'train' in key or 'val' in key or 'test' in key:
                loaders[key] = DataLoader(
                    BatteryDataset(dataset, sequence_length),
                    batch_size=batch_size,
                    shuffle='train' in key
                )
    return loaders

# ===============================================================
# Data Loading and Preparation
# ===============================================================

class DataProcessor:
    """Data processing class responsible for loading and preparing data"""
    def __init__(self, data_dir, resample='10min', test_cell_id='17', seed=42):
        self.data_dir = Path(data_dir)
        self.resample = resample
        self.test_cell_id = test_cell_id
        self.seed = seed
        self.scaler_vt = MinMaxScaler(feature_range=(0, 1))
        self.scaler_i  = MinMaxScaler(feature_range=(-1, 1))
        random.seed(seed)
        
    def load_cell_data(self):
        """Load battery data, categorize cells, then separate test cell"""
        # Gather all parquet files ordered by cell index
        parquet_files = sorted(
            [f for f in self.data_dir.glob('*.parquet') if f.is_file()],
            key=lambda x: int(x.stem.split('_')[-1])
        )
        cell_info = []

        # First pass: categorize every cell
        for fp in parquet_files:
            cell_id = fp.stem.split('_')[1]
            df = pd.read_parquet(fp)
            initial_soh = df['SOH_ZHU'].iloc[0]
            final_soh = df['SOH_ZHU'].iloc[-1]
            if final_soh > 0.8:
                category = 'normal'
            elif 0.65 < final_soh <= 0.8:
                category = 'fast'
            else:
                category = 'faster'
            cell_info.append({
                'file': fp,
                'cell_id': cell_id,
                'initial_soh': initial_soh,
                'final_soh': final_soh,
                'category': category
            })

        # Separate out test cell
        test_file = None
        remaining_info = []
        for info in cell_info:
            if info['cell_id'] == self.test_cell_id:
                test_file = info['file']
            else:
                remaining_info.append(info)

        if test_file is None:
            raise ValueError(f"Test cell ID {self.test_cell_id} not found in data directory")

        return remaining_info, test_file
    
    def assign_cells_to_phases(self, cell_info):
        """Assign batteries to different training phases"""
        # Group by category
        normal_cells = [c for c in cell_info if c['category'] == 'normal']
        fast_cells = [c for c in cell_info if c['category'] == 'fast']
        faster_cells = [c for c in cell_info if c['category'] == 'faster']
        
        # Check if there are enough normal cells
        if len(normal_cells) < 7:
            logger.warning("Warning: Not enough normal cells (%d), at least 7 needed.", len(normal_cells))
            # If not enough, supplement from the next category
            if len(normal_cells) + len(fast_cells) >= 7:
                fast_cells_sorted = sorted(fast_cells, key=lambda x: x['final_soh'], reverse=True)
                normal_cells.extend(fast_cells_sorted[:7-len(normal_cells)])
                # Remove cells transferred to normal
                fast_cells = fast_cells_sorted[7-len(normal_cells):]
            else:
                raise ValueError("Error: Not enough cells for base training.")
                
        # Randomly select 6 normal cells
        selected_normal = random.sample(normal_cells, min(6, len(normal_cells)))
        
        # Base training (5) and validation (1)
        base_train_cells = selected_normal[:5]
        base_val_cells = selected_normal[5:6]
        
        # Remaining normal cells 2
        remaining_normal = [c for c in normal_cells if c not in selected_normal]
        
        # Update1: base_val(1) + 2 normal + 2 fast as training set, 1 fast as validation set
        update1_train_normal = remaining_normal
        
        if len(fast_cells) < 3:
            logger.warning("Warning: Not enough fast cells (%d), at least 3 needed.", len(fast_cells))
        
        update1_train_fast = fast_cells[:2] if len(fast_cells) >= 2 else fast_cells
        update1_val_fast = fast_cells[2:3] if len(fast_cells) >= 3 else []
        
        # Combine update1 cells 1 + 2 normal + 2 fast
        update1_train_cells = base_val_cells + update1_train_normal + update1_train_fast
        update1_val_cells = update1_val_fast
        
        # Update2: update1_val + 2 faster as training set, remaining faster as validation set
        update2_train_faster = faster_cells[:2] if len(faster_cells) >= 2 else faster_cells
        update2_val_faster = faster_cells[2:] if len(faster_cells) >= 3 else []
        
        # Combine update2 cells
        update2_train_cells = update1_val_cells + update2_train_faster
        update2_val_cells = update2_val_faster
        
        # Print assignment summary
        logger.info("Cell Assignment Summary:")
        logger.info("Normal cells (%d): %s", len(normal_cells), [c['cell_id'] for c in normal_cells])
        logger.info("Fast cells (%d): %s", len(fast_cells), [c['cell_id'] for c in fast_cells])
        logger.info("Faster cells (%d): %s", len(faster_cells), [c['cell_id'] for c in faster_cells])
        logger.info("\nTraining Sets:")
        logger.info("Base training set: %s", [c['cell_id'] for c in base_train_cells])
        logger.info("Base validation set: %s", [c['cell_id'] for c in base_val_cells])
        logger.info("Update1 training set: %s", [c['cell_id'] for c in update1_train_cells])
        logger.info("Update1 validation set: %s", [c['cell_id'] for c in update1_val_cells])
        logger.info("Update2 training set: %s", [c['cell_id'] for c in update2_train_cells])
        logger.info("Update2 validation set: %s", [c['cell_id'] for c in update2_val_cells])
        
        return {
            'base_train': base_train_cells,
            'base_val': base_val_cells,
            'update1_train': update1_train_cells,
            'update1_val': update1_val_cells,
            'update2_train': update2_train_cells,
            'update2_val': update2_val_cells
        }
    
    def process_file(self, file_path):
        """Process a single parquet file"""
        df = pd.read_parquet(file_path)
        columns_to_keep = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 
                           'Temperature[°C]', 'SOH_ZHU']
        df_processed = df[columns_to_keep].copy()
        df_processed.dropna(inplace=True)
        
        df_processed['Testtime[s]'] = df_processed['Testtime[s]'].round().astype(int)
        start_date = pd.Timestamp("2023-02-02")
        df_processed['Datetime'] = pd.date_range(
            start=start_date,
            periods=len(df_processed),
            freq='s'
        )
        
        df_sampled = df_processed.resample(self.resample, on='Datetime').mean().reset_index(drop=False)
        df_sampled["cell_id"] = file_path.stem.split('_')[1]
        return df_sampled
    
    def prepare_data(self):
        """Prepare datasets for battery incremental learning"""
        # Load battery data
        cell_info, test_file = self.load_cell_data()
        
        # Assign cells to different phases
        phase_cells = self.assign_cells_to_phases(cell_info)
        
        # Process training and validation files
        df_base_train = pd.concat([self.process_file(c['file']) for c in phase_cells['base_train']], ignore_index=True)
        df_base_val = pd.concat([self.process_file(c['file']) for c in phase_cells['base_val']], ignore_index=True)
        
        df_update1_train = pd.concat([self.process_file(c['file']) for c in phase_cells['update1_train']], ignore_index=True)
        df_update1_val = pd.concat([self.process_file(c['file']) for c in phase_cells['update1_val']], ignore_index=True) if phase_cells['update1_val'] else pd.DataFrame()
        
        df_update2_train = pd.concat([self.process_file(c['file']) for c in phase_cells['update2_train']], ignore_index=True) if phase_cells['update2_train'] else pd.DataFrame()
        df_update2_val = pd.concat([self.process_file(c['file']) for c in phase_cells['update2_val']], ignore_index=True) if phase_cells['update2_val'] else pd.DataFrame()
        
        # Process test file
        df_test_full = self.process_file(test_file)
        
        # Split test set based on SOH values
        df_test_base = df_test_full[df_test_full['SOH_ZHU'] >= 0.9].reset_index(drop=True)
        df_test_update1 = df_test_full[(df_test_full['SOH_ZHU'] < 0.9) & (df_test_full['SOH_ZHU'] >= 0.8)].reset_index(drop=True)
        df_test_update2 = df_test_full[df_test_full['SOH_ZHU'] < 0.8].reset_index(drop=True)
        
        # Standardize data using StandardScaler
        self.scaler_vt.fit(df_base_train[['Voltage[V]', 'Temperature[°C]']])
        self.scaler_i .fit(df_base_train[['Current[A]']])
        
        # label_cols = ['SOH_ZHU']
        # self.target_scaler.fit(df_base_train[label_cols])
        
        # Apply standardization
        datasets = {
            'base_train': self.apply_scaling(df_base_train),
            'base_val': self.apply_scaling(df_base_val),
            'update1_train': self.apply_scaling(df_update1_train),
            'update1_val': self.apply_scaling(df_update1_val),
            'update2_train': self.apply_scaling(df_update2_train),
            'update2_val': self.apply_scaling(df_update2_val),
            'test_base': self.apply_scaling(df_test_base),
            'test_update1': self.apply_scaling(df_test_update1),
            'test_update2': self.apply_scaling(df_test_update2)
        }
        
        # Print dataset sizes
        logger.info("\nDataset Sizes:")
        logger.info("Base training set: %d samples, validation set: %d samples", len(datasets['base_train']), len(datasets['base_val']))
        logger.info("Update1 training set: %d samples, validation set: %d samples", len(datasets['update1_train']), len(datasets['update1_val']))
        logger.info("Update2 training set: %d samples, validation set: %d samples", len(datasets['update2_train']), len(datasets['update2_val']))

        logger.info("\nTest Sets:")
        logger.info("Test cell: %s", test_file.stem.split('_')[1])
        logger.info("Base test set (SOH >= 0.9): %d samples", len(df_test_base))
        logger.info("Update1 test set (0.8 <= SOH < 0.9): %d samples", len(df_test_update1))
        logger.info("Update2 test set (SOH < 0.8): %d samples", len(df_test_update2))
        
        return datasets
        
    def apply_scaling(self, df):
        """Apply MinMax scaling: Volt/Temp->[0,1], Current->[-1,1]"""
        if df.empty:
            return df
        df_scaled = df.copy()
        # Voltage & Temperature -> [0,1]
        df_scaled[['Voltage[V]', 'Temperature[°C]']] = self.scaler_vt.transform(df[['Voltage[V]', 'Temperature[°C]']])
        # Current -> [-1,1]
        df_scaled[['Current[A]']] = self.scaler_i.transform(df[['Current[A]']])
        return df_scaled

# ===============================================================
# LSTM Model
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
        return self.fc(h)

# ===============================================================
# EWC Implementation
# ===============================================================
class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.params = {n: p.clone().detach() for n,p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher()

    def _compute_fisher(self):
        fisher = {n: torch.zeros_like(p) for n,p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for x,y in self.dataloader:
            self.model.zero_grad()
            x,y = x.to(self.device), y.to(self.device)
            out = self.model(x).squeeze()
            loss = F.mse_loss(out, y)
            loss.backward()
            for n,p in self.model.named_parameters():
                if p.requires_grad:
                    fisher[n] += p.grad.data.clone().pow(2)
        # average
        for n in fisher: fisher[n] /= len(self.dataloader)
        return fisher

    def penalty(self, model):
        loss=0
        for n,p in model.named_parameters():
            if p.requires_grad:
                _loss = self.fisher[n] * (p - self.params[n]).pow(2)
                loss += _loss.sum()
        return loss

# ===============================================================
# Trainer with EWC
# ===============================================================
class Trainer:
    def __init__(self, model, device, config, checkpoint_dir):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.ewc_tasks = []  # list of EWC objects
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_task(self, train_loader, val_loader, task_id, apply_ewc=True, resume=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE,
                                     weight_decay=self.config.WEIGHT_DECAY)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        ckpt_last = self.checkpoint_dir / f"task{task_id}_last.pt"
        start_epoch = 0
        if resume and ckpt_last.is_file():
            ckpt = torch.load(ckpt_last, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            self.ewc_tasks = []
            for data in ckpt['ewc_tasks']:
                ewc = EWC.__new__(EWC)
                ewc.model  = self.model
                ewc.device = self.device
                ewc.params = {n: p.to(self.device) for n,p in data['params'].items()}
                ewc.fisher = {n: f.to(self.device) for n,f in data['fisher'].items()}
                self.ewc_tasks.append(ewc)
            start_epoch = ckpt['epoch'] + 1
            logger.info(f"[Task {task_id}] Resuming from epoch {start_epoch}/{self.config.EPOCHS}")

            
        best_val=float('inf')
        no_imp=0
        history={'epoch':[],'train_loss':[],'val_loss':[]}
        
        for epoch in range(start_epoch, self.config.EPOCHS):
            logger.info(f"[Task {task_id}] Starting epoch {epoch+1}/{self.config.EPOCHS}")
            self.model.train()
            train_loss=0
            for x,y in train_loader:
                x,y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x).squeeze()
                loss = F.mse_loss(out, y)
                if apply_ewc and self.ewc_tasks:
                    ewc_loss = sum([t.penalty(self.model) for t in self.ewc_tasks])
                    loss += (self.config.EWC_LAMBDA/2)*ewc_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),1)
                optimizer.step()
                train_loss += loss.item()*x.size(0)
            train_loss /= len(train_loader.dataset)
            # val
            self.model.eval(); val_loss=0
            with torch.no_grad():
                for x,y in val_loader:
                    x,y = x.to(self.device), y.to(self.device)
                    out = self.model(x).squeeze()
                    loss = F.mse_loss(out,y)
                    val_loss += loss.item()*x.size(0)
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            history['epoch'].append(epoch+1)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            logger.info(f"Task {task_id} Epoch {epoch+1}: train={train_loss:.4e}, val={val_loss:.4e}")
            
            # save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state':     self.model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'ewc_tasks': [
                    {
                        'params': {n: p.clone().cpu()   for n,p in ewc.params .items()},
                        'fisher': {n: f.clone().cpu()   for n,f in ewc.fisher.items()}
                    }
                    for ewc in self.ewc_tasks
                ]
            }, ckpt_last)
            
            if val_loss < best_val:
                best_val=val_loss
                no_imp=0
                best_state={k:v.clone() for k,v in self.model.state_dict().items()}
                torch.save({
                    'epoch': epoch,
                    'model_state': best_state,
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'ewc_tasks': [
                        {
                            'params': {n: p.clone().cpu()   for n,p in ewc.params .items()},
                            'fisher': {n: f.clone().cpu()   for n,f in ewc.fisher.items()}
                        }
                        for ewc in self.ewc_tasks
                    ]
                }, self.checkpoint_dir/f"task{task_id}_best.pt")
            else:
                no_imp+=1
                if no_imp>=self.config.PATIENCE: break

        return history

    def consolidate(self, loader):
        # compute fisher on loader and store
        self.ewc_tasks.append(EWC(self.model, loader, self.device))

    def evaluate(self, loader):
        self.model.eval(); preds=[]; targs=[]
        with torch.no_grad():
            for x,y in loader:
                out=self.model(x.to(self.device)).cpu().numpy().ravel()
                preds.append(out); targs.append(y.numpy().ravel())
        preds=np.concatenate(preds); targs=np.concatenate(targs)
        return {'RMSE':np.sqrt(mean_squared_error(targs,preds)),
                'MAE':mean_absolute_error(targs,preds),
                'R2':r2_score(targs,preds)}

if __name__=='__main__': 
    main()