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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
        self.EPOCHS = 200
        self.PATIENCE = 20
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
    save_dir = Path(__file__).parent/'models'/ "EWC" / "seq6days"
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
    # ===============================================================
    # regular LSTM training 
    # ===============================================================
    dp_lstm=DataProcessor(Path('../01_Datenaufbereitung/Output/Calculated/'), resample=config.RESAMPLE, seed=config.SEED, 
                        base_train_ids=['03','05','07','09','11','15','21','23','25','27','29'], 
                        base_val_ids=['01','13','19'],
                        update1_train_ids=[],
                        update1_val_ids=[],
                        update2_train_ids=[],
                        update2_val_ids=[]
                     )
    data_lstm=dp_lstm.prepare_data()
    loaders_lstm=create_dataloaders(data_lstm, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    # model & trainer
    model_lstm=SOHLSTM(3,config.HIDDEN_SIZE,config.NUM_LAYERS,config.DROPOUT).to(device)
    trainer_lstm = Trainer(model_lstm, device, config, checkpoint_dir=ckpt_dir)
    trainer_lstm.train_task(
        train_loader=loaders_lstm['base_train'],
        val_loader=loaders_lstm['base_val'],
        task_id=0,
        apply_ewc=False
    )
    
    
    # incremental learning data
    dp=DataProcessor(Path('../01_Datenaufbereitung/Output/Calculated/'), resample=config.RESAMPLE, seed=config.SEED)
    data=dp.prepare_data()
    loaders=create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    # model & trainer
    model=SOHLSTM(3,config.HIDDEN_SIZE,config.NUM_LAYERS,config.DROPOUT).to(device)
    trainer = Trainer(model, device, config, checkpoint_dir=ckpt_dir)
    
    # EWC training
    phases = [
        ('base',    loaders['base_train'],   loaders['base_val'],    loaders['test_base'],    False),
        ('update1', loaders['update1_train'],loaders['update1_val'], loaders['test_update1'], True),
        ('update2', loaders['update2_train'],loaders['update2_val'], loaders['test_update2'], True),
    ]

    for i, (phase_name, train_loader, val_loader, test_loader, use_ewc) in enumerate(phases):
        trained_flag = ckpt_dir / f"task{i}.trained"
        consolidated_flag = ckpt_dir / f"task{i}.consolidated"
        last_checkpoint = ckpt_dir / f"task{i}_last.pt"
        best_checkpoint = ckpt_dir / f"task{i}_best.pt"

        # Step 1: Training
        if not trained_flag.exists():
            logger.info(f"[Task {i}:{phase_name}] Starting training")
            trainer.train_task(
                train_loader=train_loader,
                val_loader=val_loader,
                task_id=i,
                apply_ewc=use_ewc,
                resume=last_checkpoint.is_file()
            )
            # Mark training complete
            trained_flag.write_text(f"trained at {datetime.now().isoformat()}")
        else:
            logger.info(f"[Task {i}:{phase_name}] Already trained, skipping training step")

        # Step 2: Consolidate EWC
        if not consolidated_flag.exists():
            try:
                logger.info(f"[Task {i}:{phase_name}] Starting EWC consolidation")
                trainer.consolidate(train_loader)
                consolidated_flag.write_text(f"consolidated at {datetime.now().isoformat()}")
            except Exception as e:
                logger.error(f"[Task {i}:{phase_name}] EWC consolidation failed: {e}")
                # Consolidation failed; next run will retry
                continue
        else:
            logger.info(f"[Task {i}:{phase_name}] EWC already consolidated, skipping consolidation")

        # Step 3: Evaluation
        # Load best checkpoint if available
        if best_checkpoint.is_file():
            checkpoint = torch.load(best_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state'])

        metrics = trainer.evaluate(test_loader)
        logger.info(f"[Task {i}:{phase_name}] Evaluation metrics: {metrics}")


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
def save_checkpoint(path, model, optimizer, ewc_tasks, epoch):
    """
    Save a checkpoint of the model and optimizer state
    """
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'ewc_fisher': [task.fisher for task in ewc_tasks],
        'ewc_params': [ {n: p.clone() for n,p in task.params.items()} for task in ewc_tasks ],
    }, path)
    print(f"[Checkpoint] saved to {path} (epoch {epoch})")


def load_checkpoint(path, model, optimizer, ewc_tasks, device):
    """
    Load a checkpoint and resume training
    """
    checkpoint = torch.load(path, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])

    saved_fishers = checkpoint.get('ewc_fisher', [])
    saved_params  = checkpoint.get('ewc_params', [])
    for task, fisher, params in zip(ewc_tasks, saved_fishers, saved_params):
        task.fisher = fisher
        task.params = params
    print(f"[Checkpoint] loaded from {path} (resuming at epoch {start_epoch})")
    return start_epoch


class DataProcessor:
    """Load and prepare battery data with fully manual phase specifications."""
    def __init__(self, data_dir, resample='10min', test_cell_id='17', seed=42,
                 base_train_ids=['01','03','05','21','27'], base_val_ids=['23'],
                 update1_train_ids=['07','09','11','19','23'], update1_val_ids=['25'],
                 update2_train_ids=['15','25','29'], update2_val_ids=['13']):
        self.data_dir = Path(data_dir)
        self.resample = resample
        self.test_cell_id = test_cell_id
        self.seed = seed
        self.scaler = StandardScaler()
        random.seed(seed)
        # Require manual phase lists
        self.base_train_ids    = base_train_ids or []
        self.base_val_ids      = base_val_ids or []
        self.update1_train_ids = update1_train_ids or []
        self.update1_val_ids   = update1_val_ids or []
        self.update2_train_ids = update2_train_ids or []
        self.update2_val_ids   = update2_val_ids or []

    def load_cell_data(self):
        """Load all cells and separate test cell."""
        files = sorted(self.data_dir.glob('*.parquet'),
                       key=lambda x: int(x.stem.split('_')[-1]))
        info = {fp.stem.split('_')[1]: fp for fp in files}
        if self.test_cell_id not in info:
            raise ValueError(f"Test cell {self.test_cell_id} not found in {self.data_dir}")
        test_fp = info.pop(self.test_cell_id)
        return info, test_fp

    def process_file(self, fp):
        df = pd.read_parquet(fp)[[
            'Testtime[s]','Voltage[V]','Current[A]','Temperature[°C]','SOH_ZHU']]
        df = df.dropna().reset_index(drop=True)
        df['Testtime[s]'] = df['Testtime[s]'].round().astype(int)
        df['Datetime']     = pd.date_range('2023-02-02', periods=len(df), freq='s')
        sampled = df.resample(self.resample, on='Datetime').mean().reset_index()
        sampled['cell_id'] = fp.stem.split('_')[1]
        return sampled

    def prepare_data(self):
        """Prepare and scale datasets for manual phases and test splits."""
        info_map, test_fp = self.load_cell_data()
        # Log selected IDs
        logger.info(f"Base train IDs: {self.base_train_ids}")
        logger.info(f"Base val IDs:   {self.base_val_ids}")
        logger.info(f"Update1 train IDs: {self.update1_train_ids}")
        logger.info(f"Update1 val IDs:   {self.update1_val_ids}")
        logger.info(f"Update2 train IDs: {self.update2_train_ids}")
        logger.info(f"Update2 val IDs:   {self.update2_val_ids}")
        
        def build(ids):
            missing = [cid for cid in ids if cid not in info_map]
            if missing:
                raise ValueError(f"Cell IDs not found: {missing}")
            dfs = [self.process_file(info_map[cid]) for cid in ids]
            return pd.concat(dfs, ignore_index=True)
        # Build each phase
        df_base_train    = build(self.base_train_ids)
        df_base_val      = build(self.base_val_ids)
        df_update1_train = build(self.update1_train_ids)
        df_update1_val   = build(self.update1_val_ids)
        df_update2_train = build(self.update2_train_ids)
        df_update2_val   = build(self.update2_val_ids)
        
        # Test splits by SOH
        df_test_full     = self.process_file(test_fp)
        df_test_base     = df_test_full[df_test_full['SOH_ZHU']>=0.9].reset_index(drop=True)
        df_test_update1  = df_test_full[(df_test_full['SOH_ZHU']<0.9)&(df_test_full['SOH_ZHU']>=0.8)].reset_index(drop=True)
        df_test_update2  = df_test_full[df_test_full['SOH_ZHU']<0.8].reset_index(drop=True)
        logger.info(
            f"test_full: {len(df_test_full)} samples, "
            f"test_base: {len(df_test_base)} samples, SOH>=0.9, "
            f"test_update1: {len(df_test_update1)}, SOH<0.9,>=0.8, "
            f"test_update2: {len(df_test_update2)}, SOH<0.8"
        )

        
        # Fit scaler on base_train
        self.scaler.fit(df_base_train[['Voltage[V]','Current[A]','Temperature[°C]']])
        def scale(df):
            df2 = df.copy()
            df2[['Voltage[V]','Current[A]','Temperature[°C]']] = \
                self.scaler.transform(df2[['Voltage[V]','Current[A]','Temperature[°C]']])
            return df2
        # Return all
        
        return {
            'base_train':    scale(df_base_train),
            'base_val':      scale(df_base_val),
            'update1_train': scale(df_update1_train),
            'update1_val':   scale(df_update1_val),
            'update2_train': scale(df_update2_train),
            'update2_val':   scale(df_update2_val),
            'test_full':     scale(df_test_full),
            'test_base':     scale(df_test_base),
            'test_update1':  scale(df_test_update1),
            'test_update2':  scale(df_test_update2)
        }


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
        was_training = self.model.training
        self.model.train()
        fisher = {n: torch.zeros_like(p) for n,p in self.model.named_parameters() if p.requires_grad}
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
        for n in fisher: 
            fisher[n] /= len(self.dataloader)
        
        if not was_training:
            self.model.eval()
            
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
            best_val = ckpt.get('best_val', float('inf'))
            no_imp = ckpt.get('no_imp', 0)
            logger.info(f"[Task {task_id}] Resuming from epoch {start_epoch}/{self.config.EPOCHS}, best_val={best_val:.4e}, no_imp={no_imp}")

            
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
            self.model.eval()
            val_loss=0
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
                ],
                'best_val': best_val,
                'no_imp':   no_imp  
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
                    ],
                    'best_val': best_val,
                    'no_imp':   no_imp  
                }, self.checkpoint_dir/f"task{task_id}_best.pt")
            else:
                no_imp+=1
                if no_imp>=self.config.PATIENCE: break

        return history

    def consolidate(self, loader):
        # compute fisher on loader and store
        self.ewc_tasks.append(EWC(self.model, loader, self.device))

    def evaluate(self, loader):
        self.model.eval()
        preds=[]
        targs=[]
        with torch.no_grad():
            for x,y in loader:
                out=self.model(x.to(self.device)).cpu().numpy().ravel()
                preds.append(out)
                targs.append(y.numpy().ravel())
            
        preds=np.concatenate(preds); targs=np.concatenate(targs)
        return {'RMSE':np.sqrt(mean_squared_error(targs,preds)),
                'MAE':mean_absolute_error(targs,preds),
                'R2':r2_score(targs,preds)}

if __name__=='__main__': 
    main()