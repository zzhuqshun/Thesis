import logging
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class BatteryDataset(Dataset):
    def __init__(self, df, seq_len):
        """
        PyTorch Dataset for battery time-series data.
        
        Args:
            df (pd.DataFrame): DataFrame containing ['Voltage[V]', 'Current[A]', 'Temperature[°C]', 'SOH_ZHU'].
            seq_len (int): Length of the input sequence for the model.
        """
        feats = df[['Voltage[V]', 'Current[A]', 'Temperature[°C]']].values
        self.X = torch.tensor(feats, dtype=torch.float32)
        self.y = torch.tensor(df['SOH_ZHU'].values, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        # Number of samples = total timesteps minus sequence length
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        """
        Returns:
            tuple of (input_sequence, next_soh):
              - input_sequence: Tensor of shape (seq_len, 3)
              - next_soh: Tensor scalar of the SOH value at t = idx + seq_len
        """
        return self.X[idx:idx + self.seq_len], self.y[idx + self.seq_len]

def create_dataloaders(datasets: dict,
                       seq_len: int,
                       batch_size: int) -> dict:
    """
    Convert a dict of pandas DataFrames into PyTorch DataLoaders.

    Args:
        datasets: mapping from split names to DataFrames (e.g., 'train', 'val', 'test')
        seq_len: length of input sequence for the dataset
        batch_size: batch size for DataLoader

    Returns:
        loaders: dict mapping split names to DataLoader instances
    """
    loaders = {}
    for split, df in datasets.items():
        if df is None or df.empty:
            continue
        # Only consider splits containing 'train', 'val', or 'test'
        if any(tag in split for tag in ('train', 'val', 'test')):
            ds = BatteryDataset(df, seq_len)
            shuffle = 'train' in split
            loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loaders

class DataProcessor:
    """Handles data loading, preprocessing, and scaling"""
    
    def __init__(self, data_dir, resample='10min', config=None):
        self.data_dir = Path(data_dir)
        self.resample = resample
        self.scaler = RobustScaler()  # Robust to outliers
        self.config = config
    
    def load_cell_data(self):
        """Load all battery cell data files"""
        files = sorted(self.data_dir.glob('*.parquet'), key=lambda x: int(x.stem.split('_')[-1]))
        return {fp.stem.split('_')[-1]: fp for fp in files}
    
    def process_file(self, fp):
        """Process single battery cell file"""
        # Load relevant columns
        df = pd.read_parquet(fp)[['Testtime[s]', 'Voltage[V]', 'Current[A]', 'Temperature[°C]', 'SOH_ZHU']]
        df = df.dropna().reset_index(drop=True)
        
        # Round timestamps and create datetime index
        df['Testtime[s]'] = df['Testtime[s]'].round().astype(int)
        df['Datetime'] = pd.date_range('2023-02-02', periods=len(df), freq='s')
        
        # Resample to reduce data size and smooth noise
        df = df.set_index('Datetime').resample(self.resample).mean().reset_index()
        df['cell_id'] = fp.stem.split('_')[-1]
        
        return df
    
    def prepare_joint_data(self, cfg):
        """Prepare data for joint training (baseline)"""
        info = self.load_cell_data()
        
        def build(ids): 
            return pd.concat([self.process_file(info[c]) for c in ids], ignore_index=True) if ids else pd.DataFrame()
        
        df_train = build(cfg['train_ids'])
        df_val = build(cfg['val_ids'])
        df_test = self.process_file(info[cfg['test_id']])
        
        logger.info("Joint training - Train IDs: %s, size: %d", 
                   cfg['train_ids'], len(df_train))
        logger.info("Joint training - Val IDs: %s, size: %d", 
                   cfg['val_ids'], len(df_val))
        logger.info("Joint training - Test ID: %s, size: %d", 
                   cfg['test_id'], len(df_test))
        
        # Fit scaler on training data only
        feat_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        self.scaler.fit(df_train[feat_cols])
        logger.info("  (Scaler) Scaler centers: %s", self.scaler.center_)
        logger.info("  (Scaler) Scaler scales: %s", self.scaler.scale_)

        
        def scale(df):
            df2 = df.copy()
            if not df2.empty:
                df2[feat_cols] = self.scaler.transform(df2[feat_cols])
            return df2
        
        return {'train': scale(df_train), 'val': scale(df_val), 'test': scale(df_test)}
    
    def prepare_incremental_data(self, cfg):
        """Prepare data for incremental learning"""
        info = self.load_cell_data()
        
        def build(ids): 
            return pd.concat([self.process_file(info[c]) for c in ids], ignore_index=True) if ids else pd.DataFrame()
        
        # Build datasets for each task
        df0t = build(cfg['task0_train_ids']); df0v_full = build(cfg['task0_val_ids'])
        df1t = build(cfg['task1_train_ids']); df1v_full = build(cfg['task1_val_ids'])
        df2t = build(cfg['task2_train_ids']); df2v_full = build(cfg['task2_val_ids'])
        df_test = self.process_file(info[cfg['test_id']])

        def split_val_test(df_full, split_ratio=0.7):
            n = len(df_full)
            split_idx = int(n * split_ratio)
            df_val  = df_full.iloc[:split_idx].reset_index(drop=True)
            df_test = df_full.iloc[split_idx:].reset_index(drop=True)
            return df_val, df_test
        
        # Split validation data into train/val for each task
        df0v, df0test = split_val_test(df0v_full)
        df1v, df1test = split_val_test(df1v_full)
        df2v, df2test = split_val_test(df2v_full)
        
        dfs_train = [df0t, df1t, df2t]
        dfs_val   = [df0v, df1v, df2v]
        dfs_test  = [df0test, df1test, df2test]

        for i in range(3):
            logger.info(
                "Incremental training - Task %d Train IDs: %s, size: %d",
                i, cfg[f'task{i}_train_ids'], len(dfs_train[i])
            )
            logger.info(
                "Incremental training - Task %d Val IDs: %s, size: %d",
                i, cfg[f'task{i}_val_ids'],   len(dfs_val[i])
            )
            logger.info(
                "Incremental training - Test Task %d size: %d",
                i, len(dfs_test[i])
            )
        
        
        # Fit scaler on first task training data only
        feat_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        self.scaler.fit(dfs_train[0][feat_cols])
        logger.info("  (Scaler) Scaler centers: %s", self.scaler.center_)
        logger.info("  (Scaler) Scaler scales: %s", self.scaler.scale_)

        
        def scale(df):
            df2 = df.copy()
            if not df2.empty:
                df2[feat_cols] = self.scaler.transform(df2[feat_cols])
            return df2
        
        return {
            'task0_train': scale(df0t), 'task0_val': scale(df0v),
            'task1_train': scale(df1t), 'task1_val': scale(df1v),
            'task2_train': scale(df2t), 'task2_val': scale(df2v),
            'test_full': scale(df_test),
            'test_task0': scale(df0test), 'test_task1': scale(df1test), 'test_task2': scale(df2test)
        }
