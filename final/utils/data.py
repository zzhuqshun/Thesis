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
    """
    Handles loading, preprocessing, and scaling of battery cell data.
    
    Workflow:
      1. Read parquet files for each cell.
      2. Drop NaNs and round timestamps.
      3. Create a datetime index and resample to reduce noise.
      4. Fit a RobustScaler on the training data.
      5. Transform all splits (train/val/test or per-task splits).
    """
    def __init__(self, data_dir, resample='10min'):
        self.data_dir = Path(data_dir)
        self.resample = resample
        self.scaler = RobustScaler()

    def load_cell_data(self) -> dict[str, Path]:
        """
        Scan the directory and map each cell ID to its parquet file path.
        """
        files = sorted(
            self.data_dir.glob('*.parquet'),
            key=lambda x: int(x.stem.split('_')[-1])
        )
        return {fp.stem.split('_')[-1]: fp for fp in files}

    def process_file(self, fp: Path) -> pd.DataFrame:
        """
        Process one cell file:
          - Read and select relevant columns.
          - Drop missing values.
          - Round 'Testtime[s]' and generate a datetime index.
          - Resample to `self.resample` frequency.
          - Add a 'cell_id' column.
        """
        df = (
            pd.read_parquet(fp)[
                ['Testtime[s]', 'Voltage[V]', 'Current[A]', 'Temperature[°C]', 'SOH_ZHU']
            ]
            .dropna()
            .reset_index(drop=True)
        )
        df['Testtime[s]'] = df['Testtime[s]'].round().astype(int)
        df['Datetime'] = pd.date_range('2023-02-02', periods=len(df), freq='s')

        df = (
            df.set_index('Datetime')
              .resample(self.resample)
              .mean()
              .reset_index()
        )
        df['cell_id'] = fp.stem.split('_')[-1]
        return df

    def prepare_joint_data(self, splits: dict) -> dict[str, pd.DataFrame]:
        """
        Prepare data for joint training (baseline):
          - Concatenate train & val across multiple cells.
          - Process the single test cell.
          - Fit scaler on train features and transform all splits.
        
        Returns:
            dict with keys 'train', 'val', 'test' mapping to DataFrames.
        """
        logger.info("Preparing joint data...")
        cells = self.load_cell_data()
        def build(ids):
            return pd.concat(
                [self.process_file(cells[i]) for i in ids],
                ignore_index=True
            )
        
        df_train = build(splits['train_ids'])
        df_val   = build(splits['val_ids'])
        df_test  = self.process_file(cells[splits['test_id']])
        logger.info("Train IDs: %s, shape: %s", splits['train_ids'], df_train.shape)
        logger.info("Val IDs: %s, shape: %s", splits['val_ids'], df_val.shape)
        logger.info("Test ID: %s, shape: %s", splits['test_id'], df_test.shape)
        
        feat_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        self.scaler.fit(df_train[feat_cols])
        logger.info("Scaler centers: %s,", self.scaler.center_)
        logger.info("Scaler scales: %s", self.scaler.scale_)
        
        for df in (df_train, df_val, df_test):
            df[feat_cols] = self.scaler.transform(df[feat_cols])

        return {'train': df_train, 'val': df_val, 'test': df_test}

    def prepare_incremental_data(self, splits: dict) -> dict[str, pd.DataFrame]:
        """
        Prepare data for incremental learning:
          1. Build train & full-val for each task.
          2. Split each full-val into val/test (e.g. 70/30 split).
          3. Fit scaler on Task 0 train features.
          4. Transform all task-specific splits and the overall test set.
        
        Returns:
            dict containing:
              - 'task0_train', 'task0_val', 'task1_train', 'task1_val', ...
              - 'test_full', 'test_task0', 'test_task1', ...
        """
        cells = self.load_cell_data()
        def build(ids):
            return pd.concat(
                [self.process_file(cells[i]) for i in ids],
                ignore_index=True
            )

        # Task datasets
        df0t, df0v_full = build(splits['task0_train_ids']), build(splits['task0_val_ids'])
        df1t, df1v_full = build(splits['task1_train_ids']), build(splits['task1_val_ids'])
        df2t, df2v_full = build(splits['task2_train_ids']), build(splits['task2_val_ids'])
        df_test_full    = self.process_file(cells[splits['test_id']])
        # Split full-val into val/test
        def split_val_test(df_full, ratio=0.7):
            n = len(df_full)
            idx = int(n * ratio)
            return (
                df_full.iloc[:idx].reset_index(drop=True),
                df_full.iloc[idx:].reset_index(drop=True)
            )

        df0v, df0tst = split_val_test(df0v_full)
        df1v, df1tst = split_val_test(df1v_full)
        df2v, df2tst = split_val_test(df2v_full)

        logger.info("Task 0 Train IDs: %s, shape: %s", splits['task0_train_ids'], df0t.shape)
        logger.info("Task 0 Val IDs: %s, shape: %s", splits['task0_val_ids'], df0v_full.shape)
        logger.info("Task 0 Test IDs: %s, shape: %s", splits['test_id'], df0tst.shape)
        logger.info("Task 1 Train IDs: %s, shape: %s", splits['task1_train_ids'], df1t.shape)
        logger.info("Task 1 Val IDs: %s, shape: %s", splits['task1_val_ids'], df1v_full.shape)
        logger.info("Task 1 Test IDs: %s, shape: %s", splits['test_id'], df1tst.shape)
        logger.info("Task 2 Train IDs: %s, shape: %s", splits['task2_train_ids'], df2t.shape)
        logger.info("Task 2 Val IDs: %s, shape: %s", splits['task2_val_ids'], df2v_full.shape)
        logger.info("Task 2 Test IDs: %s, shape: %s", splits['test_id'], df2tst.shape)
        logger.info("Full Test ID: %s, shape: %s", splits['test_id'], df_test_full.shape)


        # Fit scaler on Task 0 train
        feat_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        self.scaler.fit(df0t[feat_cols])
        logger.info("Scaler centers: %s,", self.scaler.center_)
        logger.info("Scaler scales: %s", self.scaler.scale_)

        def scale(df: pd.DataFrame) -> pd.DataFrame:
            df2 = df.copy()
            df2[feat_cols] = self.scaler.transform(df2[feat_cols])
            return df2

        return {
            'task0_train': scale(df0t), 'task0_val':  scale(df0v),
            'task1_train': scale(df1t), 'task1_val':  scale(df1v),
            'task2_train': scale(df2t), 'task2_val':  scale(df2v),
            'test_full':    scale(df_test_full),
            'test_task0':   scale(df0tst), 'test_task1': scale(df1tst), 'test_task2': scale(df2tst)
        }
