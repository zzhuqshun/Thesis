from pathlib import Path
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(data_dir: Path, resample='10min'):
    """
    Load and return four datasets: base / update1 / update2 / test
    
    Args:
        data_dir: Directory containing parquet files
        resample: Time interval for resampling data (default: '10min')
        
    Returns:
        Tuple of four dataframes (df_base, df_update1, df_update2, df_test)
    """
    parquet_files = sorted(
        [f for f in data_dir.glob('*.parquet') if f.is_file()],
        key=lambda x: int(x.stem.split('_')[-1])
    )

    # Random assignment of files to different sets
    test_file = random.choice(parquet_files)
    remaining_files = [f for f in parquet_files if f != test_file]

    # Randomly shuffle the remaining files
    random.shuffle(remaining_files)

    # Assign the remaining files to different sets
    base_files = remaining_files[:4]
    update1_files = remaining_files[4:9]
    update2_files = remaining_files[9:14]
    
    def process_file(file_path: Path):
        """Process a single parquet file into a resampled dataframe"""
        df = pd.read_parquet(file_path)
        columns_to_keep = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 
                           'Temperature[°C]', 'SOC_ZHU', 'SOH_ZHU']
        df_processed = df[columns_to_keep].copy()
        df_processed.dropna(inplace=True)
        
        df_processed['Testtime[s]'] = df_processed['Testtime[s]'].round().astype(int)
        start_date = pd.Timestamp("2023-02-02")
        df_processed['Datetime'] = pd.date_range(
            start=start_date,
            periods=len(df_processed),
            freq='s'
        )
        
        df_sampled = df_processed.resample(resample, on='Datetime').mean().reset_index(drop=False)
        df_sampled["cell_id"] = file_path.stem.split('_')[1]
        return df_sampled, file_path.name
    
    test_data = process_file(test_file)
    base_data = [process_file(f) for f in base_files]
    update1_data = [process_file(f) for f in update1_files]
    update2_data = [process_file(f) for f in update2_files]
    
    print(f"Test cell: {test_data[1]}")
    print(f"Base training cells: {[t[1] for t in base_data]}")
    print(f"Update 1 cells: {[u[1] for u in update1_data]}")
    print(f"Update 2 cells: {[u[1] for u in update2_data]}")

    df_test = test_data[0]
    df_base = pd.concat([t[0] for t in base_data], ignore_index=True)
    df_update1 = pd.concat([u[0] for u in update1_data], ignore_index=True)
    df_update2 = pd.concat([u[0] for u in update2_data], ignore_index=True)
    
    print(f"\nBase training data shape: {df_base.shape}")
    print(f"Update 1 data shape: {df_update1.shape}")
    print(f"Update 2 data shape: {df_update2.shape}")
    print(f"Test data shape: {df_test.shape}\n")
    
    return df_base, df_update1, df_update2, df_test

def split_by_cell(df, name, val_cells=1, seed=42):
    """
    Split dataset into training and validation sets based on cell_id.
    
    Args:
        df: DataFrame containing cell_id column
        name: Name identifier for printing purposes
        val_cells: Number of cells to use for validation (default: 1)
        seed: Random seed for reproducibility
        
    Returns:
        df_train, df_val: Training and validation DataFrames
    """
    np.random.seed(seed)
    cell_ids = df['cell_id'].unique().tolist()
    np.random.shuffle(cell_ids)
    # Select first val_cells for validation, rest for training
    val_ids = cell_ids[:val_cells]
    train_ids = cell_ids[val_cells:]
    df_train = df[df['cell_id'].isin(train_ids)].reset_index(drop=True)
    df_val = df[df['cell_id'].isin(val_ids)].reset_index(drop=True)
    print(f"{name} - Training cells: {train_ids}")
    print(f"{name} - Validation cells: {val_ids}")
    return df_train, df_val


def scale_data(df_base, df_update1, df_update2, df_test):
    """
    Normalize all datasets using StandardScaler fitted on base data.
    
    Args:
        df_base: Base training data
        df_update1: First update data
        df_update2: Second update data
        df_test: Test data
        
    Returns:
        Tuple of scaled dataframes (df_base_scaled, df_update1_scaled, df_update2_scaled, df_test_scaled)
    """
    features_to_scale = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
    
    df_base_scaled = df_base.copy()
    df_update1_scaled = df_update1.copy()
    df_update2_scaled = df_update2.copy()
    df_test_scaled = df_test.copy()
    
    scaler = StandardScaler()
    scaler.fit(df_base[features_to_scale])
    
    df_base_scaled[features_to_scale] = scaler.transform(df_base[features_to_scale])
    df_test_scaled[features_to_scale] = scaler.transform(df_test[features_to_scale])
    df_update1_scaled[features_to_scale] = scaler.transform(df_update1[features_to_scale])
    df_update2_scaled[features_to_scale] = scaler.transform(df_update2[features_to_scale])
    
    print('Features scaled using StandardScaler fitted on base training data.\n')
    
    return df_base_scaled, df_update1_scaled, df_update2_scaled, df_test_scaled