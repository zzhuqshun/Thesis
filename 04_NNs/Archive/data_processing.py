"""
This module contains functions for loading, processing, and visualizing the battery data.
"""
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load data from parquet files in the given directory.
    Returns a concatenated DataFrame of all loaded files.
    """
    data_path = Path(data_dir)
    parquet_files = list(data_path.glob("**/df*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")   
    
    df_list = []
    for file_path in tqdm(parquet_files, desc="Processing cells", unit="cell"):
        file_name = file_path.stem
        cell_number = file_name.replace('df_', '')
        cell_name = f'C{cell_number}'
        tqdm.write(f"Processing {cell_name} ...")
        
        # Read the parquet file
        data = pd.read_parquet(file_path)
        
        # Convert time column to datetime
        data['Absolute_Time[yyyy-mm-dd hh:mm:ss]'] = pd.to_datetime(
            data['Absolute_Time[yyyy-mm-dd hh:mm:ss]']
        )
        
        # Select columns of interest
        data = data[['Absolute_Time[yyyy-mm-dd hh:mm:ss]', 
                     'Current[A]', 'Voltage[V]', 
                     'Temperature[°C]', 'SOH_ZHU']]
        
        # Resample the data to 1-minute intervals, then interpolate
        data_selected = data.set_index('Absolute_Time[yyyy-mm-dd hh:mm:ss]').resample('10min').mean()
        data_selected.interpolate(method='linear', inplace=True)
        data_selected.dropna(inplace=True)
        data_selected.reset_index(drop=True, inplace=True)
        
        # Add a time index column
        data_selected['Testtime[min]'] = data_selected.index
        
        # Add a cell_id column
        data_selected['cell_id'] = cell_name
        
        # Reorder columns
        data_selected = data_selected[['Testtime[min]', 'Current[A]', 'Voltage[V]',
                                   'Temperature[°C]', 'cell_id', 'SOH_ZHU']]
        
        df_list.append(data_selected)
    
    # Concatenate all processed dataframes
    return pd.concat(df_list, ignore_index=True)

def visualize_data(data_df: pd.DataFrame):
    """
    Visualize the distribution of all columns in the DataFrame.
    Creates a histogram for each column.
    """
    columns = data_df.columns
    n_cols = len(columns)
    
    fig, axs = plt.subplots(n_cols, 1, figsize=(10, 4 * n_cols))
    
    # Ensure axs is iterable even if there's only one column
    for i, column in enumerate(columns):
        axs[i].hist(data_df[column], bins=50, alpha=0.7)
        axs[i].set_title(f"{column} Data Distribution")
        axs[i].set_xlabel(column)
        axs[i].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

def split_data(all_data: pd.DataFrame,
               train: int = 13,
               val: int = 1,
               test: int = 1,
               parts: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training, validation, and test sets based on the number of cells.
    Each cell's data can be further split into contiguous chunks for training.

    Parameters:
    - all_data: DataFrame containing all data, must include 'cell_id'.
    - train, val, test: Number of full cells for training, validation, and testing.
    - parts: Number of contiguous chunks to split each training cell's data.

    Returns:
    - train_df: Training DataFrame with cells split into parts.
    - val_df: Validation DataFrame (full cells).
    - test_df: Test DataFrame (full cells).
    """
    # Shuffle the cell IDs
    unique_cells = all_data['cell_id'].unique()
    np.random.seed(773)
    np.random.shuffle(unique_cells)
    
    # Partition cell IDs for train, validation, and test
    train_cells = unique_cells[:train]
    val_cells = unique_cells[train:train+val]
    test_cells = unique_cells[train+val:train+val+test]
    
    print("Cell split completed:")
    print(f"Training set: {len(train_cells)} cells")
    print(f"Validation set: {len(val_cells)} cells")
    print(f"Test set: {len(test_cells)} cells")
    
    # Split training data into parts for each cell
    train_parts_list = []
    time_col = "Absolute_Time[yyyy-mm-dd hh:mm:ss]"  # optional time column
    
    for cell in train_cells:
        cell_data = all_data[all_data['cell_id'] == cell].copy()
        # Sort by time if the column exists
        if time_col in cell_data.columns:
            cell_data.sort_values(time_col, inplace=True)
        
        # Calculate chunk size for splitting into parts
        length = len(cell_data)
        chunk_size = length // parts
        
        for i in range(parts):
            start = i * chunk_size
            # For the last part, include all remaining rows
            end = length if i == parts - 1 else (i + 1) * chunk_size
            df_chunk = cell_data.iloc[start:end].copy()
            # Append part index to cell_id to differentiate each chunk
            df_chunk['cell_id'] = f"{cell}_{i+1}"
            train_parts_list.append(df_chunk)
    
    train_df = pd.concat(train_parts_list, ignore_index=True)
    
    # Validation and test data use full cells
    val_df = all_data[all_data['cell_id'].isin(val_cells)].copy()
    test_df = all_data[all_data['cell_id'].isin(test_cells)].copy()
    
    print("Final dataset sizes:")
    print(f"Training set: {len(train_df)} rows (split into {len(train_cells)*parts} parts)")
    print(f"Validation set: {len(val_df)} rows from {len(val_cells)} cells")
    print(f"Test set: {len(test_df)} rows from {len(test_cells)} cells")
    
    return train_df, val_df, test_df

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_dataset_soh(data_df: pd.DataFrame,
                     title: str,
                     figsize=(10, 7),
                     cell_filter=None):
    """
    1) 画出所有 cell 的 SOH 曲线  
    2) 按 LOW/MID/HIGH 三类分别着色画出同样的曲线

    参数同原版，多了内部的 categories 定义
    """
    # 如果提供了 cell_filter，则先过滤
    if cell_filter is not None:
        if isinstance(cell_filter, (list, tuple, set)):
            data_df = data_df[data_df['cell_id'].isin(cell_filter)]
        else:
            data_df = data_df[data_df['cell_id'] == cell_filter]

    # ---------- 1) 所有曲线 ---------
    plt.figure(figsize=figsize)
    for cell, group in data_df.groupby('cell_id'):
        plt.plot(group['Testtime[min]'], group['SOH_ZHU'], label=cell)
    plt.title(f'{title} Set SOH Curves (all cells)')
    plt.xlabel('Time (10 min intervals)')
    plt.ylabel('SOH')
    plt.grid(True)
    plt.legend(loc='upper right', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.show()

    # ---------- 2) 按类别着色的曲线 ---------
    # 定义三类 ID
    categories = {
        'LOW':  ['C01', 'C03', 'C05', 'C07', 'C27'],
        'MID':  ['C17', 'C19', 'C21', 'C23', 'C25'],
        'HIGH': ['C09', 'C11', 'C13', 'C15', 'C29'],
    }
    # 给每类指定颜色+线型
    styles = {
        'LOW':  {'color': 'tab:blue',   'linestyle': '-'},
        'MID':  {'color': 'tab:orange', 'linestyle': '-'},
        'HIGH': {'color': 'tab:green',  'linestyle': '-'},
    }

    plt.figure(figsize=figsize)
    for cat, ids in categories.items():
        for cell in ids:
            subset = data_df[data_df['cell_id'] == cell]
            # 如果过滤后某些 IDs 不存在，就跳过
            if subset.empty:
                continue
            plt.plot(
                subset['Testtime[min]'],
                subset['SOH_ZHU'],
                label=cell,
                **styles[cat]
            )

    # 构造一个分类图例
    legend_handles = [
        Line2D([0], [0],
               color=styles[cat]['color'],
               linestyle=styles[cat]['linestyle'],
               lw=2)
        for cat in categories
    ]
    plt.legend(handles=legend_handles,
               labels=categories.keys(),
               title='Category',
               loc='upper right')

    plt.title(f'{title} Set SOH Curves (by LOW/MID/HIGH)')
    plt.xlabel('Time (10 min intervals)')
    plt.ylabel('SOH')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def scale_data(train_df: pd.DataFrame, 
               val_df: pd.DataFrame, 
               test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scale the specified columns in the dataset using StandardScaler fitted on the training set.
    """
    columns_to_scale = ['Current[A]', 'Temperature[°C]', 'Voltage[V]']
    
    # Fit the scaler on the training data
    scaler = StandardScaler()
    scaler.fit(train_df[columns_to_scale])
    
    # Transform the data
    train_scaled = scaler.transform(train_df[columns_to_scale])
    val_scaled = scaler.transform(val_df[columns_to_scale])
    test_scaled = scaler.transform(test_df[columns_to_scale])
    
    # Convert the numpy arrays back to DataFrame and merge with unscaled columns
    train_df_scaled = train_df.copy()
    train_df_scaled[columns_to_scale] = train_scaled
    
    val_df_scaled = val_df.copy()
    val_df_scaled[columns_to_scale] = val_scaled
    
    test_df_scaled = test_df.copy()
    test_df_scaled[columns_to_scale] = test_scaled
    
    return train_df_scaled, val_df_scaled, test_df_scaled
