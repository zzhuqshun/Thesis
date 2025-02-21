'''
This module contains functions for loading, processing, and visualizing the battery data.
'''
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_data(data_dir: str) -> pd.DataFrame:
    '''
    Load data from parquet files in the given directory.
    '''
    data_path = Path(data_dir)
    parquet_files = list(data_path.glob("**/df*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")   
    
    df_list = []
    for file_path in tqdm(parquet_files, desc="Processing cells", unit="cell"):
        file_name = file_path.stem
        cell_number = file_name.replace('df_', '')
        cell_name = f'C{cell_number}'
        tqdm.write(f"Processing {cell_name} ...")
        
        data = pd.read_parquet(file_path)
        data['Absolute_Time[yyyy-mm-dd hh:mm:ss]'] = pd.to_datetime(data['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        data = data[['Absolute_Time[yyyy-mm-dd hh:mm:ss]', 'Current[A]', 'Voltage[V]', 'Temperature[°C]', 'EFC', 'Q_sum', 'SOH_ZHU']]
        
        data["dV"] = data["Voltage[V]"].diff().fillna(0)
        data["dI"] = data["Current[A]"].diff().fillna(0)

        data["InternalResistance[Ohms]"] = np.where(
            data["dI"].abs() > 0.5,
            data["dV"] / data["dI"],
            np.nan
        )
        
        data_hourly = data.set_index('Absolute_Time[yyyy-mm-dd hh:mm:ss]').resample('h').mean()
        
        # fill missing values
        data_hourly.interpolate(method='linear', inplace=True)
        data_hourly = data_hourly.bfill()
        data_hourly.reset_index(drop=True, inplace=True)
        
        # add time_idx column
        data_hourly['Testtime[h]'] = data_hourly.index
        
        # add cell_id column
        data_hourly['cell_id'] = cell_name
        
        data_hourly = data_hourly[['Testtime[h]','Current[A]', 'Voltage[V]','Temperature[°C]', 'cell_id',
                                   'Q_sum','EFC', 'InternalResistance[Ohms]','SOH_ZHU']]
        
        df_list.append(data_hourly)
    
    return pd.concat(df_list, ignore_index=True)

def visualize_data(data_df: pd.DataFrame):
    """
    Visualize the distribution of all columns in the DataFrame.
    """
    # Get all columns from the DataFrame
    columns = data_df.columns
    n_cols = len(columns)
    
    # Create subplots dynamically: one subplot per column
    fig, axs = plt.subplots(n_cols, 1, figsize=(10, 4 * n_cols))
    
    # If there's only one column, ensure axs is iterable
    for i, column in enumerate(columns):
        axs[i].hist(data_df[column], bins=50, alpha=0.7)
        axs[i].set_title(f"{column} Data Distribution")
        axs[i].set_xlabel(column)
        axs[i].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
    
def inspect_data_ranges(data_df: pd.DataFrame) -> None:
    '''
    Inspect the ranges of the data columns.
    '''
    for cell_id, cell_data in data_df.groupby('cell_id'):
        print(f"\n=== {cell_id} ===")
        
        time_range = (
            cell_data['Absolute_Time[yyyy-mm-dd hh:mm:ss]'].min(), 
            cell_data['Absolute_Time[yyyy-mm-dd hh:mm:ss]'].max()
        )
        print(f"Time Range: {time_range[0]} to {time_range[1]}")
        
        for column in cell_data.columns:
            values = cell_data[column]
            print(f"\n{column}:")
            print(f"Value Range: {values.min():.4f} to {values.max():.4f}")
            print(f"Number of Data Points: {len(values)}")
            

def split_data(all_data: pd.DataFrame,
                    train: int = 13,
                    val: int = 1,
                    test: int = 1,
                    parts: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training, validation, and test sets.
    For training cells, further split each cell's data into 'parts' contiguous chunks.
    
    Parameters:
    - all_data: DataFrame containing all data; must include a 'cell_id' column.
    - train, val, test: Number of full cells for training, validation, and testing.
    - parts: Number of contiguous chunks to split each training cell's data into.
    
    Returns:
    - train_df: DataFrame for training, where each cell is split into parts.
    - val_df: DataFrame for validation (full cells).
    - test_df: DataFrame for testing (full cells).
    """
    # Get unique cell IDs and shuffle them.
    unique_cells = all_data['cell_id'].unique()
    np.random.seed(773)
    np.random.shuffle(unique_cells)
    
    # Split cell IDs into train, validation, and test groups.
    train_cells = unique_cells[:train]
    val_cells = unique_cells[train:train+val]
    test_cells = unique_cells[train+val:train+val+test]
    
    print("Cell split completed:")
    print(f"Training set: {len(train_cells)} cells")
    print(f"Validation set: {len(val_cells)} cells")
    print(f"Test set: {len(test_cells)} cells")
    
    # Build training data by splitting each cell's data into contiguous parts.
    train_parts_list = []
    # Optional: specify time column for sorting (if exists)
    time_col = "Absolute_Time[yyyy-mm-dd hh:mm:ss]"
    for cell in train_cells:
        cell_data = all_data[all_data['cell_id'] == cell].copy()
        # Sort by time if the column exists.
        if time_col in cell_data.columns:
            cell_data.sort_values(time_col, inplace=True)
        
        # Calculate chunk size for splitting cell_data into 'parts' chunks.
        l = len(cell_data)
        chunk_size = l // parts
        
        for i in range(parts):
            start = i * chunk_size
            # For the last part, include all remaining rows.
            end = len(cell_data) if i == parts - 1 else (i + 1) * chunk_size
            df_chunk = cell_data.iloc[start:end].copy()
            # Add part identifier columns.
            # df_chunk['cell_part'] = i + 1
            df_chunk['cell_id'] = f"{cell}_{i+1}"
            
            train_parts_list.append(df_chunk)
    
    # Concatenate all training parts.
    train_df = pd.concat(train_parts_list, ignore_index=True)
    
    # Validation and test data: use full cell data.
    val_df = all_data[all_data['cell_id'].isin(val_cells)].copy()
    test_df = all_data[all_data['cell_id'].isin(test_cells)].copy()
    
    print("Final dataset sizes:")
    print(f"Training set: {len(train_df)} rows (split into {len(train_cells)*parts} parts)")
    print(f"Validation set: {len(val_df)} rows from {len(val_cells)} cells")
    print(f"Test set: {len(test_df)} rows from {len(test_cells)} cells")
    
    return train_df, val_df, test_df


def plot_dataset_soh(data_df: pd.DataFrame, title: str, figsize=(10, 7)):
    '''
    Plot the SOH curves for the given dataset.
    '''
    
    plt.figure(figsize=figsize)
    
    # Group data by cell_id and plot each group's SOH curve
    for cell, group in data_df.groupby('cell_id'):
        plt.plot(group['Testtime[h]'], group['SOH_ZHU'], label=cell)
    
    plt.title(f'{title} Set SOH Curves')
    plt.xlabel('Time')
    plt.ylabel('SOH')
    plt.grid(True)
    plt.legend(loc='upper right')
    # plt.tight_layout()
    plt.show()
    

def scale_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Scale the data using a robust/standard scaler
    '''
    scaler = RobustScaler()
    scaler.fit(train_df[['Current[A]', 'Temperature[°C]', 'Voltage[V]', 'Q_sum', 'EFC', "InternalResistance[Ohms]"]])
    
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy[['Current[A]', 'Temperature[°C]', 'Voltage[V]', 'Q_sum', 'EFC',"InternalResistance[Ohms]"]] = scaler.transform(
            df_copy[['Current[A]', 'Temperature[°C]', 'Voltage[V]', 'Q_sum', 'EFC',"InternalResistance[Ohms]"]]
        )
        return df_copy

    train_scaled = transform(train_df)
    val_scaled = transform(val_df)
    test_scaled = transform(test_df)
    
    return train_scaled, val_scaled, test_scaled