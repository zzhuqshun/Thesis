# %%
## Packages
# %matplotlib widget
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import optuna
from tqdm import tqdm
from typing import Dict, Tuple

from darts.metrics import rmse, mae
import warnings
warnings.filterwarnings('ignore')


# %%
def split_data_into_parts(data: pd.DataFrame, parts: int = 15) -> Dict[str, pd.DataFrame]:
    # Split data into parts
    chunk_size = len(data) // parts
    return {f"{idx+1}": data.iloc[idx * chunk_size:(idx + 1) * chunk_size] for idx in range(parts)}

def load_data(data_dir: str) -> Tuple[Dict, Dict]:
    data_path = Path(data_dir)
    all_data = {}

    # Find all parquet files
    parquet_files = list(data_path.glob("**/df*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    for file_path in tqdm(parquet_files, desc="Processing cells", unit="cell"):
        # Extract cell number from parent directory name
        file_name = file_path.stem  
        cell_number = file_name.replace('df_', '')  
        cell_name = f'C{cell_number}'  
        tqdm.write(f"Processing {cell_name} ...")
            
        # Load and process data
        data = pd.read_parquet(file_path)
        data['Absolute_Time[yyyy-mm-dd hh:mm:ss]'] = pd.to_datetime(data['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        
        # Select relevant columns
        data = data[['Absolute_Time[yyyy-mm-dd hh:mm:ss]', 'Current[A]', 'Voltage[V]', 
                    'Temperature[°C]', 'SOH_ZHU']]
        
        # Resample to hourly
        data.set_index('Absolute_Time[yyyy-mm-dd hh:mm:ss]', inplace=True)
        data_hourly = data.resample('h').mean().reset_index()
        
        # Fill missing values
        data_hourly.interpolate(method='linear', inplace=True)
        data_hourly['SOH_ZHU'] = data_hourly['SOH_ZHU'].fillna(1)
        
        # Convert to time series for full data
        target_series_full = TimeSeries.from_dataframe(data_hourly, 'Absolute_Time[yyyy-mm-dd hh:mm:ss]', 'SOH_ZHU')
        covariates_full = TimeSeries.from_dataframe(data_hourly, 'Absolute_Time[yyyy-mm-dd hh:mm:ss]', ['Current[A]', 'Voltage[V]', 'Temperature[°C]'])
        target_series_full, covariates_full = target_series_full.slice_intersect(covariates_full), covariates_full.slice_intersect(target_series_full)
        scaler_full = Scaler(scaler=MinMaxScaler(feature_range=(-1,1)))
        covariates_scaled_full = scaler_full.fit_transform(covariates_full)
        
        all_data[cell_name] = {'target': target_series_full, 'covariates_scaled': covariates_scaled_full, 'df': data_hourly }

    return all_data

data_dir = "../01_Datenaufbereitung/Output/Calculated/"
all_data = load_data(data_dir)

# %%
def inspect_data_ranges(data_dict: dict):
   """
   Inspect time ranges and value ranges for each battery in the data dictionary
   """
   for cell_name, cell_data in data_dict.items():
       print(f"\n=== {cell_name} ===")
       
       # Get target data range
       target = cell_data['target']
       target_values = target.values().flatten()  # Flatten array for calculation
       print("\nTarget (SOH_ZHU):")
       print(f"Time Range: {target.start_time()} to {target.end_time()}")
       print(f"Value Range: {target_values.min():.4f} to {target_values.max():.4f}")
       print(f"Number of Data Points: {len(target)}")
       
       # Get covariates data range
       covariates = cell_data['covariates_scaled']
       cov_values = covariates.values()
       print("\nCovariates (scaled):")
       for i, feature in enumerate(covariates.components):
           values = cov_values[:, i].flatten()
           print(f"{feature}:")
           print(f"Value Range: {values.min():.4f} to {values.max():.4f}")

# View all data ranges
print("All Data Ranges:")
inspect_data_ranges(all_data)

# %%
def split_cell_data(all_data: dict, train=13, val=1, test=1, parts=15) -> Tuple[Dict, Dict, Dict]:
    """Splits the dataset into training, validation, and test sets, then further divides train and val into parts."""
    
    cell_names = list(all_data.keys())
    np.random.seed(773)
    np.random.shuffle(cell_names)

    # 1. **Split data at the cell level into train/val/test sets**
    train_cells = cell_names[:train]
    val_cells = cell_names[train:train + val]
    test_cells = cell_names[train + val:train + val + test]

    print(f"Cell split completed:")
    print(f"Training set: {len(train_cells)} cells")
    print(f"Validation set: {len(val_cells)} cells")
    print(f"Test set: {len(test_cells)} cells")

    train_parts = []
    val_parts = []

    # 2. **Split training data into smaller parts**
    for cell in train_cells:
        split_data = split_data_into_parts(all_data[cell]['df'], parts=parts)
        for part_idx, df_part in split_data.items():
            part_name = f"{cell}_{part_idx}"
            target_series_part = TimeSeries.from_dataframe(df_part, 'Absolute_Time[yyyy-mm-dd hh:mm:ss]', 'SOH_ZHU')
            covariates_part = TimeSeries.from_dataframe(df_part, 'Absolute_Time[yyyy-mm-dd hh:mm:ss]', ['Current[A]', 'Voltage[V]', 'Temperature[°C]'])
            target_series_part, covariates_part = target_series_part.slice_intersect(covariates_part), covariates_part.slice_intersect(target_series_part)
            scaler_part = Scaler(scaler=MinMaxScaler(feature_range=(-1,1)))
            covariates_scaled_part = scaler_part.fit_transform(covariates_part)
            train_parts.append((part_name, {'target': target_series_part, 'covariates_scaled': covariates_scaled_part}))

    # 3. **Split validation data into smaller parts**
    for cell in val_cells:
        split_data = split_data_into_parts(all_data[cell]['df'], parts=parts)
        for part_idx, df_part in split_data.items():
            part_name = f"{cell}_{part_idx}"
            target_series_part = TimeSeries.from_dataframe(df_part, 'Absolute_Time[yyyy-mm-dd hh:mm:ss]', 'SOH_ZHU')
            covariates_part = TimeSeries.from_dataframe(df_part, 'Absolute_Time[yyyy-mm-dd hh:mm:ss]', ['Current[A]', 'Voltage[V]', 'Temperature[°C]'])
            target_series_part, covariates_part = target_series_part.slice_intersect(covariates_part), covariates_part.slice_intersect(target_series_part)
            scaler_part = Scaler(scaler=MinMaxScaler(feature_range=(-1,1)))
            covariates_scaled_part = scaler_part.fit_transform(covariates_part)
            val_parts.append((part_name, {'target': target_series_part, 'covariates_scaled': covariates_scaled_part}))

    # 4. **Combine and shuffle train and validation parts**
    all_parts = train_parts + val_parts
    np.random.shuffle(all_parts)

    # 5. **Reassign train and validation parts**
    new_train_size = train * parts
    new_val_size = val * parts

    train_data = dict(all_parts[:new_train_size])
    val_data = dict(all_parts[new_train_size:new_train_size + new_val_size])

    # 6. **Keep test data as full cells without splitting**
    test_data = {cell: all_data[cell] for cell in test_cells}

    print(f"Data split completed:")
    print(f"Training set: {len(train_data)} parts")
    print(f"Validation set: {len(val_data)} parts")
    print(f"Test set: {len(test_data)} full cells")

    return train_data, val_data, test_data

# Execute data split
train_data, val_data, test_data = split_cell_data(all_data)

# Inspect training data
inspect_data_ranges(train_data)


# %%
def plot_dataset_soh(data_dict: dict, title: str, figsize=(10, 7)):
    plt.figure(figsize=figsize)
    
    # Plot each cell's SOH
    for cell_name, cell_data in data_dict.items():
        target = cell_data['target']
        plt.plot(target.time_index, target.values().flatten(), label=cell_name)
    
    plt.title(f'{title} Set SOH Curves')
    plt.xlabel('Time')
    plt.ylabel('SOH_ZHU')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Plot all three datasets
plot_dataset_soh(train_data, "Training")
plot_dataset_soh(val_data, "Validation")
plot_dataset_soh(test_data, "Test")

# %%
def prepare_data(data):
    targets = []
    covariates = []
    for cell_data in data.values():
        targets.append(cell_data['target'])
        covariates.append(cell_data['covariates_scaled'])
    
    series = targets[0]
    cov = covariates[0]
    for i in range(1, len(targets)):
        series = series.concatenate(targets[i], ignore_time_axis=True)
        cov = cov.concatenate(covariates[i], ignore_time_axis=True)
    return series, cov

# %%
def objective(trial):
    # Define hyperparameter search space
    # 1. Search - Basic structure
    input_chunk_length = trial.suggest_int("input_chunk_length", 24, 48, step=12)
    output_chunk_length = trial.suggest_int("output_chunk_length", 1, 24, step=12)
    num_blocks = trial.suggest_int("num_blocks", 2, 5)
    num_stacks = trial.suggest_int("num_stacks", 2, 5)
    activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU"])
    
    # 2. Search - Training parameters
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    layer_widths = trial.suggest_categorical("layer_widths", [128, 256, 512])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    expansion_coefficient_dim = trial.suggest_int("expansion_coefficient_dim", 8, 32, step=8)

    # Define and train model
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        num_blocks=num_blocks,
        num_stacks=num_stacks,
        batch_size=batch_size,
        layer_widths=layer_widths,
        dropout=dropout_rate,
        expansion_coefficient_dim=expansion_coefficient_dim, 
        random_state=773,
        activation=activation,
        pl_trainer_kwargs={
        "accelerator": "gpu",
        "devices": 1,
        "callbacks": [
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
            EarlyStopping(monitor="val_loss", patience=20, mode="min", verbose=True),  
            ],
        "enable_checkpointing": True
        },
        optimizer_kwargs={"lr": 0.0001}, 
        lr_scheduler_cls=ReduceLROnPlateau,
        lr_scheduler_kwargs={
            "mode": "min",  
            "factor": 0.5,  
            "patience": 15,  
            "min_lr": 1e-6  
    }
    )
    
    train_series, train_cov = prepare_data(train_data)
    val_series, val_cov = prepare_data(val_data)

    model.fit(series=train_series, past_covariates=train_cov, 
              val_series=val_series, val_past_covariates=val_cov, epochs=100)  
    
    # Retrieve best validation loss directly from the training process
    best_val_loss = model.trainer.checkpoint_callback.best_model_score.item() 
    
    return best_val_loss

# %%
# Optuna call with progress bar
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100) 

# Best trial
print("Best trial:")
trial = study.best_trial
print(f"  Value (MAE): {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# %%



