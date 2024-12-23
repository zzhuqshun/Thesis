# %%
## Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from sklearn.metrics import mean_absolute_error
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.callbacks import EarlyStopping
from pathlib import Path
import optuna
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple




# %%
def load_data(data_dir: str, feature_range=(-1, 1)) -> dict:
    data_path = Path(data_dir)
    processed_data = {}

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
        
        # Convert to time series
        target_series = TimeSeries.from_dataframe(
            data_hourly, 'Absolute_Time[yyyy-mm-dd hh:mm:ss]', 'SOH_ZHU'
        )
        covariates = TimeSeries.from_dataframe(
            data_hourly, 'Absolute_Time[yyyy-mm-dd hh:mm:ss]', 
            ['Current[A]', 'Voltage[V]', 'Temperature[°C]']
        )
        
        # Time align
        target_series, covariates = target_series.slice_intersect(covariates), covariates.slice_intersect(target_series)
        
        # Scale covariates
        scaler = Scaler(scaler=MinMaxScaler(feature_range=(-1,1)))
        covariates_scaled = scaler.fit_transform(covariates)
        
        processed_data[cell_name] = {
            'target': target_series,
            'covariates_scaled': covariates_scaled
        }
    
    return processed_data

data_dir = "../01_Datenaufbereitung/Output/Calculated/"
processed_data = load_data(data_dir)

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
inspect_data_ranges(processed_data)

# %%
def split_cell_data(processed_data: dict, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2) -> Tuple[Dict, Dict, Dict]:
   # Get all cell numbers
   cell_names = list(processed_data.keys())
   
   # Calculate number of cells needed for each set
   n_cells = len(cell_names)
   n_train = int(n_cells * train_ratio)
   n_val = int(n_cells * val_ratio)
   
   # Randomly shuffle cell order
   np.random.seed(773)
   np.random.shuffle(cell_names)
   
   # Split cell numbers
   train_cells = cell_names[:n_train]
   val_cells = cell_names[n_train:n_train + n_val]
   test_cells = cell_names[n_train + n_val:]
   
   # Create dataset dictionaries
   train_data = {cell: processed_data[cell] for cell in train_cells}
   val_data = {cell: processed_data[cell] for cell in val_cells}
   test_data = {cell: processed_data[cell] for cell in test_cells}
   
   print(f"Data split completed:")
   print(f"Training set: {len(train_data)} cells {sorted(train_cells)}")
   print(f"Validation set: {len(val_data)} cells {sorted(val_cells)}")
   print(f"Test set: {len(test_data)} cells {sorted(test_cells)}")
   
   return train_data, val_data, test_data

# Usage example:
train_data, val_data, test_data = split_cell_data(processed_data)
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
def prepare_data(train_data, val_data):
    # Concatenate training data
    train_targets = []
    train_covariates = []
    for cell_data in train_data.values():
        train_targets.append(cell_data['target'])
        train_covariates.append(cell_data['covariates_scaled'])
    
    train_series = train_targets[0]
    train_cov = train_covariates[0]
    for i in range(1, len(train_targets)):
        train_series = train_series.concatenate(train_targets[i], ignore_time_axis=True)
        train_cov = train_cov.concatenate(train_covariates[i], ignore_time_axis=True)
    
    # Concatenate validation data
    val_targets = []
    val_covariates = []
    for cell_data in val_data.values():
        val_targets.append(cell_data['target'])
        val_covariates.append(cell_data['covariates_scaled'])
    
    val_series = val_targets[0]
    val_cov = val_covariates[0]
    for i in range(1, len(val_targets)):
        val_series = val_series.concatenate(val_targets[i], ignore_time_axis=True)
        val_cov = val_cov.concatenate(val_covariates[i], ignore_time_axis=True)
    
    return train_series, train_cov, val_series, val_cov

# %%
# Optuna objective function
def objective(trial):
    # Define hyperparameter search space
    # 1. Search - Basic structure
    input_chunk_length = trial.suggest_int("input_chunk_length", 12, 24)
    output_chunk_length = trial.suggest_int("output_chunk_length", 1, 12)
    num_blocks = trial.suggest_int("num_blocks", 2, 5)
    num_stacks = trial.suggest_int("num_stacks", 2, 5)
    activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU"])
    # 2. Search - Training parameters
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    expansion_coefficient_dim = 8
    trend_polynomial_degree = 2

    # Define and train model
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        num_blocks=num_blocks,
        num_stacks=num_stacks,
        batch_size=batch_size,
        expansion_coefficient_dim=expansion_coefficient_dim, 
        trend_polynomial_degree=trend_polynomial_degree, 
        optimizer_kwargs={"lr": learning_rate},
        random_state=773,
        activation = activation,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": 1, 
            "callbacks": [
              ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
              EarlyStopping(monitor="val_loss", patience=10, mode="min")
            ],
            "enable_checkpointing": True
        }
    )
    train_series, ctrain_cov, val_series, val_cov = prepare_data(train_data, val_data)
    
    model.fit(series=train_series, past_covariates=ctrain_cov, 
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
# best_params = {'input_chunk_length': 15, 'output_chunk_length': 1, 'batch_size': 16, 'num_blocks': 2, 'num_stacks': 2}

# %%
# best_params = trial.params
# best_model = NBEATSModel(
#     input_chunk_length=best_params["input_chunk_length"],
#     output_chunk_length=best_params["output_chunk_length"],
#     batch_size=best_params["batch_size"],
#     num_blocks=best_params["num_blocks"],
#     num_stacks=best_params["num_stacks"],
#     random_state=42,
#     save_checkpoints=True
# )

# best_model.fit(series=train_series, past_covariates=cov_train, 
#                val_series=val_series, val_past_covariates=cov_val, epochs=200, verbose=True)


# %%
# model = NBEATSModel.load_from_checkpoint('in22_out1_bs16_nb3_ns2')

# pred_series = model.predict(len(val_series), series=train_series, past_covariates=cov_val)

# plt.figure(figsize=(8, 5))
# target_series.plot(label="Actual")
# pred_series.plot(label="Forecast")
# plt.title("SOH Forecast using NBEATS Model")
# plt.xlabel("Time")
# plt.legend()
# plt.show()

# plt.figure(figsize=(8, 5)) 
# train_series.plot(label="train")
# val_series.plot(label="true")
# pred_series.plot(label="forecast")
# plt.legend()
# plt.xlabel('Time')
# plt.show()

# %%
# param_grid = {
#     'input_chunk_length': [12, 24], # Half day or full day
#     'output_chunk_length': [1, 3, 6, 12], # One Hour or more
#     'batch_size': [16, 32, 64], # Training speed
#     'num_blocks': [2, 3], # Depth and nonliniarity
#     'num_stacks': [2, 3] # Different nonlinear mode 
# }

# %%
# def grid_search_nbeats(param_grid, train_series, val_series, cov_train=None, cov_val=None):
#     best_params = None
#     best_score = float("inf")
#     best_model = None 

#     keys, values = zip(*param_grid.items())
#     param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#     for params in tqdm(param_combinations, desc="Grid Search Progress"):
#         model = NBEATSModel(
#             input_chunk_length=params['input_chunk_length'],
#             output_chunk_length=params['output_chunk_length'],
#             batch_size=params['batch_size'],
#             num_blocks=params['num_blocks'],
#             num_stacks=params['num_stacks'],
#             random_state=42
#         )

#         # Training
#         model.fit(series=train_series, past_covariates=cov_train, epochs=200)

#         # Predict
#         pred_series = model.predict(len(val_series), series=train_series, past_covariates=cov_val)
#         score = mean_absolute_error(val_series.values(), pred_series.values())

#         print(f"Params: {params} - MAE: {score}")

#         if score < best_score:
#             best_score = score
#             best_params = params
#             best_model = model
#     if best_model is not None:
#         best_model.save_model("best_nbeats_model.pth")
#         print("Best model saved as 'best_nbeats_model.pth'")
        
#     print(f"Best Params: {best_params} with MAE: {best_score}")
#     return best_params, best_score

# best_params, best_score = grid_search_nbeats(param_grid, train_series, val_series, cov_train=cov_train, cov_val=cov_val)



