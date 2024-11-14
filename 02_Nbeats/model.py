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
import optuna


# %%
## Read data
data = pd.read_parquet("/home/users/z/zzhuqshun/Thesis/01_Datenaufbereitung/Output/Calculated/df_15.parquet")
data['Absolute_Time[yyyy-mm-dd hh:mm:ss]'] = pd.to_datetime(data['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
data = data[['Absolute_Time[yyyy-mm-dd hh:mm:ss]', 'Current[A]', 'Voltage[V]', 'Temperature[°C]', 'SOH_ZHU']]

## Resample to hourly
data.set_index('Absolute_Time[yyyy-mm-dd hh:mm:ss]', inplace=True)
data_hourly = data.resample('h').mean().reset_index()

## Fill missing values
numeric_cols = data_hourly.select_dtypes(include=[np.number]).columns  
data_hourly[numeric_cols] = data_hourly[numeric_cols].interpolate(method='linear')
data_hourly['SOH_ZHU'] = data_hourly['SOH_ZHU'].fillna(1)

# %%
## Data to time series
target_series = TimeSeries.from_dataframe(data_hourly, 'Absolute_Time[yyyy-mm-dd hh:mm:ss]', 'SOH_ZHU')
covariates = TimeSeries.from_dataframe(data_hourly, 'Absolute_Time[yyyy-mm-dd hh:mm:ss]', ['Current[A]', 'Voltage[V]', 'Temperature[°C]'])

## Time align
target_series, covariates = target_series.slice_intersect(covariates), covariates.slice_intersect(target_series)

## Covariates normalization
scaler = Scaler() # Scale data [min,max] to [0,1]
## Don't scale SOH
covariates_scaled = scaler.fit_transform(covariates)

## Data split
train_series, val_series = target_series.split_after(0.8)
cov_train, cov_val = covariates_scaled.split_after(0.8)

# Time align
required_start_time = train_series.start_time() - pd.Timedelta(hours=12) 
if cov_train.start_time() > required_start_time:
    cov_train = covariates_scaled.slice(required_start_time, cov_train.end_time())
if cov_val.start_time() > required_start_time:
    cov_val = covariates_scaled.slice(required_start_time, cov_val.end_time())


# %%
# Optuna objective function
def objective(trial):
    # Define hyperparameter search space
    input_chunk_length = trial.suggest_int("input_chunk_length", 12, 24)
    output_chunk_length = trial.suggest_int("output_chunk_length", 1, 7)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_blocks = trial.suggest_int("num_blocks", 2, 3)
    num_stacks = trial.suggest_int("num_stacks", 2, 3)

    # Define and train model
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        batch_size=batch_size,
        num_blocks=num_blocks,
        num_stacks=num_stacks,
        random_state=42,
        pl_trainer_kwargs={
            "callbacks": [ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)],
            "enable_checkpointing": True
        }
    )

    model.fit(series=train_series, past_covariates=cov_train, 
              val_series=val_series, val_past_covariates=cov_val, epochs=200)  
    
    # Retrieve best validation loss directly from the training process
    best_val_loss = model.trainer.checkpoint_callback.best_model_score.item() 
    
    return best_val_loss



# %%
# Optuna call with progress bar
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)  

# Best trial
print("Best trial:")
trial = study.best_trial
print(f"  Value (MAE): {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# %%
best_params = trial.params
best_model = NBEATSModel(
    input_chunk_length=best_params["input_chunk_length"],
    output_chunk_length=best_params["output_chunk_length"],
    batch_size=best_params["batch_size"],
    num_blocks=best_params["num_blocks"],
    num_stacks=best_params["num_stacks"],
    random_state=42,
    save_checkpoints=True
)

best_model.fit(series=train_series, past_covariates=cov_train, 
               val_series=val_series, val_past_covariates=cov_val, epochs=200, verbose=True)


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



