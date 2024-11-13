# %%
%pip install darts
%matplotlib widget

# %%
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm 
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, r2_score
from darts import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


# %% [markdown]
# # Data Import

# %%
## Read data
data = pd.read_parquet(r"..\01_Datenaufbereitung\Output\Calculated\df_15.parquet")
data['Absolute_Time[yyyy-mm-dd hh:mm:ss]'] = pd.to_datetime(data['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
data = data[['Absolute_Time[yyyy-mm-dd hh:mm:ss]', 'Current[A]', 'Voltage[V]', 'Temperature[°C]', 'SOH_ZHU']]

## Resample to hourly
data.set_index('Absolute_Time[yyyy-mm-dd hh:mm:ss]', inplace=True)
data_hourly = data.resample('h').mean().reset_index()

## Fill missing values
data_hourly.interpolate(method='linear', inplace=True)
data_hourly['SOH_ZHU'] = data_hourly['SOH_ZHU'].fillna(1)
data_hourly

# %%
## Data to time series
target_series = TimeSeries.from_dataframe(data_hourly, 'Absolute_Time[yyyy-mm-dd hh:mm:ss]', 'SOH_ZHU')
covariates = TimeSeries.from_dataframe(data_hourly, 'Absolute_Time[yyyy-mm-dd hh:mm:ss]', ['Current[A]', 'Voltage[V]', 'Temperature[°C]'])

## Time align
target_series, covariates = target_series.slice_intersect(covariates), covariates.slice_intersect(target_series)

## Covariates normalization
scaler = Scaler()
## Don't scale SOH
covariates_scaled = scaler.fit_transform(covariates)

## Data split
train_series, val_series = target_series.split_after(0.8)
cov_train, cov_val = covariates_scaled.split_after(0.8)
plt.figure(figsize=(8, 5))
train_series.plot(label="training")
val_series.plot(label="validation")
plt.title("SOH Over Time (hourly)")
plt.xlabel("Time")

# %% [markdown]
# # Model training

# %% [markdown]
# ## Model define

# %%
## Model definition
model_name = "NBeats"
model = NBEATSModel(
    input_chunk_length=24,
    output_chunk_length=6,
    random_state=42,
    save_checkpoints=True
)

# %%
model.fit(series=train_series, past_covariates=cov_train, 
          val_series=val_series, val_past_covariates=cov_val, epochs=200, verbose=True)


# %%
model_nbeats = NBEATSModel.load_from_checkpoint('in24_out6')
pred_series = model_nbeats.historical_forecasts(start = train_series.end_time() - pd.Timedelta(hours=1),  
                                   series = target_series,
                                   past_covariates = covariates_scaled,
                                   forecast_horizon=1,
                                   stride=1,
                                   last_points_only=False, 
                                   retrain=False
                                   )

pred_series = concatenate(pred_series) 

plt.figure(figsize=(8, 5)) 
train_series.plot(label="train")
val_series.plot(label="true")
pred_series.plot(label="forecast")
plt.legend()
plt.xlabel('Time')
plt.show()

# %% [markdown]
# ## Grid search

# %%
param_grid = {
    'input_chunk_length': [12, 24], # Half day or full day
    'output_chunk_length': [1, 3, 6, 12], # One Hour or more
    'batch_size': [16, 32, 64], # Training speed
    'num_blocks': [2, 3], # Depth and nonliniarity
    'num_stacks': [2, 3] # Different nonlinear mode 
}

# %%
def grid_search_nbeats(param_grid, train_series, val_series, cov_train=None, cov_val=None):
    best_params = None
    best_score = float("inf")

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for params in tqdm(param_combinations, desc="Grid Search Progress"):
        model = NBEATSModel(
            input_chunk_length=params['input_chunk_length'],
            output_chunk_length=params['output_chunk_length'],
            batch_size=params['batch_size'],
            num_blocks=params['num_blocks'],
            num_stacks=params['num_stacks'],
            random_state=42
        )

        # Training
        model.fit(series=train_series, past_covariates=cov_train, epochs=200)

        # Time align
        required_start_time = train_series.start_time() - pd.Timedelta(hours=params['input_chunk_length']) 
        if cov_train.start_time() > required_start_time:
            cov_train = covariates_scaled.slice(required_start_time, cov_train.end_time())
        if cov_val.start_time() > required_start_time:
            cov_val = covariates_scaled.slice(required_start_time, cov_val.end_time())

        # Predict
        pred_series = model.predict(len(val_series), series=train_series, past_covariates=cov_val)
        score = mean_absolute_error(val_series.values(), pred_series.values())

        print(f"Params: {params} - MAE: {score}")

        if score < best_score:
            best_score = score
            best_params = params

    print(f"Best Params: {best_params} with MAE: {best_score}")
    return best_params, best_score

best_params, best_score = grid_search_nbeats(param_grid, train_series, val_series, cov_train=cov_train, cov_val=cov_val)
