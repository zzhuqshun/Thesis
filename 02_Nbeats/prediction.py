## Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.callbacks import EarlyStopping
from pathlib import Path
import optuna
from tqdm import tqdm
from typing import Dict, Tuple

from darts.metrics import rmse
import warnings
warnings.filterwarnings('ignore')

best_params = {'input_chunk_length': 33, 
               'output_chunk_length': 1, 
               'num_blocks': 4, 
               'num_stacks': 3, 
               'activation': 'LeakyReLU', 
               'batch_size': 32, 
               'learning_rate': 0.00041328603854847963,
               "expansion_coefficient_dim":16,
               "trend_polynomial_degree":2
               }

best_model = NBEATSModel(
    input_chunk_length=best_params["input_chunk_length"],
    output_chunk_length=best_params["output_chunk_length"],
    num_blocks=best_params["num_blocks"],
    num_stacks=best_params["num_stacks"],
    batch_size=best_params["batch_size"],
    expansion_coefficient_dim=best_params["expansion_coefficient_dim"],  
    trend_polynomial_degree=best_params["trend_polynomial_degree"],  
    optimizer_kwargs={"lr": best_params["learning_rate"]},  
    random_state=773,  
    activation=best_params["activation"],
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

train_series, train_cov = prepare_data(train_data)
val_series, val_cov = prepare_data(val_data)
best_model.fit(series=train_series, past_covariates=train_cov, 
               val_series=val_series, val_past_covariates=val_cov, epochs=500, verbose=True)

best_model.save('best_nbeats_model')

def prepare_test_data(data):
    series = []
    cov = []
    for cell_data in data.values():
        series.append(cell_data['target'])
        cov.append(cell_data['covariates_scaled'])
    return series, cov

def prdictions(model_path:str, test_data):
    model = NBEATSModel.load(model_path)

    test_series, test_cov = [], []
    for cell_data in test_data.values():
        test_series.append(cell_data['target'])
        test_cov.append(cell_data['covariates_scaled'])
        
    start = model.input_chunk_length + model.output_chunk_length
    pred_test = model.historical_forecasts(
        series=test_series,
        past_covariates=test_cov,
        start=start,
        forecast_horizon=model.output_chunk_length,
        retrain=False,
        verbose=True
    )

    backtest_rmse = rmse(test_series, pred_test)
    print(f'Backtest RMSE of Testdaten = {backtest_rmse}')
    
    cell_ids = test_data.keys()  
    colors = ['blue', 'green', 'red']

    plt.figure(figsize=(12, 6))
    for i, (cell_id, pred) in enumerate(zip(cell_ids, pred_test)):
        pred.plot(label=f'{cell_id} Forecast', 
                    color=colors[i], 
                    linestyle='--')
    
        test_data[cell_id]['target'].plot(label=f'{cell_id} Actual', 
                                            color=colors[i], 
                                            linestyle='-',
                                            alpha=0.7)

    plt.title("SOH Forecast vs Actual")    
    plt.xlabel("Time")
    plt.ylabel("SOH")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

model_path = r'best\01_val3test3\best_nbeats_model'
# model_path = 'best_nbeats_model'

prdictions(model_path, test_data)
