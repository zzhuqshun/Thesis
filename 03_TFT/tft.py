# %%
%matplotlib widget

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner



from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import RMSE, MAE
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder


def load_data(data_dir: str) -> pd.DataFrame:
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
        data = data[['Absolute_Time[yyyy-mm-dd hh:mm:ss]', 'Current[A]', 'Voltage[V]', 
                     'Temperature[°C]', 'SOH_ZHU']]
        
        # resample to hourly data
        data.set_index('Absolute_Time[yyyy-mm-dd hh:mm:ss]', inplace=True)
        data_hourly = data.resample('h').mean().reset_index()
        
        # fill missing values
        data_hourly.interpolate(method='linear', inplace=True)
        data_hourly= data_hourly.dropna()
        data_hourly.reset_index(drop=True, inplace=True)
        
        # add time_idx column
        data_hourly['time_idx'] = np.arange(len(data_hourly))
        
        # add cell_id column
        data_hourly['cell_id'] = cell_name
        
        df_list.append(data_hourly)
    
    all_df = pd.concat(df_list, ignore_index=True)
    return all_df

data_dir = "../01_Datenaufbereitung/Output/Calculated/"
all_data = load_data(data_dir)

def split_cell_data(all_data: pd.DataFrame, 
                             train: int = 13, 
                             val: int = 1, 
                             test: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Get and shuffle unique cell IDs
    unique_cells = all_data['cell_id'].unique()
    np.random.seed(773)
    np.random.shuffle(unique_cells)
    
    # Split cell IDs into train, validation, and test sets
    train_cells = unique_cells[:train]
    val_cells = unique_cells[train:train+val]
    test_cells = unique_cells[train+val:train+val+test]
    
    # Process training data: simply filter by cell_id (no splitting)
    train_df = all_data[all_data['cell_id'].isin(train_cells)].copy()
    
    # Process validation and test data: simply filter by cell_id
    val_df = all_data[all_data['cell_id'].isin(val_cells)].copy()
    test_df = all_data[all_data['cell_id'].isin(test_cells)].copy()
    
    # Print final dataset sizes
    print("Final dataset sizes:")
    print(f"Training set: {len(train_cells)} full cells")
    print(f"Validation set: {len(val_cells)} full cells")
    print(f"Test set: {len(test_cells)} full cells")
    
    return train_df, val_df, test_df

# Example usage:
train_df, val_df, test_df = split_cell_data(all_data, train=13, val=1, test=1)

def scale_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    robust_scaler = StandardScaler()
    robust_scaler.fit(train_df[['Current[A]', 'Temperature[°C]', 'Voltage[V]']])
    
    def robust_transform(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy[['Current[A]', 'Temperature[°C]', 'Voltage[V]']] = robust_scaler.transform(
            df_copy[['Current[A]', 'Temperature[°C]', 'Voltage[V]']]
        )
        return df_copy

    train_scaled = robust_transform(train_df)
    val_scaled = robust_transform(val_df)
    test_scaled = robust_transform(test_df)
    
    return train_scaled, val_scaled, test_scaled

# Example usage:
train_scaled, val_scaled, test_scaled = scale_data(train_df, val_df, test_df)


max_encoder_length = 36
max_prediction_length = 6

train_dataset = TimeSeriesDataSet(
        train_scaled,
        time_idx="time_idx",
        group_ids=["cell_id"],
        target="SOH_ZHU",
        max_prediction_length=max_prediction_length,
        max_encoder_length=max_encoder_length,
        time_varying_known_reals=["Current[A]", "Voltage[V]", "Temperature[°C]"],
        time_varying_unknown_reals=[],
)

validation = TimeSeriesDataSet.from_dataset(train_dataset, val_scaled)

batch_size = 16
train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size)


train_batch = next(iter(train_dataloader))
train_inputs, train_targets = train_batch  
print("Inputs shape:", train_inputs["encoder_cont"].shape)
print("Inputs keys:", train_inputs.keys())
print( train_inputs["encoder_cont"].shape)
print(train_inputs["encoder_target"].shape)
print(train_inputs["decoder_target"].shape)
print(train_inputs["decoder_cont"].shape)
train_inputs["decoder_target"]

tft = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=0.03,
    hidden_size=32, # most important hyperparameter apart from learning rate
    hidden_continuous_size=16,
    attention_head_size=6,
    lstm_layers=4,
    dropout=0.1,
    loss=MAE(),
    optimizer="adam",
)

trainer = pl.Trainer(
    accelerator="cpu",
    gradient_clip_val=0.1,
)

res = Tuner(trainer).lr_find(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    max_lr=10,
    min_lr=1e-6,
)

print(f"suggested learning rate: {res.suggestion()}")
res.plot(suggest=True)
plt.show()
# lr = 2e-5

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",     
    mode="min",             
    save_top_k=1,           
    filename="best-checkpoint"
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=10,
    verbose=False,
    mode="min"
)

trainer = pl.Trainer(
    max_epochs=1,
    accelerator="gpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback, checkpoint_callback]
)

tft = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=1e-5,
    hidden_size=32,
    attention_head_size=6,
    dropout=0.1,
    hidden_continuous_size=16, # set to <= hidden_size
    loss=MAE(),
    optimizer="adam"
)

trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

def autoregressive_predict(
    best_tft,
    train_dataset: TimeSeriesDataSet,  # 用于继承配置
    test_scaled: pd.DataFrame,         # 单个 cell 的测试数据
    max_encoder_length: int = 36,
    max_prediction_length: int = 6
) -> pd.DataFrame:
    """
    自回归预测：
    1) 只将未来要预测的部分置为 NaN，历史观测值保留
    2) 每次预测 max_prediction_length 步，并将预测值写回
    3) 不生成新的 version 文件夹或日志
    """
    # 1) 复制 + 排序
    df_ar = test_scaled.copy()
    df_ar.sort_values("time_idx", inplace=True)
    df_ar.reset_index(drop=True, inplace=True)

    total_len = len(df_ar)
    if total_len <= max_encoder_length:
        print("Warning: total_len <= max_encoder_length, no prediction needed.")
        return df_ar

    # 2) **只修改未来需要预测的部分**
    df_ar.loc[df_ar["time_idx"] >= max_encoder_length, "SOH_ZHU"] = np.nan

    # 3) 确保数据类型正确
    df_ar["time_idx"] = df_ar["time_idx"].astype(int)
    df_ar["cell_id"] = df_ar["cell_id"].astype(str)

    # Fill missing values in SOH_ZHU with forward fill method
    df_ar["SOH_ZHU"].fillna(method='ffill', inplace=True)

    # 4) 自回归循环：从第 max_encoder_length 条开始，每次预测 6 步
    current_position = max_encoder_length
    while current_position < total_len:
        # 计算本次预测步长
        prediction_steps = min(max_prediction_length, total_len - current_position)
        
        # 构建动态数据集
        temp_dataset = TimeSeriesDataSet.from_dataset(
            train_dataset,
            df_ar,
            predict=True,
            stop_randomization=True
        )
        temp_loader = temp_dataset.to_dataloader(train=False, batch_size=16)

        # 执行预测
        raw_preds = best_tft.predict(
            temp_loader,
            mode="prediction",
            return_x=False,
            trainer_kwargs=dict(logger=False, enable_checkpointing=False)
        )

        # 提取当前窗口预测值
        window_predictions = raw_preds[0].cpu().numpy().squeeze()[:prediction_steps]

        # 更新数据
        for step in range(prediction_steps):
            target_idx = current_position + step
            if target_idx >= total_len:
                break
            df_ar.loc[df_ar["time_idx"] == target_idx, "SOH_ZHU"] = window_predictions[step]

        # 移动窗口位置
        current_position += prediction_steps

        # 打印进度
        print(f"Predicted up to time_idx {min(current_position-1, total_len-1)}/{total_len-1}")

    return df_ar


# ========== 使用示例 ==========
df_autoreg = autoregressive_predict(
    best_tft, train_dataset, test_scaled,
    max_encoder_length=36,
    max_prediction_length=6
)


test_dataset = TimeSeriesDataSet.from_dataset(train_dataset, test_scaled,  stop_randomization=True)
test_dataloader = test_dataset.to_dataloader(train=False, batch_size=16, num_workers=0)

batch = next(iter(test_dataloader))
inputs, targets = batch  
print("Inputs keys:", inputs.keys())
print( inputs["encoder_cont"].shape)
print(inputs["encoder_target"].shape)
print(inputs["decoder_target"].shape)
print(inputs["decoder_cont"].shape)
inputs["decoder_target"]

# %%
predictions = best_tft.predict(test_dataloader, return_y=True, mode="prediction", trainer_kwargs=dict(accelerator="gpu", logger=False))
MAE()(predictions.output, predictions.y)

# Extract tensors from the tuples
pred = predictions.output[0].squeeze().cpu().numpy()  # => (batch_size, max_prediction_length)
true = predictions.y[0].squeeze().cpu().numpy()       # => (batch_size, max_prediction_length)

print(f"y_pred shape: {len(pred)}")
print(f"y_true shape: {len(true)}")

min_len = min(len(pred), len(true))
y_pred = pred[:min_len]
y_true = true[:min_len]

# Evaluate the model
mae = mean_absolute_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
print(f" MAE = {mae:.3e}, RMSE = {rmse:.3e}")

# visualize the prediction
time_index = np.array(test_dataset.data["time"]).flatten()[:min_len]  

pred_df = pd.DataFrame({"time_idx": time_index, "true_SOH_ZHU": y_true, "predicted_SOH_ZHU": y_pred})
pred_df.sort_values("time_idx", inplace=True)


plt.figure(figsize=(12, 6))
plt.plot(pred_df["time_idx"], pred_df["true_SOH_ZHU"], label="True SOH_ZHU")
plt.plot(pred_df["time_idx"], pred_df["predicted_SOH_ZHU"], label="Predicted SOH_ZHU")
plt.xlabel("Time Index")
plt.ylabel("SOH_ZHU")
plt.title("TFT Model - SOH_ZHU Test Prediction")
plt.text(0.80, 0.85, f"MAE = {mae:.3e}\nRMSE = {rmse:.3e}", 
         transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
plt.legend()
plt.grid()
plt.show()

# %%
print(type(test_predictions), len(test_predictions))

# %%



