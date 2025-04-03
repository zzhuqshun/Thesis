import random
from soh_lstm import set_seed
from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

set_seed(42)


def load_data(data_dir: Path, resample='10min'):
    """
    读取 data_dir 中所有的 .parquet 文件，按指定时间间隔（resample）重采样后合并。
    不再划分 train/val/test，直接返回一个合并后的 DataFrame。
    
    :param data_dir: 包含 .parquet 文件的目录
    :param resample: 重采样频率，如 "1min", "10min", "h" 等
    :return: 处理并重采样后的总 DataFrame
    """
    # 找到并排序所有 parquet 文件
    parquet_files = sorted(
        [f for f in data_dir.glob('*.parquet') if f.is_file()],
        key=lambda x: int(x.stem.split('_')[-1])  # 假设文件名里最后的数字可排序
    )
    print(f"共找到 {len(parquet_files)} 个文件，准备合并...")

    # 准备一个列表，用于存放处理后的 DataFrame
    data_frames = []

    for file_path in parquet_files:
        df = pd.read_parquet(file_path)

        # 只保留需要的列，减少内存使用
        columns_to_keep = ['Testtime[s]', 'Voltage[V]', 'Current[A]',
                           'Temperature[°C]', 'SOC_ZHU', 'SOH_ZHU']
        df_processed = df[columns_to_keep].copy()
        df_processed.dropna(inplace=True)

        # 将 Testtime[s] 转为整型秒，再生成对应 Datetime 列
        df_processed['Testtime[s]'] = df_processed['Testtime[s]'].round().astype(int)
        start_date = pd.Timestamp("2023-02-02")
        df_processed['Datetime'] = pd.date_range(
            start=start_date,
            periods=len(df_processed),
            freq='s'
        )

        # 按指定频率重采样（默认 10 分钟），并 reset index
        df_sampled = df_processed.resample(resample, on='Datetime').mean().reset_index(drop=False)

        data_frames.append(df_sampled)

    # 将所有文件的数据合并
    df_all = pd.concat(data_frames, ignore_index=True)

    print(f"合并后 DataFrame 形状: {df_all.shape}")
    return df_all

data_dir = Path("../01_Datenaufbereitung/Output/Calculated/")
df_min = load_data(data_dir, resample='min')
df_10min = load_data(data_dir, resample='10min')
df_h = load_data(data_dir, resample='h')


def plot_acf_for_df(df, scale_label):
    """
    对 DataFrame 中的 SOH_ZHU 列计算并绘制自相关函数图。
    
    :param df: 包含 "Datetime" 和 "SOH_ZHU" 列的 DataFrame
    :param scale_label: 字符串，用于图表标题，表示当前的重采样尺度
    """
    # 按时间排序，并将 Datetime 设置为索引
    df_sorted = df.sort_values('Datetime').set_index('Datetime')
    # 提取 SOH_ZHU 序列，并去除空值
    soh_series = df_sorted['SOH_ZHU'].dropna()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_acf(soh_series, lags=50, ax=ax)
    plt.title(f'Autocorrelation of SOH_ZHU ({scale_label} scale)')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.grid(linestyle='--', alpha=0.6)
    plt.show()

# 对三种不同重采样尺度的数据进行自相关分析
plot_acf_for_df(df_min, 'min')
plot_acf_for_df(df_10min, '10min')
plot_acf_for_df(df_h, 'h')

