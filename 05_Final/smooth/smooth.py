import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 引入训练时定义的模块：
# 确保 train.py（或你的主训练脚本）在同一目录，或已安装为模块
from ewc import DataProcessor, create_dataloaders, SOHLSTM, get_predictions


def main():
    # 根目录为本文件所在目录
    base_dir = Path(__file__).parent
    # 1. 读取 config.json
    config_path = base_dir / 'model/Regular_LSTM/Joint_Training'/'config.json'
    with open(config_path) as f:
        cfg = json.load(f)

    # 提取超参
    seq_len    = cfg['SEQUENCE_LENGTH']
    batch_size = cfg['BATCH_SIZE']
    resample   = cfg.get('RESAMPLE', '10min')
    alpha      = 0.1  # EWMA 衰减系数，可根据需要调整

    # 2. 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. 加载并准备数据（Joint Training 配置）
    dp = DataProcessor(
        data_dir='../01_Datenaufbereitung/Output/Calculated/',
        resample=resample,
        base_train_ids=['03','05','07','09','11','15','21','23','25','27','29'],
        base_val_ids=['01','19','13'],
        update1_train_ids=[], update1_val_ids=[],
        update2_train_ids=[], update2_val_ids=[],
        test_cell_id='17'
    )
    datasets = dp.prepare_data()
    loaders = create_dataloaders(datasets, seq_len, batch_size)
    test_loader = loaders['test_full']
    df_test = datasets['test_full']

    # 4. 加载模型及最优 checkpoint
    model = SOHLSTM(
        input_size=3,
        hidden_size=cfg['HIDDEN_SIZE'],
        num_layers=cfg['NUM_LAYERS'],
        dropout=cfg['DROPOUT']
    ).to(device)
    ckpt_path = base_dir / 'model/Regular_LSTM/Joint_Training/regular'/ 'checkpoints' / 'task0_best.pt'
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state'])
    model.eval()

    # 5. 原始预测
    preds, tgts = get_predictions(model, test_loader, device)

    # 6. EWMA 平滑
    preds_smooth = pd.Series(preds).ewm(alpha=alpha, adjust=False).mean().to_numpy()

    # 7. 计算指标
    metrics_raw = {
        'RMSE': np.sqrt(mean_squared_error(tgts, preds)),
        'MAE':  mean_absolute_error(tgts, preds),
        'R2':   r2_score(tgts, preds)
    }
    metrics_smooth = {
        'RMSE': np.sqrt(mean_squared_error(tgts, preds_smooth)),
        'MAE':  mean_absolute_error(tgts, preds_smooth),
        'R2':   r2_score(tgts, preds_smooth)
    }
    print("Raw metrics:", metrics_raw)
    print("Smoothed metrics:", metrics_smooth)

    # 8. 绘图
    out_dir = base_dir / 'results'
    out_dir.mkdir(exist_ok=True)

    dates = df_test['Datetime'].iloc[seq_len:].values
    plt.figure(figsize=(10, 6))
    plt.plot(dates, tgts, label='Actual', linewidth=2)
    plt.plot(dates, preds, label='Predicted', alpha=0.5)
    plt.plot(dates, preds_smooth, label=f'EWMA (α={alpha})', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('SOH')
    # 添加指标到标题下方
    subtitle = (
        f"Raw RMSE: {metrics_raw['RMSE']:.4f}, MAE: {metrics_raw['MAE']:.4f}\n"
        f"Smoothed RMSE: {metrics_smooth['RMSE']:.4f}, MAE: {metrics_smooth['MAE']:.4f}"
    )
    plt.title('SOH Predictions — Raw vs EWMA Smoothed\n' + subtitle)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / 'smoothed_predictions.png')
    plt.close()


if __name__ == '__main__':
    main()
