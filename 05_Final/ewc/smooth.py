import numpy as np
import pandas as pd
import torch
import tqdm
from pathlib import Path

from ewc import SOHLSTM, create_dataloaders, DataProcessor, set_seed, Config, Trainer


def find_best_alpha(model_path: Path,
                    config: Config,
                    test_loader: torch.utils.data.DataLoader,
                    alphas: np.ndarray):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(config.SEED)
    # 1. 加载模型
    model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()

    # 2. 包装一个 Trainer 用它的 evaluate 方法
    trainer = Trainer(model, device, config, task_dir=None)

    best_alpha = None
    best_mae_smooth = float('inf')
    records = []

    # 3. 扫描所有 α
    for alpha in tqdm.tqdm(alphas, desc="Searching best alpha"):
        _, _, metrics = trainer.evaluate(test_loader, alpha=alpha, log=False)
        mae        = metrics['MAE']
        mae_smooth = metrics['MAE_smooth']
        records.append({'alpha': alpha, 'MAE': mae, 'MAE_smooth': mae_smooth})
        if mae_smooth < best_mae_smooth:
            best_mae_smooth = mae_smooth
            best_alpha = alpha

    # 4. 返回最佳 α + 全部记录
    df = pd.DataFrame(records)
    return best_alpha, df

if __name__ == '__main__':
    # 加载配置
    config = Config()
    config.MODE = 'joint'  # joint 模式
    config.BASE_DIR = Path.cwd() / "strategies" / "joint"
    # 准备 joint 测试集 DataLoader
    dp = DataProcessor(config.DATA_DIR, config.RESAMPLE, config)
    data = dp.prepare_joint_data(config.joint_datasets)
    loaders = create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    test_loader = loaders['test']

    # joint 模式下 best checkpoint 的默认路径
    ckpt_path = config.BASE_DIR / 'checkpoints' / 'task0_best.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {ckpt_path}")

    # 在 0.01 到 0.50 间取 50 个点搜索最佳 α
    alphas = np.arange(0.01, 0.51, 0.01)

    best_alpha, df_results = find_best_alpha(ckpt_path, config, test_loader, alphas)
    print(f"最佳 smoothing α = {best_alpha:.3f} (MAE_smooth = {df_results.loc[df_results.alpha==best_alpha, 'MAE_smooth'].values[0]:.4e})")

    # 保存结果到 disk
    out_csv = config.BASE_DIR / 'smooth_search_results.csv'
    df_results.to_csv(out_csv, index=False)
    print(f"所有 α 的评估结果已保存到: {out_csv}")
