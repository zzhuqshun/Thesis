import shutil
from pathlib import Path
import optuna
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error

from lstm import (
    Config,
    DataProcessor,
    create_dataloaders,
    SOHLSTM,
    Trainer,
    EWC,
    set_seed,
)

DATA_DIR = "../01_Datenaufbereitung/Output/Calculated/"
TMP_ROOT = Path("inc_search")
TMP_ROOT.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BASE_MODEL_PARAMS = {
    "SEQUENCE_LENGTH": 720,
    "HIDDEN_SIZE": 128,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.3,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-6,
    "EPOCHS": 200,
    "PATIENCE": 20,
    "SEED": 42,
    "SCALER": "RobustScaler",
    "RESAMPLE": "10min",
    "MODE": "incremental"
}

def build_loaders(cfg: Config, data_cached=None):
    if data_cached is None:
        processor = DataProcessor(DATA_DIR, cfg)
        data_cached = processor.prepare_data()
    return create_dataloaders(data_cached, cfg), data_cached

def get_mae(model, loader, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            preds.extend(y_pred.cpu().numpy())
            tgts.extend(y.cpu().numpy())
    return mean_absolute_error(tgts, preds)

def train_and_save_base(cfg, data_cached, save_path):
    model = SOHLSTM(len(cfg.FEATURES_COLS), cfg.HIDDEN_SIZE, cfg.NUM_LAYERS, cfg.DROPOUT).to(device)
    trainer = Trainer(model, device, cfg)
    loaders, _ = build_loaders(cfg, data_cached)
    
    history = trainer.train_task(loaders['base_train'], loaders['base_val'], task_id=0, apply_ewc=False, alpha_lwf=0.0)
    best_val_loss = min(h['val_loss'] for h in history)
    print(f"[INFO] Best validation loss for base task: {best_val_loss:.4e}")
    
    
    trainer.consolidate(loaders['base_train'], lam=1.0)
    torch.save({
        'model_state': trainer.model.state_dict(),
        'ewc_params': [ewc.params for ewc in trainer.ewc_tasks],
        'ewc_fisher': [ewc.fisher for ewc in trainer.ewc_tasks],
    }, save_path)

def incremental_objective(trial, cfg_dict, data_cached, base_ckpt_path):
    set_seed(cfg_dict["SEED"] + trial.number)
    cfg = Config(**cfg_dict)

    lambda0 = trial.suggest_float("lambda0", 1e1, 1e4, log=True)
    lambda1 = trial.suggest_float("lambda1", 1e1, 1e4, log=True)

    loaders, _ = build_loaders(cfg, data_cached)
    model = SOHLSTM(len(cfg.FEATURES_COLS), cfg.HIDDEN_SIZE, cfg.NUM_LAYERS, cfg.DROPOUT).to(device)
    trainer = Trainer(model, device, cfg)

    # 加载 base 阶段参数和 Fisher
    base_state = torch.load(base_ckpt_path, map_location=device)
    trainer.model.load_state_dict(base_state['model_state'])
    trainer.ewc_tasks = []
    for params, fisher in zip(base_state['ewc_params'], base_state['ewc_fisher']):
        ewc = EWC.__new__(EWC)
        ewc.params = {n: p.to(device) for n, p in params.items()}
        ewc.fisher = {n: f.to(device) for n, f in fisher.items()}
        ewc.lam = lambda0
        ewc.device = device
        trainer.ewc_tasks.append(ewc)

    # --- update1 phase ---
    trainer.train_task(
        loaders['update1_train'], loaders['update1_val'],
        task_id=1, apply_ewc=True, alpha_lwf=0.0
    )
    trainer.consolidate(loaders['update1_train'], lambda1)

    # --- update2 phase ---
    trainer.train_task(
        loaders['update2_train'], loaders['update2_val'],
        task_id=2, apply_ewc=True, alpha_lwf=0.0
    )

    # 评估
    mae_base = get_mae(trainer.model, loaders['test_base'], device)
    mae_u1 = get_mae(trainer.model, loaders['test_update1'], device)
    mae_u2 = get_mae(trainer.model, loaders['test_update2'], device)
    acc = (mae_base + mae_u1 + mae_u2) / 3.0

    trial.set_user_attr("mae_base", mae_base)
    trial.set_user_attr("mae_update1", mae_u1)
    trial.set_user_attr("mae_update2", mae_u2)
    return acc 
#(sum_Q)

if __name__ == "__main__":
    print("[INFO] Caching data ...")
    processor = DataProcessor(DATA_DIR, Config(**BASE_MODEL_PARAMS))
    data_cached = processor.prepare_data()
    print("[INFO] Data cached.")

    # 只训练一次 base
    base_ckpt_path = TMP_ROOT / "base_ckpt.pth"
    if not base_ckpt_path.exists():
        print("[INFO] Training base task and saving base checkpoint ...")
        train_and_save_base(Config(**BASE_MODEL_PARAMS), data_cached, base_ckpt_path)
        print("[INFO] Base checkpoint saved:", base_ckpt_path)
    else:
        print("[INFO] Base checkpoint already exists.")

    study = optuna.create_study(
        direction="minimize",
    )
    study.optimize(lambda t: incremental_objective(t, BASE_MODEL_PARAMS, data_cached, base_ckpt_path), n_trials=50)

    results_dir = TMP_ROOT / "optuna_results"
    results_dir.mkdir(exist_ok=True)
    df_all = study.trials_dataframe(attrs=("number", "values", "params", "user_attrs"))
    df_all.to_csv(results_dir / "all_trials.csv", index=False)
    print("\n[INFO] Search finished. Results saved to", results_dir.resolve())
