'''
lambda_search.py  
Tune **lambda0** (base-task EWC penalty) & **lambda1** (update-1 penalty) with Optuna.
No EWC is applied **after** update-2, so lambda2 is implicitly 0 / None.
Data are cached once in RAM; every trial only re-instantiates DataLoaders & models.
'''

import shutil
from pathlib import Path
from typing import Tuple

import optuna
import torch
from sklearn.metrics import mean_absolute_error

from lstm import (
    Config,
    DataProcessor,
    create_dataloaders,
    SOHLSTM,
    Trainer,
    set_seed,
    EWC)

# ──────────────────────────
# 0.  Paths & constants
# ──────────────────────────
DATA_DIR = "../01_Datenaufbereitung/Output/Calculated/"
TMP_ROOT = Path("lambda_optuna_search")
TMP_ROOT.mkdir(parents=True, exist_ok=True)

INC_BASE_TRAIN_IDS = ["03", "05", "07", "27"]
INC_BASE_VAL_IDS = ["01"]
INC_UPDATE1_TRAIN_IDS = ["21", "23", "25"]
INC_UPDATE1_VAL_IDS = ["19"]
INC_UPDATE2_TRAIN_IDS = ["09", "11", "15", "29"]
INC_UPDATE2_VAL_IDS = ["13"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ──────────────────────────
# 1.  Model hyper-parameters (already tuned)
# ──────────────────────────
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
    "RESAMPLE": "10min"
}

# ──────────────────────────
# 2.  ONE-SHOT: load + resample + scale  (cached in RAM)
# ──────────────────────────
print("[INFO] Loading parquet files & caching …")
_dp_cached = DataProcessor(
    DATA_DIR,
    BASE_MODEL_PARAMS.get("RESAMPLE", "10min"),
    Config(**BASE_MODEL_PARAMS),
    INC_BASE_TRAIN_IDS,
    INC_BASE_VAL_IDS,
    INC_UPDATE1_TRAIN_IDS,
    INC_UPDATE1_VAL_IDS,
    INC_UPDATE2_TRAIN_IDS,
    INC_UPDATE2_VAL_IDS,
)
data_cached = _dp_cached.prepare_data()
print("[INFO] Data cached in memory.")

def build_loaders(cfg: Config):
    """Create fresh DataLoaders from the cached DataFrames for one trial."""
    return create_dataloaders(data_cached, cfg.SEQUENCE_LENGTH, cfg.BATCH_SIZE)

# ──────────────────────────
# 3.  Train **base task only** once; return checkpoint & baseline MAE
# ──────────────────────────

from pathlib import Path
from typing import Tuple

import torch
from sklearn.metrics import mean_absolute_error

def prepare_base(cfg: Config) -> Tuple[Path, float]:
    base_dir = TMP_ROOT / "base"
    base_dir.mkdir(exist_ok=True)

    # 1. 把 best_ckpt 提前声明，后续逻辑都会用到
    best_ckpt = base_dir / "task0_best.pt"

    # 2. 数据 & 模型 & Trainer 始终要先构建，好让后面能做评估
    loaders = build_loaders(cfg)

    model   = SOHLSTM(3, cfg.HIDDEN_SIZE, cfg.NUM_LAYERS, cfg.DROPOUT).to(device)
    trainer = Trainer(model, device, cfg, checkpoint_dir=base_dir)

    if best_ckpt.exists():
        # ---- 已有模型：跳过训练，直接载入参数 ----
        print("[INFO] Found existing checkpoint — skipping training.")
        state = torch.load(best_ckpt, map_location=device)
        trainer.model.load_state_dict(state["model_state"])
    else:
        # ---- 没有模型：常规训练流程 ----
        print("[INFO] Preparing base task (no EWC)…")
        set_seed(cfg.SEED)
        trainer.train_task(
            train_loader=loaders["base_train"],
            val_loader=loaders["base_val"],
            task_id=0,
            apply_ewc=False,
            resume=False,
        )

        # Fisher for base; weight will be overwritten inside each trial
        trainer.consolidate(loaders["base_train"], task_id=0, lam=0.0)

        # 训练完会在内部保存 best_ckpt，无需手动调用

    # 3. 统一做测试评估
    preds, tgts = get_predictions(trainer.model, loaders["test_base"], device)
    mae_base = mean_absolute_error(tgts, preds)
    print(f"[INFO] Base ready. MAE(test_base) = {mae_base:.6f}")

    return best_ckpt, mae_base


# ──────────────────────────
# 4.  Incremental routine for **one** (λ0, λ1) pair
# ──────────────────────────

def train_incremental(
    cfg: Config,
    lambda0: float,
    lambda1: float,
    alpha1: float,
    alpha2: float,
    trial_id: int,
    base_ckpt: Path,
    loaders,
):
    """Return ACC & BWT for the given lambda pair."""
    tmp_ckpt = TMP_ROOT / f"trial_{trial_id}"
    trainer = Trainer(
        SOHLSTM(3, cfg.HIDDEN_SIZE, cfg.NUM_LAYERS, cfg.DROPOUT).to(device),
        device,
        cfg,
        checkpoint_dir=tmp_ckpt,
    )

    # —— restore base weights & Fisher ——
    state = torch.load(base_ckpt, map_location=device)
    trainer.model.load_state_dict(state["model_state"])
    trainer.ewc_tasks = []
    for e_data in state["ewc_tasks"]:
        e = EWC.__new__(EWC)
        e.params = {n: p.to(device) for n, p in e_data["params"].items()}
        e.fisher = {n: f.to(device) for n, f in e_data["fisher"].items()}
        e.lam = lambda0          # ★ overwrite with current λ0
        trainer.ewc_tasks.append(e)
    
    import copy
    trainer.old_model = copy.deepcopy(trainer.model).to(device).eval()
    for p in trainer.old_model.parameters():
        p.requires_grad_(False)

    # ——— metrics BEFORE any update ———
    preds_b0, tgts_b0 = get_predictions(trainer.model, loaders["test_base"], device)
    mae_b0 = mean_absolute_error(tgts_b0, preds_b0)

    # ────────────────── Phase-1 : update-1 ──────────────────
    trainer.train_task(
        loaders["update1_train"], loaders["update1_val"],
        task_id=1, apply_ewc=True, 
        alpha_lwf=alpha1, resume=False,
    )
    best1 = trainer.checkpoint_dir / "task1_best.pt"
    state1 = torch.load(best1, map_location=device)
    trainer.model.load_state_dict(state1["model_state"])
    trainer.consolidate(loaders["update1_train"], task_id=1, lam=lambda1)

    preds_u1_after, tgts_u1_after = get_predictions(trainer.model, loaders["test_update1"], device)
    mae_u1_after = mean_absolute_error(tgts_u1_after, preds_u1_after)

    # ────────────────── Phase-2 : update-2  (no extra penalty) ──────────────────
    trainer.train_task(
        loaders["update2_train"], loaders["update2_val"],
        task_id=2, apply_ewc=True, 
        alpha_lwf=alpha2, resume=False,  # will use existing λ0 & λ1 only
    )
    # 不再 consolidate update-2，因为之后不再继续训练
    best2 = trainer.checkpoint_dir / "task2_best.pt"
    if best2.exists():
        trainer.model.load_state_dict(torch.load(best2, map_location=device)["model_state"])
        
    preds_b2, tgts_b2 = get_predictions(trainer.model, loaders["test_base"], device)
    mae_b2 = mean_absolute_error(tgts_b2, preds_b2)
    preds_u1_2, tgts_u1_2 = get_predictions(trainer.model, loaders["test_update1"], device)
    mae_u1_2 = mean_absolute_error(tgts_u1_2, preds_u1_2)
    preds_u2, tgts_u2 = get_predictions(trainer.model, loaders["test_update2"], device)
    mae_u2 = mean_absolute_error(tgts_u2, preds_u2)

    # ────────────────── metrics ──────────────────
    ACC = -(mae_b2 + mae_u1_2 + mae_u2) / 3.0  # 越大越好（即 MAE 越小）
    BWT = ((mae_b2 - mae_b0) + (mae_u1_2 - mae_u1_after)) / 2.0  # 越小越好

    # clean tmp files to save disk
    shutil.rmtree(tmp_ckpt, ignore_errors=True)
    torch.cuda.empty_cache()

    return ACC, BWT

# ──────────────────────────
# 5.  Optuna objective
# ──────────────────────────

def objective(trial: optuna.trial.Trial, cfg_dict, base_ckpt: Path):
    set_seed(cfg_dict["SEED"] + trial.number)
    cfg = Config(**cfg_dict)

    lambda0 = trial.suggest_float("lambda0", 1e1, 1e4, log=True)
    lambda1 = trial.suggest_float("lambda1", 1e1, 1e4, log=True)
    
    # alpha1  = trial.suggest_float("alpha1", 0.05, 5.0, log=True)   # task1
    # alpha2  = trial.suggest_float("alpha2", 0.05, 5.0, log=True)   # task2
    alpha1 = 0.0
    alpha2 = 0.0

    loaders = build_loaders(cfg)
    ACC, BWT = train_incremental(cfg, lambda0, lambda1, 
                                 alpha1, alpha2,
                                 trial.number, base_ckpt, loaders)

    trial.set_user_attr("ACC", ACC)
    trial.set_user_attr("BWT", BWT)
    return ACC, BWT  # directions: ["maximize", "minimize"]

# ──────────────────────────
# 6.  Run search & save results
# ──────────────────────────
if __name__ == "__main__":
    base_ckpt, _ = prepare_base(Config(**BASE_MODEL_PARAMS))

    from optuna.samplers import NSGAIISampler

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=NSGAIISampler(seed=0),
    )

    study.optimize(lambda t: objective(t, BASE_MODEL_PARAMS, base_ckpt), n_trials=50)

    # ─── save results ───
    results_dir = TMP_ROOT / "optuna_results"
    results_dir.mkdir(exist_ok=True)

    df_all = study.trials_dataframe(attrs=("number", "values", "params", "user_attrs"))
    df_all.to_csv(results_dir / "all_trials.csv", index=False)

    df_pareto = df_all.loc[[t.number for t in study.best_trials]]
    df_pareto.to_csv(results_dir / "pareto_front.csv", index=False)

    print("\n[INFO] Search finished. Results saved to", results_dir.resolve())
