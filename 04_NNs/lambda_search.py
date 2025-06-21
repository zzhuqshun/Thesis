"""
lambda_search.py
Searches optimal EWC lambda1 / lambda2 with Optuna while caching all data
to avoid heavy disk I/O on every trial.
"""

import optuna
import torch
from pathlib import Path
from model import (
    Config, DataProcessor, create_dataloaders, SOHLSTM,
    Trainer, set_seed, EWC, get_predictions
)
from sklearn.metrics import mean_absolute_error

# ──────────────────────────
# 0.  Paths & constants
# ──────────────────────────
DATA_DIR = "../01_Datenaufbereitung/Output/Calculated/"
TMP_ROOT = Path("lambda_optuna")
TMP_ROOT.mkdir(parents=True, exist_ok=True)

INC_BASE_TRAIN_IDS    = ["03", "05", "07", "27"]
INC_BASE_VAL_IDS      = ["01"]
INC_UPDATE1_TRAIN_IDS = ["21", "23", "25"]
INC_UPDATE1_VAL_IDS   = ["19"]
INC_UPDATE2_TRAIN_IDS = ["09", "11", "15", "29"]
INC_UPDATE2_VAL_IDS   = ["13"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ──────────────────────────
# 1.  Hyper-parameters you already tuned for the model itself
# ──────────────────────────
best_params = {
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
}
VAL_SCALE   = 0.005
DELTA_SCALE = 0.002 
# ──────────────────────────
# 2.  ONE-SHOT: load + resample + scale ALL data  (no more I/O later)
# ──────────────────────────
print("Loading all parquet files once …")
dp_cached = DataProcessor(
    DATA_DIR,
    best_params.get("RESAMPLE", "10min"),
    Config(**best_params),
    INC_BASE_TRAIN_IDS, INC_BASE_VAL_IDS,
    INC_UPDATE1_TRAIN_IDS, INC_UPDATE1_VAL_IDS,
    INC_UPDATE2_TRAIN_IDS, INC_UPDATE2_VAL_IDS,
)
data_cached = dp_cached.prepare_data()
print("Data cached in memory.")

def build_loaders(cfg: Config):
    """Convert cached DataFrames → fresh DataLoaders for this trial."""
    return create_dataloaders(
        data_cached,
        cfg.SEQUENCE_LENGTH,
        cfg.BATCH_SIZE,
    )

# ──────────────────────────
# 3.  Prepare base model once, get baseline₀
# ──────────────────────────
def prepare_base(config: Config):
    print("Preparing base task (no EWC) …")
    base_dir = TMP_ROOT / "base"
    base_dir.mkdir(exist_ok=True)

    loaders = build_loaders(config)

    model   = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    trainer = Trainer(model, device, config, checkpoint_dir=base_dir)

    set_seed(config.SEED)
    trainer.train_task(
        train_loader=loaders["base_train"],
        val_loader=loaders["base_val"],
        task_id=0,
        apply_ewc=False,
        resume=False,
    )
    trainer.consolidate(loaders["base_train"])

    ckpt0 = base_dir / "task0_best.pt"
    state = torch.load(ckpt0, map_location=device)
    torch.save(
        {
            "model_state": state["model_state"],
            "ewc_tasks": [
                {
                    "params": {n: p.cpu() for n, p in e.params.items()},
                    "fisher": {n: f.cpu() for n, f in e.fisher.items()},
                }
                for e in trainer.ewc_tasks
            ],
        },
        base_dir / "base_checkpoint.pt",
    )

    preds0, tgts0 = get_predictions(trainer.model, loaders["test_base"], device)
    baseline0 = mean_absolute_error(tgts0, preds0)

    print(f"Base prepared. MAE(test_base) = {baseline0:.6f}")
    return base_dir / "base_checkpoint.pt", baseline0

# ──────────────────────────
# 4.  Incremental training for a given λ₁/λ₂
# ──────────────────────────
def train_incremental(
    config: Config,
    lambda1: float,
    lambda2: float,
    trial_id: int,
    base_ckpt,
    baseline0,
    loaders,        # cached loaders
):
    # Load base model + Fisher
    trainer = Trainer(
        SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device),
        device, config, checkpoint_dir=None
    )
    base_state = torch.load(base_ckpt, map_location=device)
    trainer.model.load_state_dict(base_state["model_state"])
    trainer.ewc_tasks = []
    for ewc_data in base_state["ewc_tasks"]:
        e = EWC.__new__(EWC)
        e.params  = {n: p.to(device) for n, p in ewc_data["params"].items()}
        e.fisher  = {n: f.to(device) for n, f in ewc_data["fisher"].items()}
        trainer.ewc_tasks.append(e)

    # ── Phase 1 : Update-1 ──────────────────────────────────────────
    set_seed(config.SEED)
    config.EWC_LAMBDA = lambda1
    trainer.config    = config
    up1_dir = TMP_ROOT / f"trial_{trial_id}" / "update1"
    up1_dir.mkdir(parents=True, exist_ok=True)
    trainer.checkpoint_dir = up1_dir

    hist1 = trainer.train_task(
        train_loader=loaders["update1_train"],
        val_loader=loaders["update1_val"],
        task_id=1,
        apply_ewc=True,
        resume=False,
    )
    val1 = min(hist1["val_loss"])
    trainer.consolidate(loaders["update1_train"])

    best1 = up1_dir / "task1_best.pt"
    trainer.model.load_state_dict(torch.load(best1, map_location=device)["model_state"])
    preds1_up1, tgts1_up1 = get_predictions(trainer.model, loaders["test_update1"], device)
    baseline1 = mean_absolute_error(tgts1_up1, preds1_up1)

    preds0_1, tgts0_1 = get_predictions(trainer.model, loaders["test_base"], device)
    delta0_after1 = mean_absolute_error(tgts0_1, preds0_1) - baseline0

    # ── Phase 2 : Update-2 ──────────────────────────────────────────
    set_seed(config.SEED)
    config.EWC_LAMBDA = lambda2
    trainer.config    = config
    up2_dir = TMP_ROOT / f"trial_{trial_id}" / "update2"
    up2_dir.mkdir(parents=True, exist_ok=True)
    trainer.checkpoint_dir = up2_dir

    hist2 = trainer.train_task(
        train_loader=loaders["update2_train"],
        val_loader=loaders["update2_val"],
        task_id=2,
        apply_ewc=True,
        resume=False,
    )
    val2 = min(hist2["val_loss"])
    trainer.consolidate(loaders["update2_train"])

    best2 = up2_dir / "task2_best.pt"
    trainer.model.load_state_dict(torch.load(best2, map_location=device)["model_state"])

    preds0_2, tgts0_2 = get_predictions(trainer.model, loaders["test_base"], device)
    delta0_after2 = mean_absolute_error(tgts0_2, preds0_2) - baseline0

    preds1_2, tgts1_2 = get_predictions(trainer.model, loaders["test_update1"], device)
    delta1_after2 = mean_absolute_error(tgts1_2, preds1_2) - baseline1

    # free memory (important for many trials on GPU)
    del trainer
    torch.cuda.empty_cache()

    return val1, val2, delta0_after1, delta0_after2, delta1_after2

# ──────────────────────────
# 5.  Optuna objective
# ──────────────────────────
def objective_incremental(trial, best_config, base_ckpt, baseline0):
    config = Config(**best_config)

    lambda1 = trial.suggest_float("lambda1", 1e-2, 1e5, log=True)
    lambda2 = trial.suggest_float("lambda2", 1e-2, 1e5, log=True)

    loaders = build_loaders(config)

    val1, val2, d01, d02, d12 = train_incremental(
        config, lambda1, lambda2, trial.number,
        base_ckpt, baseline0, loaders
    )

    score = (
        (val1         / VAL_SCALE)   +
        (val2         / VAL_SCALE)   +
        (abs(d01)     / DELTA_SCALE) +
        (abs(d02)     / DELTA_SCALE) +
        (abs(d12)     / DELTA_SCALE)
    ) / 5.0

    # log raw metrics
    trial.set_user_attr("val_update1",   val1)
    trial.set_user_attr("val_update2",   val2)
    trial.set_user_attr("delta0_after1", d01)
    trial.set_user_attr("delta0_after2", d02)
    trial.set_user_attr("delta1_after2", d12)

    return score

# ──────────────────────────
# 6.  Main entry
# ──────────────────────────
if __name__ == "__main__":
    base_ckpt, baseline0 = prepare_base(Config(**best_params))

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: objective_incremental(t, best_params, base_ckpt, baseline0),
        n_trials=50,
    )

    df = study.trials_dataframe()
    df.to_csv(TMP_ROOT / "incremental_trials.csv", index=False)
    print("Search finished.")
    print("Best λ values:", study.best_trial.params)
    print("Best score:", study.best_value)
