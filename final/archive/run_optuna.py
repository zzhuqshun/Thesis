import argparse
import logging
from pathlib import Path
from typing import Dict
import optuna
import torch
import torch.nn.functional as F
import yaml  # 新增

from utils.config import Config
from utils.base import SOHLSTM
from utils.data import DataProcessor, create_dataloaders
from utils.si import SITrainer
from utils.utils import set_seed, setup_logging

logger = logging.getLogger("si-kd-optuna")


@torch.no_grad()
def eval_mae(model, loader):
    """Mean Absolute Error over a loader."""
    model.eval()
    device = next(model.parameters()).device
    mae_sum, n = 0.0, 0
    for x, y in loader:
        pred = model(x.to(device))
        mae_sum += F.l1_loss(pred, y.to(device), reduction="sum").item()
        n += y.numel()
    return mae_sum / max(1, n)


@torch.no_grad()
def evaluate_checkpoints(trial_dir: Path, loaders: Dict[str, torch.utils.data.DataLoader], cfg: Config, device):
    """Load task{t}_best.pt and compute val MAEs."""
    def _load(ckpt: Path):
        model = SOHLSTM(input_size=3, hidden_size=cfg.HIDDEN_SIZE, dropout=cfg.DROPOUT).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
        return model.eval()

    metrics = {}
    for t in range(3):
        ck = trial_dir / f"task{t}" / f"task{t}_best.pt"
        if not ck.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ck}")
        metrics[f"val_mae_task{t}"] = eval_mae(_load(ck), loaders[f"task{t}_val"])
    return metrics


def cleanup_checkpoints(trial_dir: Path):
    for t in range(3):
        p = trial_dir / f"task{t}" / f"task{t}_best.pt"
        if p.exists():
            p.unlink()


def run_three_tasks(loaders, cfg: Config, device, si_lambda, si_epsilon, kd_alpha):
    """Train Task0->Task1->Task2 using SI + KD."""
    model = SOHLSTM(input_size=3, hidden_size=cfg.HIDDEN_SIZE, dropout=cfg.DROPOUT).to(device)
    trainer = SITrainer(model, device, cfg, None, si_epsilon)

    for t in range(3):
        set_seed(cfg.SEED + t)
        trainer.task_dir = cfg.BASE_DIR / f"task{t}"
        trainer.task_dir.mkdir(parents=True, exist_ok=True)

        alpha_lwf = kd_alpha if t > 0 else 0.0
        lambda_si = si_lambda if t > 0 else 0.0
        logger.info("==== Task %d: alpha=%f, lambda_si=%f ====", t, alpha_lwf, lambda_si)

        trainer.train_task(loaders[f"task{t}_train"], loaders[f"task{t}_val"], t,
                           alpha_lwf=alpha_lwf, lambda_si=lambda_si, adapter_scale=None)
        trainer.consolidate(t)
    return cfg.BASE_DIR


def make_objective(cfg: Config, device_str: str, loaders, outdir: Path):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.Trial):
        trial_dir = outdir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        cfg_trial = Config()
        cfg_trial.__dict__.update(cfg.__dict__)
        cfg_trial.BASE_DIR, cfg_trial.MODE = trial_dir, "incremental"

        si_lambda = trial.suggest_float("si_lambda", 1e-3, 1e2, log=True)
        si_epsilon = trial.suggest_float("si_epsilon", 1e-4, 1e-1, log=True)
        kd_alpha   = trial.suggest_float("kd_alpha",   1e-4, 1.0, log=True)
        logger.info("Trial %d: si_lambda=%.6f, si_epsilon=%.6f, kd_alpha=%.6f",
                    trial.number, si_lambda, si_epsilon, kd_alpha)
        run_three_tasks(loaders, cfg_trial, device, si_lambda, si_epsilon, kd_alpha)
        metrics = evaluate_checkpoints(trial_dir, loaders, cfg_trial, device)

        trial.set_user_attr("metrics", metrics)
        cleanup_checkpoints(trial_dir)

        return sum(metrics.values()) / 3.0  # Macro MAE

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna: search SI(lambda,epsilon) & KD(alpha).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--study_name", default="si_kd_search_sohlstm")
    parser.add_argument("--storage", default=None)
    parser.add_argument("--outdir", default="si_kd_optuna")
    args = parser.parse_args()

    cfg = Config()
    outdir = Path(args.outdir)
    set_seed(cfg.SEED)
    outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir, logging.INFO, "optuna.log")

    dp = DataProcessor(cfg.DATA_DIR, cfg.RESAMPLE, cfg)
    loaders = create_dataloaders(dp.prepare_incremental_data(cfg.incremental_datasets),
                                 cfg.SEQUENCE_LENGTH, cfg.BATCH_SIZE)

    study = optuna.create_study(
        study_name=args.study_name, direction="minimize", load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, n_startup_trials=10),
    )
    study.optimize(make_objective(cfg, args.device, loaders, outdir),
                   n_trials=args.n_trials, gc_after_trial=True)

    logger.info("Best macro MAE: %.6f", study.best_value)
    logger.info("Best params:\n%s", yaml.safe_dump(study.best_trial.params, sort_keys=False))  # YAML 日志

    # 保存 YAML 格式
    with open(outdir / "best_params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(study.best_trial.params, f, sort_keys=False)

    with open(outdir / "trials.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump([
            {
                "number": t.number,
                "state": str(t.state),
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs
            }
            for t in study.trials
        ], f, sort_keys=False)


if __name__ == "__main__":
    main()
