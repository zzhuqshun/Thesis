# run_pure_fine_tuning.py
import logging
from pathlib import Path

import torch
import pandas as pd

from utils.config import Config
from utils.utils import set_seed, setup_logging, print_model_summary
from utils.data import DataProcessor, create_dataloaders
from utils.base import SOHLSTMAdapter
from utils.trainer import Trainer  # <- unified trainer
from utils.evaluate import plot_losses, evaluate_incremental_learning

logger = logging.getLogger(__name__)


def fine_tuning(config: Config, dynamic_kl: bool = True):
    """
    Fine-tuning with unified Trainer.
    - dynamic_kl=True: KL-driven schedules (KD↓, SI↓, Adapter↑)
    - dynamic_kl=False: fixed alpha/lambda/adapter (baseline pure fine-tuning)
    """
    logger.info("==== Incremental Training (Adapter + %s) ====",
                "KL-dynamic schedules" if dynamic_kl else "fixed weights")

    inc_dir = config.BASE_DIR
    inc_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    dp = DataProcessor(config.DATA_DIR, config.RESAMPLE, config)
    frames = dp.prepare_incremental_data(config.incremental_datasets)
    loaders = create_dataloaders(frames, config.SEQUENCE_LENGTH, config.BATCH_SIZE)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model: Adapter version
    # input_size=3 for C/V/T; keep your hidden/dropout from config
    model = SOHLSTMAdapter(
        input_size=3,
        hidden_size=config.HIDDEN_SIZE,
        dropout=config.DROPOUT,
        adapter_bottleneck=32,
        adapter_dropout=0.0
    ).to(device)

    # Trainer: unified, switch by dynamic_kl
    # NOTE: You can tune these schedule hyperparams via Optuna later.
    if dynamic_kl:
        trainer = Trainer(
            model=model,
            device=device,
            config=config,
            task_dir=None,
            si_epsilon=1e-3,
            use_dynamic=True,
            # KD: alpha = alpha_max * exp(-alpha_c * KL)
            alpha_max=1.0, alpha_c=2.0,
            # SI: lambda = lam_base * exp(-lam_c * KL), floored by lam_min
            lam_base=1.0, lam_c=1.0, lam_min=0.0,
            # Adapter: scale = adapt_max * sigmoid(adapt_k * (KL - adapt_tau))
            adapt_max=1.0, adapt_k=5.0, adapt_tau=0.1,
            # KL smoothing/clipping
            kl_smooth=0.1,   # EMA smoothing; set 0.0 to disable
            kl_clip=None     # e.g., 10.0 to hard clip
        )
    else:
        # Fixed weights baseline (pure fine-tuning style)
        trainer = Trainer(
            model=model,
            device=device,
            config=config,
            task_dir=None,
            si_epsilon=1e-3,
            use_dynamic=False,
            fixed_alpha=0.0,     # no KD
            fixed_lambda=0.0,    # no SI
            fixed_adapter=0.0    # adapter closed (residual only)
        )

    print_model_summary(model)

    # Train over tasks
    num_tasks = config.NUM_TASKS
    for task_idx in range(num_tasks):
        task_dir = inc_dir / f"task{task_idx}"
        task_dir.mkdir(parents=True, exist_ok=True)
        trainer.task_dir = task_dir

        # Different seed per task for reproducibility
        set_seed(config.SEED + task_idx)

        history = trainer.train_task(
            train_loader=loaders[f"task{task_idx}_train"],
            val_loader=loaders[f"task{task_idx}_val"],
            task_id=task_idx
        )

        # Persist history and plots
        pd.DataFrame(history).to_csv(task_dir / "training_history.csv", index=False)
        plot_losses(history, task_dir)

        logger.info("Task %d completed.", task_idx)

    logger.info("==== Incremental Training Complete ====")
    # Full evaluation across tasks
    evaluate_incremental_learning(config, None, inc_dir, loaders, device)


def main():
    config = Config()
    config.MODE = "incremental"
    set_seed(config.SEED)

    # Output directory for this run
    config.BASE_DIR = Path.cwd() / "adapter_ft"
    config.BASE_DIR.mkdir(parents=True, exist_ok=True)
    config.ADAPTER = False
    setup_logging(config.BASE_DIR)

    
    # Save config snapshot
    config.save(config.BASE_DIR / "config.yaml")

    # Switch True/False to compare dynamic vs fixed quickly
    fine_tuning(config, dynamic_kl=config.ADAPTER)


if __name__ == "__main__":
    main()
