# run_inc.py  (you can keep the filename run_inc_adapter.py for now)
import logging
from pathlib import Path

import torch
import pandas as pd

from utils.config import Config
from utils.utils import set_seed, setup_logging, print_model_summary
from utils.data import DataProcessor, create_dataloaders
from utils.base import SOHLSTM               # <- use plain LSTM (no adapter)
from utils.trainer import Trainer            # unified trainer (we'll de-adapterize next step)
from utils.evaluate import plot_losses, evaluate_incremental_learning

logger = logging.getLogger(__name__)


def fine_tuning(config: Config, dynamic_kl: bool = True):
    """
    Incremental fine-tuning with unified Trainer (NO adapter).
    - dynamic_kl=True: KL-driven schedules for KD/SI
    - dynamic_kl=False: fixed alpha/lambda (pure fine-tuning baseline)
    """
    logger.info("==== Incremental Training (SI + KD; %s) ====",
                "KL-dynamic schedules" if dynamic_kl else "fixed weights")

    inc_dir = config.BASE_DIR
    inc_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    dp = DataProcessor(config.DATA_DIR, config.RESAMPLE, config)
    frames = dp.prepare_incremental_data(config.incremental_datasets)
    loaders = create_dataloaders(frames, config.SEQUENCE_LENGTH, config.BATCH_SIZE)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model: plain LSTM backbone, no adapter
    model = SOHLSTM(
        input_size=3,
        hidden_size=config.HIDDEN_SIZE,
        dropout=config.DROPOUT
    ).to(device)

    # Trainer (adapter args removed; we'll also clean trainer.py in step 2)
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
            # KL smoothing/clipping
            kl_smooth=0.1,
            kl_clip=None
        )
    else:
        # Fixed weights baseline (pure fine-tuning)
        trainer = Trainer(
            model=model,
            device=device,
            config=config,
            task_dir=None,
            si_epsilon=1e-3,
            use_dynamic=False,
            fixed_alpha=0.0,   # no KD
            fixed_lambda=0.0   # no SI
        )

    print_model_summary(model)

    # Train over tasks
    num_tasks = config.NUM_TASKS
    for task_idx in range(num_tasks):
        task_dir = inc_dir / f"task{task_idx}"
        logger.info("--- %s ---", f"task{task_idx}")
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
    # Full evaluation across tasks (plain SOHLSTM)
    evaluate_incremental_learning(config, SOHLSTM, inc_dir, loaders, device)


def main():
    config = Config()
    config.MODE = "incremental"
    set_seed(config.SEED)

    # Output directory for this run
    config.BASE_DIR = Path.cwd() / "inc_sikd"   # <- rename output dir
    config.BASE_DIR.mkdir(parents=True, exist_ok=True)

    setup_logging(config.BASE_DIR)

    # Save config snapshot
    config.save(config.BASE_DIR / "config.yaml")

    # Switch True/False to compare dynamic vs fixed quickly
    fine_tuning(config, dynamic_kl=True)


if __name__ == "__main__":
    main()
