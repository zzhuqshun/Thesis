# run_pure_fine_tuning_with_sitrainer.py
# Pure incremental fine-tuning using the SAME training path as SITrainer.
# Key: alpha_lwf = 0.0, lambda_si = 0.0, and DO NOT call consolidate().
# This keeps the baseline comparable to SI+KD while preserving your logging/eval pipeline.

import logging
import torch
import pandas as pd
from pathlib import Path

from utils.config import Config
from utils.utils import set_seed, setup_logging, print_model_summary
from utils.data import DataProcessor, create_dataloaders
from utils.base import SOHLSTM
from utils.si import SITrainer     # <-- use SITrainer instead of base.Trainer
from utils.evaluate import plot_losses, evaluate_incremental_learning

logger = logging.getLogger(__name__)

def fine_tuning(config: Config):
    """
    Pure incremental fine-tuning (no SI regularization, no KD) using SITrainer.
    We keep the same control flow and logging as your original baseline.
    """
    logger.info("==== Incremental Training (Pure Fine-tuning via SITrainer) ====")

    # Setup directories
    inc_dir = config.BASE_DIR
    inc_dir.mkdir(parents=True, exist_ok=True)

    # Number of tasks
    num_tasks = config.NUM_TASKS
    logger.info("Number of tasks: %d", num_tasks)

    # Data
    dp = DataProcessor(config.DATA_DIR, config.RESAMPLE, config)
    frames = dp.prepare_incremental_data(config.incremental_datasets)
    loaders = create_dataloaders(frames, config.SEQUENCE_LENGTH, config.BATCH_SIZE)

    # Model & trainer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SOHLSTM(3, config.HIDDEN_SIZE, config.DROPOUT).to(device)
    trainer = SITrainer(model=model, device=device, config=config, task_dir=None, si_epsilon=1e-3)
    print_model_summary(model)

    # Train sequentially over tasks
    for task_idx in range(num_tasks):
        task_name = f"task{task_idx}"
        logger.info("--- %s ---", task_name)

        # Task directory
        task_dir = inc_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        trainer.task_dir = task_dir

        # Deterministic per-task seed (same as your baseline)
        set_seed(config.SEED + task_idx)

        # Train current task with SI/KD turned off
        # NOTE: We intentionally DO NOT call trainer.consolidate() for pure fine-tuning.
        history = trainer.train_task(
            train_loader=loaders[f"{task_name}_train"],
            val_loader=loaders[f"{task_name}_val"],
            task_id=task_idx,
            alpha_lwf=0.0,     # KD disabled
            lambda_si=0.0,     # SI penalty disabled
            adapter_scale=None # adapter off (no effect for plain SOHLSTM)
        )

        # Save history & plots
        pd.DataFrame(history).to_csv(task_dir / 'training_history.csv', index=False)
        plot_losses(history, task_dir)

        logger.info("Task %d completed.", task_idx)

    logger.info("==== Incremental Training Complete ====")
    # Keep your existing evaluation entrypoint (same loaders/device)
    evaluate_incremental_learning(config, None, inc_dir, loaders, device)


def main():
    """
    Entry point for pure fine-tuning via SITrainer.
    Keeps the same logging/seeding structure as your original script.
    """
    config = Config()
    config.MODE = "incremental"
    set_seed(config.SEED)

    # Output base directory
    config.BASE_DIR = Path.cwd() / 'pure_fine_tuning_sitrainer'
    config.BASE_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(config.BASE_DIR)

    # Save config snapshot
    config.save(config.BASE_DIR / 'config.yaml')

    # Run
    fine_tuning(config)


if __name__ == '__main__':
    main()
