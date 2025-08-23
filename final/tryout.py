# run_inc.py
import os
import logging
import torch
import copy
import pandas as pd
from pathlib import Path

from utils.config import Config
from utils.utils import set_seed, setup_logging, print_model_summary
from utils.data import DataProcessor, create_dataloaders
from utils.base import SOHLSTM, IncTrainer          # <-- use IncTrainer from utils.base
from utils.evaluate import plot_losses, evaluate_incremental_learning

logger = logging.getLogger(__name__)

def inc_training(config: Config):
    """
    Incremental training using IncTrainer (SI + KD with KL-driven scheduling).
    This follows your original control flow: per-task loop, save history, plot, evaluate later.
    """
    logger.info("==== Incremental Training (SI + KD via IncTrainer) ====")

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
    
    # IncTrainer handles SI+KD scheduling internally based on config.*
    trainer = IncTrainer(
        model=model,
        device=device,
        config=config,
        inc_dir=inc_dir
    )
    print_model_summary(model)
    # Sequential tasks
    for task_idx in range(num_tasks):
        task_name = f"task{task_idx}"
        task_dir = inc_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # per-task seed
        set_seed(config.SEED + task_idx)
        # if task_idx == 1:
        #     config.SI_LAMBDA = 0.0015831525202184133
        #     config.SI_EPSILON = 0.013481575603601416
        #     config.KD_LAMBDA = 0.001171142666747359
       
        # if task_idx == 2:
        #     config.SI_LAMBDA = 0.0001776934210981567
        #     config.SI_EPSILON = 0.00801632436628857
        #     config.KD_LAMBDA = 0.0021646846539953237
        # >>> train one task (IncTrainer already saves best checkpoint under inc_dir/task{t}/task{t}_best.pt)
        hist = trainer.train_task(
            train_loader=loaders[f"{task_name}_train"],
            val_loader=loaders[f"{task_name}_val"],
            task_id=task_idx
        )
        
        # >>> save history csv (for plotting kd_pct/si_pct/kl curves)
        csv_path = task_dir / "history.csv"
        pd.DataFrame(hist).to_csv(csv_path, index=False)
        plot_losses(csv_path, task_dir, title=f"Task {task_idx} Training")

    logger.info("==== Incremental Training Complete ====")
    logger.info("Evaluating final model...")
    # Evaluate all checkpoints with your existing evaluator
    evaluate_incremental_learning(config, SOHLSTM, inc_dir, loaders, device)


def main():
    """
    Main entrypoint for running the incremental training.
    """
    config = Config()
    config.MODE = "incremental"

    # config.SKIP_TASK0 = True
    
    # ---- toggles: enable/disable components here ----
    # SI / KD / KL switches
    config.USE_SI = True
    config.USE_KD = True
    # config.USE_KL = True
    # config.USE_SI = False
    # config.USE_KD = False
    config.USE_KL = False

    # KL mode & temperature
    config.KL_MODE = 'both'   # 'input' | 'hidden' | 'both'
    config.TAU = 0.5
    # config.S_MIN = 0.15
    # config.S_MAX = None

    # SI strength range (auto-scheduled by KL)
    config.SI_LAMBDA = 0.003002606845241161
    config.SI_EPSILON = 0.0951554140693468
    config.SI_WARMUP_EPOCHS = 10
    # KD strength range (auto-scheduled by KL)
    config.KD_LAMBDA = 0.009542713658753916
    config.KD_LOSS  = 'mse'    # 'mse' | 'l1' | 'smoothl1'
    config.KD_WARMUP_EPOCHS = 10
    # Training basics (keep your existing settings)
    # config.EPOCHS, config.BATCH_SIZE, config.LEARNING_RATE, config.WEIGHT_DECAY, config.PATIENCE, etc.

    set_seed(config.SEED)

    # Output directory
    # job_id = os.getenv("SLURM_JOB_ID") or os.getenv("SLURM_JOBID")
    # config.BASE_DIR = Path.cwd() / f"tryout_{job_id}" / "tryout_optuna_init_tau10"
    config.BASE_DIR = Path.cwd() / "tryout_trial14"
    config.BASE_DIR.mkdir(parents=True, exist_ok=True)

    setup_logging(config.BASE_DIR)
    config.save(config.BASE_DIR / "config.yaml")

    inc_training(config)


if __name__ == "__main__":
    main()
