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

def inc_training(config: Config):
    """
    Incremental training using SITrainer with SI + KD.
    This function follows the same control flow as your original baseline.
    """
    logger.info("==== Incremental Training (SI + KD via SITrainer) ====")

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
    trainer = SITrainer(model=model, device=device, config=config, task_dir=None, si_epsilon=config.SI_EPSILON)
    print_model_summary(model)
    
    # Train sequentially over tasks
    for task_idx in range(num_tasks):
        task_name = f"task{task_idx}"
        lwf_alpha = config.KD_ALPHA
        si_lambda = config.SI_LAMBDA
        si_epsilon = config.SI_EPSILON
        logger.info("--- %s (LWF alpha=%.4f, SI lambda=%.4f, SI epsilon=%.4f) ---",
                    task_name, lwf_alpha, si_lambda, si_epsilon)
                
        # Task directory
        task_dir = inc_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        trainer.task_dir = task_dir
        # Deterministic per-task seed (same as your baseline)
        set_seed(config.SEED + task_idx)

        # Train current task with SI/KD enabled
        history = trainer.train_task(
            train_loader=loaders[f"{task_name}_train"],
            val_loader=loaders[f"{task_name}_val"],
            task_id=task_idx,
            alpha_lwf=lwf_alpha,
            lambda_si=si_lambda,
            adapter_scale=None
        )
        
        trainer.consolidate(task_idx)

        # Save training history
        pd.DataFrame(history).to_csv(task_dir / "history.csv", index=False)
        plot_losses(history, task_dir)
    
    logger.info("==== Incremental Training Complete ====")
    logger.info("Evaluating final model...")
    evaluate_incremental_learning(config, None, inc_dir, loaders, device)

def main():
    """
    Main entrypoint for running the incremental training.
    """
    config = Config()
    config.MODE = "incremental"
    # config.SI_LAMBDA = 0.0019517224641449498
    # config.SI_EPSILON = 0.0396760507705299
    # config.KD_ALPHA = 0.02537815508265665
    config.SI_LAMBDA = 0.0
    config.SI_EPSILON = 0.0
    config.KD_ALPHA = 0.0

    set_seed(config.SEED)
    
    # Output directory
    config.BASE_DIR = Path.cwd() / "inc_fine_tuning"
    config.BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    setup_logging(config.BASE_DIR)
    config.save(config.BASE_DIR / "config.yaml")
    
    inc_training(config)
    
if __name__ == "__main__":
    main()
    
    