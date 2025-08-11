import logging
import torch
import pandas as pd
from pathlib import Path

from utils.config import Config
from utils.utils import set_seed, setup_logging, print_model_summary
from utils.data import DataProcessor, create_dataloaders
from utils.base import SOHLSTM, Trainer
from utils.si import SI, SITrainer
from utils.evaluate import plot_losses, evaluate_incremental_learning
logger = logging.getLogger(__name__)

def fine_tuning(config):
    """
    Pure incremental fine-tuning: train on tasks sequentially without regularization.
    """
    logger.info("==== Incremental Training (Pure Fine-tuning) ====")
    
    # Setup directories
    inc_dir = config.BASE_DIR
    inc_dir.mkdir(parents=True, exist_ok=True)
    
    # Get number of tasks from config
    num_tasks = config.NUM_TASKS
    logger.info("Number of tasks: %d", num_tasks)
    
    # Prepare incremental learning data
    dp = DataProcessor(config.DATA_DIR, config.RESAMPLE, config)
    data = dp.prepare_incremental_data(config.incremental_datasets)
    loaders = create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    
    # Initialize model and trainer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SOHLSTM(3, config.HIDDEN_SIZE, config.DROPOUT).to(device)
    trainer = Trainer(model, device, config, inc_dir)
    print_model_summary(model)
    
    # Sequential task training
    for task_idx in range(num_tasks):
        task_name = f"task{task_idx}"
        
        logger.info("--- %s ---", task_name)
        
        # Setup task directory
        task_dir = inc_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        trainer.task_dir = task_dir
        
        # Set task-specific random seed for reproducibility
        set_seed(config.SEED + task_idx)
        
        # Train on current task
        history = trainer.train_task(
            loaders[f"{task_name}_train"], 
            loaders[f"{task_name}_val"], 
            task_idx
        )
        
        pd.DataFrame(history).to_csv(task_dir / 'training_history.csv', index=False)
        plot_losses(history, task_dir)
        
        logger.info("Task %d completed.", task_idx)
    
    logger.info("==== Incremental Training Complete ====")
    evaluate_incremental_learning(config, None, inc_dir, loaders, device)

def main():
    """Run pure fine-tuning for the model.
    This function initializes the configuration, sets the random seed for reproducibility,
    sets up logging, and starts the pure fine-tuning process.
    """
    # Initialize configuration
    config = Config()
    config.MODE = "incremental"  # Set mode to incremental for fine-tuning
    set_seed(config.SEED)
    
    config.BASE_DIR = Path.cwd() / 'pure_fine_tuning'
    config.BASE_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(config.BASE_DIR)

    config.save(config.BASE_DIR / 'config.json')
    
    # Start pure fine-tuning
    fine_tuning(config)

if __name__ == '__main__':
    main()