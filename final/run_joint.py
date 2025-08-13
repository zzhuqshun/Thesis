import logging

from pathlib import Path
import torch
import pandas as pd
from utils.config import Config
from utils.utils import set_seed, setup_logging
from utils.data import DataProcessor, create_dataloaders
from utils.base import SOHLSTM,Trainer
from utils.evaluate import evaluate, plot_losses, plot_predictions, plot_prediction_scatter
from utils.utils import print_model_summary

logger = logging.getLogger(__name__)

def joint(config: Config):
    """Run joint training over all data as baseline."""
    logger.info("==== Starting Joint Training ====")
    base_dir = Path(config.BASE_DIR) 
    ckpt_dir = base_dir / 'checkpoints'
    res_dir = base_dir / 'results'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Data prep
    dp = DataProcessor(config.DATA_DIR, config.RESAMPLE, config)
    datasets = dp.prepare_joint_data(config.joint_datasets)
    loaders = create_dataloaders(datasets, config.SEQUENCE_LENGTH, config.BATCH_SIZE)

    # Model and Trainer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SOHLSTM(3, config.HIDDEN_SIZE, config.DROPOUT).to(device)
    trainer = Trainer(model, device, config, ckpt_dir)
    print_model_summary(model)
    # Train
    history = trainer.train_task(loaders['train'], loaders['val'], task_id=0)
    pd.DataFrame(history).to_csv(ckpt_dir / 'training_history.csv', index=False)
    plot_losses(history, res_dir)

    # Evaluate
    preds, tgts, metrics = evaluate(model, loaders['test'], alpha=config.ALPHA)
    logger.info("Test metrics -> RMSE: %.4e, MAE: %.4e, R2: %.4f", 
                metrics['RMSE'], metrics['MAE'], metrics['R2'])
    plot_predictions(preds, tgts, metrics, res_dir, alpha=config.ALPHA)
    plot_prediction_scatter(preds, tgts, res_dir, alpha=config.ALPHA)
    pd.DataFrame([metrics]).to_csv(res_dir / 'test_metrics.csv', index=False)

    logger.info("==== Joint Training Complete ====")
    
def main():
    """Run joint training for the model.
    This function initializes the configuration, sets the random seed for reproducibility,
    sets up logging, and starts the joint training process.
    """
    # Initialize configuration
    config = Config()
    config.MODE = "joint" 
    config.SEED = 42
    set_seed(config.SEED)
    
    config.BASE_DIR = Path.cwd() / 'joint_swa'
    config.BASE_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(config.BASE_DIR)

    config.save(config.BASE_DIR / 'config.yaml')
    
    # Start joint training
    joint(config)
    
if __name__ == '__main__':
    main()
