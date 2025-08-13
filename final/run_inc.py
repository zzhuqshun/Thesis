import logging
import torch
import pandas as pd
from pathlib import Path
from utils.config import Config
from utils.utils import set_seed, setup_logging, print_model_summary
from utils.data import DataProcessor, create_dataloaders
from utils.base import SOHLSTM
from utils.si import SITrainer, RegConfig     # <-- use SITrainer instead of base.Trainer
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
    reg_cfg = RegConfig(
        p_kd_max=0.55, p_si_max=0.30,
        p_kd_floor=0.15, p_si_floor=0.08,
        kd_feat_weight=0.5,   # 生效
        kd_delta=1e-2,
        kl_tau=2.0,
        scale_clip=5.0,
        warmup_epochs=5
    )

    trainer = SITrainer(
        model=model,
        device=device,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        grad_clip=1.0,
        si_epsilon=1e-3,
        reg_cfg=reg_cfg,
        val_metric="mse"
    )

    print_model_summary(model)

    # Sequential tasks
    for task_idx in range(num_tasks):
        task_name = f"task{task_idx}"
        logger.info("--- %s (KL-scheduled KD+SI, eps=%.4g) ---", task_name, trainer.si.epsilon)

        # per-task dir
        task_dir = inc_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        # per-task seed
        set_seed(config.SEED + task_idx)

        # >>> train one task
        result = trainer.train_one_task(
            train_loader=loaders[f"{task_name}_train"],
            val_loader=loaders[f"{task_name}_val"],
            epochs=config.EPOCHS,
            task_id=task_idx
        )

        # >>> save best checkpoint for this task
        ckpt_path = task_dir / f"task{task_idx}_best.pt"
        torch.save({'model_state': trainer.model.state_dict()}, ckpt_path)
        logger.info("Saved best checkpoint: %s", ckpt_path)
        if task_idx == 0:
            trainer.prime_kl_ref(loaders[f"task0_train"])
        # >>> save history csv if available
        if isinstance(result, dict) and "history" in result:
            pd.DataFrame(result["history"]).to_csv(task_dir / "history.csv", index=False)
        plot_losses(task_dir / "history.csv", task_dir, title=f"Task {task_idx} Training")


    logger.info("==== Incremental Training Complete ====")
    logger.info("Evaluating final model...")
    evaluate_incremental_learning(config, None, inc_dir, loaders, device)

def main():
    """
    Main entrypoint for running the incremental training.
    """
    config = Config()
    config.MODE = "incremental"

    set_seed(config.SEED)
    
    # Output directory
    config.BASE_DIR = Path.cwd() / "inc_dynamical"
    config.BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    setup_logging(config.BASE_DIR)
    config.save(config.BASE_DIR / "config.yaml")
    
    inc_training(config)
    
if __name__ == "__main__":
    main()
    
    