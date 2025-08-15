# run_inc.py
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
        
        ckpt_path0 = Path("task0_best.pt")
        if task_idx == 0 and getattr(config, "SKIP_TASK0", False) and ckpt_path0.exists():
            ckpt = torch.load(ckpt_path0, map_location=device)

            # 1) 恢复模型
            trainer.model.load_state_dict(ckpt["model_state"])
            logger.info("[Skip] Loaded Task0 best from %s", ckpt_path0)

            # 2) 恢复 SI（如有保存）
            si_state = ckpt.get("si_state")
            if config.USE_SI and si_state:
                trainer.si.omega = {n: t.to(device) for n, t in si_state.get("omega", {}).items()}
                trainer.si.theta_star = {n: t.to(device) for n, t in si_state.get("theta_star", {}).items()}
                logger.info("[Skip] Restored SI (omega/theta_star).")
            else:
                trainer.si.omega = {}
                trainer.si.theta_star = {}

            # 3) 为下一任务构建 teacher 与 KL 参考统计（最简单：直接复算）
            if config.USE_KL:
                trainer._after_task_finalize(loaders["task0_train"])
                logger.info("[Skip] Built teacher & KL stats from task0_train.")
            else:
                trainer.prev_stats = None
                trainer.prev_model = copy.deepcopy(trainer.model).to(device)
                trainer.prev_model.eval()
                for p in trainer.prev_model.parameters():
                    p.requires_grad_(False)

            logger.info("[Skip] Task0 training skipped. Start Task1.")
            continue
            
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

    config.SKIP_TASK0 = True
    
    # ---- toggles: enable/disable components here ----
    # SI / KD / KL switches
    config.USE_SI = True
    config.USE_KD = True
    config.USE_KL = True
    # config.USE_SI = False
    # config.USE_KD = False
    # config.USE_KL = False

    # KL mode & temperature
    config.KL_MODE = 'both'   # 'input' | 'hidden' | 'both'
    config.TAU = 50          # s = KL / (KL + TAU)

    # SI strength range (auto-scheduled by KL)
    config.SI_FLOOR = 0.0
    config.SI_MAX   = 0.10
    config.SI_EPSILON = 1e-3
    config.SI_WARMUP_EPOCHS = 10
    # KD strength range (auto-scheduled by KL)
    config.KD_FLOOR = 0.1
    config.KD_MAX   = 0.5
    config.KD_LOSS  = 'mse'    # 'mse' | 'l1' | 'smoothl1'
    config.KD_WARMUP_EPOCHS = 10
    # Training basics (keep your existing settings)
    # config.EPOCHS, config.BATCH_SIZE, config.LEARNING_RATE, config.WEIGHT_DECAY, config.PATIENCE, etc.

    set_seed(config.SEED)

    # Output directory
    config.BASE_DIR = Path.cwd() / "tryout_tau50_warmup_smalelr_si_load_task0"
    config.BASE_DIR.mkdir(parents=True, exist_ok=True)

    setup_logging(config.BASE_DIR)
    config.save(config.BASE_DIR / "config.yaml")

    inc_training(config)


if __name__ == "__main__":
    main()
