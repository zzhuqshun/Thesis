import random
import numpy as np
import torch
import logging
from pathlib import Path
import torch.nn as nn

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def setup_logging(log_dir: Path,
                  level: int = logging.INFO,
                  log_filename: str = 'train.log') -> logging.Logger:
    """
    Configure root logger to write INFO-level logs to both console and a file.

    Args:
        log_dir: directory where the log file will be stored
        level: logging level (e.g., logging.INFO)
        log_filename: name of the log file

    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(level)

    # File handler
    log_path = log_dir / log_filename
    if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path
               for h in logger.handlers):
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setLevel(level)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)

    return logger

def print_model_summary(
    model: nn.Module,
    only_trainable: bool = True,
    logger: logging.Logger = None
) -> None:
    """
    Print a summary of the model architecture and parameters.
    Args:
        model: PyTorch model instance
        only_trainable: if True, only print trainable parameters
        logger: optional logger to use; if None, uses the root logger
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info("========== Model Structure ==========")
    logger.info("\n%s", model)
    logger.info("=====================================")

    total_params = 0
    logger.info("====== Model Parameters Details ======")
    for name, param in model.named_parameters():
        if only_trainable and not param.requires_grad:
            continue
        n = param.numel()
        total_params += n
        status = "trainable" if param.requires_grad else "frozen"
        logger.info("%-50s | %-9s | %8d params", name, status, n)

    logger.info("-" * 80)
    scope = "only trainable" if only_trainable else "all"
    logger.info("%-50s   %8d params (%s)", "Total params:", total_params, scope)
    logger.info("=====================================")