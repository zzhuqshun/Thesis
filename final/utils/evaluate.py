import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Assuming SOHLSTM is defined in model.py
from base import SOHLSTM

logger = logging.getLogger(__name__)


def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             alpha: float = 0.1,
             log: bool = True):
    """
    Evaluate model performance on a dataset.

    Args:
        model: PyTorch model to evaluate
        loader: DataLoader providing (inputs, targets)
        alpha: smoothing factor for exponential smoothing
        log: whether to log metrics via logger

    Returns:
        preds: numpy array of predictions
        targets: numpy array of true values
        metrics: dict containing RMSE, MAE, R2 and their smoothed equivalents
    """
    model.eval()
    # Determine device from model parameters
    device = next(model.parameters()).device
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Compute primary metrics
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(targets, preds)),
        'MAE': mean_absolute_error(targets, preds),
        'R2': r2_score(targets, preds)
    }

    # Compute smoothed metrics
    preds_smooth = pd.Series(preds).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    metrics.update({
        'RMSE_smooth': np.sqrt(mean_squared_error(targets, preds_smooth)),
        'MAE_smooth': mean_absolute_error(targets, preds_smooth),
        'R2_smooth': r2_score(targets, preds_smooth)
    })

    if log:
        logger.info(
            "Eval -> RMSE: %.4e, MAE: %.4e, R2: %.4f", 
            metrics['RMSE'], metrics['MAE'], metrics['R2']
        )

    return preds, targets, metrics


# Visualization utilities
def plot_losses(history: dict, out_dir: Path):
    """Plot training and validation loss curves."""
    df = pd.DataFrame(history)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.semilogy(df['epoch'], df['train_loss'], label='Train Loss')
    plt.semilogy(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_predictions(preds: np.ndarray,
                     targets: np.ndarray,
                     metrics: dict,
                     out_dir: Path,
                     alpha: float = 0.1):
    """Plot time series of predictions vs actuals with metrics in title."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.arange(len(targets))

    preds_smooth = pd.Series(preds).ewm(alpha=alpha, adjust=False).mean().to_numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(idx, targets, label='Actual')
    plt.plot(idx, preds, label='Predicted')
    plt.plot(idx, preds_smooth, label='Predicted (Smooth)')
    plt.xlabel('Index')
    plt.ylabel('Value')
    title = (
        f"RMSE: {metrics['RMSE']:.4e}, MAE: {metrics['MAE']:.4e}, R2: {metrics['R2']:.4f}\n"
        f"RMSE_s: {metrics['RMSE_smooth']:.4e}, MAE_s: {metrics['MAE_smooth']:.4e}, R2_s: {metrics['R2_smooth']:.4f}"
    )
    plt.title('Predictions vs Actuals\n' + title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / 'predictions.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_scatter(preds: np.ndarray,
                            targets: np.ndarray,
                            out_dir: Path,
                            alpha: float = 0.1):
    """Plot scatter of predictions vs actuals (original and smoothed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_smooth = pd.Series(preds).ewm(alpha=alpha, adjust=False).mean().to_numpy()

    plt.figure(figsize=(12, 5))
    # Original
    plt.subplot(1, 2, 1)
    plt.scatter(targets, preds, alpha=0.6)
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    plt.plot(lims, lims, '--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Original Scatter')
    plt.grid(True)

    # Smoothed
    plt.subplot(1, 2, 2)
    plt.scatter(targets, preds_smooth, alpha=0.6)
    lims = [min(targets.min(), preds_smooth.min()), max(targets.max(), preds_smooth.max())]
    plt.plot(lims, lims, '--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted (Smooth)')
    plt.title('Smoothed Scatter')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_incremental_learning(config,
                                  inc_dir: Path,
                                  num_tasks: int,
                                  loaders: dict):
    """
    Comprehensive evaluation of incremental learning performance.

    Args:
        config: experiment configuration
        inc_dir: base directory for incremental outputs
        num_tasks: number of incremental tasks
        loaders: dict of DataLoaders, keys include 'test_full', 'test_task{i}'

    Returns:
        continual_learning_metrics: dict with BWT, FWT, ACC
        R_matrix: numpy array of performance values
    """
    logger.info("==== Starting Comprehensive Evaluation ====")
    eval_dir = inc_dir / 'metrics'
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Initialize R matrix and summary list
    R_matrix = np.zeros((num_tasks, num_tasks))
    summary = []

    for i in range(num_tasks):
        # Load checkpoint
        ckpt = inc_dir / f"task{i}" / f"task{i}_best.pt"
        if not ckpt.exists():
            logger.error("Checkpoint not found: %s", ckpt)
            continue

        # Rebuild model and load weights
        model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
        model.load_state_dict(torch.load(ckpt, map_location='cpu')['model_state'])
        preds_full, tgts_full, metrics_full = evaluate(model, loaders['test_full'], alpha=config.ALPHA, log=False)
        logger.info("Full Test after Task %d -> MAE: %.4e, R2: %.4f", i, metrics_full['MAE'], metrics_full['R2'])
        plot_predictions(preds_full, tgts_full, metrics_full, inc_dir / f"task{i}", alpha=config.ALPHA)
        summary.append({
            'trained': i, 'eval': 'full', **metrics_full, 'R_value': -metrics_full['MAE']
        })

        for j in range(num_tasks):
            key = f'test_task{j}'
            preds_j, tgts_j, metrics_j = evaluate(model, loaders[key], alpha=config.ALPHA, log=False)
            R_matrix[i, j] = -metrics_j['MAE']
            summary.append({
                'trained': i, 'eval': j, **metrics_j, 'R_value': R_matrix[i, j]
            })
            logger.info("Task %d->Task %d: MAE=%.4e, R2=%.4f", i, j, metrics_j['MAE'], metrics_j['R2'])

    # Baseline performance
    baseline_R = []
    torch.manual_seed(config.SEED + 999)
    for j in range(num_tasks):
        model0 = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
        _, _, m0 = evaluate(model0, loaders[f'test_task{j}'], alpha=config.ALPHA, log=False)
        baseline_R.append(-m0['MAE'])
        logger.info("Baseline Task %d R: %.4f", j, baseline_R[-1])

    # Compute CL metrics
    # BWT
    if num_tasks > 1:
        bwt = np.mean([R_matrix[-1, k] - R_matrix[k, k] for k in range(num_tasks-1)])
        fwt = np.mean([R_matrix[k-1, k] - baseline_R[k] for k in range(1, num_tasks)])
    else:
        bwt = fwt = 0.0
    acc = np.mean(R_matrix[-1])
    cl_metrics = {'BWT': bwt, 'FWT': fwt, 'ACC': acc}

    # Save results
    pd.DataFrame(summary).to_csv(eval_dir / 'detailed_results.csv', index=False)
    pd.DataFrame([cl_metrics]).to_csv(eval_dir / 'cl_metrics.csv', index=False)
    pd.DataFrame(R_matrix, columns=[f"t{j}" for j in range(num_tasks)]).to_csv(eval_dir / 'R_matrix.csv', index=False)
    pd.DataFrame({'baseline': baseline_R}).to_csv(eval_dir / 'baseline.csv', index=False)

    logger.info("CL Metrics: %s", cl_metrics)
    return cl_metrics, R_matrix
