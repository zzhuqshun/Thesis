import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Assuming SOHLSTM is defined in model.py
from utils.config import Config
from utils.base import SOHLSTM

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
def plot_losses(history, out_dir, title=None):
    df = pd.read_csv(history) if isinstance(history, (str, Path)) else history
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if title is None: title = "Training"
    t = f"{title}\nKL={df['kl'].mean():.3f} | λ: KD={df['lam_kd'].mean():.3f} SI={df['lam_si'].mean():.3f}"

    plt.figure(figsize=(9,5))
    plt.semilogy(df['epoch'], df['train_loss'], label='Train Loss')
    plt.semilogy(df['epoch'], df['train_sup'], label='Task Loss')
    plt.semilogy(df['epoch'], df['train_kd'], label='KD Loss')
    plt.semilogy(df['epoch'], df['train_si'], label='SI Loss')
    plt.semilogy(df['epoch'], df['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(t)
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "loss_curves.png", dpi=300, bbox_inches="tight")
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

def evaluate_incremental_learning(config: Config,
                                  model_class: type = None,
                                  inc_dir: Path = None,
                                  loaders: dict = None,
                                  device: torch.device = None):
    """
    Comprehensive evaluation of incremental learning performance.
    """
    logger.info("==== Starting Comprehensive Evaluation ====")
    
    # Create evaluation directory
    eval_dir = inc_dir / 'metrics'
    eval_dir.mkdir(parents=True, exist_ok=True)
    num_tasks = config.NUM_TASKS
    
    # Performance matrix R[i][j] = performance of model after task i on test set of task j
    R_matrix = np.zeros((num_tasks, num_tasks))
    metrics_summary = []
    model_class = SOHLSTM if model_class is None else model_class
    # Evaluate each trained model on all test sets
    for trained_task_idx in range(num_tasks):
        logger.info("Evaluating model trained after task %d...", trained_task_idx)
        
        # Load model checkpoint from this training stage
        checkpoint_path = inc_dir / f"task{trained_task_idx}" / f"task{trained_task_idx}_best.pt"
        if not checkpoint_path.exists():
            logger.error("Checkpoint not found: %s", checkpoint_path)
            continue
        
        # Create fresh model and load trained weights
        eval_model = model_class(3, hidden_size=config.HIDDEN_SIZE, dropout=config.DROPOUT).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        eval_model.load_state_dict(checkpoint['model_state'])
        eval_model.eval()

        # Evaluate on full test set (complete battery degradation curve)
        full_preds, full_targets, full_metrics = evaluate(eval_model,
            loaders['test_full'], alpha=config.ALPHA, log=False
        )
        logger.info("Full test set evaluation: MAE=%.4e, R2=%.4f", 
                    full_metrics['MAE'], full_metrics['R2'])
        
        # Save full test set predictions plot
        plot_predictions(
            full_preds, full_targets, full_metrics,
            inc_dir / f"task{trained_task_idx}", alpha=config.ALPHA
        )
        
        # Record full test performance
        metrics_summary.append({
            "trained_after_task": f"task{trained_task_idx}",
            "evaluated_on_task": "full_test",
            "trained_task_idx": trained_task_idx,
            "eval_task_idx": -1,  # -1 indicates full test set
            "MAE": full_metrics['MAE'],
            "MAE_smooth": full_metrics['MAE_smooth'],
            "RMSE": full_metrics['RMSE'],
            "RMSE_smooth": full_metrics['RMSE_smooth'],
            "R2": full_metrics['R2'],
            "R2_smooth": full_metrics['R2_smooth'],
            "R_value": -full_metrics['MAE']  # Negative MAE for maximization
        })
        
        # Evaluate on each task-specific test set
        for eval_task_idx in range(num_tasks):
            test_loader_key = f'test_task{eval_task_idx}'
            test_loader = loaders[test_loader_key]
            
            # Get predictions and metrics for this task
            _, _, task_metrics = evaluate(eval_model, test_loader, alpha=config.ALPHA, log=False)
            
            # Store performance in R matrix
            R_matrix[trained_task_idx][eval_task_idx] = -task_metrics['MAE']
            
            # Record detailed metrics
            metrics_summary.append({
                "trained_after_task": f"task{trained_task_idx}",
                "evaluated_on_task": f"test_task{eval_task_idx}",
                "trained_task_idx": trained_task_idx,
                "eval_task_idx": eval_task_idx,
                "MAE": task_metrics['MAE'],
                "MAE_smooth": task_metrics['MAE_smooth'],
                "RMSE": task_metrics['RMSE'],
                "RMSE_smooth": task_metrics['RMSE_smooth'],
                "R2": task_metrics['R2'],
                "R2_smooth": task_metrics['R2_smooth'],
                "R_value": R_matrix[trained_task_idx][eval_task_idx]
            })
            
            logger.info("  Task %d -> Test Task %d: MAE=%.4e, R2=%.4f", 
                       trained_task_idx, eval_task_idx, 
                       task_metrics['MAE'], task_metrics['R2'])
    
    # ===============================================================
    # Calculate Continual Learning Metrics
    # ===============================================================
    logger.info("==== Computing Continual Learning Metrics ====")
    
    # Compute baseline performance for Forward Transfer calculation
    logger.info("Computing random initialization baselines...")
    torch.manual_seed(config.SEED + 999)  # Different seed for baseline
    baseline_model = model_class(3, hidden_size=config.HIDDEN_SIZE, dropout=config.DROPOUT).to(device)

    baseline_performance = np.zeros(num_tasks)
    for j in range(num_tasks):
        test_loader = loaders[f'test_task{j}']
        _, _, baseline_metrics = evaluate(baseline_model, test_loader, alpha=config.ALPHA, log=False)
        baseline_performance[j] = -baseline_metrics['MAE']
        logger.info("  Baseline Task %d: R=%.4f", j, baseline_performance[j])
    
    # Calculate BWT (Backward Transfer)
    # BWT measures how much old task performance degrades after learning new tasks
    if num_tasks > 1:
        bwt_scores = []
        for i in range(num_tasks - 1):  # Tasks 0 to T-2
            final_perf = R_matrix[num_tasks - 1, i]  # Performance after all tasks
            when_learned_perf = R_matrix[i, i]       # Performance when task was learned
            bwt_scores.append(final_perf - when_learned_perf)
        BWT = np.mean(bwt_scores)
    else:
        BWT = 0.0
    
    # Calculate FWT (Forward Transfer)  
    # FWT measures how much learning previous tasks helps with new tasks
    if num_tasks > 1:
        fwt_scores = []
        for i in range(1, num_tasks):  # Tasks 1 to T-1
            when_learned_perf = R_matrix[i - 1, i]  # Performance on task i after learning task i-1
            baseline_perf = baseline_performance[i]  # Random initialization performance
            fwt_scores.append(when_learned_perf - baseline_perf)
        FWT = np.mean(fwt_scores)
    else:
        FWT = 0.0
    
    # Calculate ACC (Average Accuracy)
    # ACC measures overall performance: average final performance across all tasks
    ACC = np.mean(R_matrix[num_tasks - 1, :])
    
    # Compile continual learning metrics
    continual_learning_metrics = {
        "BWT": BWT,  # Backward Transfer (negative = forgetting)
        "FWT": FWT,  # Forward Transfer (positive = beneficial transfer)
        "ACC": ACC,  # Average final accuracy
        "num_tasks": num_tasks
    }
    
    # Log results
    logger.info("==== Continual Learning Results ====")
    logger.info("BWT (Backward Transfer): %.4f %s", BWT, 
               "(negative = forgetting)" if BWT < 0 else "(positive = backward gain)")
    logger.info("FWT (Forward Transfer): %.4f %s", FWT,
               "(positive = beneficial transfer)" if FWT > 0 else "(negative = interference)")
    logger.info("ACC (Average Accuracy): %.4f", ACC)
    
    # Print R matrix for detailed inspection
    logger.info("==== Performance Matrix R[i][j] ====")
    logger.info("Rows: trained after task i, Columns: evaluated on task j")
    header = "       " + " ".join([f"Task{j:2d}" for j in range(num_tasks)])
    logger.info(header)
    for i in range(num_tasks):
        row_values = " ".join([f"{R_matrix[i,j]:7.4f}" for j in range(num_tasks)])
        logger.info("Task%2d: %s", i, row_values)
    
    # ===============================================================
    # Save All Results
    # ===============================================================
    
    # Save detailed evaluation metrics
    summary_df = pd.DataFrame(metrics_summary)
    summary_df.to_csv(eval_dir / 'detailed_evaluation_results.csv', index=False)
    
    # Save continual learning metrics summary
    cl_metrics_df = pd.DataFrame([continual_learning_metrics])
    cl_metrics_df.to_csv(eval_dir / 'continual_learning_metrics.csv', index=False)
    
    # Save performance matrix
    r_matrix_df = pd.DataFrame(
        R_matrix, 
        index=[f"after_task{i}" for i in range(num_tasks)],
        columns=[f"eval_task{j}" for j in range(num_tasks)]
    )
    r_matrix_df.to_csv(eval_dir / 'R_matrix.csv')
    
    # Save baseline performance for reference
    baseline_df = pd.DataFrame({
        'task': [f'task{i}' for i in range(num_tasks)],
        'baseline_performance': baseline_performance
    })
    baseline_df.to_csv(eval_dir / 'baseline_performance.csv', index=False)
    
    logger.info("==== Evaluation Complete ====")
    logger.info("All results saved to: %s", eval_dir)
    
    return continual_learning_metrics, R_matrix