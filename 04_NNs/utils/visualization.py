from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

def evaluate_model(model, data_loader):
    """
    Generic model evaluation function (for non-PNN models)
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader containing test data
        
    Returns:
        predictions: Model predictions
        targets: Ground truth values
        metrics: Dictionary of evaluation metrics
        total_loss: Mean loss on the dataset
    """
    model.eval()
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    total_loss = 0.0

    all_predictions, all_targets = [], []
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    total_loss /= len(data_loader)
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    metrics = calc_metrics(predictions, targets)

    return predictions, targets, metrics, total_loss

def evaluate_pnn(model, data_loader, task_id):
    """
    Specialized function to evaluate a specific task/column of PNN
    
    Args:
        model: The PNN model
        data_loader: DataLoader containing test data
        task_id: The task/column to evaluate
        
    Returns:
        predictions: Model predictions
        targets: Ground truth values
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    total_loss = 0.0

    all_predictions, all_targets = [], []
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features, task_id=task_id)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Handle case with empty data loader
    if all_predictions and all_targets:
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        metrics = calc_metrics(predictions, targets)
    else:
        # Handle empty lists case
        predictions = np.array([])
        targets = np.array([])
        metrics = {"RMSE": float('nan'), "MAE": float('nan'), 
                  "MAPE": float('nan'), "R²": float('nan')}

    return predictions, targets, metrics

def calc_metrics(predictions, targets):
    """
    Calculate evaluation metrics
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        
    Returns:
        Dictionary of metrics (RMSE, MAE, MAPE, R²)
    """
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    mape = mean_absolute_percentage_error(targets, predictions) * 100
    r2 = r2_score(targets, predictions)
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    }
    return metrics

def plot_results(save_dir, method_name, df_test, seq_len,
                base_pred, update1_pred, update2_pred, 
                base_metrics, update1_metrics, update2_metrics):
    """
    Plot continuous curves of true vs predicted values with metrics for each phase
    
    Args:
        save_dir: Directory to save results (Path object or string)
        method_name: Name of the method used, for filename generation
        df_test: DataFrame containing test data with 'Datetime' and target columns
        seq_len: Sequence length used for prediction
        base_pred: Predictions for base phase
        update1_pred: Predictions for update1 phase
        update2_pred: Predictions for update2 phase
        base_metrics, update1_metrics, update2_metrics: Dictionaries of metrics for each phase
    """
    # Extract x-coordinates and true target values from test data
    sequence_length = seq_len
    datetime_vals = df_test['Datetime'].iloc[sequence_length:].values
    true_vals = df_test['SOH_ZHU'].iloc[sequence_length:].values

    # Convert predictions to numpy arrays
    base_pred = np.array(base_pred)
    update1_pred = np.array(update1_pred)
    update2_pred = np.array(update2_pred)
    
    # Determine segment boundaries based on prediction lengths
    n_base = len(base_pred)
    n_update1 = len(update1_pred)
    n_update2 = len(update2_pred)
    
    # Split true values and dates for each phase
    base_true = true_vals[:n_base]
    update1_true = true_vals[n_base:n_base+n_update1]
    update2_true = true_vals[n_base+n_update1:n_base+n_update1+n_update2]
    
    x_base = datetime_vals[:n_base]
    x_update1 = datetime_vals[n_base:n_base+n_update1]
    x_update2 = datetime_vals[n_base+n_update1:n_base+n_update1+n_update2]
    
    # Concatenate predictions for the overall curve
    all_pred = np.concatenate([base_pred, update1_pred, update2_pred])
    
    # Create the main plot
    plt.figure(figsize=(15, 6))
    plt.plot(datetime_vals[:len(true_vals)], true_vals, label='True Values', marker='o', linestyle='-')
    plt.plot(datetime_vals[:len(all_pred)], all_pred, label='Predicted Values', marker='x', linestyle='--')
    
    # Function to annotate each segment with metrics
    def annotate_segment(x_segment, seg_true, seg_pred, metrics, phase_name):
        if len(x_segment) == 0:
            return
            
        # Use middle of segment for annotation position
        mid_x = x_segment[len(x_segment) // 2]
        # Use mean of true and predicted values for y-position
        y_mean = np.mean(np.concatenate([seg_true, seg_pred]))
        
        # Format metrics text
        text = (f"{phase_name}\n"
                f"RMSE: {metrics['RMSE']:.4f}\n"
                f"MAE: {metrics['MAE']:.4f}\n"
                f"MAPE: {metrics['MAPE']:.2f}%\n"
                f"R²: {metrics['R²']:.4f}")
                
        plt.text(mid_x, y_mean, text, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7),
                 horizontalalignment='center', verticalalignment='center')
    
    # Add annotations for each phase
    annotate_segment(x_base, base_true, base_pred, base_metrics, "Base Model")
    annotate_segment(x_update1, update1_true, update1_pred, update1_metrics, "Update 1")
    annotate_segment(x_update2, update2_true, update2_pred, update2_metrics, "Update 2")
    
    # Set plot labels and title
    plt.xlabel("Datetime")
    plt.ylabel("SOH")
    plt.title(f"{method_name} - True vs Predicted SOH Across Phases")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Create output directory and save figure
    results_dir = Path(save_dir) / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(results_dir / f"{method_name}_true_pred_plot.png")
    plt.close()

    # Create metrics comparison bar chart
    plt.figure(figsize=(12, 6))
    metrics_names = ['RMSE', 'MAE', 'MAPE', 'R²']
    phases = ['Base Model', 'Update 1', 'Update 2']
    
    # Set up bar positions
    x = np.arange(len(metrics_names))
    width = 0.25
    
    # Plot bars for each phase
    plt.bar(x - width, [base_metrics[m] for m in metrics_names], width, label='Base Model')
    plt.bar(x, [update1_metrics[m] for m in metrics_names], width, label='Update 1')
    plt.bar(x + width, [update2_metrics[m] for m in metrics_names], width, label='Update 2')
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title(f'{method_name} - Performance Metrics Comparison')
    plt.xticks(x, metrics_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(results_dir / f"{method_name}_metrics_comparison.png")
    plt.close()

    print(f"Results saved to {results_dir}")