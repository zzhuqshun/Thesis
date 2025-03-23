import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import random
import json
from datetime import datetime
from collections import deque
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history, plot_parallel_coordinate
import joblib
from soh_lstm import *

# Set configurations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = Path(__file__).parent / f"models/hpo_{timestamp}"
save_dir.mkdir(exist_ok=True, parents=True)

# Create a directory for storing Optuna study results
optuna_dir = save_dir / "optuna"
optuna_dir.mkdir(exist_ok=True)

# Constants for early stopping during hyperparameter search
HPO_EPOCHS = 25  # Maximum epochs during hyperparameter optimization
HPO_PATIENCE = 5  # Early stopping patience during hyperparameter search
N_TRIALS = 50    # Number of trials for hyperparameter optimization

def objective(trial):
    """
    Objective function for Optuna to minimize.
    
    Args:
        trial: An Optuna trial object
        
    Returns:
        float: Validation loss (to be minimized)
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Suggest hyperparameters
    hyperparams = {
        "SEQUENCE_LENGTH": trial.suggest_categorical("SEQUENCE_LENGTH", [144, 216, 288, 360, 432]),
        "HIDDEN_SIZE": trial.suggest_categorical("HIDDEN_SIZE", [32, 64, 128]),
        "NUM_LAYERS": trial.suggest_int("NUM_LAYERS", 2, 5),
        "DROPOUT": trial.suggest_categorical("DROPOUT", [0.1, 0.2, 0.3, 0.4, 0.5]),
        "BATCH_SIZE": trial.suggest_categorical("BATCH_SIZE", [32, 64, 128]),
        "LEARNING_RATE": trial.suggest_categorical("LEARNING_RATE", [1e-4, 1e-3]),
        "WEIGHT_DECAY": trial.suggest_categorical("WEIGHT_DECAY", [0, 1e-5, 1e-4])
    }
    
    # Print current trial hyperparameters
    trial_str = f"Trial {trial.number}: {hyperparams}"
    print(f"\n{'-' * len(trial_str)}")
    print(trial_str)
    print(f"{'-' * len(trial_str)}")
    
    # Load data
    data_dir = Path("../01_Datenaufbereitung/Output/Calculated/")
    df_train, df_val, df_test = load_data(data_dir)
    
    # Scale the data
    df_train_scaled, df_val_scaled, df_test_scaled = scale_data(df_train, df_val, df_test, scaler_type='standard')
    
    # Create datasets with current sequence length
    train_dataset = BatteryDataset(df_train_scaled, hyperparams["SEQUENCE_LENGTH"])
    val_dataset = BatteryDataset(df_val_scaled, hyperparams["SEQUENCE_LENGTH"])
    
    # Create dataloaders with current batch size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=hyperparams["BATCH_SIZE"], 
        shuffle=True, 
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=hyperparams["BATCH_SIZE"], 
        shuffle=False, 
        pin_memory=torch.cuda.is_available()
    )
    
    # Create model with current hyperparameters
    model = SOHLSTM(
        input_size=3,  # Voltage, current, temperature
        hidden_size=hyperparams["HIDDEN_SIZE"],
        num_layers=hyperparams["NUM_LAYERS"],
        dropout=hyperparams["DROPOUT"]
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=hyperparams["LEARNING_RATE"], 
        weight_decay=hyperparams["WEIGHT_DECAY"]
    )
    
    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Training loop
    for epoch in range(HPO_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Report intermediate metric
        trial.report(val_loss, epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{HPO_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= HPO_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
    
    # Return the best validation loss as the objective value
    return best_val_loss

def run_hyperparameter_optimization():
    """
    Run the hyperparameter optimization process using Optuna.
    """
    print("\n" + "="*50)
    print("Starting Hyperparameter Optimization")
    print("="*50)
    
    # Create a study name
    study_name = f"soh_lstm_optimization_{timestamp}"
    
    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS, timeout=None)
    
    # Get best trial
    best_trial = study.best_trial
    
    print("\n" + "="*50)
    print(f"Best trial: {best_trial.number}")
    print(f"Best validation loss: {best_trial.value:.6f}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print("="*50)
    
    # Save study results
    study_path = optuna_dir / f"{study_name}.pkl"
    joblib.dump(study, study_path)
    print(f"Study saved to {study_path}")
    
    # Save best hyperparameters to JSON
    best_params = best_trial.params
    best_params["MODEL"] = "LSTM SOH Optimized"
    best_params["device"] = str(device)
    best_params["EPOCHS"] = 50  # For full training
    best_params["PATIENCE"] = 10  # For full training
    
    best_params_path = save_dir / "best_hyperparameters.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Best hyperparameters saved to {best_params_path}")
    
    # Create visualizations if more than 1 trial
    if len(study.trials) > 1:
        try:
            # Parameter importance plot
            param_importance_fig = plot_param_importances(study)
            param_importance_fig.write_image(str(optuna_dir / "param_importance.png"))
            
            # Optimization history plot
            history_fig = plot_optimization_history(study)
            history_fig.write_image(str(optuna_dir / "optimization_history.png"))
            
            # Parallel coordinate plot
            parallel_fig = plot_parallel_coordinate(study)
            parallel_fig.write_image(str(optuna_dir / "parallel_coordinate.png"))
            
            print(f"Visualization plots saved to {optuna_dir}")
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    return best_params

def train_with_best_params(best_params):
    """
    Train the final model with the best hyperparameters.
    
    Args:
        best_params: Dictionary of best hyperparameters
    """
    print("\n" + "="*50)
    print("Training Final Model with Best Hyperparameters")
    print("="*50)
    
    # Extract hyperparameters
    SEQUENCE_LENGTH = best_params["SEQUENCE_LENGTH"]
    HIDDEN_SIZE = best_params["HIDDEN_SIZE"]
    NUM_LAYERS = best_params["NUM_LAYERS"]
    DROPOUT = best_params["DROPOUT"]
    BATCH_SIZE = best_params["BATCH_SIZE"]
    LEARNING_RATE = best_params["LEARNING_RATE"]
    WEIGHT_DECAY = best_params["WEIGHT_DECAY"]
    EPOCHS = best_params["EPOCHS"]
    PATIENCE = best_params["PATIENCE"]
    
    # Load and preprocess data
    data_dir = Path("../01_Datenaufbereitung/Output/Calculated/")
    df_train, df_val, df_test = load_data(data_dir)
    df_train_scaled, df_val_scaled, df_test_scaled = scale_data(df_train, df_val, df_test, scaler_type='standard')
    
    # Create datasets and dataloaders
    train_dataset = BatteryDataset(df_train_scaled, SEQUENCE_LENGTH)
    val_dataset = BatteryDataset(df_val_scaled, SEQUENCE_LENGTH)
    test_dataset = BatteryDataset(df_test_scaled, SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())
    
    # Create model
    model = SOHLSTM(
        input_size=3,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Define paths for saving models and history
    save_path = {
        'best': save_dir / 'best_soh_model.pth',
        'last': save_dir / 'last_soh_model.pth',
        'history': save_dir / 'train_history.parquet'
    }
    
    # Train and validate model
    history = train_and_validate_model(model, train_loader, val_loader, save_path, 
                                      epochs=EPOCHS, patience=PATIENCE, 
                                      learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Evaluate the best model on the test set
    print("\nEvaluating the best model on the test set...")
    model.load_state_dict(torch.load(save_path['best'], map_location=device))
    predictions, targets, metrics = evaluate_model(model, test_loader)
    
    # Print evaluation metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    if save_path['history'].exists():
        plot_losses(pd.read_parquet(save_path['history']), save_dir)
    plot_results(predictions, targets, metrics, df_test_scaled, save_dir)
    
    return metrics

def train_and_validate_model(model, train_loader, val_loader, save_path, 
                            epochs=50, patience=10, learning_rate=1e-3, weight_decay=0.0):
    """
    Train and validate the model with early stopping.
    Modified to accept hyperparameters as arguments.
    """
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Define learning rate scheduler and early stopping
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Initialize the early stopping parameters
    epochs_no_improve = 0
    best_val_loss = float('inf')

    # Define the history to store training and validation loss
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': []
    }

    # Start training
    print('\nStart training...')
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', leave=False) as pbar:
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                # Clear the previous gradients
                optimizer.zero_grad()
                # Forward propagation and calculate the loss
                outputs = model(features)
                loss = criterion(outputs, labels)
                # Backward propagation and clip the gradients
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                # Update the parameters of the model
                optimizer.step()
                # Update the training loss and progress bar
                train_loss += loss.item()
                pbar.update(1)

        # Update the average training loss
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        # Start the validation loop
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                # Forward propagation and calculate the loss
                outputs = model(features)
                loss = criterion(outputs, labels)
                # Update the validation loss
                val_loss += loss.item()

        # Update the average validation loss and epoch
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        history['epoch'].append(epoch + 1)

        # Update the learning rate after validation
        scheduler.step(val_loss)

        # Display the training and validation loss and learning rate
        print(f'Epoch {epoch + 1}/{epochs} | '
              f'Training Loss: {train_loss:.3e} | '
              f'Validation Loss: {val_loss:.3e} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # Check for early stopping and save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path['best'])
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs!')
                break
        
    # Save the last model (whether early stopping or not)
    torch.save(model.state_dict(), save_path['last'])
    print(f'\nLast model saved to {save_path["last"]}')
    # Save the training history
    history_df = pd.DataFrame(history)
    history_df.to_parquet(save_path['history'], index=False)
    print(f'Training history saved to {save_path["history"]}')

    return history
    
# Main function for hyperparameter optimization
def main_hpo():
    # Set seed for reproducibility
    set_seed(42)
    
    print(f"Using device: {device}")
    
    # Run hyperparameter optimization
    best_params = run_hyperparameter_optimization()
    
    # Train the final model with the best hyperparameters
    final_metrics = train_with_best_params(best_params)
    
    print("\n" + "="*50)
    print("Hyperparameter Optimization Completed")
    print(f"Best hyperparameters saved to {save_dir / 'best_hyperparameters.json'}")
    print(f"Best model saved to {save_dir / 'best_soh_model.pth'}")
    print("Final metrics:")
    for metric, value in final_metrics.items():
        print(f"    {metric}: {value:.4f}")
    print("="*50)

# Run this script for hyperparameter optimization
if __name__ == "__main__":
    main_hpo()
