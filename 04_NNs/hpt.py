import json
import optuna
import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

# ===== Import from your original script =====
from base import (
    set_seed,
    scale_data,
    BatteryDataset,
    SOHLSTM,
    train_and_validate_model,
    device
)
config = {
    "INFO": [
        "Hyperparameter tuning",
        "LSTM(10 min resampling)",
        "val_id:['01', '15', '17']",
        "test_id:[random]",
        "Standard scaled ['Voltage[V]', 'Current[A]', 'Temperature[°C]', 'SOH_ZHU']"
        ],
    "Search": {
        "seq_length": [144, 288, 432, 576, 720, 864, 1008],
        "hidden_size": [32, 64, 128, 256],
        "num_layers": [2, 3, 4, 5],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "learning_rate": 1e-4,
        "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
        "batch_size": [16, 32, 64, 128],
        "epochs": 100,
        "patience": 10
    }
    }

# Create directory for optimization results
optuna_dir = Path(__file__).parent / "models/HPT"
optuna_dir.mkdir(exist_ok=True, parents=True)

with open(optuna_dir / "config.json", "w") as f:
    json.dump(config, f, indent=4)

def consistent_load_data(data_dir: Path, resample='10min', split_file_path='dataset_split.json'):
    """
    Load training, validation, and test data with persistent dataset split.
    Saves the split on first run, then reuses the same split on subsequent runs.
    
    Args:
        data_dir: Directory containing the parquet files
        resample: Time interval for resampling data
        split_file_path: Path to save/load the dataset split configuration
    """
    # Get all parquet files and sort them by the numeric part of the filename
    parquet_files = sorted(
        [f for f in data_dir.glob('*.parquet') if f.is_file()],
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    # Check if split file exists (from a previous run)
    split_file = Path(split_file_path)
    if split_file.exists():
        # Load the existing split configuration
        print(f"Loading existing dataset split from {split_file_path}")
        with open(split_file, 'r') as f:
            split_config = json.load(f)
            
        # Convert string paths back to Path objects
        test_file = Path(split_config['test_file'])
        train_files = [Path(f) for f in split_config['train_files']]
        val_files = [Path(f) for f in split_config['val_files']]
        
        print("Using predefined split")
    else:
        # First run - create a new split
        print(f"Creating new dataset split (will be saved to {split_file_path})")
        
        # Set seed to ensure the split is reproducible
        random.seed(42)
        
        # Define test and validation cell IDs
        test_cell_id = '17'  # Cell 17 for test
        val_cell_ids = ['01', '13', '19']  # Cells 1, 13, 19 for validation
        
        # Find test file based on cell ID
        test_file = next((f for f in parquet_files if f.stem.split('_')[1] == test_cell_id), None)
        if test_file is None:
            raise ValueError(f"Test cell ID '{test_cell_id}' not found in dataset")
        
        # Find validation files based on cell IDs
        val_files = [f for f in parquet_files if f.stem.split('_')[1] in val_cell_ids]
        if len(val_files) != len(val_cell_ids):
            found_ids = [f.stem.split('_')[1] for f in val_files]
            missing_ids = [vid for vid in val_cell_ids if vid not in found_ids]
            print(f"Warning: Could not find all validation cell IDs. Missing: {missing_ids}")
        
        # Remaining files are for training
        train_files = [f for f in parquet_files if f not in [test_file] and f not in val_files]
        
        # Save the split configuration for future runs
        split_config = {
            'test_file': str(test_file),
            'train_files': [str(f) for f in train_files],
            'val_files': [str(f) for f in val_files]
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(split_file) if os.path.dirname(split_file) else '.', exist_ok=True)
        
        with open(split_file, 'w') as f:
            json.dump(split_config, f, indent=4)
        
        print("Created and saved new dataset split")

    # Print filenames for each dataset
    print(f"Training files: {[f.name for f in train_files]}")
    print(f"Validation files: {[f.name for f in val_files]}")
    print(f"Testing file: {test_file.name}")

    # Process the files
    def process_file(file_path: Path):
        """Internal function to read and process each parquet file."""
        df = pd.read_parquet(file_path)
        
        # Keep only needed columns to reduce memory usage
        columns_to_keep = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 
                           'Temperature[°C]', 'SOH_ZHU']

        df_processed = df[columns_to_keep].copy()
        df_processed.dropna(inplace=True)
        
        # Process time column into integers and generate corresponding Datetime column
        df_processed['Testtime[s]'] = df_processed['Testtime[s]'].round().astype(int)
        
        # Convert to datetime using Testtime offset from a reference
        df_processed['Datetime'] = pd.to_datetime(
            df_processed['Testtime[s]'], unit='s', origin=pd.Timestamp("2023-02-02")
        )
        
        # Sample data at specified interval
        df_sampled = df_processed.set_index('Datetime').resample(resample).mean().reset_index()
        
        # Add cell_id column
        df_sampled["cell_id"] = file_path.stem.split('_')[1]
        
        return df_sampled, file_path.name

    # Process training, validation, and test files
    print("\nProcessing test file...")
    test_data = process_file(test_file)
    
    print("Processing validation files...")
    val_data = [process_file(f) for f in val_files]
    
    print("Processing training files...")
    train_data = [process_file(f) for f in train_files]

    # Combine data
    df_train = pd.concat([t[0] for t in train_data], ignore_index=True)
    df_val = pd.concat([v[0] for v in val_data], ignore_index=True)
    df_test = test_data[0]

    print(f"\nDataset statistics:")
    print(f"Training set: {df_train.shape[0]} samples, {len(train_files)} cells")
    print(f"Validation set: {df_val.shape[0]} samples, {len(val_files)} cells")
    print(f"Testing set: {df_test.shape[0]} samples, 1 cell (ID: {test_cell_id})")
    
    # Print unique cell IDs in each set
    train_cells = sorted(df_train['cell_id'].unique())
    val_cells = sorted(df_val['cell_id'].unique())
    test_cells = sorted(df_test['cell_id'].unique())
    
    print(f"\nTraining cells: {train_cells}")
    print(f"Validation cells: {val_cells}")
    print(f"Test cell: {test_cells}")

    return df_train, df_val, df_test


def objective(trial):
    """
    Optuna objective function, called once per trial.
    Samples a set of hyperparameters for each trial, trains the model,
    and returns the best validation loss to Optuna.
    """
    # 1. Set random seed for reproducibility
    set_seed(42)

    # 2. Sample hyperparameters for this trial
    seq_length = trial.suggest_int("SEQUENCE_LENGTH", 144, 1008, step=144)
    hidden_size = trial.suggest_categorical("HIDDEN_SIZE", [32, 64, 128, 256])
    num_layers = trial.suggest_int("NUM_LAYERS", 2, 5)
    dropout = trial.suggest_float("DROPOUT", 0.0, 0.5, step=0.1)
    weight_decay = trial.suggest_categorical("WEIGHT_DECAY", [0.0, 1e-6, 1e-5, 1e-4])
    batch_size = trial.suggest_categorical("BATCH_SIZE", [16, 32, 64])

    # 3. Define hyperparameter dictionary
    hyperparams = {
        "SEQUENCE_LENGTH": seq_length,
        "HIDDEN_SIZE": hidden_size,
        "NUM_LAYERS": num_layers,
        "DROPOUT": dropout,
        "LEARNING_RATE": 1e-4,
        "WEIGHT_DECAY": weight_decay,
        "BATCH_SIZE": batch_size,
        "EPOCHS": 100,
        "PATIENCE": 10
    }

    # 4. Load and preprocess data with consistent splitting
    data_dir = Path("../01_Datenaufbereitung/Output/Calculated/")
    df_train, df_val, df_test = consistent_load_data(data_dir, split_file_path='optuna_dataset_split.json')

    df_train_scaled, df_val_scaled, _ = scale_data(df_train, df_val, df_test)

    train_dataset = BatteryDataset(df_train_scaled, hyperparams["SEQUENCE_LENGTH"])
    val_dataset = BatteryDataset(df_val_scaled, hyperparams["SEQUENCE_LENGTH"])

    train_loader = DataLoader(train_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=False)

    # 5. Create a separate directory for this trial to save models and training history
    trial_save_dir = optuna_dir / f"trial_{trial.number}"
    trial_save_dir.mkdir(exist_ok=True, parents=True)

    save_path = {
        'best': trial_save_dir / 'best_soh_model.pth',
        'last': trial_save_dir / 'last_soh_model.pth',
        'history': trial_save_dir / 'train_history.parquet'
    }

    # 6. Define and train model, get history and best validation loss
    model = SOHLSTM(
        input_size=3,
        hidden_size=hyperparams["HIDDEN_SIZE"],
        num_layers=hyperparams["NUM_LAYERS"],
        dropout=hyperparams["DROPOUT"]
    ).to(device)

    history, best_val_loss = train_and_validate_model(
        model,
        train_loader,
        val_loader,
        save_path
    )
    
    # 7. Save hyperparameters and best_val_loss to trial folder
    hyperparam_path = trial_save_dir / "hyperparams.json"
    with open(hyperparam_path, "w") as f:
        # Add necessary information to the dict for saving
        hyperparams["best_val_loss"] = best_val_loss
        json.dump(hyperparams, f, indent=4)
    
    # 8. Store the best epoch number as additional information in the trial
    if isinstance(history, dict) and 'val_loss' in history:
        best_epoch = np.argmin(history['val_loss']) + 1
        trial.set_user_attr('best_epoch', int(best_epoch))
    
    # 9. Return best_val_loss to Optuna
    return best_val_loss


def main():
    """
    Run Optuna hyperparameter search and output the best results.
    """
    # Create study and set direction to minimize - use fixed random seed
    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=50, timeout=None)

    print("\n============================")
    print("      Search Finished!      ")
    print("============================")
    print(f"Best trial ID: {study.best_trial.number}")
    print(f"Best trial value (Val. Loss): {study.best_trial.value:.4e}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("============================")

    # Save best hyperparameters
    best_params_path = optuna_dir / "best_hyperparams.json"
    with open(best_params_path, "w") as f:
        # Add extra information to best parameters
        best_params = study.best_trial.params.copy()
        best_params["best_val_loss"] = study.best_trial.value
        best_params["trial_number"] = study.best_trial.number
        if 'best_epoch' in study.best_trial.user_attrs:
            best_params["best_epoch"] = study.best_trial.user_attrs['best_epoch']
        json.dump(best_params, f, indent=4)

    # Save complete optimization history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(optuna_dir / "optuna_history.csv", index=False)
    
    # Save complete study object (containing all trials information)
    with open(optuna_dir / "study.pkl", "wb") as f:
        pickle.dump(study, f)


if __name__ == "__main__":
    main()