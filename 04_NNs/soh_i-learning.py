from pathlib import Path
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import from project modules
from models.pnn import ProgressiveNN
from models.ewc import EWC  # Import the EWC class
from utils.data_processing import load_and_prepare_data, scale_data, split_by_cell
from utils.common import set_seed
from utils.training import train_model
from utils.evaluation import evaluate_model, evaluate_pnn, plot_results, calc_metrics, plot_pnn_learning_curves
from soh_lstm import SOHLSTM, BatteryDataset

def main(model_type="parameter_based(EWC)"):
    """
    Main function to run incremental learning experiments
    
    Args:
        model_type: Type of incremental learning approach
            - "structure_based(PNN)": Progressive Neural Network
            - "parameter_based(EWC)": Elastic Weight Consolidation (EWC)
            - "data_based": Replay Buffer (not implemented)
    """
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(__file__).parent / f"models/EWC"
    save_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hyperparams = {
        "MODEL_TYPE": model_type,
        "SEQUENCE_LENGTH": 1008,
        "HIDDEN_SIZE": 128,
        "NUM_LAYERS": 3,
        "DROPOUT": 0.5,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 1e-4,
        "EPOCHS": 100,
        "PATIENCE": 10,
        "WEIGHT_DECAY": 0.0, 
        "EWC_LAMBDA": 5000,  # EWC regularization strength
        "device": str(device)
    }
    
    with open(save_dir / "hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=4)
    
    print(f"Using device: {device}")
    print(f"Current model_type: {model_type}\n")

    set_seed(42)

    data_dir = Path("../01_Datenaufbereitung/Output/Calculated/")
    # Load data
    df_base, df_update1, df_update2, df_test = load_and_prepare_data(data_dir)
    # Normalize data
    (df_base_scaled, df_update1_scaled,
     df_update2_scaled, df_test_scaled) = scale_data(
        df_base, df_update1, df_update2, df_test
    )
    
    # Split into train/validation sets by cell
    df_base_train, df_base_val = split_by_cell(df_base_scaled, "Base", val_cells=1)
    df_update1_train, df_update1_val = split_by_cell(df_update1_scaled, "Update1", val_cells=1)
    df_update2_train, df_update2_val = split_by_cell(df_update2_scaled, "Update2", val_cells=1)

    # Create datasets
    base_train_dataset = BatteryDataset(df_base_train, hyperparams["SEQUENCE_LENGTH"])
    base_val_dataset = BatteryDataset(df_base_val, hyperparams["SEQUENCE_LENGTH"])
    update1_train_dataset = BatteryDataset(df_update1_train, hyperparams["SEQUENCE_LENGTH"])
    update1_val_dataset = BatteryDataset(df_update1_val, hyperparams["SEQUENCE_LENGTH"])
    update2_train_dataset = BatteryDataset(df_update2_train, hyperparams["SEQUENCE_LENGTH"])
    update2_val_dataset = BatteryDataset(df_update2_val, hyperparams["SEQUENCE_LENGTH"])

    # Create data loaders
    base_train_loader = DataLoader(base_train_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    base_val_loader = DataLoader(base_val_dataset, batch_size=hyperparams["BATCH_SIZE"])
    update1_train_loader = DataLoader(update1_train_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    update1_val_loader = DataLoader(update1_val_dataset, batch_size=hyperparams["BATCH_SIZE"])
    update2_train_loader = DataLoader(update2_train_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    update2_val_loader = DataLoader(update2_val_dataset, batch_size=hyperparams["BATCH_SIZE"])
    
    # Split test set into 3 segments for each phase
    timestamps = df_test_scaled['Datetime'].values

    # Assuming timestamps are sorted, we can split them into 3 segments
    time_boundary1 = timestamps[len(timestamps) // 3]  
    time_boundary2 = timestamps[2 * len(timestamps) // 3]  

    # Create boolean masks for each segment
    test_idx1 = df_test_scaled['Datetime'] < time_boundary1
    test_idx2 = (df_test_scaled['Datetime'] >= time_boundary1) & (df_test_scaled['Datetime'] < time_boundary2)
    test_idx3 = df_test_scaled['Datetime'] >= time_boundary2

    # Create datasets for each segment
    test_dataset1 = BatteryDataset(df_test_scaled[test_idx1], hyperparams["SEQUENCE_LENGTH"])
    test_dataset2 = BatteryDataset(df_test_scaled[test_idx2], hyperparams["SEQUENCE_LENGTH"])
    test_dataset3 = BatteryDataset(df_test_scaled[test_idx3], hyperparams["SEQUENCE_LENGTH"])

    base_test_loader = DataLoader(test_dataset1, batch_size=hyperparams["BATCH_SIZE"])
    update1_test_loader = DataLoader(test_dataset2, batch_size=hyperparams["BATCH_SIZE"])
    update2_test_loader = DataLoader(test_dataset3, batch_size=hyperparams["BATCH_SIZE"])

    # Choose incremental learning method
    if model_type == "structure_based(PNN)":
        """
        Progressive Neural Network
        """
        print("=" * 80)
        print("Method: Progressive Neural Networks (PNN)")
        print("=" * 80)

        # Initialize PNN
        input_size = 3  # Voltage, Current, Temperature
        pnn_model = ProgressiveNN(
            input_size=input_size,
            hidden_size=hyperparams["HIDDEN_SIZE"],
            num_layers=hyperparams["NUM_LAYERS"],
            dropout=hyperparams["DROPOUT"]
        ).to(device)

        # Train base column (first task)
        print("\nTraining base column...")
        pnn_model, base_history = train_model(
            model=pnn_model,
            train_loader=base_train_loader,
            val_loader=base_val_loader,
            epochs=hyperparams["EPOCHS"],
            lr=hyperparams["LEARNING_RATE"],
            weight_decay=hyperparams["WEIGHT_DECAY"],
            patience=hyperparams["PATIENCE"]
        )

        # Save base model weights
        torch.save(pnn_model.state_dict(), save_dir / "pnn_base_model.pt")
        
        base_history_df = pd.DataFrame(base_history)
        base_history_df.to_parquet(save_dir / "pnn_base_history.parquet", index=False)
        
        # Evaluate base model
        print("\nEvaluating base model...")
        base_pred, base_targets, base_metrics = evaluate_pnn(pnn_model, base_test_loader, task_id=0)
        print(f"Base model metrics: {base_metrics}")

        # Add column for first update (second task)
        print("\nAdding column for first update...")
        task1_idx = pnn_model.add_column()
        print(f"New column added at index {task1_idx}")
        
        # Train second column while keeping first column frozen
        print("Training column for first update...")
        pnn_model, update1_history = train_model(
            model=pnn_model,
            train_loader=update1_train_loader,
            val_loader=update1_val_loader,
            epochs=hyperparams["EPOCHS"],
            lr=hyperparams["LEARNING_RATE"],
            weight_decay=hyperparams["WEIGHT_DECAY"],      
            patience=hyperparams["PATIENCE"]
        )

        # Save updated model
        torch.save(pnn_model.state_dict(), save_dir / "pnn_update1_model.pt")
        
        update1_history_df = pd.DataFrame(update1_history)
        update1_history_df.to_parquet(save_dir / "pnn_update1_history.parquet", index=False)
        
        # Evaluate after first update
        print("\nEvaluating after first update...")
        update1_pred, update1_targets, update1_metrics = evaluate_pnn(pnn_model, update1_test_loader, task_id=task1_idx)
        print(f"Update 1 metrics: {update1_metrics}")

        # Add column for second update (third task)
        print("\nAdding column for second update...")
        task2_idx = pnn_model.add_column()
        print(f"New column added at index {task2_idx}")
        
        # Train third column while keeping first and second columns frozen
        print("Training column for second update...")
        pnn_model, update2_history = train_model(
            model=pnn_model,
            train_loader=update2_train_loader,
            val_loader=update2_val_loader,
            epochs=hyperparams["EPOCHS"],
            lr=hyperparams["LEARNING_RATE"],
            weight_decay=hyperparams["WEIGHT_DECAY"],
            patience=hyperparams["PATIENCE"]
        )

        # Save final model
        torch.save(pnn_model.state_dict(), save_dir / "pnn_update2_model.pt")
        
        update2_history_df = pd.DataFrame(update2_history)
        update2_history_df.to_parquet(save_dir / "pnn_update2_history.parquet", index=False)
        
        # Evaluate after second update
        print("\nEvaluating after second update...")
        update2_pred, update2_targets, update2_metrics = evaluate_pnn(pnn_model, update2_test_loader, task_id=task2_idx)
        print(f"Update 2 metrics: {update2_metrics}")

        # Plot results and save figures
        plot_results(save_dir, "Structure based(PNN)", df_test_scaled, hyperparams["SEQUENCE_LENGTH"], 
                     base_pred, update1_pred, update2_pred, base_metrics, update1_metrics, update2_metrics)
        
        # Plot learning curves
        plot_pnn_learning_curves(
            base_history, update1_history, update2_history, 
            save_dir, "pnn_learning_curves.png"
        )
        
        print("PNN Incremental Learning Finished!")

    elif model_type == "parameter_based(EWC)":
        """
        Elastic Weight Consolidation (EWC)
        """
        print("=" * 80)
        print("Method: Elastic Weight Consolidation (EWC)")
        print("=" * 80)
        
        # Initialize the standard LSTM model
        input_size = 3  # Voltage, Current, Temperature
        base_model = SOHLSTM(
            input_size=input_size,
            hidden_size=hyperparams["HIDDEN_SIZE"],
            num_layers=hyperparams["NUM_LAYERS"],
            dropout=hyperparams["DROPOUT"]
        ).to(device)
        
        # Train on base data
        print("\nTraining on base data...")
        base_model, base_history = train_model(
            model=base_model,
            train_loader=base_train_loader,
            val_loader=base_val_loader,
            epochs=hyperparams["EPOCHS"],
            lr=hyperparams["LEARNING_RATE"],
            weight_decay=hyperparams["WEIGHT_DECAY"],
            patience=hyperparams["PATIENCE"]
        )
        
        # Save base model
        torch.save(base_model.state_dict(), save_dir / "ewc_base_model.pt")
        
        # Convert history to DataFrame and save
        base_history_df = pd.DataFrame(base_history)
        base_history_df.to_parquet(save_dir / "ewc_base_history.parquet", index=False)
        
        # Evaluate base model
        print("\nEvaluating base model...")
        base_pred, base_targets, base_metrics, _ = evaluate_model(base_model, base_test_loader)
        print(f"Base model metrics: {base_metrics}")
        
        # Initialize EWC after training on base data
        ewc = EWC(base_model, device, lambda_ewc=hyperparams["EWC_LAMBDA"])
        
        # Compute Fisher information matrix using base data
        print("\nComputing Fisher information matrix...")
        ewc.compute_fisher(base_train_loader)
        
        # Update optimal parameters
        ewc.update_optimal_params()
        
        # Train on update1 data with EWC regularization
        print("\nTraining on update1 data with EWC regularization...")
        base_model, update1_history = train_model(
            model=base_model,
            train_loader=update1_train_loader,
            val_loader=update1_val_loader,
            epochs=hyperparams["EPOCHS"],
            lr=hyperparams["LEARNING_RATE"],
            weight_decay=hyperparams["WEIGHT_DECAY"],
            patience=hyperparams["PATIENCE"],
            ewc=ewc  # Pass EWC object for regularization
        )
        
        # Save model after first update
        torch.save(base_model.state_dict(), save_dir / "ewc_update1_model.pt")
        
        # Convert history to DataFrame and save
        update1_history_df = pd.DataFrame(update1_history)
        update1_history_df.to_parquet(save_dir / "ewc_update1_history.parquet", index=False)
        
        # Evaluate after first update
        print("\nEvaluating after first update...")
        update1_pred, update1_targets, update1_metrics, _ = evaluate_model(base_model, update1_test_loader)
        print(f"Update 1 metrics: {update1_metrics}")
        
        # Update Fisher information matrix and optimal parameters for second update
        print("\nUpdating Fisher information matrix...")
        # Combine base and update1 data for Fisher calculation
        combined_loader = DataLoader(
            torch.utils.data.ConcatDataset([base_train_dataset, update1_train_dataset]),
            batch_size=hyperparams["BATCH_SIZE"],
            shuffle=True
        )
        ewc.compute_fisher(combined_loader)
        ewc.update_optimal_params()
        
        # Train on update2 data with updated EWC regularization
        print("\nTraining on update2 data with EWC regularization...")
        base_model, update2_history = train_model(
            model=base_model,
            train_loader=update2_train_loader,
            val_loader=update2_val_loader,
            epochs=hyperparams["EPOCHS"],
            lr=hyperparams["LEARNING_RATE"],
            weight_decay=hyperparams["WEIGHT_DECAY"],
            patience=hyperparams["PATIENCE"],
            ewc=ewc  # Pass updated EWC object
        )
        
        # Save final model
        torch.save(base_model.state_dict(), save_dir / "ewc_update2_model.pt")
        
        # Convert history to DataFrame and save
        update2_history_df = pd.DataFrame(update2_history)
        update2_history_df.to_parquet(save_dir / "ewc_update2_history.parquet", index=False)
        
        # Evaluate after second update
        print("\nEvaluating after second update...")
        update2_pred, update2_targets, update2_metrics, _ = evaluate_model(base_model, update2_test_loader)
        print(f"Update 2 metrics: {update2_metrics}")
        
        # Plot results
        plot_results(save_dir, "Parameter based(EWC)", df_test_scaled, hyperparams["SEQUENCE_LENGTH"],
                     base_pred, update1_pred, update2_pred, base_metrics, update1_metrics, update2_metrics)
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        
        # Plot base training and validation loss
        plt.plot(base_history["epoch"], base_history["train_loss"], label="Base Training Loss")
        plt.plot(base_history["epoch"], base_history["val_loss"], label="Base Validation Loss")
        
        # Plot update1 training and validation loss (shifted epochs)
        epoch_shift = max(base_history["epoch"])
        plt.plot([e + epoch_shift for e in update1_history["epoch"]], 
                 update1_history["train_loss"], label="Update1 Training Loss")
        plt.plot([e + epoch_shift for e in update1_history["epoch"]], 
                 update1_history["val_loss"], label="Update1 Validation Loss")
        
        # Plot update2 training and validation loss (shifted epochs)
        epoch_shift += max(update1_history["epoch"])
        plt.plot([e + epoch_shift for e in update2_history["epoch"]], 
                 update2_history["train_loss"], label="Update2 Training Loss")
        plt.plot([e + epoch_shift for e in update2_history["epoch"]], 
                 update2_history["val_loss"], label="Update2 Validation Loss")
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title("EWC Training and Validation Loss")
        
        results_dir = Path(save_dir) / "results"
        results_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(results_dir / "ewc_learning_curves.png")
        plt.close()
        
        print("EWC Incremental Learning Finished!")
        
    elif model_type == "data_based":
        """
        Data-Based Method (e.g., Replay Buffer)
        """
        print("=" * 80)
        print("Method: Data-Based Method - Not Implemented")
        print("This option will be implemented in a future update.")
        print("=" * 80)
        
    else:
        print(f"Unknown model_type: {model_type}. Please choose 'structure_based(PNN)', 'parameter_based(EWC)', or 'data_based'.")

    print("\nAll Done.")

if __name__ == "__main__":
    main("parameter_based(EWC)")  # Default to PNN