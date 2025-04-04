from pathlib import Path
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# Import from project modules
from models.pnn import ProgressiveNN
from utils.data_processing import load_and_prepare_data, scale_data, split_by_cell
from utils.common import set_seed
from utils.training import train_model
from utils.visualization import evaluate_pnn, plot_results
from soh_lstm import SOHLSTM, BatteryDataset


def main(model_type="structure_based(PNN)"):
    """
    Main function to run incremental learning experiments
    
    Args:
        model_type: Type of incremental learning approach
            - "structure_based(PNN)": Progressive Neural Network
            - "parameter_based": Elastic Weight Consolidation (EWC)
            - "data_based": Replay Buffer
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(__file__).parent / f"models/incremental_learning_{timestamp}"
    save_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hyperparams = {
        "MODEL_TYPE": model_type,
        "SEQUENCE_LENGTH": 1008,
        "HIDDEN_SIZE": 128,
        "NUM_LAYERS": 3,
        "DROPOUT": 0.4,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 1e-4,
        "EPOCHS": 100,
        "PATIENCE": 10,
        "WEIGHT_DECAY": 1e-4, 
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
    test_dataset = BatteryDataset(df_test_scaled, hyperparams["SEQUENCE_LENGTH"])

    # Create data loaders
    base_train_loader = DataLoader(base_train_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    base_val_loader = DataLoader(base_val_dataset, batch_size=hyperparams["BATCH_SIZE"])
    update1_train_loader = DataLoader(update1_train_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    update1_val_loader = DataLoader(update1_val_dataset, batch_size=hyperparams["BATCH_SIZE"])
    update2_train_loader = DataLoader(update2_train_dataset, batch_size=hyperparams["BATCH_SIZE"], shuffle=True)
    update2_val_loader = DataLoader(update2_val_dataset, batch_size=hyperparams["BATCH_SIZE"])
    
    # Split test set into 3 segments for each phase
    n = len(test_dataset)
    seg1 = Subset(test_dataset, list(range(0, n // 3)))
    seg2 = Subset(test_dataset, list(range(n // 3, 2 * n // 3)))
    seg3 = Subset(test_dataset, list(range(2 * n // 3, n)))
    base_test = DataLoader(seg1, batch_size=hyperparams["BATCH_SIZE"])
    update1_test = DataLoader(seg2, batch_size=hyperparams["BATCH_SIZE"])
    update2_test = DataLoader(seg3, batch_size=hyperparams["BATCH_SIZE"])

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
        
        # Evaluate base model
        print("\nEvaluating base model...")
        base_pred, base_targets, base_metrics = evaluate_pnn(pnn_model, base_test, task_id=0)
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
            epochs=hyperparams["EPOCHS"] // 2,
            lr=hyperparams["LEARNING_RATE"],
            weight_decay=hyperparams["WEIGHT_DECAY"],      
            patience=hyperparams["PATIENCE"]
        )

        # Save updated model
        torch.save(pnn_model.state_dict(), save_dir / "pnn_update1_model.pt")
        
        # Evaluate after first update
        print("\nEvaluating after first update...")
        update1_pred, update1_targets, update1_metrics = evaluate_pnn(pnn_model, update1_test, task_id=task1_idx)
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
            epochs=hyperparams["EPOCHS"] // 2,
            lr=hyperparams["LEARNING_RATE"],
            weight_decay=hyperparams["WEIGHT_DECAY"],
            patience=hyperparams["PATIENCE"]
        )

        # Save final model
        torch.save(pnn_model.state_dict(), save_dir / "pnn_update2_model.pt")
        
        # Evaluate after second update
        print("\nEvaluating after second update...")
        update2_pred, update2_targets, update2_metrics = evaluate_pnn(pnn_model, update2_test, task_id=task2_idx)
        print(f"Update 2 metrics: {update2_metrics}")

        # Plot results and save figures
        plot_results(save_dir, "Structure based(PNN)", df_test_scaled, hyperparams["SEQUENCE_LENGTH"], 
                     base_pred, update1_pred, update2_pred, base_metrics, update1_metrics, update2_metrics)
        
        # Save learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(base_history["epoch"], base_history["train_loss"], label="Base Training Loss")
        plt.plot(base_history["epoch"], base_history["val_loss"], label="Base Validation Loss")
        plt.plot([e + max(base_history["epoch"]) for e in update1_history["epoch"]], 
                 update1_history["train_loss"], label="Update1 Training Loss")
        plt.plot([e + max(base_history["epoch"]) for e in update1_history["epoch"]], 
                 update1_history["val_loss"], label="Update1 Validation Loss")
        plt.plot([e + max(base_history["epoch"]) + max(update1_history["epoch"]) for e in update2_history["epoch"]], 
                 update2_history["train_loss"], label="Update2 Training Loss")
        plt.plot([e + max(base_history["epoch"]) + max(update1_history["epoch"]) for e in update2_history["epoch"]], 
                 update2_history["val_loss"], label="Update2 Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title("PNN Training and Validation Loss")
        plt.savefig(save_dir / "pnn_learning_curves.png")
        
        print("PNN Incremental Learning Finished!")

    elif model_type != "structure_based(PNN)":
        # Initialize the standard LSTM model for other methods
        input_size = 3  # Voltage, Current, Temperature
        base_model = SOHLSTM(
            input_size=input_size,
            hidden_size=hyperparams["HIDDEN_SIZE"],
            num_layers=hyperparams["NUM_LAYERS"],
            dropout=hyperparams["DROPOUT"]
        ).to(device)
        
        # Train on base data
        base_model, _ = train_model(
            model=base_model,
            train_loader=base_train_loader,
            val_loader=base_val_loader,
            epochs=hyperparams["EPOCHS"],
            lr=hyperparams["LEARNING_RATE"],
            weight_decay=hyperparams["WEIGHT_DECAY"],
            patience=hyperparams["PATIENCE"]
        )
        
        # Note: The commented code for EWC and Replay Buffer would go here
        # Left commented as it wasn't part of the improvement request

    else:
        print(f"Unknown model_type: {model_type}. Please choose 'structure_based(PNN)', 'parameter_based', or 'data_based'.")

    print("\nAll Done.")

if __name__ == "__main__":
    main("structure_based(PNN)")  # Default to PNN