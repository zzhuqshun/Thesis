import copy
import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs, lr=1e-4, weight_decay=1e-4, patience=10,
                ewc=None, replay_buffer=None, replay_ratio=0.3):
    """
    Generic model training function with optional EWC or Replay Buffer support.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Maximum number of training epochs
        lr: Learning rate (default: 1e-4)
        weight_decay: L2 regularization strength (default: 1e-4)
        patience: Early stopping patience in epochs (default: 10)
        ewc: Optional EWC object for regularization
        replay_buffer: Optional replay buffer for experience replay
        replay_ratio: Ratio of replay data to current data (default: 0.3)
        
    Returns:
        model: Trained model with best validation performance
        history: Dictionary containing training history (epochs, train_loss, val_loss)
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    # Initialize training history dictionary
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', leave=False) as pbar:
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                
                # Mix in old task data if replay buffer is available
                if replay_buffer is not None and len(replay_buffer) > 0:
                    replay_batch_size = int(features.shape[0] * replay_ratio)
                    if replay_batch_size > 0:
                        old_feat, old_lbl = replay_buffer.get_batch(replay_batch_size)
                        old_feat, old_lbl = old_feat.to(device), old_lbl.to(device)
                        features = torch.cat([features, old_feat], dim=0)
                        labels = torch.cat([labels, old_lbl], dim=0)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Add EWC regularization if available
                if ewc is not None:
                    loss += ewc.ewc_loss()
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                
                train_loss += loss.item()
                pbar.update(1)

                # Add current batch to replay buffer if available 
                if replay_buffer is not None:
                    features_cpu = features.detach().cpu()
                    labels_cpu = labels.detach().cpu()
                    replay_buffer.add_batch(features_cpu, labels_cpu)

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.3e} | "
              f"Val Loss: {val_loss:.3e} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Record metrics for current epoch
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Save best model and check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
    
    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history