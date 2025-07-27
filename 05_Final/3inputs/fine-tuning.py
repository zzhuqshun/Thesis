#!/usr/bin/env python3
"""
Simple fine-tuning implementation for battery SOH prediction
Split into 3 tasks, training only
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import from existing model.py
from model import (
    Config,
    BatteryDataset,
    DataProcessor,
    SOHLSTM,
    set_seed,
    create_dataloaders
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ===============================================================
# Training Function
# ===============================================================
def train_task(model, train_loader, val_loader, config, task_name, device):
    """Train model for one task"""
    logger.info(f"Training {task_name}")
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += F.mse_loss(pred, y).item() * x.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4e} | "
            f"Val Loss: {val_loss:.4e} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), config.BASE_DIR / f'{task_name}_best.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= config.PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(config.BASE_DIR / f'{task_name}_best.pt'))
    return model

# ===============================================================
# Main
# ===============================================================
def main():
    # Configuration
    config = Config()
    config.MODE = "finetune"
    config.BASE_DIR = Path.cwd() / "fine_tuning_results"
    # config.LWF_ALPHAS = [0.0, 0.0, 0.0]
    # config.EWC_LAMBDAS = [0.0, 0.0, 0.0]
    
    config.BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    set_seed(config.SEED)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = SOHLSTM(
        input_size=3,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    # Data processor
    dp = DataProcessor(
        data_dir=config.DATA_DIR,
        resample=config.RESAMPLE,
        config=config
    )
    
    # Prepare incremental data
    datasets = dp.prepare_incremental_data(config.incremental_datasets)
    loaders = create_dataloaders(datasets, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    
    # Train through tasks sequentially
    for i in range(3):

        task_name = f'task{i}'
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting {task_name}")
        logger.info(f"{'='*50}")
        
        train_loader = loaders[f'{task_name}_train']
        val_loader = loaders[f'{task_name}_val']
        set_seed(config.SEED + i)
        # Train
        model = train_task(model, train_loader, val_loader, config, task_name, device)
        
        logger.info(f"{task_name} training completed")
    
    logger.info(f"\nAll tasks completed!")

if __name__ == '__main__':
    main()