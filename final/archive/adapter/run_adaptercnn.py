import logging
import torch
import pandas as pd
from pathlib import Path

from utils.config import Config
from utils.utils import set_seed, setup_logging, print_model_summary
from utils.data import DataProcessor, create_dataloaders
from utils.adapter import AdapterCNN, AdapterTrainer
from utils.evaluate import plot_losses, evaluate_incremental_learning

logger = logging.getLogger(__name__)

def adapter_learning(config):
    """
    Incremental learning with cumulative adapters.
    Task0: Train LSTM + Adapters together
    Task1-2: Freeze LSTM, only train adapters (cumulative learning)
    """
    logger.info("==== Incremental Learning with Adapters ====")
    
    # Setup directories
    inc_dir = config.BASE_DIR
    inc_dir.mkdir(parents=True, exist_ok=True)
    
    # Get number of tasks
    num_tasks = config.NUM_TASKS
    logger.info("Number of tasks: %d", num_tasks)
    
    # Prepare incremental learning data
    dp = DataProcessor(config.DATA_DIR, config.RESAMPLE, config)
    data = dp.prepare_incremental_data(config.incremental_datasets)
    loaders = create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    
    # Initialize model with adapters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AdapterCNN(
        input_size=3, 
        hidden_size=config.HIDDEN_SIZE,
        dropout=config.DROPOUT,
        kernel_size=5,
        reduction_factor=8  # Bottleneck to hidden_size/8
    ).to(device)
    
    # Initialize trainer
    trainer = AdapterTrainer(model, device, config, inc_dir)
    
    # Print model summary
    logger.info("Model architecture with adapters:")
    print_model_summary(model)
    
    # Log adapter configuration
    logger.info("Adapter configuration:")
    logger.info("  Kernel size: 5")
    logger.info("  Reduction factor: 8")
    logger.info("  Bottleneck size: %d", config.HIDDEN_SIZE // 8)
    
    # Sequential task training with cumulative adapter learning
    for task_idx in range(num_tasks):
        task_name = f"task{task_idx}"
        
        logger.info("--- Training %s ---", task_name)
        
        # Setup task directory
        task_dir = inc_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        trainer.task_dir = task_dir
        
        # Set task-specific random seed for reproducibility
        set_seed(config.SEED + task_idx)
        
        # Determine whether to freeze base model
        # Task0: Train LSTM

        # Task1-2: Freeze LSTM, only train adapters
        freeze_base = (task_idx > 0)
        
        if freeze_base:
            logger.info("Task %d: Freezing base LSTM, training adapters + FC", task_idx)
        else:
            logger.info("Task %d: Training base LSTM + FC", task_idx)
        
        # Train on current task
        history = trainer.train_task(
            loaders[f"{task_name}_train"], 
            loaders[f"{task_name}_val"], 
            task_idx,
            freeze_base=freeze_base
        )
        
        # Save training history and visualizations
        pd.DataFrame(history).to_csv(task_dir / 'training_history.csv', index=False)
        plot_losses(history, task_dir)
        
        # Log task completion
        logger.info("Task %d completed. Best val loss: %.4e", 
                   task_idx, min(history['val_loss']))
        
        # Evaluate on current task's test set
        if f'test_{task_name}' in loaders:
            from utils.evaluate import evaluate
            _, _, metrics = evaluate(
                model, 
                loaders[f'test_{task_name}'], 
                alpha=config.ALPHA,
                log=False
            )
            logger.info("Task %d test performance - MAE: %.4e, R2: %.4f",
                       task_idx, metrics['MAE'], metrics['R2'])
    
    logger.info("==== Incremental Training with Adapters Complete ====")
    
    # Comprehensive evaluation on all test sets
    evaluate_incremental_learning(config, inc_dir, loaders, device)
    
    # Additional analysis: Compare adapter parameters across tasks
    analyze_adapter_evolution(inc_dir, num_tasks, device, config)

def analyze_adapter_evolution(inc_dir, num_tasks, device, config):
    """
    Analyze how adapter parameters evolved across tasks.
    This helps understand what the adapters learned.
    """
    logger.info("==== Analyzing Adapter Evolution ====")
    
    analysis_dir = inc_dir / 'adapter_analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoints and compare adapter parameters
    adapter_stats = []
    
    for task_idx in range(num_tasks):
        checkpoint_path = inc_dir / f"task{task_idx}" / f"task{task_idx}_best.pt"
        if not checkpoint_path.exists():
            continue
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state = checkpoint['model_state']
        
        # Extract adapter parameters
        adapter1_gate = model_state['adapter1.gate'].item()
        adapter2_gate = model_state['adapter2.gate'].item()
        
        # Calculate parameter norms
        adapter1_norm = sum(
            model_state[k].norm().item() 
            for k in model_state.keys() 
            if k.startswith('adapter1.') and 'gate' not in k
        )
        adapter2_norm = sum(
            model_state[k].norm().item() 
            for k in model_state.keys() 
            if k.startswith('adapter2.') and 'gate' not in k
        )
        
        adapter_stats.append({
            'task': task_idx,
            'adapter1_gate': adapter1_gate,
            'adapter2_gate': adapter2_gate,
            'adapter1_norm': adapter1_norm,
            'adapter2_norm': adapter2_norm
        })
        
        logger.info("Task %d - Adapter1 gate: %.4f, Adapter2 gate: %.4f",
                   task_idx, adapter1_gate, adapter2_gate)
    
    # Save adapter statistics
    if adapter_stats:
        stats_df = pd.DataFrame(adapter_stats)
        stats_df.to_csv(analysis_dir / 'adapter_evolution.csv', index=False)
        
        # Plot adapter evolution
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Gate values
        axes[0, 0].plot(stats_df['task'], stats_df['adapter1_gate'], 'o-', label='Adapter1')
        axes[0, 0].plot(stats_df['task'], stats_df['adapter2_gate'], 's-', label='Adapter2')
        axes[0, 0].set_xlabel('Task')
        axes[0, 0].set_ylabel('Gate Value')
        axes[0, 0].set_title('Adapter Gate Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Parameter norms
        axes[0, 1].plot(stats_df['task'], stats_df['adapter1_norm'], 'o-', label='Adapter1')
        axes[0, 1].plot(stats_df['task'], stats_df['adapter2_norm'], 's-', label='Adapter2')
        axes[0, 1].set_xlabel('Task')
        axes[0, 1].set_ylabel('Parameter Norm')
        axes[0, 1].set_title('Adapter Parameter Norm Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Gate ratio
        axes[1, 0].bar(stats_df['task'], 
                      stats_df['adapter2_gate'] / (stats_df['adapter1_gate'] + 1e-8))
        axes[1, 0].set_xlabel('Task')
        axes[1, 0].set_ylabel('Gate Ratio (Adapter2/Adapter1)')
        axes[1, 0].set_title('Relative Adapter Importance')
        axes[1, 0].grid(True)
        
        # Norm ratio
        axes[1, 1].bar(stats_df['task'], 
                      stats_df['adapter2_norm'] / (stats_df['adapter1_norm'] + 1e-8))
        axes[1, 1].set_xlabel('Task')
        axes[1, 1].set_ylabel('Norm Ratio (Adapter2/Adapter1)')
        axes[1, 1].set_title('Relative Parameter Magnitude')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(analysis_dir / 'adapter_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Adapter evolution analysis saved to: %s", analysis_dir)

def main():
    """
    Run incremental learning with adapters.
    This function initializes configuration, sets up the environment,
    and starts the adapter-based incremental learning process.
    """
    # Initialize configuration
    config = Config()
    config.MODE = "incremental"
    set_seed(config.SEED)
    
    # Set up directories
    config.BASE_DIR = Path.cwd() / 'cnn_adapter_learning'
    config.BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(config.BASE_DIR)
    
    # Save configuration
    config.save(config.BASE_DIR / 'config.json')
    
    # Log experimental setup
    logger.info("=" * 60)
    logger.info("Incremental Learning with Cumulative Adapters")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("  Base directory: %s", config.BASE_DIR)
    logger.info("  Number of tasks: %d", config.NUM_TASKS)
    logger.info("  Sequence length: %d", config.SEQUENCE_LENGTH)
    logger.info("  Hidden size: %d", config.HIDDEN_SIZE)
    logger.info("  Batch size: %d", config.BATCH_SIZE)
    logger.info("  Learning rate: %.2e", config.LEARNING_RATE)
    logger.info("  Epochs: %d", config.EPOCHS)
    logger.info("  Patience: %d", config.PATIENCE)
    logger.info("=" * 60)
    
    # Start adapter-based incremental learning
    adapter_learning(config)
    
    logger.info("=" * 60)
    logger.info("Experiment completed successfully!")
    logger.info("Results saved to: %s", config.BASE_DIR)
    logger.info("=" * 60)

if __name__ == '__main__':
    main()