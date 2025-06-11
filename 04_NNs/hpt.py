import optuna
import torch
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from model import Config, DataProcessor, create_dataloaders, SOHLSTM, Trainer, set_seed, EWC

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define static dataset splits
REG_TRAIN_IDS = ['03','05','07','09','11','15','21','23','25','27','29']
REG_VAL_IDS   = ['01','13','19']

# Data directory (adjust as needed)
DATA_DIR = '../01_Datenaufbereitung/Output/Calculated/'

# Temporary checkpoint and results directory for Optuna trials
TMP_ROOT = Path('hyperparams_search')
TMP_ROOT.mkdir(parents=True, exist_ok=True)

# -------- Regular Training Function & Objective --------
def train_regular(config, trial_dir):
    set_seed(config.SEED)  # Set seed for reproducibility
    dp = DataProcessor(
        data_dir=DATA_DIR,
        resample=config.RESAMPLE,
        config=config,
        base_train_ids=REG_TRAIN_IDS,
        base_val_ids=REG_VAL_IDS,
        update1_train_ids=[], update1_val_ids=[],
        update2_train_ids=[], update2_val_ids=[]
    )
    data = dp.prepare_data()
    loaders = create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    trainer = Trainer(model, device, config, checkpoint_dir=trial_dir)
    history = trainer.train_task(
        train_loader=loaders['base_train'],
        val_loader=loaders['base_val'],
        task_id=0,
        apply_ewc=False,
        resume=False
    )
    return history['val_loss'][-1]


def objective_regular(trial):
    # Suggest hyperparameters
    seq_length    = trial.suggest_categorical('seq_length', [144,288,432,576,720,864,1008])
    hidden_size   = trial.suggest_categorical('hidden_size', [32,64,128,256])
    num_layers    = trial.suggest_categorical('num_layers', [2,3,4,5])
    dropout       = trial.suggest_categorical('dropout', [0.0,0.1,0.2,0.3,0.4,0.5])
    weight_decay  = trial.suggest_categorical('weight_decay', [0.0,1e-6,1e-5,1e-4])
    batch_size    = trial.suggest_categorical('batch_size', [16,32,64,128])
    # Fixed learning rate for regular phase
    learning_rate = 1e-4
    epochs        = 200
    patience      = 20

    # Build config
    config = Config(
        SEQUENCE_LENGTH=seq_length,
        HIDDEN_SIZE=hidden_size,
        NUM_LAYERS=num_layers,
        DROPOUT=dropout,
        BATCH_SIZE=batch_size,
        LEARNING_RATE=learning_rate,
        WEIGHT_DECAY=weight_decay,
        EPOCHS=epochs,
        PATIENCE=patience,
        SCALER = 'RobustScaler',  # Fixed scaler for regular phase
        SEED = 42,  # Fixed seed for reproducibility
    )

    # Unique trial checkpoint directory
    trial_dir = TMP_ROOT  / f'trial_{trial.number}'
    trial_dir.mkdir(parents=True, exist_ok=True)

    val_loss = train_regular(config, trial_dir)
    return val_loss


if __name__ == '__main__':
    # Regular LSTM hyperparameters
    study_reg = optuna.create_study(direction='minimize')
    study_reg.optimize(objective_regular, n_trials=50)

    # Save all trial params and values
    df_reg = study_reg.trials_dataframe()
    df_reg.to_csv(TMP_ROOT / 'regular_trials.csv', index=False)

    # Print summary for regular
    print('Regular study completed.')
    print(f"Min validation loss: {study_reg.best_value:.6f}")
    print('Best parameters:', study_reg.best_params)
    
