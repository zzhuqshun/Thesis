import optuna
import torch
from pathlib import Path
from lstm import Config, DataProcessor, create_dataloaders, SOHLSTM, Trainer, set_seed
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '../01_Datenaufbereitung/Output/Calculated/'
TMP_ROOT = Path('hyperparams_cellsplit')
TMP_ROOT.mkdir(parents=True, exist_ok=True)

def train_regular(config, trial_dir):
    set_seed(config.SEED)
    dp = DataProcessor(
        data_dir=DATA_DIR,
        config=config
    )
    datasets = dp.prepare_data()
    loaders = create_dataloaders(datasets, config)
    model = SOHLSTM(len(config.FEATURES_COLS), config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    trainer = Trainer(model, device, config)
    history = trainer.train_task(
        train_loader=loaders['joint_train'],
        val_loader=loaders['joint_val'],
        task_id=0,
        apply_ewc=False,
        alpha_lwf=0.0
    )

    # Save history and config in trial_dir
    trial_dir.mkdir(parents=True, exist_ok=True)
    config.save(trial_dir / 'config.json')
    with open(trial_dir / 'history.json', 'w') as f:
        json.dump(history, f)

    best_val = min(h['val_loss'] for h in history)
    return best_val

def objective_regular(trial):
    seq_length    = trial.suggest_int('sequence_length', 6, 1008, step=6)
    hidden_size   = trial.suggest_categorical('hidden_size', [32,64,128,256])
    num_layers    = trial.suggest_categorical('num_layers', [2,3,4,5])
    batch_size    = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout       = trial.suggest_categorical('dropout', [0.0,0.1,0.2,0.3,0.4,0.5])
    weight_decay  = trial.suggest_categorical('weight_decay', [0.0,1e-6,1e-5,1e-4])
    learning_rate = 1e-4
    epochs        = 200
    patience      = 20

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
        SCALER='RobustScaler',
        SEED=42,
        MODE='joint',
    )

    trial_dir = TMP_ROOT / f'trial_{trial.number}'
    best_val = train_regular(config, trial_dir)
    print(f"[Trial {trial.number}] best val={best_val:.4e}, params={trial.params}")
    return best_val 

if __name__ == '__main__':
    study_reg = optuna.create_study(direction='minimize')
    study_reg.optimize(objective_regular, n_trials=50)

    df_reg = study_reg.trials_dataframe()
    df_reg.to_csv(TMP_ROOT / 'regular_trials.csv', index=False)

    print('Regular study completed.')
    print(f"Min validation loss: {study_reg.best_value:.4e}")
    print('Best parameters:', study_reg.best_params)
