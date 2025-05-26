import optuna
import torch
from pathlib import Path
from model import Config, DataProcessor, create_dataloaders, SOHLSTM, Trainer

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define static dataset splits
# REG_TRAIN_IDS = ['03','05','07','09','11','15','21','23','25','27','29']
# REG_VAL_IDS   = ['01','13','19']

INC_BASE_TRAIN_IDS    = ['01', '03', '05', '21', '27']
INC_BASE_VAL_IDS      = ['23']
INC_UPDATE1_TRAIN_IDS = ['07','09', '11','19', '23']
INC_UPDATE1_VAL_IDS   = ['25']
INC_UPDATE2_TRAIN_IDS = ['15', '25', '29']
INC_UPDATE2_VAL_IDS   = ['13']

# Data directory (adjust as needed)
DATA_DIR = '../01_Datenaufbereitung/Output/Calculated/'

# Temporary checkpoint and results directory for Optuna trials
tmp_dir = Path('optuna_tmp_inclemental')
# (tmp_dir / 'regular').mkdir(parents=True, exist_ok=True)
(tmp_dir / 'incremental').mkdir(parents=True, exist_ok=True)

# # -------- Regular Training Function & Objective --------
# def train_regular(config, trial_dir):
#     dp = DataProcessor(
#         data_dir=DATA_DIR,
#         resample=config.RESAMPLE,
#         seed=config.SEED,
#         base_train_ids=REG_TRAIN_IDS,
#         base_val_ids=REG_VAL_IDS,
#         update1_train_ids=[], update1_val_ids=[],
#         update2_train_ids=[], update2_val_ids=[]
#     )
#     data = dp.prepare_data()
#     loaders = create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
#     model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
#     trainer = Trainer(model, device, config, checkpoint_dir=trial_dir)
#     history = trainer.train_task(
#         train_loader=loaders['base_train'],
#         val_loader=loaders['base_val'],
#         task_id=0,
#         apply_ewc=False,
#         resume=False
#     )
#     return history['val_loss'][-1]


# def objective_regular(trial):
#     # Suggest hyperparameters
#     seq_length    = trial.suggest_categorical('seq_length', [144,288,432,576,720,864,1008])
#     hidden_size   = trial.suggest_categorical('hidden_size', [32,64,128,256])
#     num_layers    = trial.suggest_categorical('num_layers', [2,3,4,5])
#     dropout       = trial.suggest_categorical('dropout', [0.0,0.1,0.2,0.3,0.4,0.5])
#     weight_decay  = trial.suggest_categorical('weight_decay', [0.0,1e-6,1e-5,1e-4])
#     batch_size    = trial.suggest_categorical('batch_size', [16,32,64,128])
#     # Fixed learning rate for regular phase
#     learning_rate = 1e-4
#     epochs        = 200
#     patience      = 20

#     # Build config
#     config = Config(
#         SEQUENCE_LENGTH=seq_length,
#         HIDDEN_SIZE=hidden_size,
#         NUM_LAYERS=num_layers,
#         DROPOUT=dropout,
#         BATCH_SIZE=batch_size,
#         LEARNING_RATE=learning_rate,
#         WEIGHT_DECAY=weight_decay,
#         EPOCHS=epochs,
#         PATIENCE=patience
#     )

#     # Unique trial checkpoint directory
#     trial_dir = tmp_dir / 'regular' / f'trial_{trial.number}'
#     trial_dir.mkdir(parents=True, exist_ok=True)

#     val_loss = train_regular(config, trial_dir)
#     return val_loss

# -------- Incremental EWC Objective --------
def train_incremental(config, lambda1, lambda2, trial_dir):
    dp = DataProcessor(
        data_dir=DATA_DIR,
        resample=config.RESAMPLE,
        config = config,
        base_train_ids=INC_BASE_TRAIN_IDS,
        base_val_ids=INC_BASE_VAL_IDS,
        update1_train_ids=INC_UPDATE1_TRAIN_IDS,
        update1_val_ids=INC_UPDATE1_VAL_IDS,
        update2_train_ids=INC_UPDATE2_TRAIN_IDS,
        update2_val_ids=INC_UPDATE2_VAL_IDS
    )
    data = dp.prepare_data()
    loaders = create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    model = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    trainer = Trainer(model, device, config, checkpoint_dir=trial_dir)

    # Phase 0: base
    trainer.train_task(loaders['base_train'], loaders['base_val'], task_id=0, apply_ewc=False, resume=False)
    # Phase 1: update1 with EWC lambda1
    config.EWC_LAMBDA = lambda1
    trainer.config = config
    hist1 = trainer.train_task(loaders['update1_train'], loaders['update1_val'], task_id=1, apply_ewc=True, resume=False)
    val1 = hist1['val_loss'][-1]
    trainer.consolidate(loaders['update1_train'])
    # Phase 2: update2 with EWC lambda2
    config.EWC_LAMBDA = lambda2
    trainer.config = config
    hist2 = trainer.train_task(loaders['update2_train'], loaders['update2_val'], task_id=2, apply_ewc=True, resume=False)
    val2 = hist2['val_loss'][-1]
    return val1 + val2


def objective_incremental(trial, best_config_params):
    # Load best regular config (includes fixed learning rate)
    config = Config(**best_config_params)
    # Suggest EWC lambdas
    lambda1 = trial.suggest_categorical('ewc_lambda_update1', [100, 300, 1000, 3000, 10000])
    lambda2 = trial.suggest_categorical('ewc_lambda_update2', [100, 300, 1000, 3000, 10000] )

    # Unique trial checkpoint directory
    trial_dir = tmp_dir / 'incremental' / f'trial_{trial.number}'
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Incremental uses same learning rate as regular
    return train_incremental(config, lambda1, lambda2, trial_dir)

if __name__ == '__main__':
    # # Study 1: Regular LSTM hyperparameters
    # study_reg = optuna.create_study(direction='minimize')
    # study_reg.optimize(objective_regular, n_trials=50)

    # # Save all trial params and values
    # df_reg = study_reg.trials_dataframe()
    # df_reg.to_csv(tmp_dir / 'regular_trials.csv', index=False)

    # # Print summary for regular
    # print('Regular study completed.')
    # print(f"Min validation loss: {study_reg.best_value:.6f}")
    # print('Best parameters:', study_reg.best_params)

    # # Prepare best config for incremental search
    # best_params = study_reg.best_params.copy()
    # best_params.update({
    #     'LEARNING_RATE': 1e-4,
    #     'EPOCHS': 200,
    #     'PATIENCE': 20
    # })
    
    best_params = {
        'SEQUENCE_LENGTH': 720,
        'HIDDEN_SIZE': 128,
        'NUM_LAYERS': 2,
        'DROPOUT': 0.3,
        'BATCH_SIZE': 32,
        'LEARNING_RATE': 1e-4,
        'WEIGHT_DECAY': 1e-6,
        'EPOCHS': 200,
        'PATIENCE': 20,
        'SEED': 42,
    }

    # Study 2: Incremental EWC lambdas
    study_inc = optuna.create_study(direction='minimize')
    study_inc.optimize(lambda trial: objective_incremental(trial, best_params), n_trials=50)

    # Save all trial params and values for incremental
    df_inc = study_inc.trials_dataframe()
    df_inc.to_csv(tmp_dir / 'incremental_trials.csv', index=False)

    # Print summary for incremental
    print('Incremental study completed.')
    print(f"Min combined validation loss: {study_inc.best_value:.6f}")
    print('Best lambdas:', study_inc.best_params)
