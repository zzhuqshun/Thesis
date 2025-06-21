import optuna
import torch
from pathlib import Path
from model import Config, DataProcessor, create_dataloaders, SOHLSTM, Trainer, set_seed, EWC

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATA_DIR = '../01_Datenaufbereitung/Output/Calculated/'
TMP_ROOT = Path('lambda_search_Seq576')
TMP_ROOT.mkdir(parents=True, exist_ok=True)

INC_BASE_TRAIN_IDS    = ['03', '05', '07', '27']
INC_BASE_VAL_IDS      = ['01']
INC_UPDATE1_TRAIN_IDS = ['21', '23', '25']
INC_UPDATE1_VAL_IDS   = ['19']
INC_UPDATE2_TRAIN_IDS = ['09', '11', '15', '29']
INC_UPDATE2_VAL_IDS   = ['13']

# ———— 1. 先固定跑一次 Base 阶段 ————
def prepare_base(config):
    base_dir = TMP_ROOT / 'base'
    base_dir.mkdir(parents=True, exist_ok=True)

    dp = DataProcessor(
        data_dir=DATA_DIR,
        resample=config.RESAMPLE,
        config=config,
        base_train_ids=INC_BASE_TRAIN_IDS,
        base_val_ids=INC_BASE_VAL_IDS,
        update1_train_ids=[], update1_val_ids=[],
        update2_train_ids=[], update2_val_ids=[]
    )
    data = dp.prepare_data()
    loaders = create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)

    model   = SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    trainer = Trainer(model, device, config, checkpoint_dir=base_dir)

    # 用固定 seed
    set_seed(config.SEED)
    trainer.train_task(
        train_loader=loaders['base_train'],
        val_loader=loaders['base_val'],
        task_id=0,
        apply_ewc=False,
        resume=False
    )
    # Consolidate Base model
    trainer.consolidate(loaders['base_train'])
    # 保存 Base 的状态和空 ewc_tasks
    ckpt0 = base_dir / 'task0_best.pt'
    state = torch.load(ckpt0, map_location=device)
    torch.save({
        'model_state': state['model_state'],
        'ewc_tasks': [
            { 'params': {n:p.clone().cpu() for n,p in e.params.items()},
              'fisher': {n:f.clone().cpu() for n,f in e.fisher.items()} }
            for e in trainer.ewc_tasks
        ],
            
    }, base_dir / 'base_checkpoint.pt')

    return  base_dir / 'base_checkpoint.pt'

# ———— 2. 修改 train_incremental：不再跑 Base 了 ————
def train_incremental(config, lambda1, lambda2, trial_id, base_ckpt_path):
    # 加载 Base checkpoint & ewc_tasks
    trainer = Trainer(SOHLSTM(3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device),
                      device, config, checkpoint_dir=None)
    base_state = torch.load(base_ckpt_path, map_location=device)
    trainer.model.load_state_dict(base_state['model_state'])
    trainer.ewc_tasks = []
    for data in base_state['ewc_tasks']:
        e = EWC.__new__(EWC)
        e.params = {n:p.to(device)  for n,p in data['params'].items()}
        e.fisher = {n:f.to(device)  for n,f in data['fisher'].items()}
        trainer.ewc_tasks.append(e)

    # 复用 DataProcessor 只拿 loaders
    dp = DataProcessor(
        data_dir=DATA_DIR,
        resample=config.RESAMPLE,
        config=config,
        base_train_ids=INC_BASE_TRAIN_IDS, base_val_ids=INC_BASE_VAL_IDS,
        update1_train_ids=INC_UPDATE1_TRAIN_IDS, update1_val_ids=INC_UPDATE1_VAL_IDS,
        update2_train_ids=INC_UPDATE2_TRAIN_IDS, update2_val_ids=INC_UPDATE2_VAL_IDS
    )
    data = dp.prepare_data()
    loaders = create_dataloaders(data, config.SEQUENCE_LENGTH, config.BATCH_SIZE)

    # — Phase 1：Update1 + EWC(lambda1)
    set_seed(config.SEED)       # 固定 seed
    config.EWC_LAMBDA = lambda1
    trainer.config = config
    up1_dir = TMP_ROOT / f"trial_{trial_id}" / "update1"
    up1_dir.mkdir(parents=True, exist_ok=True)
    trainer.checkpoint_dir = up1_dir
        

    hist1 = trainer.train_task(
        train_loader=loaders['update1_train'],
        val_loader=loaders['update1_val'],
        task_id=1,
        apply_ewc=True,
        resume=False
    )
    val1 = hist1['val_loss'][-1]
    trainer.consolidate(loaders['update1_train'])

    # — Phase 2：Update2 + EWC(lambda2)
    set_seed(config.SEED)
    config.EWC_LAMBDA = lambda2
    trainer.config = config
    
    up2_dir = TMP_ROOT / f"trial_{trial_id}" / "update2"
    up2_dir.mkdir(parents=True, exist_ok=True)
    trainer.checkpoint_dir = up2_dir

    hist2 = trainer.train_task(
        train_loader=loaders['update2_train'],
        val_loader=loaders['update2_val'],
        task_id=2,
        apply_ewc=True,
        resume=False
    )
    val2 = hist2['val_loss'][-1]

    return val1, val2

# ———— 3. 精简 objective_incremental，只跑 Update1/2 ————
def objective_incremental(trial, best_config_params, base_ckpt_path):
    config = Config(**best_config_params)
    lambda1 = trial.suggest_float('lambda1', 10, 10_000, log=True)
    lambda2 = trial.suggest_float('lambda2', 10, 10_000, log=True)

    val1, val2 = train_incremental(
        config, lambda1, lambda2, trial.number, base_ckpt_path
    )
    
    trial.set_user_attr('val_update1', val1)
    trial.set_user_attr('val_update2', val2)

    return val2

if __name__ == '__main__':
    best_params = {
        'SEQUENCE_LENGTH': 576,
        'HIDDEN_SIZE': 128,
        'NUM_LAYERS': 2,
        'DROPOUT': 0.1,
        'BATCH_SIZE': 16,
        'LEARNING_RATE': 1e-4,
        'WEIGHT_DECAY': 1e-6,
        'EPOCHS': 200,
        'PATIENCE': 20,
        'SEED': 42,
        'SCALER': 'RobustScaler',
    }
    
    # 1) 准备 Base checkpoint
    base_ckpt = prepare_base(Config(**best_params))

    # 2) 只做 Update1/2 的 Optuna 搜索
    study_inc = optuna.create_study(direction='minimize')
    study_inc.optimize(
        lambda trial: objective_incremental(trial, best_params, base_ckpt),
        n_trials=50
    )
    
    # Save all trial params and values for incremental
    df_inc = study_inc.trials_dataframe()
    df_inc.to_csv(TMP_ROOT / 'incremental_trials.csv', index=False)

    # Print summary for incremental
    print('Incremental study completed.')
    print("Best trial:")
    print(study_inc.best_trial.params)
    print("  Best loss =", study_inc.best_value)