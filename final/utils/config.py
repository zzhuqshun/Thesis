from pathlib import Path
import random, torch
import json

class Config:
    def __init__(self, **kwargs):
        # Mode: 'joint' or 'incremental'
        self.MODE = kwargs.get('MODE', 'joint')
        
        # Paths
        self.BASE_DIR = Path()
        self.DATA_DIR = Path(__file__).resolve().parent.parent / '01_Datenaufbereitung' / 'Output' / 'Calculated'

        # Model & training hyperparams
        self.SEQUENCE_LENGTH = 720
        self.HIDDEN_SIZE = 128
        self.NUM_LAYERS = 2
        self.DROPOUT = 0.3
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-4
        self.EPOCHS = 200
        self.PATIENCE = 20
        self.WEIGHT_DECAY = 1e-6
        
        # Data & smoothing
        self.RESAMPLE = '10min'
        self.ALPHA = 0.1

        # Continual learning
        self.NUM_TASKS = 3
        self.SEED = 42

        # Dataset splits
        self.joint_datasets = {
            'train_ids': ['03', '05', '07', '09', '11', '15', '21', '23', '25', '27', '29'],
            'val_ids':   ['01', '19', '13'],
            'test_id':   '17'
        }
        self.incremental_datasets = self._create_incremental_splits()

        # GPU info
        self.Info = {}
        if torch.cuda.is_available():
            self.Info['gpu_model'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        else:
            self.Info['gpu_model'] = ['CPU']

    def _create_incremental_splits(self):
        random.seed(self.SEED)
        normal = ['03','05','07','27']
        fast = ['21','23','25']
        faster = ['09','11','15','29']
        # Task 0: mixed sample
        t0 = random.sample(normal, 3) + random.sample(fast, 1) + random.sample(faster, 1)
        # Task 1: the remaining fast + normal cells
        t1 = [c for c in fast + normal if c not in t0]
        # Task 2: the remaining faster cells
        t2 = [c for c in faster if c not in t0]
        return {
            'task0_train_ids': t0, 'task0_val_ids': ['01'],
            'task1_train_ids': t1, 'task1_val_ids': ['19'],
            'task2_train_ids': t2, 'task2_val_ids': ['13'],
            'test_id': '17'
        }
    
    def save(self, path):
        """Save configuration to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
