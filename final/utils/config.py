from pathlib import Path
import random, torch
import yaml

class Config:
    def __init__(self, **kwargs):
        # Mode: 'joint' or 'incremental'
        self.MODE = kwargs.get('MODE', 'joint')
        
        # Paths
        self.BASE_DIR = Path()
        self.DATA_DIR = Path(__file__).resolve().parent.parent.parent / '01_Datenaufbereitung' / 'Output' / 'Calculated'

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
        cells = normal + fast + faster
        # Task 0: mixed sample
        t0_n = [str(item) for item in random.sample(normal, 3)]  # 确保转换为字符串
        t0_f = [str(item) for item in random.sample(fast, 1)]
        t0_fer = [str(item) for item in random.sample(faster, 1)]
        t0 = t0_n + t0_f + t0_fer
        
        # Task 1: Random 3 cells
        rest_cells = [str(item) for item in [c for c in cells if c not in t0]]  # 确保转换为字符串
        random.shuffle(rest_cells)
        
        # Task 1: Random 3 cells
        t1 = rest_cells[:3]
        # Task 2: Random 3 cells
        t2 = rest_cells[3:6]
        
        return {
            'task0_train_ids': t0, 'task0_val_ids': ['01'],
            'task1_train_ids': t1, 'task1_val_ids': ['19'],
            'task2_train_ids': t2, 'task2_val_ids': ['13'],
            'test_id': '17'
        }
    
    def save(self, path):
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare config dict
        config_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                config_dict[k] = str(v)
            else:
                config_dict[k] = v
        
        # Custom representer for compact list formatting
        def represent_list(dumper, data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        
        yaml.add_representer(list, represent_list)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, allow_unicode=True, sort_keys=False)