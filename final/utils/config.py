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
            
        # ---- Continual / Incremental learning switches & hyperparams ----
        # Toggles
        self.USE_SI = kwargs.get('USE_SI', True)        # turn SI penalty on/off
        self.USE_KD = kwargs.get('USE_KD', True)        # turn KD (LwF-style) on/off
        self.USE_KL = kwargs.get('USE_KL', True)        # turn KL-driven scheduling on/off

        # KL scheduling
        self.KL_MODE = kwargs.get('KL_MODE', 'input')   # 'input' | 'hidden' | 'both'
        self.TAU = float(kwargs.get('TAU', 60.0))       # temperature for KL->[0,1] mapping
        # self.S_MIN = float(kwargs.get('S_MIN', 0.15))
        # self.S_MAX = float(kwargs.get('S_MAX', 1.0))
        self.KL_EPS = float(kwargs.get('KL_EPS', 1e-8)) # numerical stability for variances

        # SI (Synaptic Intelligence)
        self.SI_EPSILON = float(kwargs.get('SI_EPSILON', 1e-3))  # SI denominator epsilon
        self.SI_LAMBDA = float(kwargs.get('SI_LAMBDA', 0.0))      
        self.SI_WARMUP_EPOCHS = int(kwargs.get('SI_WARMUP_EPOCHS', 5.0))

        # KD (teacher-student distillation for regression)
        self.KD_LAMBDA = float(kwargs.get('KD_LAMBDA', 0.0))    
        self.KD_LOSS  = kwargs.get('KD_LOSS', 'mse')             # 'mse' | 'l1' | 'smoothl1'
        self.KD_WARMUP_EPOCHS = int(kwargs.get('KD_WARMUP_EPOCHS', 5.0))
        

        # Training safety / misc
        self.MAX_GRAD_NORM = float(kwargs.get('MAX_GRAD_NORM', 1.0))  # grad clipping


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
        
        # # Task 1: Random 3 cells
        # rest_cells = [str(item) for item in [c for c in cells if c not in t0]]  # 确保转换为字符串
        # random.shuffle(rest_cells)
        
        # # Task 1: Random 3 cells
        # t1 = rest_cells[:3]
        # # Task 2: Random 3 cells
        # t2 = rest_cells[3:6]
        
        # Decay distribution
        rem_normal = [c for c in normal  if c not in t0_n]   # 1 left (since 4N - 3N used)
        rem_fast   = [c for c in fast    if c not in t0_f]   # all fast except the 1 picked for Task0
        rem_faster = [c for c in faster  if c not in t0_fer] # all faster except the 1 picked for Task0

        # Task 1: all remaining fast + one remaining normal
        t1 = rem_fast + rem_normal

        # Task 2: all remaining faster
        t2 = rem_faster
        
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