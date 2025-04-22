import random
import torch
import numpy as np

def set_seed(seed):
    """
    Set random seeds for reproducibility across torch, numpy, and random.
    
    Args:
        seed (int): Seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True