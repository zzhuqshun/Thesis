import random
import torch
from collections import deque

class ReplayBuffer:
    """
    数据增量: 经验回放 (Replay Buffer)
    """
    def __init__(self, max_size=5000):
        self.buffer = deque(maxlen=max_size)
        
    def add_batch(self, features, labels):
        for i in range(len(features)):
            self.buffer.append((features[i], labels[i]))
    
    def get_batch(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        features = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        return features, labels
    
    def __len__(self):
        return len(self.buffer)
