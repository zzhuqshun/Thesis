import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class EWC:
    """
    参数增量: Elastic Weight Consolidation (EWC)
    """
    def __init__(self, model, dataset, lambda_param=5000):
        self.model = model
        self.dataset = dataset
        self.lambda_param = lambda_param
        self.fisher_information = {}
        self.optimal_params = {}
        
    def compute_fisher_information(self, n_batches=32):
        self.model.eval()
        loader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        fisher_information = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        count = 0
        for batch_idx, (features, labels) in enumerate(loader):
            if batch_idx >= n_batches:
                break
            count += 1

            features, labels = features.to(next(self.model.parameters()).device), labels.to(next(self.model.parameters()).device)
            self.model.zero_grad()
            outputs = self.model(features)
            loss = F.mse_loss(outputs, labels)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_information[n] += p.grad.detach() ** 2
        
        # 取平均
        for n in fisher_information:
            fisher_information[n] /= max(count, 1)
            
        return fisher_information
    
    def register_task(self):
        # 记录最优参数
        self.optimal_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        # 计算 Fisher 信息
        self.fisher_information = self.compute_fisher_information()
        
    def ewc_loss(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher_information and p.requires_grad:
                loss += (self.fisher_information[n] * (p - self.optimal_params[n]) ** 2).sum()
        return self.lambda_param * loss / 2
