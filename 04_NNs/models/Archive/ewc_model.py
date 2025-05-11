import torch
import torch.nn as nn

class EWC:
    """
    Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting in neural networks.
    
    EWC adds a regularization term to the loss function that penalizes changes to parameters
    that were important for previous tasks, based on the Fisher information matrix.
    
    Attributes:
        model (nn.Module): The PyTorch model to apply EWC to.
        device (torch.device): Device to use for computation.
        fisher (dict): Fisher information matrix for each parameter.
        optimal_params (dict): Optimal parameters from previous tasks.
        lambda_ewc (float): Regularization strength.
    """
    
    def __init__(self, model, device, lambda_ewc=5000):
        """
        Initialize EWC.
        
        Args:
            model (nn.Module): The PyTorch model to apply EWC to.
            device (torch.device): Device to use for computation.
            lambda_ewc (float): Regularization strength (default: 5000).
        """
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        
        # Initialize dictionaries for Fisher information and optimal parameters
        self.fisher = {}
        self.optimal_params = {}
        
        # Register hooks to track Fisher information
        for n, p in self.model.named_parameters():
            # Initialize Fisher information as zeros
            self.fisher[n] = torch.zeros_like(p, device=self.device)
            # Initialize optimal parameters 
            self.optimal_params[n] = p.clone().detach()
    
    def compute_fisher(self, data_loader, num_samples=None):
        """
        Compute the Fisher Information Matrix using the data loader.
        
        Args:
            data_loader: DataLoader containing data for computing Fisher information.
            num_samples (int, optional): Number of samples to use. If None, use all samples.
        """
        # Set model to training mode
        self.model.train()
        
        # Initialize parameter gradients
        for n, p in self.model.named_parameters():
            self.fisher[n] = torch.zeros_like(p, device=self.device)
        
        # MSE loss for regression
        criterion = nn.MSELoss()
        
        # Limit number of samples if specified
        sample_count = 0
        
        # Process batches
        for features, labels in data_loader:
            # Check if we've processed enough samples
            if num_samples is not None and sample_count >= num_samples:
                break
                
            features, labels = features.to(self.device), labels.to(self.device)
            sample_count += features.size(0)
            
            # Forward pass
            self.model.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Update Fisher information with squared gradients
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.pow(2).detach()
        
        # Normalize Fisher information by number of samples
        if sample_count > 0:
            for n in self.fisher:
                self.fisher[n] /= sample_count
    
    def update_optimal_params(self):
        """
        Update the optimal parameter values after training on a task.
        """
        for n, p in self.model.named_parameters():
            self.optimal_params[n] = p.clone().detach()
    
    def ewc_loss(self):
        """
        Compute the EWC regularization loss.
        
        Returns:
            torch.Tensor: The EWC regularization loss.
        """
        loss = 0
        for n, p in self.model.named_parameters():
            # Skip parameters without Fisher information
            if n not in self.fisher:
                continue
                
            # Compute squared difference between current and optimal parameters
            # weighted by Fisher information
            loss += (self.fisher[n] * (p - self.optimal_params[n]).pow(2)).sum()
        
        return self.lambda_ewc * 0.5 * loss