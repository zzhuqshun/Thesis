import copy
import time
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
logger = logging.getLogger(__name__)

class EWC:
    """
    Elastic Weight Consolidation (EWC) for continual learning.
    
    EWC estimates parameter importance using the Fisher Information Matrix
    and applies regularization to prevent important parameters from changing
    too much in subsequent tasks.
    
    Reference: Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks" (2017)
    """
    
    def __init__(self, model, dataloader, device, ewc_lambda=1.0):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.ewc_lambda = ewc_lambda
        
        # Store parameters from previous task
        self.params = {n: p.clone().detach() 
                      for n, p in model.named_parameters() if p.requires_grad}
        
        # Compute Fisher Information Matrix for this task
        self.fisher = self._compute_fisher()
    
    def _compute_fisher(self):
        """
        Compute Fisher Information Matrix.
        The Fisher matrix approximates the second derivative of the loss
        with respect to parameters, indicating parameter importance.
        """
        # Create a copy of the model for Fisher computation
        model_copy = copy.deepcopy(self.model).to(self.device)
        model_copy.train()

        # Disable dropout for Fisher computation to get consistent gradients
        for m in model_copy.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
            if isinstance(m, nn.LSTM):
                m.dropout = 0.0

        # Initialize Fisher matrix
        fisher = {n: torch.zeros_like(p, device=self.device)
                 for n, p in model_copy.named_parameters() if p.requires_grad}

        n_processed = 0

        # Accumulate Fisher information across the dataset
        for x, y in self.dataloader:
            x, y = x.to(self.device), y.to(self.device)

            model_copy.zero_grad(set_to_none=True)
            output = model_copy(x)
            loss = F.mse_loss(output, y)
            loss.backward()

            bs = x.size(0)
            n_processed += bs

            # Accumulate squared gradients (Fisher approximation)
            with torch.no_grad():
                for n, p in model_copy.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.pow(2) * bs

        # Normalize by total number of samples
        for n in fisher:
            fisher[n] /= float(n_processed)

        # Clean up
        del model_copy
        torch.cuda.empty_cache()

        return fisher
    
    def penalty(self, model):
        """
        Compute EWC regularization penalty.
        Penalizes changes to important parameters from previous tasks.
        """
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher and p.requires_grad:
                # Calculate parameter change from previous task
                delta = p - self.params[n]
                
                # Add Fisher-weighted penalty: λ * F * (θ - θ_prev)^2
                loss += self.ewc_lambda * (self.fisher[n] * delta**2).sum()
        
        return loss
    
class EWCTrainer:
    """Main training class with support for continual learning"""
    
    def __init__(self, model, device, config, task_dir=None):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.ewc_tasks = []     # List of EWC regularizers from previous tasks
        self.old_model = None   # Previous model for knowledge distillation
        self.task_dir = Path(task_dir) if task_dir else None
        if self.task_dir: 
            self.task_dir.mkdir(parents=True, exist_ok=True)
    
    def train_task(self, train_loader, val_loader, task_id, alpha_lwf=0.0):
        """
        Train model on a single task with continual learning regularization.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            task_id: Current task identifier
            alpha_lwf: Learning without Forgetting weight (knowledge distillation)
        """
        # Setup optimizer and scheduler
        opt = torch.optim.Adam(self.model.parameters(), 
                              lr=self.config.LEARNING_RATE,
                              weight_decay=self.config.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)
        
        # Early stopping variables
        best_val = float('inf')
        no_imp = 0
        best_state = None
        
        # Training history tracking
        history = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'lr', 'time', 
                                  'task_loss', 'kd_loss', 'ewc_loss']}
        
        # Training loop
        for epoch in range(self.config.EPOCHS):
            start = time.time()
            self.model.train()
            
            # Loss components tracking
            tot_loss = 0
            sum_task = sum_kd = sum_ewc = 0
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                
                # Forward pass
                yp = self.model(x)
                
                # Task-specific loss (MSE for regression)
                task_loss = F.mse_loss(yp, y)
                
                # Knowledge distillation loss (Learning without Forgetting)
                kd_loss = torch.zeros((), device=self.device)
                if alpha_lwf > 0 and self.old_model is not None:
                    with torch.no_grad():
                        old_output = self.old_model(x)
                    kd_loss = F.mse_loss(yp, old_output)
                
                # EWC regularization loss
                ewc_loss = torch.zeros((), device=self.device)
                if self.ewc_tasks:
                    ewc_loss = sum(ewc_reg.penalty(self.model) for ewc_reg in self.ewc_tasks)
                
                # Total loss combination
                loss = task_loss + alpha_lwf * kd_loss + ewc_loss
                
                # Backward pass and parameter update
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                opt.step()
                
                # Track loss components
                bs = x.size(0)
                sum_task += task_loss.item() * bs
                sum_kd += kd_loss.item() * bs
                sum_ewc += ewc_loss.item() * bs
                tot_loss += loss.item() * bs
            
            # Calculate epoch averages
            n = len(train_loader.dataset)
            train_loss = tot_loss / n
            
            if sum_task > 0:
                reg_ratio = sum_ewc / sum_task * 100
                logger.info("Epoch %d: EWC loss is %.2f%% of task loss", epoch, reg_ratio)

            # Record training history
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['task_loss'].append(sum_task / n)
            history['kd_loss'].append(sum_kd / n)
            history['ewc_loss'].append(sum_ewc / n)
            
            lr_cur = opt.param_groups[0]['lr']
            history['lr'].append(lr_cur)
            history['time'].append(time.time() - start)
            
            # Validation evaluation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    val_loss += F.mse_loss(self.model(x.to(self.device)), 
                                         y.to(self.device)).item() * x.size(0)
            
            val_loss = val_loss / len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            sched.step(val_loss)
            
            # Logging
            logger.info("Epoch %d task=%.4e kd=%.4e ewc=%.4e val=%.4e lr=%.2e time=%.2fs",
                       epoch, sum_task/n, sum_kd/n, sum_ewc/n, val_loss, lr_cur, history['time'][-1])
            
            # Early stopping check
            if val_loss < best_val:
                best_val = val_loss
                no_imp = 0
                best_state = copy.deepcopy(self.model.state_dict())
                
                # Save best model checkpoint
                if self.task_dir: 
                    torch.save({'model_state': best_state}, 
                              self.task_dir / f"task{task_id}_best.pt")
            else:
                no_imp += 1
                if no_imp >= self.config.PATIENCE:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
        
        # Restore best model
        if best_state: 
            self.model.load_state_dict(best_state)
        
        return history
    
    def consolidate(self, loader, task_id=None, ewc_lambda=0.0):
        """
        Consolidate knowledge after task completion using EWC.
        
        Args:
            loader: Data loader for Fisher computation
            task_id: Task identifier
            ewc_lambda: EWC regularization strength
        """
        # Create EWC regularizer for this task
        ewc_reg = EWC(self.model, loader, self.device, ewc_lambda)
        
        all_fishers = torch.cat([v.flatten() for v in ewc_reg.fisher.values()])
        scale = all_fishers.mean().clamp(min=1e-12)
        for n in ewc_reg.fisher:
            ewc_reg.fisher[n] /= scale
        self.ewc_tasks.append(ewc_reg)

        logger.info("EWC fisher normalized with scale %.4f", scale.item())
                    
        # Save model for knowledge distillation
        self.old_model = copy.deepcopy(self.model).to(self.device)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)
        
        logger.info("Task %s consolidated with EWC lambda=%.4f", task_id, ewc_lambda)    