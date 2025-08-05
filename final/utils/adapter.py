import logging
import copy
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.config import Config

logger = logging.getLogger(__name__)

class WindowedAttentionAdapter(nn.Module):
    """
    Windowed self-attention adapter with bottleneck structure.
    Uses overlapping windows to maintain temporal continuity while reducing computation.
    """
    def __init__(self, hidden_size=128, window_size=144, overlap=72, reduction_factor=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        
        # Bottleneck architecture to reduce parameters
        bottleneck_size = hidden_size // reduction_factor  # 128/8 = 16
        
        # Down-projection
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        
        # Multi-head attention in bottleneck dimension
        self.attention = nn.MultiheadAttention(
            embed_dim=bottleneck_size,
            num_heads=4,  # 16/4 = 4 dims per head
            dropout=0.1,
            batch_first=True
        )
        
        # Up-projection
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        
        # Layer normalization and residual connection
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Learnable gate for controlling adapter contribution
        self.gate = nn.Parameter(torch.ones(1) * 0.1)  # Initialize with small value
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            Output tensor with same shape as input
        """
        B, T, H = x.shape
        residual = x
        
        # Down projection to bottleneck
        x = self.down_proj(x)  # [B, T, bottleneck_size]
        
        # Apply windowed attention for long sequences
        if T > self.window_size:
            x = self._windowed_attention(x)
        else:
            # For short sequences, use standard attention
            x, _ = self.attention(x, x, x)
        
        # Up projection back to hidden size
        x = self.up_proj(x)  # [B, T, H]
        
        # Gated residual connection
        gate = torch.sigmoid(self.gate)
        output = residual + gate * x
        
        # Layer normalization
        return self.layer_norm(output)
    
    def _windowed_attention(self, x):
        """
        Efficient windowed attention using unfold + fold to batch compute all windows.
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        B, T, D = x.shape
        W, S = self.window_size, self.stride

        # 1. Extract overlapping windows: [B, num_w, W, D]
        windows = x.unfold(1, W, S)
        num_w = windows.size(1)

        # 2. Batch all windows → [B*num_w, W, D]
        flat = windows.contiguous().view(-1, W, D)

        # 3. One-shot attention
        out_flat, _ = self.attention(flat, flat, flat)

        # 4. Reshape back → [B, num_w, W, D]
        out = out_flat.view(B, num_w, W, D)

        # 5. Fold windows into full sequence
        out_perm = out.permute(0, 3, 2, 1).contiguous()   # [B, D, W, num_w]
        out_reshaped = out_perm.view(B, D*W, num_w)       # [B, D*W, num_w]
        fold = nn.Fold(output_size=(T,1), kernel_size=(W,1), stride=(S,1))
        recon = fold(out_reshaped)                       # [B, D, T, 1]
        recon = recon.squeeze(-1).permute(0, 2, 1)       # [B, T, D]

        # 6. Build counts and fold them
        ones = torch.ones(B, num_w, W, 1, device=x.device)
        ones_perm = ones.permute(0, 3, 2, 1).contiguous() # [B,1,W,num_w]
        ones_reshaped = ones_perm.view(B, W, num_w)       # [B, W, num_w]
        fold_counts = nn.Fold(output_size=(T,1), kernel_size=(W,1), stride=(S,1))
        counts = fold_counts(ones_reshaped)               # [B,1,T,1]
        counts = counts.squeeze(-1).permute(0, 2, 1)      # [B, T,1]

        # 7. Average overlapping regions
        recon = recon / counts.clamp(min=1)
        return recon


class AdapterSOHLSTM(nn.Module):
    """
    LSTM model with cumulative attention-based adapters for incremental learning.
    Adapters are placed after each LSTM layer to adapt representations.
    """
    def __init__(self, input_size=3, hidden_size=128, dropout=0.3, 
                 window_size=144, overlap=72, reduction_factor=8):
        super().__init__()
        
        # Base LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.adapter1 = WindowedAttentionAdapter(
            hidden_size=hidden_size,
            window_size=window_size,
            overlap=overlap,
            reduction_factor=reduction_factor
        )
        self.dropout_between = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.adapter2 = WindowedAttentionAdapter(
            hidden_size=hidden_size,
            window_size=window_size,
            overlap=overlap,
            reduction_factor=reduction_factor
        )
        
        # Final prediction layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        """
        Forward pass through LSTM with adapters.
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            predictions: [batch]
        """
        # LSTM1 + Adapter1
        h1, _ = self.lstm1(x)
        h1 = self.adapter1(h1)  # Apply adapter after LSTM1
        h1 = self.dropout_between(h1)
        
        # LSTM2 + Adapter2
        h2, _ = self.lstm2(h1)
        h2 = self.adapter2(h2)  # Apply adapter after LSTM2
        
        # Use last timestep for prediction
        return self.fc(h2[:, -1, :]).squeeze(-1)
    
    def freeze_base_model(self):
        """Freeze LSTM parameters, keep adapters and FC trainable."""
        # Freeze LSTM layers
        for param in self.lstm1.parameters():
            param.requires_grad = False
        for param in self.lstm2.parameters():
            param.requires_grad = False
        
            
        logger.info("Base model LSTM frozen. Adapters and FC are trainable.")
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen.")
    
    def get_trainable_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_adapter_params(self):
        """Get adapter parameters for regularization or analysis."""
        adapter_params = []
        adapter_params.extend(list(self.adapter1.parameters()))
        adapter_params.extend(list(self.adapter2.parameters()))
        return adapter_params


class AdapterTrainer:
    """
    Trainer for incremental learning with adapters.
    Supports cumulative adapter learning where adapters are continuously updated across tasks.
    """
    def __init__(self, model: AdapterSOHLSTM, device: torch.device, config: Config, task_dir: Path = None):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.task_dir = task_dir
        if self.task_dir:
            self.task_dir.mkdir(parents=True, exist_ok=True)
        
        # Store initial adapter state for potential regularization
        self.prev_adapter_state = None
        
    def train_task(self, train_loader: DataLoader, val_loader: DataLoader, task_id: int, 
                   freeze_base: bool = False):
        """
        Train model on a single task.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            task_id: Current task identifier
            freeze_base: Whether to freeze base model (LSTM+FC)
        """
        # Configure model parameters
        if freeze_base:
            self.model.freeze_base_model()
        else:
            self.model.unfreeze_all()
        
        # Log trainable parameters
        logger.info("Task %d - Trainable parameters: %d", task_id, self.model.get_trainable_params())
        
        # Setup optimizer (only for trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(
            trainable_params,
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, 'min', factor=0.5, patience=5
        )
        
        # Training variables
        best_val = float('inf')
        no_imp = 0
        best_state = None
        history = {
            'epoch': [], 'train_loss': [], 'val_loss': [], 
            'lr': [], 'time': []
        }
        
        # Training loop
        for epoch in range(self.config.EPOCHS):
            start = time.time()
            self.model.train()
            tot_loss = 0.0
            
            # Training
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                
                # Forward pass
                yp = self.model(x)
                loss = F.mse_loss(yp, y)
                
                # Add L2 regularization on adapter changes if not first task
                if self.prev_adapter_state is not None and task_id > 0:
                    reg_loss = self._adapter_regularization()
                    loss = loss + 0.01 * reg_loss  # Small regularization weight
                
                # Backward pass
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                
                tot_loss += loss.item() * x.size(0)
            
            train_loss = tot_loss / len(train_loader.dataset)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    yp = self.model(x)
                    val_loss += F.mse_loss(yp, y).item() * x.size(0)
            val_loss /= len(val_loader.dataset)
            
            # Record history
            lr_cur = opt.param_groups[0]['lr']
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(lr_cur)
            history['time'].append(time.time() - start)
            
            logger.info("Epoch %d train=%.4e val=%.4e lr=%.2e time=%.2fs",
                       epoch, train_loss, val_loss, lr_cur, history['time'][-1])
            
            # Learning rate scheduling
            sched.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val:
                best_val = val_loss
                no_imp = 0
                best_state = copy.deepcopy(self.model.state_dict())
                
                # Save checkpoint
                if self.task_dir:
                    checkpoint = {
                        'model_state': best_state,
                        'task_id': task_id,
                        'epoch': epoch,
                        'val_loss': val_loss
                    }
                    torch.save(checkpoint, self.task_dir / f"task{task_id}_best.pt")
            else:
                no_imp += 1
                if no_imp >= self.config.PATIENCE:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
        
        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
        
        # Store adapter state for next task
        self._store_adapter_state()
        
        return history
    
    def _adapter_regularization(self):
        """
        Compute L2 regularization to prevent adapters from changing too much.
        """
        reg_loss = 0.0
        current_adapter_params = self.model.get_adapter_params()
        
        for curr_param, prev_param in zip(current_adapter_params, self.prev_adapter_state):
            reg_loss += ((curr_param - prev_param) ** 2).sum()
        
        return reg_loss
    
    def _store_adapter_state(self):
        """
        Store current adapter parameters for regularization in next task.
        """
        self.prev_adapter_state = [
            param.clone().detach() for param in self.model.get_adapter_params()
        ]