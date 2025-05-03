import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import logging


logger = logging.getLogger(__name__)


# ===============================================================
# Configuration Class
# ===============================================================

class Config:
    """Configuration class for managing hyperparameters"""
    def __init__(self, **kwargs):
        # Default parameters
        self.SEQUENCE_LENGTH = 864
        self.HIDDEN_SIZE = 256
        self.NUM_LAYERS = 2
        self.DROPOUT = 0.4
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 1e-4
        self.EPOCHS = 100
        self.PATIENCE = 10
        self.WEIGHT_DECAY = 1e-6
        self.SEED = 42
        self.RESAMPLE = '10min'
        
        # Update with custom parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def save(self, path):
        """Save configuration to JSON file"""
        config_dict = {key: value for key, value in self.__dict__.items()}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, path):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

# ===============================================================
# Dataset Definition
# ===============================================================

class BatteryDataset(Dataset):
    """Battery SOH prediction dataset"""
    def __init__(self, df, sequence_length):
        self.sequence_length = sequence_length
        self.features = torch.tensor(df[['Voltage[V]', 'Current[A]', 'Temperature[°C]']].values, dtype=torch.float32)
        self.targets = torch.tensor(df['SOH_ZHU'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length - 1]
        return x, y

# ===============================================================
# Data Loading and Preparation
# ===============================================================

class DataProcessor:
    """Data processing class responsible for loading and preparing data"""
    def __init__(self, data_dir, resample='10min', test_cell_id='17', seed=42):
        self.data_dir = Path(data_dir)
        self.resample = resample
        self.test_cell_id = test_cell_id
        self.seed = seed
        self.scaler = StandardScaler() 
        # self.target_scaler = MinMaxScaler()
        random.seed(seed)
        
    def load_cell_data(self):
        """Load battery data and categorize cells"""
        parquet_files = sorted(
            [f for f in self.data_dir.glob('*.parquet') if f.is_file()],
            key=lambda x: int(x.stem.split('_')[-1])
        )
        
        # Process files to get final SOH values and categorize batteries
        cell_info = []
        test_file = None
        
        for fp in parquet_files:
            cell_id = fp.stem.split('_')[1]
            
            # Use cell 17 as test set
            if cell_id == self.test_cell_id:
                test_file = fp
                continue
                
            # Read file to get SOH values
            df = pd.read_parquet(fp)
            initial_soh = df['SOH_ZHU'].iloc[0]
            final_soh = df['SOH_ZHU'].iloc[-1]
            
            # Categorize batteries based on final SOH
            if final_soh > 0.8:
                category = 'normal'
            elif 0.65 < final_soh <= 0.8:
                category = 'fast'
            else:
                category = 'faster'
                
            cell_info.append({
                'file': fp,
                'cell_id': cell_id,
                'initial_soh': initial_soh,
                'final_soh': final_soh,
                'category': category
            })
        
        return cell_info, test_file
    
    def assign_cells_to_phases(self, cell_info):
        """Assign batteries to different training phases"""
        # Group by category
        normal_cells = [c for c in cell_info if c['category'] == 'normal']
        fast_cells = [c for c in cell_info if c['category'] == 'fast']
        faster_cells = [c for c in cell_info if c['category'] == 'faster']
        
        # Check if there are enough normal cells
        if len(normal_cells) < 7:
            logger.warning("Warning: Not enough normal cells (%d), at least 7 needed.", len(normal_cells))
            # If not enough, supplement from the next category
            if len(normal_cells) + len(fast_cells) >= 7:
                fast_cells_sorted = sorted(fast_cells, key=lambda x: x['final_soh'], reverse=True)
                normal_cells.extend(fast_cells_sorted[:7-len(normal_cells)])
                # Remove cells transferred to normal
                fast_cells = fast_cells_sorted[7-len(normal_cells):]
            else:
                raise ValueError("Error: Not enough cells for base training.")
                
        # Randomly select 7 normal cells
        selected_normal = random.sample(normal_cells, min(7, len(normal_cells)))
        
        # Base training (5) and validation (2)
        base_train_cells = selected_normal[:5]
        base_val_cells = selected_normal[5:7]
        
        # Remaining normal cells
        remaining_normal = [c for c in normal_cells if c not in selected_normal]
        
        # Update1: base_val(2) + 1 normal + 2 fast as training set, 1 fast as validation set
        update1_train_normal = remaining_normal[:1] if remaining_normal else []
        
        if len(fast_cells) < 3:
            logger.warning("Warning: Not enough fast cells (%d), at least 3 needed.", len(fast_cells))
        
        update1_train_fast = fast_cells[:2] if len(fast_cells) >= 2 else fast_cells
        update1_val_fast = fast_cells[2:3] if len(fast_cells) >= 3 else []
        
        # Combine update1 cells
        update1_train_cells = base_val_cells + update1_train_normal + update1_train_fast
        update1_val_cells = update1_val_fast
        
        # Update2: update1_val + 2 faster as training set, remaining faster as validation set
        update2_train_faster = faster_cells[:2] if len(faster_cells) >= 2 else faster_cells
        update2_val_faster = faster_cells[2:] if len(faster_cells) >= 3 else []
        
        # Combine update2 cells
        update2_train_cells = update1_val_cells + update2_train_faster
        update2_val_cells = update2_val_faster
        
        # Print assignment summary
        logger.info("Cell Assignment Summary:")
        logger.info("Normal cells (%d): %s", len(normal_cells), [c['cell_id'] for c in normal_cells])
        logger.info("Fast cells (%d): %s", len(fast_cells), [c['cell_id'] for c in fast_cells])
        logger.info("Faster cells (%d): %s", len(faster_cells), [c['cell_id'] for c in faster_cells])
        logger.info("\nTraining Sets:")
        logger.info("Base training set: %s", [c['cell_id'] for c in base_train_cells])
        logger.info("Base validation set: %s", [c['cell_id'] for c in base_val_cells])
        logger.info("Update1 training set: %s", [c['cell_id'] for c in update1_train_cells])
        logger.info("Update1 validation set: %s", [c['cell_id'] for c in update1_val_cells])
        logger.info("Update2 training set: %s", [c['cell_id'] for c in update2_train_cells])
        logger.info("Update2 validation set: %s", [c['cell_id'] for c in update2_val_cells])
        
        return {
            'base_train': base_train_cells,
            'base_val': base_val_cells,
            'update1_train': update1_train_cells,
            'update1_val': update1_val_cells,
            'update2_train': update2_train_cells,
            'update2_val': update2_val_cells
        }
    
    def process_file(self, file_path):
        """Process a single parquet file"""
        df = pd.read_parquet(file_path)
        columns_to_keep = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 
                           'Temperature[°C]', 'SOH_ZHU']
        df_processed = df[columns_to_keep].copy()
        df_processed.dropna(inplace=True)
        
        df_processed['Testtime[s]'] = df_processed['Testtime[s]'].round().astype(int)
        start_date = pd.Timestamp("2023-02-02")
        df_processed['Datetime'] = pd.date_range(
            start=start_date,
            periods=len(df_processed),
            freq='s'
        )
        
        df_sampled = df_processed.resample(self.resample, on='Datetime').mean().reset_index(drop=False)
        df_sampled["cell_id"] = file_path.stem.split('_')[1]
        return df_sampled
    
    def prepare_data(self):
        """Prepare datasets for battery incremental learning"""
        # Load battery data
        cell_info, test_file = self.load_cell_data()
        
        # Assign cells to different phases
        phase_cells = self.assign_cells_to_phases(cell_info)
        
        # Process training and validation files
        df_base_train = pd.concat([self.process_file(c['file']) for c in phase_cells['base_train']], ignore_index=True)
        df_base_val = pd.concat([self.process_file(c['file']) for c in phase_cells['base_val']], ignore_index=True)
        
        df_update1_train = pd.concat([self.process_file(c['file']) for c in phase_cells['update1_train']], ignore_index=True)
        df_update1_val = pd.concat([self.process_file(c['file']) for c in phase_cells['update1_val']], ignore_index=True) if phase_cells['update1_val'] else pd.DataFrame()
        
        df_update2_train = pd.concat([self.process_file(c['file']) for c in phase_cells['update2_train']], ignore_index=True) if phase_cells['update2_train'] else pd.DataFrame()
        df_update2_val = pd.concat([self.process_file(c['file']) for c in phase_cells['update2_val']], ignore_index=True) if phase_cells['update2_val'] else pd.DataFrame()
        
        # Process test file
        df_test_full = self.process_file(test_file)
        
        # Split test set based on SOH values
        df_test_base = df_test_full[df_test_full['SOH_ZHU'] >= 0.9].reset_index(drop=True)
        df_test_update1 = df_test_full[(df_test_full['SOH_ZHU'] < 0.9) & (df_test_full['SOH_ZHU'] >= 0.8)].reset_index(drop=True)
        df_test_update2 = df_test_full[df_test_full['SOH_ZHU'] < 0.8].reset_index(drop=True)
        
        # Standardize data using StandardScaler
        feature_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        self.scaler.fit(df_base_train[feature_cols])
        
        # label_cols = ['SOH_ZHU']
        # self.target_scaler.fit(df_base_train[label_cols])
        
        # Apply standardization
        datasets = {
            'base_train': self.apply_scaling(df_base_train),
            'base_val': self.apply_scaling(df_base_val),
            'update1_train': self.apply_scaling(df_update1_train),
            'update1_val': self.apply_scaling(df_update1_val),
            'update2_train': self.apply_scaling(df_update2_train),
            'update2_val': self.apply_scaling(df_update2_val),
            'test_base': self.apply_scaling(df_test_base),
            'test_update1': self.apply_scaling(df_test_update1),
            'test_update2': self.apply_scaling(df_test_update2)
        }
        
        # Print dataset sizes
        logger.info("\nDataset Sizes:")
        logger.info("Base training set: %d samples, validation set: %d samples", len(datasets['base_train']), len(datasets['base_val']))
        logger.info("Update1 training set: %d samples, validation set: %d samples", len(datasets['update1_train']), len(datasets['update1_val']))
        logger.info("Update2 training set: %d samples, validation set: %d samples", len(datasets['update2_train']), len(datasets['update2_val']))

        logger.info("\nTest Sets:")
        logger.info("Test cell: %s", test_file.stem.split('_')[1])
        logger.info("Base test set (SOH ≥ 0.9): %d samples", len(df_test_base))
        logger.info("Update1 test set (0.8 ≤ SOH < 0.9): %d samples", len(df_test_update1))
        logger.info("Update2 test set (SOH < 0.8): %d samples", len(df_test_update2))
        
        return datasets
        
    def apply_scaling(self, df):
        """Apply standardization transformation"""
        if df.empty:
            return df
        df_scaled = df.copy()
        feature_cols = ['Voltage[V]', 'Current[A]', 'Temperature[°C]']
        df_scaled[feature_cols] = self.scaler.transform(df[feature_cols])
        
        # df_scaled[['SOH_ZHU']] = self.target_scaler.transform(df[['SOH_ZHU']])
        
        return df_scaled

# ===============================================================
# Model Definition
# ===============================================================

class LateralConnection(nn.Module):
    """
    Lateral connection module for Progressive Neural Networks.
    Maps activations from previous columns to the current column.
    """
    def __init__(self, input_dim, output_dim, adapter_type='non_linear'):
        super(LateralConnection, self).__init__()
        self.adapter_type = adapter_type
        self.adapter = nn.Linear(input_dim, output_dim)
        
        # Initialize with small weights as recommended in the paper
        init_scale = min(0.1, 1.0 / (input_dim ** 0.5))
        nn.init.normal_(self.adapter.weight, std=init_scale)
        nn.init.zeros_(self.adapter.bias)
        
    def forward(self, x):
        out = self.adapter(x)
        # Apply non-linearity if specified
        if self.adapter_type == 'non_linear':
            out = F.relu(out)
        return out


class LSTMLayerWithLaterals(nn.Module):
    """
    LSTM layer with lateral connections from previous columns
    """
    def __init__(self, input_size, hidden_size, prev_columns=None, adapter_type='non_linear'):
        super(LSTMLayerWithLaterals, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.prev_columns = prev_columns or []
        
        # Main LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Lateral connections (if there are previous columns)
        if prev_columns:
            self.lateral_connections = nn.ModuleList([
                LateralConnection(col.hidden_size, hidden_size, adapter_type) 
                for col in prev_columns
            ])
            
            # Optional: Add a combiner network for lateral connections
            # This allows more complex integration of lateral inputs
            self.lateral_combiner = nn.Linear(hidden_size * (1 + len(prev_columns)), hidden_size)
    
    def forward(self, x, prev_activations=None, hx=None):
        batch_size = x.size(0)
        device = x.device  # Get device from input tensor
        
        # Initialize LSTM hidden state if not provided
        if hx is None:
            h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
            hx = (h0, c0)
        
        # Get output from LSTM
        output, (hn, cn) = self.lstm(x, hx)
        
        # Add lateral connections from previous columns if any
        if hasattr(self, 'lateral_connections') and prev_activations:
            # Collect all lateral outputs including current output
            lateral_outputs = [output]
            
            for i, lateral in enumerate(self.lateral_connections):
                if i < len(prev_activations) and prev_activations[i] is not None:
                    # Apply lateral connection
                    lateral_contrib = lateral(prev_activations[i])
                    lateral_outputs.append(lateral_contrib)
            
            # If we have a combiner, use it to integrate all lateral inputs
            if hasattr(self, 'lateral_combiner') and len(lateral_outputs) > 1:
                # Reshape for concatenation if needed
                shapes = [lo.shape for lo in lateral_outputs]
                if not all(s == shapes[0] for s in shapes):
                    # Ensure all tensors have the same shape
                    for i in range(len(lateral_outputs)):
                        if lateral_outputs[i].shape != shapes[0]:
                            # Adjust shapes if needed (e.g., broadcasting)
                            lateral_outputs[i] = lateral_outputs[i].expand_as(lateral_outputs[0])
                
                # Concatenate all outputs along the last dimension
                combined = torch.cat(lateral_outputs, dim=-1)
                
                # Apply the combiner
                output = self.lateral_combiner(combined)
            else:
                # Simple summation (as in your original implementation)
                for lo in lateral_outputs[1:]:  # Skip the first one which is the current output
                    output = output + lo
        
        return output, (hn, cn)


class ColumnLSTM(nn.Module):
    """
    A single column in the Progressive Neural Network
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=None, 
                 dropout=0.2, prev_columns=None, adapter_type='non_linear'):
        super(ColumnLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.adapter_type = adapter_type
        
        # Create stack of LSTM layers
        self.layers = nn.ModuleList()
        
        # First layer takes the input
        self.layers.append(LSTMLayerWithLaterals(
            input_size=input_size,
            hidden_size=hidden_size,
            prev_columns=[col.layers[0] for col in prev_columns] if prev_columns else None,
            adapter_type=adapter_type
        ))
        
        # Subsequent layers 
        for i in range(1, num_layers):
            layer_input_size = hidden_size
            layer = LSTMLayerWithLaterals(
                input_size=layer_input_size,
                hidden_size=hidden_size,
                prev_columns=[col.layers[i] for col in prev_columns] if prev_columns else None,
                adapter_type=adapter_type
            )
            self.layers.append(layer)
        
        # Output layer - now optional and configurable
        if output_size is not None:
            self.fc_output = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, output_size)
            )
    
    def forward(self, x, prev_columns_activations=None):
        batch_size, seq_len, _ = x.size()
        
        # Initialize activations list for each layer
        layer_activations = []
        
        # Get previous activations for each layer if provided
        if prev_columns_activations is None:
            prev_columns_activations = [None] * self.num_layers
        
        # Process each layer
        current_input = x
        h_states = []
        
        for i, layer in enumerate(self.layers):
            # Get previous activations for this layer
            prev_acts = None
            if prev_columns_activations and i < len(prev_columns_activations) and prev_columns_activations[i]:
                prev_acts = [acts[i] for acts in prev_columns_activations if i < len(acts)]
            
            # Apply dropout between layers (except after the last layer)
            if i > 0 and self.dropout > 0:
                current_input = F.dropout(current_input, p=self.dropout, training=self.training)
            
            # Process through LSTM layer with laterals
            output, (hn, cn) = layer(current_input, prev_acts)
            current_input = output
            layer_activations.append(output)
            h_states.append((hn, cn))
        
        # Use the output from the last sequence step
        final_output = output[:, -1, :]
        
        # Pass through the fully connected layers if output_size is specified
        if hasattr(self, 'fc_output'):
            prediction = self.fc_output(final_output)
            return prediction, layer_activations
        else:
            # Just return the LSTM outputs
            return final_output, layer_activations


class ProgressiveNN(nn.Module):
    """
    Progressive Neural Network implementation based on the paper
    "Progressive Neural Networks" by Rusu et al.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, 
                 dropout=0.2, adapter_type='non_linear'):
        super(ProgressiveNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.adapter_type = adapter_type
        
        # List to store columns
        self.columns = nn.ModuleList()
        
        # List to store task-specific output heads
        self.output_heads = nn.ModuleList()
        
        # Initialize with the first column
        self.add_column()
    
    def add_column(self, freeze_previous=True):
        """
        Add a new column to the network.
        
        Args:
            freeze_previous: Whether to freeze parameters of previous columns
            
        Returns:
            The index of the newly added column.
        """
        # Freeze parameters of previous columns if requested
        if freeze_previous and self.columns:
            for col in self.columns:
                for param in col.parameters():
                    param.requires_grad = False
        
        column_idx = len(self.columns)
        prev_columns = list(self.columns) if column_idx > 0 else None
        device = next(self.parameters()).device if self.columns else torch.device("cpu")
        
        # Create new column with connections to previous columns
        # The column itself doesn't have an output layer
        new_column = ColumnLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=None,  # No output layer in the column
            dropout=self.dropout,
            prev_columns=prev_columns,
            adapter_type=self.adapter_type
        ).to(device)
        
        # Create a task-specific output head
        output_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.output_size)
        ).to(device)
        
        self.columns.append(new_column)
        self.output_heads.append(output_head)
        
        return column_idx
    
    def forward(self, x, task_id=None):
        """
        Forward pass with option to specify which task (column) to use.
        
        Args:
            x: Input tensor
            task_id: Column to use for prediction. If None, uses the latest column.
        
        Returns:
            Prediction from the specified column
        """
        if task_id is None:
            task_id = len(self.columns) - 1
        
        if task_id >= len(self.columns) or task_id < 0:
            raise ValueError(f"Invalid task_id {task_id}. Must be between 0 and {len(self.columns) - 1}")
        
        # Ensure input tensor is on same device as model
        device = next(self.parameters()).device
        x = x.to(device)
        
        # For the first column, there are no lateral connections
        if task_id == 0:
            features, _ = self.columns[0](x, None)
            output = self.output_heads[0](features)
        else:
            # For subsequent columns, we need to collect activations from previous columns
            prev_columns_activations = []
            
            with torch.no_grad():  # No gradient needed for previous columns
                for i in range(task_id):
                    _, activations = self.columns[i](x, None if i == 0 else prev_columns_activations[:i])
                    prev_columns_activations.append(activations)
            
            # Forward through the target column with lateral connections
            features, _ = self.columns[task_id](x, prev_columns_activations)
            
            # Apply task-specific output head
            output = self.output_heads[task_id](features)
        
        # CRITICAL FIX: Always return the right shape for outputs 
        # For single-valued outputs, always make sure the output is flat (no extra dimension)
        if self.output_size == 1:
            # This ensures output is [batch_size] not [batch_size, 1]
            return output
        else:
            return output
    
    def to(self, *args, **kwargs):
        """
        Override the to() method to ensure all columns are moved to the target device
        """
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        
        # Call the parent to() method
        super(ProgressiveNN, self).to(*args, **kwargs)
        
        # Ensure all columns and output heads are on the same device
        if device is not None:
            for column in self.columns:
                column.to(device)
            for head in self.output_heads:
                head.to(device)
        
        return self


# ===============================================================
# Model Training and Evaluation
# ===============================================================

class PNNTrainer:
    """PNN model trainer"""
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.criterion = nn.MSELoss()
        
    def train(self, train_loader, val_loader):
        """Train the model"""
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.LEARNING_RATE, 
            weight_decay=self.config.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        best_state = None
        no_improve = 0
        history = {'epoch': [], 'train_loss': [], 'val_loss': []}

        for epoch in range(self.config.EPOCHS):
            # Training phase
            self.model.train()
            train_loss = 0.0
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{self.config.EPOCHS}', leave=False) as pbar:
                for features, labels in train_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                    optimizer.step()
                    train_loss += loss.item()
                    pbar.update(1)
            train_loss /= len(train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = self.model(features)
                    val_loss += self.criterion(outputs, labels).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            history['epoch'].append(epoch+1)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            logger.info('Epoch %d/%d | Train Loss: %.3e | Val Loss: %.3e', epoch+1, self.config.EPOCHS, train_loss, val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.config.PATIENCE:
                    logger.info('Early stopping at epoch %d', epoch+1)
                    break

        # Load best model
        if best_state is not None:
            for k, v in best_state.items():
                best_state[k] = v.to(self.device)
            self.model.load_state_dict(best_state)
            
        return history
    
    def evaluate(self, data_loader, task_id=None):
        """Evaluate the model"""
        self.model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        all_pred, all_targets = [], []
        
        with torch.no_grad():
            for features, labels in data_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features, task_id=task_id)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                all_pred.append(outputs.cpu().numpy().ravel())
                all_targets.append(labels.cpu().numpy().ravel())
                
        total_loss /= len(data_loader)
        predictions = np.concatenate(all_pred)
        targets = np.concatenate(all_targets)
        
        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        
        return predictions, targets, metrics, total_loss
    
    @staticmethod
    def plot_history(history, task_id, save_dir):
        """Plot training and validation loss curves"""
        plt.figure()
        epochs = history['epoch']
        plt.semilogy(epochs, history['train_loss'], label='Train Loss')
        plt.semilogy(epochs, history['val_loss'], label='Val Loss')
        plt.title(f'Task {task_id} Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / f'loss_task{task_id}.png')
        plt.close()

# ===============================================================
# Helper Functions
# ===============================================================

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dataloaders(datasets, sequence_length, batch_size):
    """Create data loaders"""
    loaders = {}
    
    # Create training and validation loaders
    for key, dataset in datasets.items():
        if not dataset.empty:
            if 'train' in key or 'val' in key or 'test' in key:
                loaders[key] = DataLoader(
                    BatteryDataset(dataset, sequence_length),
                    batch_size=batch_size,
                    shuffle='train' in key
                )
    
    return loaders

# ===============================================================
# Main Function
# ===============================================================

def main():
    """Main function for incremental learning"""
    # Create save directory
    save_dir = Path(__file__).parent / 'models' / 'incLearning_PNN' / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_file = save_dir / 'inclearning.log'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    if logger.handlers:
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    logger.info('Logs will be saved to: %s', log_file)
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Set configuration
    config = Config()
    config.save(save_dir / 'hyperparams.json')
    
    # Set random seed
    set_seed(config.SEED)

    # Load data
    data_dir = Path('../01_Datenaufbereitung/Output/Calculated/')
    data_processor = DataProcessor(data_dir, resample=config.RESAMPLE, seed=config.SEED)
    datasets = data_processor.prepare_data()
    
    # Create data loaders
    loaders = create_dataloaders(datasets, config.SEQUENCE_LENGTH, config.BATCH_SIZE)
    
    # PNN incremental learning
    logger.info('='*80)
    logger.info('Method: Progressive Neural Network (PNN)')
    logger.info('='*80)

    # Initialize PNN
    input_size = 3  # Voltage, Current, Temperature
    pnn_model = ProgressiveNN(
        input_size=input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    # Create trainer
    trainer = PNNTrainer(pnn_model, device, config)

    # Phase 1: Train on base data
    logger.info('\nPhase 1: Training on base data for column 0...')
    base_history = trainer.train(loaders['base_train'], loaders['base_val'])
    torch.save(pnn_model.state_dict(), save_dir / 'pnn_column0.pt')
    trainer.plot_history(base_history, 0, save_dir)
    
    _, _, base_metrics, _ = trainer.evaluate(loaders['test_base'], task_id=0)
    logger.info(f'Column 0 metrics: {base_metrics}')

    # Phase 2: Add column 1 and train
    logger.info('\nPhase 2: Adding column 1 and training on update1 data...')
    pnn_model.add_column()
    update1_history = trainer.train(loaders['update1_train'], loaders['update1_val'])
    torch.save(pnn_model.state_dict(), save_dir / 'pnn_column1.pt')
    trainer.plot_history(update1_history, 1, save_dir)
    
    _, _, update1_metrics, _ = trainer.evaluate(loaders['test_update1'], task_id=1)
    logger.info(f'Column 1 metrics: {update1_metrics}')

    # Phase 3: Add column 2 and train
    logger.info('\nPhase 3: Adding column 2 and training on update2 data...')
    pnn_model.add_column()
    update2_history = trainer.train(loaders['update2_train'], loaders['update2_val'])
    torch.save(pnn_model.state_dict(), save_dir / 'pnn_column2.pt')
    trainer.plot_history(update2_history, 2, save_dir)
    
    _, _, update2_metrics, _ = trainer.evaluate(loaders['test_update2'], task_id=2)
    logger.info(f'Column 2 metrics: {update2_metrics}')

    logger.info(f'\nPNN incremental learning completed! Models and results saved to: {save_dir}')

if __name__ == '__main__':
    main()