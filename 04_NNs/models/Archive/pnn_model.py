import torch
import torch.nn as nn
import torch.nn.functional as F

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
            return output.squeeze(-1)
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
