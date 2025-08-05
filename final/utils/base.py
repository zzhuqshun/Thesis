import torch.nn as nn

class SOHLSTM(nn.Module):
    """LSTM model for State of Health (SOH) prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), 
            nn.LeakyReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        # LSTM forward pass
        out, _ = self.lstm(x)
        # Use only the last time step output
        return self.fc(out[:, -1, :]).squeeze(-1)