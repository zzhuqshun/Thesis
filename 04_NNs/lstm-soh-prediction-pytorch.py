import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# 设置随机种子以保证可重复性
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 定义参数
look_back = 48  # 用于预测的历史时间窗口大小 [t-n,t]
horizon = 12     # 预测未来的时间步数 [t+1,t+m]
features = ['Current[A]', 'Voltage[V]', 'Temperature[°C]']  # 输入特征
target = 'SOH_ZHU'  # 目标变量
hidden_size = 4
num_layers = 2
dropout_rate = 0.5
learning_rate = 0.0001
batch_size = 16
num_epochs = 100
patience = 10

# 确定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据准备函数 - 创建时间序列窗口
def create_sequences(data, cell_ids, look_back, horizon):
    X, y, cell_id_list = [], [], []
    
    # 按电池ID分组处理
    for cell_id in np.unique(cell_ids):
        cell_data = data[cell_ids == cell_id]
        
        if len(cell_data) <= look_back + horizon:
            continue
            
        for i in range(len(cell_data) - look_back - horizon + 1):
            X.append(cell_data[i:i+look_back, :-1])
            y.append(cell_data[i+look_back:i+look_back+horizon, -1])  # SOH是最后一列
            cell_id_list.append(cell_id)
            
    return np.array(X), np.array(y), np.array(cell_id_list)

# 创建Teacher Forcing训练数据
def create_tf_sequences(data, cell_ids, look_back, horizon):
    X, y_true, decoder_inputs, cell_id_list = [], [], [], []
    
    for cell_id in np.unique(cell_ids):
        cell_data = data[cell_ids == cell_id]
        
        if len(cell_data) <= look_back + horizon:
            continue
            
        for i in range(len(cell_data) - look_back - horizon + 1):
            # 编码器输入：当前时间窗口的特征
            X.append(cell_data[i:i+look_back, :-1])  # 不包括SOH
            
            # 解码器输入：前一个时间步的SOH (t+0 到 t+horizon-1)
            if horizon > 1:
                # 取t+look_back-1(当前)到t+look_back+horizon-2(倒数第二个预测)的SOH
                decoder_input = cell_data[i+look_back-1:i+look_back+horizon-1, -1]
                decoder_inputs.append(decoder_input)
            else:
                # 如果只预测一步，则使用当前SOH
                decoder_inputs.append(np.array([cell_data[i+look_back-1, -1]]))
            
            # 真实目标：t+1到t+horizon的SOH
            y_true.append(cell_data[i+look_back:i+look_back+horizon, -1])
            
            cell_id_list.append(cell_id)
            
    return np.array(X), np.array(y_true), np.array(decoder_inputs), np.array(cell_id_list)

# 加载数据
def load_data():
    import data_processing as dp  # 引入 data_processing 模块
    
    # 指定数据所在目录（请根据实际情况调整路径）
    data_dir = "./01_Datenaufbereitung/Output/Calculated/"
    
    # 加载所有电池数据（parquet 格式）
    all_data = dp.load_data(data_dir)
    
    # 划分数据集：训练集、验证集和测试集
    # 参数 train/val/test 和 parts 可根据需求调整
    train_df, val_df, test_df = dp.split_data(all_data, train=13, val=1, test=1, parts=1)
    
    # 对数据进行标准化（仅对特定特征进行缩放）
    train_scaled, val_scaled, test_scaled = dp.scale_data(train_df, val_df, test_df)
    
    # 提取特征和目标变量
    X_train = train_scaled[features].values
    y_train = train_scaled[target].values.reshape(-1, 1)
    cell_ids_train = train_scaled['cell_id'].values

    X_val = val_scaled[features].values
    y_val = val_scaled[target].values.reshape(-1, 1)
    cell_ids_val = val_scaled['cell_id'].values

    X_test = test_scaled[features].values
    y_test = test_scaled[target].values.reshape(-1, 1)
    cell_ids_test = test_scaled['cell_id'].values

    # 合并特征与目标变量，方便后续创建时间序列窗口
    train_data = np.hstack((X_train, y_train))
    val_data = np.hstack((X_val, y_val))
    test_data = np.hstack((X_test, y_test))

    
    # 根据数据创建时间序列窗口（验证和测试集采用 create_sequences，训练集采用 Teacher Forcing 序列）
    X_train_enc, y_train_true, X_train_dec, train_cell_ids_tf = create_tf_sequences(train_data, cell_ids_train, look_back, horizon)
    X_val, y_val, val_cell_ids = create_sequences(val_data, cell_ids_val, look_back, horizon)
    X_test, y_test, test_cell_ids = create_sequences(test_data, cell_ids_test, look_back, horizon)
    
    return (X_train_enc, X_train_dec, y_train_true, train_cell_ids_tf,
            X_val, y_val, val_cell_ids,
            X_test, y_test, test_cell_ids)


# 创建PyTorch数据集
class SOHDataset(Dataset):
    def __init__(self, X_enc, X_dec, y):
        self.X_enc = torch.FloatTensor(X_enc)
        self.X_dec = None   
        self.y = None  
        
        if X_dec is not None and y is not None:
            self.X_dec = torch.FloatTensor(X_dec).unsqueeze(-1)  # 添加特征维度
            self.y = torch.FloatTensor(y).unsqueeze(-1)  # 添加特征维度
            self.has_labels = True
        else:
            self.has_labels = False
    
    def __len__(self):
        return len(self.X_enc)
    
    def __getitem__(self, idx):
        if self.has_labels:
            return self.X_enc[idx], self.X_dec[idx], self.y[idx]
        else:
            return self.X_enc[idx]

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

# 定义解码器
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hidden, cell):
        # x shape: [batch_size, 1, input_dim]
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

# 定义Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super(Seq2Seq, self).__init__()
        
        self.encoder = Encoder(input_dim, hidden_size, num_layers, dropout)
        self.decoder = Decoder(1, hidden_size, num_layers, dropout)
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src shape: [batch_size, src_seq_len, input_dim]
        # trg shape: [batch_size, trg_seq_len, 1]
        
        batch_size = src.shape[0]
        trg_seq_len = trg.shape[1]
        device = src.device
        
        # 初始化用于存储所有解码器输出的张量
        outputs = torch.zeros(batch_size, trg_seq_len, 1).to(device)
        
        # 编码器前向传播
        hidden, cell = self.encoder(src)
        
        # 首先使用真实的SOH启动解码器
        decoder_input = trg[:, 0, :].unsqueeze(1)  # [batch_size, 1, 1]
        
        for t in range(1, trg_seq_len):
            # 解码器前向传播
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            
            # 将预测存储到输出张量
            outputs[:, t, :] = output.squeeze(1)
            
            # 决定是使用教师强制还是使用自己的预测
            teacher_force = random.random() < teacher_forcing_ratio
            
            # 如果使用教师强制，使用真实值；否则使用预测值
            if teacher_force:
                decoder_input = trg[:, t, :].unsqueeze(1)
            else:
                decoder_input = output
        
        # 对第一个时间步使用已知的当前SOH
        outputs[:, 0, :] = trg[:, 0, :]
        
        return outputs
    
    def predict(self, src, initial_soh, horizon):
        # src shape: [batch_size, src_seq_len, input_dim]
        # initial_soh shape: [batch_size, 1]
        
        batch_size = src.shape[0]
        device = src.device
        
        # 初始化用于存储所有解码器输出的张量
        outputs = torch.zeros(batch_size, horizon, 1).to(device)
        
        # 编码器前向传播
        hidden, cell = self.encoder(src)
        
        # 首先使用真实的SOH启动解码器
        decoder_input = initial_soh.unsqueeze(-1).unsqueeze(1)  # [batch_size, 1, 1]
        
        # 第一步直接使用初始SOH
        outputs[:, 0, :] = decoder_input.squeeze(1)
        
        for t in range(1, horizon):
            # 解码器前向传播
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            
            # 将预测存储到输出张量
            outputs[:, t, :] = output.squeeze(1)
            
            # 使用当前预测作为下一步的输入
            decoder_input = output
        
        return outputs

# 训练函数
def train_epoch(model, dataloader, optimizer, criterion, device, scheduled_sampling_prob):
    model.train()
    epoch_loss = 0
    
    for X_enc, X_dec, y_true in tqdm(dataloader, desc="Training"):
        X_enc = X_enc.to(device)
        X_dec = X_dec.to(device)
        y_true = y_true.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        output = model(X_enc, X_dec, teacher_forcing_ratio=1.0 - scheduled_sampling_prob)
        
        # 计算损失
        loss = criterion(output, y_true)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

# 验证函数
def validate(model, X_val, y_val, criterion, device):
    model.eval()
    
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(-1).to(device)
        
        batch_size = X_val_tensor.shape[0]
        
        # 使用最后一个历史SOH值作为初始输入
        initial_soh = torch.FloatTensor([X_val[i, -1, -1] for i in range(batch_size)]).to(device)
        
        # 预测
        predictions = model.predict(X_val_tensor, initial_soh, horizon)
        
        # 计算损失
        val_loss = criterion(predictions, y_val_tensor).item()
        
        # 计算MAE
        val_mae = torch.mean(torch.abs(predictions - y_val_tensor)).item()
    
    return val_loss, val_mae, predictions.cpu().numpy()

# 主函数
def main():
    # 加载数据
    X_train_enc, X_train_dec, y_train_true, train_cell_ids, \
    X_val, y_val, val_cell_ids, \
    X_test, y_test, test_cell_ids = load_data()
    
    # 输入特征维度（C,V,T）
    input_dim = X_train_enc.shape[2]
    
    # 创建数据集和数据加载器
    train_dataset = SOHDataset(X_train_enc, X_train_dec, y_train_true)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = Seq2Seq(input_dim, hidden_size, num_layers, dropout_rate).to(device)
    
    # 初始化优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    # 初始化早停参数
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        # 调整scheduled sampling概率
        scheduled_sampling_prob = min(1.0, epoch / (num_epochs * 0.75))
        print(f"\nEpoch {epoch+1}/{num_epochs} - Sampling probability: {scheduled_sampling_prob:.3f}")
        
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduled_sampling_prob)
        
        # 验证
        val_loss, val_mae, _ = validate(model, X_val, y_val, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4e} - Val Loss: {val_loss:.4e} - Val MAE: {val_mae:.4e}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_soh_model_pytorch.pt')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('best_soh_model_pytorch.pt'))
    
    # 测试模型
    test_loss, test_mae, test_predictions = validate(model, X_test, y_test, criterion, device)
    print(f"Test Loss: {test_loss:.4f} - Test MAE: {test_mae:.4f}")
    
    # 可视化一些预测结果
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(sample_indices):
        plt.subplot(len(sample_indices), 1, i+1)
        plt.plot(range(horizon), y_test[idx], 'b-', label='实际SOH')
        plt.plot(range(horizon), test_predictions[idx, :, 0], 'r--', label='预测SOH')
        plt.title(f'电池 {test_cell_ids[idx]} 的SOH预测')
        plt.ylabel('SOH')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('soh_predictions_pytorch.png')
    plt.show()

if __name__ == "__main__":
    main()
