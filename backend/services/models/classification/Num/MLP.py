import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import time

# 定义简单的 MLP 分类器
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# 自定义 CSV 数据集加载类
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        # 读取 CSV 文件
        self.df = pd.read_csv(csv_file)
        
        # 特征部分（假设前几列是特征）
        self.X = self.df.iloc[:, :-1].values  # 所有列，最后一列为标签列
        self.y = self.df.iloc[:, -1].values  # 标签列
        
        # 使用 LabelEncoder 将标签转换为从0开始的整数
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)  # 将标签转换为整数
        
        # 打印标签和映射
        print(f"标签类别映射: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        # 确保标签的范围在 0 到 num_classes-1 之间
        num_classes = len(np.unique(self.y))

        # 打印调试信息，查看标签范围
        print(f"标签范围: {np.min(self.y)} 到 {np.max(self.y)}")

        # 检查标签是否超出类别数
        if np.min(self.y) < 0 or np.max(self.y) >= num_classes:
            raise ValueError(f"标签超出类别范围。标签最大值：{np.max(self.y)}, 类别数：{num_classes}")

        # 特征标准化（标准化数值特征）
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)  # 对数值特征进行标准化

        # 打印类别信息，用于调试
        print(f"Unique classes: {np.unique(self.y)}")  # 打印唯一标签类别
        print(f"Number of classes: {num_classes}")  # 打印类别数量

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)








# 设置设备（如果有 GPU 则使用 GPU，否则使用 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 输出具体显卡型号，而不是 'cuda'
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)  # 获取设备名称（第一个CUDA设备）
    print(f"使用设备: {device_name}")
else:
    print("使用设备: CPU")

def train_mlp(csv_file, epochs=50, batch_size=16, learning_rate=0.001, eval_metric="accuracy", task_id=None):
    # 创建日志文件，保存训练日志
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_filename = f"traininglogs/{task_id}_{current_time}_mlp.log"
    
    # 打开日志文件
    log_file = open(log_filename, 'w')
    
    # 记录训练模型和超参数
    log_file.write(f"Model: MLP\n")
    log_file.write(f"Epochs: {epochs}\n")
    log_file.write(f"Batch Size: {batch_size}\n")
    log_file.write(f"Learning Rate: {learning_rate}\n")
    log_file.write(f"Evaluation Metric: {eval_metric}\n")
    
    # 加载数据集
    dataset = CSVDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = dataset.X.shape[1]  # 特征维度
    num_classes = len(np.unique(dataset.y))  # 类别数
    hidden_dim = 64  # 隐藏层大小
    
    # 初始化模型，并移动到设备（GPU/CPU）
    model = MLPClassifier(input_dim, hidden_dim, num_classes)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    logs = []
    # 开始训练
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
        
        epoch_loss = running_loss / len(dataset)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # 根据用户选择的评价指标计算指标
        if eval_metric.lower() == "accuracy":
            epoch_metric = accuracy_score(all_labels, all_preds)
        elif eval_metric.lower() == "precision":
            epoch_metric = precision_score(all_labels, all_preds, average="macro")
        elif eval_metric.lower() == "recall":
            epoch_metric = recall_score(all_labels, all_preds, average="macro")
        elif eval_metric.lower() in ["f1", "f1-score"]:
            epoch_metric = f1_score(all_labels, all_preds, average="macro")
        else:
            epoch_metric = accuracy_score(all_labels, all_preds)
        
        logs.append({"epoch": epoch, "loss": epoch_loss, "metric": round(epoch_metric, 4)})
        
        # 记录每个 epoch 的损失和评估指标
        log_file.write(f"Epoch {epoch}: Loss={epoch_loss:.4f}, {eval_metric.capitalize()}={epoch_metric:.4f}\n")
        print(f"[MLP] Epoch {epoch}: Loss={epoch_loss:.4f}, {eval_metric.capitalize()}={epoch_metric:.4f}")
    
    log_file.close()  # 关闭日志文件
    
    return {
        "model": "MLP",
        "epochs": epochs,
        "final_metric": logs[-1]["metric"],
        "training_logs": logs,
        "message": "MLP训练完成"
    }
