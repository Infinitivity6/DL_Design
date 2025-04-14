# backend/services/models/classification/Num/CNN1D.py
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

# 自定义 CSV 数据集加载类
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        """
        读取 CSV 文件，准备数据集
        1. 特征部分：假设所有列除了最后一列是特征
        2. 标签部分：最后一列为标签
        3. 对标签进行编码：将标签从字符串转换为整数
        4. 特征标准化：使用标准化方法（均值为0，方差为1）进行标准化
        """
        self.df = pd.read_csv(csv_file)
        self.X = self.df.iloc[:, :-1].values  # 特征部分
        self.y = self.df.iloc[:, -1].values  # 标签列

        # 使用 LabelEncoder 将标签转换为整数
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)

        # 特征标准化（标准化数值特征）
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

    def __len__(self):
        """返回数据集的大小"""
        return len(self.df)

    def __getitem__(self, idx):
        """获取数据集的某一条数据"""
        # 对于CNN1D，需要将特征reshape为(channels, sequence_length)的形式
        # 这里假设只有1个channel，所有特征都是序列
        features = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return features, label

# 定义 CNN1D 模型
class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_filters=64, kernel_sizes=[3, 5, 7], dropout_rate=0.5):
        """
        初始化 CNN1D 模型
        input_dim: 输入特征的维度
        num_classes: 输出类别数
        num_filters: 卷积层的滤波器数量
        kernel_sizes: 卷积核的大小列表，使用多种大小的卷积核捕获不同长度的特征
        dropout_rate: Dropout层的比率，用于防止过拟合
        """
        super(CNN1DClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        
        # 多个卷积层，每个使用不同大小的卷积核
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, 
                      out_channels=num_filters, 
                      kernel_size=k, 
                      padding=(k-1)//2) 
            for k in kernel_sizes
        ])
        
        # 批量归一化层
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) 
            for _ in kernel_sizes
        ])
        
        # 池化层
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        前向传播函数
        x: 输入数据，形状为 [batch_size, 1, input_dim]
        """
        # 应用多个卷积层
        conv_results = []
        for i, conv in enumerate(self.convs):
            # 卷积 -> 批量归一化 -> ReLU激活 -> 池化
            conv_out = conv(x)
            conv_out = self.batch_norms[i](conv_out)
            conv_out = torch.relu(conv_out)
            conv_out = self.pool(conv_out).squeeze(-1)  # [batch_size, num_filters]
            conv_results.append(conv_out)
        
        # 连接多个卷积结果
        x = torch.cat(conv_results, dim=1)  # [batch_size, num_filters * len(kernel_sizes)]
        
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 设置设备：如果有 GPU 则使用 GPU，否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_cnn1d(csv_file, epochs=50, batch_size=16, learning_rate=0.001, eval_metric="accuracy", task_id=None, task_manager=None):
    """
    训练 CNN1D 模型
    csv_file: 输入的 CSV 数据文件
    epochs: 训练的轮次
    batch_size: 每个 batch 的大小
    learning_rate: 学习率
    eval_metric: 评估指标，通常是 accuracy 或 f1 等
    task_id: 当前任务的 ID
    task_manager: 任务管理器，用于更新任务状态
    """
    # 创建日志文件，保存训练过程的日志
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_filename = f"traininglogs/{task_id}_{current_time}_cnn1d.log" if task_id else f"traininglogs/{current_time}_cnn1d.log"
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    log_file = open(log_filename, 'w')

    # 记录训练模型和超参数
    log_file.write(f"Model: CNN1D\n")
    log_file.write(f"Epochs: {epochs}\n")
    log_file.write(f"Batch Size: {batch_size}\n")
    log_file.write(f"Learning Rate: {learning_rate}\n")
    log_file.write(f"Evaluation Metric: {eval_metric}\n")

    # 加载数据集
    dataset = CSVDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = dataset.X.shape[1]  # 特征的维度
    num_classes = len(np.unique(dataset.y))  # 类别数目

    # 初始化 CNN1D 模型
    model = CNN1DClassifier(input_dim, num_classes)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam 优化器

    logs = []  # 用于记录每个 epoch 的日志
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        for X_batch, y_batch in dataloader:
            # 将数据和标签转移到 GPU 或 CPU
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()  # 清空梯度
            outputs = model(X_batch)  # 模型输出
            loss = criterion(outputs, y_batch)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            epoch_loss += loss.item() * X_batch.size(0)  # 记录当前批次的损失
            _, preds = torch.max(outputs, 1)  # 获取预测结果
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

        epoch_loss /= len(dataset)  # 计算每个 epoch 的平均损失
        all_preds = np.concatenate(all_preds)  # 合并所有预测结果
        all_labels = np.concatenate(all_labels)  # 合并所有真实标签

        # 计算评估指标
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

        # 记录每个 epoch 的损失和评估指标
        logs.append({"epoch": epoch, "loss": epoch_loss, "metric": round(epoch_metric, 4)})

        # 记录每个 epoch 的损失和评估指标到日志文件
        log_file.write(f"Epoch {epoch}: Loss={epoch_loss:.4f}, {eval_metric.capitalize()}={epoch_metric:.4f}\n")
        print(f"[CNN1D] Epoch {epoch}: Loss={epoch_loss:.4f}, {eval_metric.capitalize()}={epoch_metric:.4f}")

        # 更新任务状态（如果有任务管理器）
        if task_manager and task_id:
            task_manager.update_task(task_id, {'training_logs': logs})

    # 关闭日志文件
    log_file.close()

    # 返回训练结果
    return {
        "model": "CNN1D",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "eval_metric": eval_metric,
        "final_metric": logs[-1]["metric"],  # 最终评估指标
        "training_logs": logs,  # 训练日志
        "final_accuracy": logs[-1]["metric"],  # 最终准确率
        "status": "completed",  # 任务完成标志
        "message": "CNN1D训练完成"
    }