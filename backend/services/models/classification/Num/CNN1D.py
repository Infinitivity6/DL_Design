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
from services.task_manager import TaskManager

# 自定义 CSV 数据集加载类
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.X = self.df.iloc[:, :-1].values  # 特征部分
        self.y = self.df.iloc[:, -1].values  # 标签列
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)
        
        # 特征标准化
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# 定义 1D 卷积网络
class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * (input_dim - 2) // 2, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        return x

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_cnn1d(csv_file, epochs=50, batch_size=16, learning_rate=0.001, eval_metric="accuracy", task_id=None, task_manager=None):
    # 创建日志文件
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_filename = f"traininglogs/{task_id}_{current_time}_cnn1d.log"
    log_file = open(log_filename, 'w')

    log_file.write(f"Model: CNN1D\n")
    log_file.write(f"Epochs: {epochs}\n")
    log_file.write(f"Batch Size: {batch_size}\n")
    log_file.write(f"Learning Rate: {learning_rate}\n")
    log_file.write(f"Evaluation Metric: {eval_metric}\n")

    dataset = CSVDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = dataset.X.shape[1]
    num_classes = len(np.unique(dataset.y))

    # 初始化 CNN1D 模型
    model = CNN1DClassifier(input_dim, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logs = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            X_batch = X_batch.unsqueeze(1)  # 添加通道维度
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

        epoch_loss /= len(dataset)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

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

        log_file.write(f"Epoch {epoch}: Loss={epoch_loss:.4f}, {eval_metric.capitalize()}={epoch_metric:.4f}\n")
        print(f"[CNN1D] Epoch {epoch}: Loss={epoch_loss:.4f}, {eval_metric.capitalize()}={epoch_metric:.4f}")

        task_manager.update_task(task_id, {'training_logs': logs})

    log_file.close()

    return {
        "model": "CNN1D",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "eval_metric": eval_metric,
        "final_metric": logs[-1]["metric"],
        "training_logs": logs,
        "final_accuracy": logs[-1]["metric"],
        "status": "completed",
        "message": "CNN1D训练完成"
}
