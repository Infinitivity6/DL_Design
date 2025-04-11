# cnn1d_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# 定义简单的一维卷积分类器
class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN1DClassifier, self).__init__()
        # 将输入数据视为 1 维序列
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 计算池化后特征维度
        self.fc = nn.Linear((input_dim // 2) * 16, num_classes)
    
    def forward(self, x):
        # x.shape: [batch, input_dim] -> [batch, 1, input_dim]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.X = self.df.iloc[:, :-1].values.astype(np.float32)
        self.y = self.df.iloc[:, -1].values
        if self.y.dtype.kind in 'OU':
            unique_labels = np.unique(self.y)
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.y = np.array([self.label_to_idx[label] for label in self.y], dtype=np.int64)
        else:
            self.y = self.y.astype(np.int64)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def train_cnn1d(csv_file, epochs=50, batch_size=16, learning_rate=0.001):
    dataset = CSVDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = dataset.X.shape[1]
    num_classes = len(np.unique(dataset.y))
    
    model = CNN1DClassifier(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    logs = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        logs.append({"epoch": epoch, "loss": epoch_loss, "accuracy": round(epoch_acc, 4)})
        print(f"[1D CNN] Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")
    
    return {
        "model": "1DCNN",
        "epochs": epochs,
        "final_accuracy": logs[-1]["accuracy"],
        "training_logs": logs,
        "message": "1D CNN训练完成"
    }

if __name__ == '__main__':
    result = train_cnn1d("../uploads/train.csv", epochs=50, batch_size=16, learning_rate=0.001)
    print(result)
