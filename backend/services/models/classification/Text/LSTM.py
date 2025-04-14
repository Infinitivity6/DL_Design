# backend/services/models/classification/Text/LSTM.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os
import time
import jieba

# 定义文本数据集（同BERT版本，复用分词逻辑）
class TextDataset(Dataset):
    def __init__(self, csv_dir, language):
        """
        构建文本数据集
        参数：
          csv_dir: 存放 CSV 文件的目录
          language: 文本语言 ('zh' 或 'en')
        """
        files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if len(files) == 0:
            raise ValueError(f"目录 {csv_dir} 中没有找到 CSV 文件")
        df_list = [pd.read_csv(f) for f in files]
        self.df = pd.concat(df_list, ignore_index=True)
        self.language = language
        self.texts = self.df.iloc[:, 0].astype(str).values
        self.labels = self.df.iloc[:, -1].values

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

        # 构建词汇表
        vocab = set()
        for text in self.texts:
            if self.language.lower() == 'zh':
                words = jieba.lcut(text)
            else:
                words = text.split()
            vocab.update(words)
        self.vocab = {word: idx+1 for idx, word in enumerate(vocab)}
        self.vocab_size = len(self.vocab) + 1
        self.max_len = 100

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.language.lower() == 'zh':
            tokens = jieba.lcut(text)
        else:
            tokens = text.split()
        token_ids = [self.vocab.get(word, 0) for word in tokens]
        if len(token_ids) < self.max_len:
            token_ids += [0] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        label = self.labels[idx]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# 定义基于 LSTM 的文本分类模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        embeds = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        lstm_out, (h_n, _) = self.lstm(embeds)  # h_n: [num_layers, batch_size, hidden_dim]
        final_feature = h_n[-1]  # 取最后一层的隐状态
        logits = self.fc(final_feature)
        return logits

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_lstm(csv_dir, language, epochs=50, batch_size=16, learning_rate=0.001, eval_metric="accuracy", task_id=None, task_manager=None):
    """
    训练基于 LSTM 的文本分类模型
    参数同 train_bert
    """
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_filename = f"traininglogs/{task_id}_{current_time}_lstm.log" if task_id else f"traininglogs/{current_time}_lstm.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    log_file = open(log_filename, 'w')
    log_file.write(f"Model: LSTM\nEpochs: {epochs}\nBatch Size: {batch_size}\nLearning Rate: {learning_rate}\nEvaluation Metric: {eval_metric}\nLanguage: {language}\n")
    
    dataset = TextDataset(csv_dir, language)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    num_classes = len(np.unique(dataset.labels))
    vocab_size = dataset.vocab_size
    embed_dim = 128
    hidden_dim = 256
    num_layers = 1

    model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes, num_layers)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logs = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        all_preds = []
        all_labels = []
        for tokens, labels in dataloader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * tokens.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        epoch_loss = running_loss / len(dataset)
        if eval_metric.lower() == "accuracy":
            epoch_metric = accuracy_score(all_labels, all_preds)
        elif eval_metric.lower() == "precision":
            epoch_metric = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        elif eval_metric.lower() == "recall":
            epoch_metric = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        elif eval_metric.lower() in ["f1", "f1-score"]:
            epoch_metric = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        else:
            epoch_metric = accuracy_score(all_labels, all_preds)
        log_entry = {"epoch": epoch, "loss": round(epoch_loss, 4), "metric": round(epoch_metric, 4)}
        logs.append(log_entry)
        log_file.write(f"Epoch {epoch}: Loss={epoch_loss:.4f}, {eval_metric.capitalize()}={epoch_metric:.4f}\n")
        print(f"[LSTM] Epoch {epoch}: Loss={epoch_loss:.4f}, {eval_metric.capitalize()}={epoch_metric:.4f}")
        if task_manager and task_id:
            task_manager.update_task(task_id, {'training_logs': logs})
    log_file.close()
    return {
        "model": "LSTM",
        "language": language,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "eval_metric": eval_metric,
        "final_metric": logs[-1]["metric"],
        "training_logs": logs,
        "final_accuracy": logs[-1]["metric"],
        "status": "completed",
        "message": "LSTM训练完成"
    }
