#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
numProject_MLPClassifier.py

说明：
- 多分类 MLP，单隐藏层 + ReLU + Softmax 输出 + 交叉熵损失。
- 自动跳过CSV第一行(表头)。
- 如果没有单独测试集文件，就自动拆分 80%训练、20%测试。
"""

import numpy as np
import csv

class MLPClassifier:
    """
    多分类 MLP：单隐藏层, Softmax输出, 交叉熵损失, 全量梯度下降
    """
    def __init__(self, hidden_size=10, lr=0.01, epochs=500):
        """
        参数：
        hidden_size: 隐藏层神经元个数
        lr: 学习率
        epochs: 训练轮数
        """
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.num_classes = None

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_grad(self, x):
        return (x>0).astype(float)

    def _softmax(self, z):
        # z: (n_samples, num_classes)
        expz = np.exp(z - np.max(z, axis=1, keepdims=True))
        return expz / np.sum(expz, axis=1, keepdims=True)

    def fit(self, X, y):
        """
        训练MLP:
        X: (n_samples, n_features)
        y: (n_samples,) in {0,1,...,K-1}
        """
        n_samples, n_features = X.shape
        self.num_classes = len(np.unique(y))

        # 初始化参数
        self.W1 = 0.01*np.random.randn(n_features, self.hidden_size)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = 0.01*np.random.randn(self.hidden_size, self.num_classes)
        self.b2 = np.zeros(self.num_classes)

        # 构造one-hot标签
        Y_onehot = np.zeros((n_samples, self.num_classes))
        for i, lab in enumerate(y):
            Y_onehot[i, lab] = 1

        for epoch in range(self.epochs):
            # forward
            z1 = X.dot(self.W1) + self.b1
            a1 = self._relu(z1)
            z2 = a1.dot(self.W2) + self.b2
            probs = self._softmax(z2)

            # 交叉熵损失
            loss = -np.mean(np.sum(Y_onehot*np.log(probs+1e-9), axis=1))

            # backward
            dz2 = (probs - Y_onehot)/n_samples
            dW2 = a1.T.dot(dz2)
            db2 = np.sum(dz2, axis=0)

            da1 = dz2.dot(self.W2.T)
            dz1 = da1*self._relu_grad(z1)
            dW1 = X.T.dot(dz1)
            db1 = np.sum(dz1, axis=0)

            # 更新参数
            self.W2 -= self.lr*dW2
            self.b2 -= self.lr*db2
            self.W1 -= self.lr*dW1
            self.b1 -= self.lr*db1

            if epoch%100==0:
                print(f"[MLP] epoch={epoch}, loss={loss:.4f}")

    def predict(self, X):
        """
        预测：前向传播 => softmax => argmax
        """
        z1 = X.dot(self.W1) + self.b1
        a1 = self._relu(z1)
        z2 = a1.dot(self.W2) + self.b2
        probs = self._softmax(z2)
        return np.argmax(probs, axis=1)

# ============ 下面是可独立运行的 main ============

def load_data(file_path):
    """
    加载带表头CSV：
    - 跳过第一行
    - 最后一列是标签
    - 其余列数值化
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))
    # 跳过第一行表头
    rows = reader[1:]
    data = np.array(rows)

    X = data[:, :-1].astype(float)
    y_raw = data[:, -1]
    classes = np.unique(y_raw)
    label_map = {}
    for i, cls in enumerate(classes):
        label_map[cls] = i
    y = np.array([label_map[v] for v in y_raw], dtype=int)
    return X, y

def split_data(X, y, test_ratio=0.2):
    np.random.seed(42)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    n_test = int(len(y)*test_ratio)
    X_test = X[:n_test]
    y_test = y[:n_test]
    X_train = X[n_test:]
    y_train = y[n_test:]
    return X_train, y_train, X_test, y_test

def accuracy_score(y_true, y_pred):
    return np.mean(y_true==y_pred)

def main():
    # 可在此修改默认变量
    train_file = "../uploads/train.csv"
    test_file = None
    hidden_size = 10
    lr = 0.01
    epochs = 500

    print("=== 多分类 MLP 示例 ===")
    print("训练文件:", train_file)
    print("如果 test_file=None，则自动拆分部分数据做测试。")

    X_all, y_all = load_data(train_file)
    if test_file is None:
        print("未指定测试集 => 自动拆分 80% 训练, 20% 测试。")
        X_train, y_train, X_test, y_test = split_data(X_all, y_all, test_ratio=0.2)
    else:
        print("使用独立测试集:", test_file)
        X_train, y_train = X_all, y_all
        X_test, y_test = load_data(test_file)

    print(f"训练集大小: {len(y_train)}, 测试集大小: {len(y_test)}")

    model = MLPClassifier(hidden_size=hidden_size, lr=lr, epochs=epochs)
    print("开始训练 MLPClassifier...")
    model.fit(X_train, y_train)
    print("训练结束。")

    print("在测试集上评估...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {acc:.4f}")

if __name__=='__main__':
    main()
