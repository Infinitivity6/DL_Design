#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
numProject_SVMClassifier.py

说明：
- 多分类线性SVM，采用 One-vs-Rest(OvR) 策略。
- 带表头的CSV，自动跳过第一行。
- 最后一列是标签，其余列都是数值特征。
- 如果 test_file=None，则自动拆分训练集的80%训练、20%测试。
- 可处理标签为数字或字母（都会映射成0,1,2,...）。
"""

import numpy as np
import csv

class SVMClassifier:
    """
    多分类线性SVM：使用 One-vs-Rest (OvR) 策略
    """
    def __init__(self, lr=0.01, epochs=100, C=1.0):
        """
        参数：
        lr: 学习率
        epochs: 训练轮数
        C: 惩罚系数(越大则对分类错误越敏感)
        """
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.ovr_models = []  # [(w, b, class_label), ...]

    def fit(self, X, y):
        """
        训练多分类SVM:
        X: (n_samples, n_features)
        y: (n_samples,) in {0,1,...,K-1}
        """
        self.ovr_models = []
        classes = np.unique(y)
        for cls in classes:
            # 构造二分类标签：+1 if y_i==cls, else -1
            y_ = np.where(y==cls, 1, -1)
            w, b = self._train_one(X, y_)
            self.ovr_models.append((w, b, cls))

    def _train_one(self, X, y_):
        """
        训练一个二分类SVM (标签+1/-1)，返回 (w, b)
        简化的 hinge loss + L2正则，全量梯度下降
        """
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        for epoch in range(self.epochs):
            margin = y_*(X.dot(w) + b)
            mask = (margin<1).astype(float)
            dw = w.copy()  # L2 => w
            db = 0.0
            for i in range(n_samples):
                if mask[i]>0:
                    dw += -self.C * y_[i] * X[i]
                    db += -self.C * y_[i]
            w -= self.lr*dw
            b -= self.lr*db

            if epoch%10==0:
                loss = self._compute_loss(X, y_, w, b)
                print(f"[SVM] epoch={epoch}, loss={loss:.4f}")

        return w, b

    def _compute_loss(self, X, y_, w, b):
        margin = y_*(X.dot(w)+b)
        hinge = np.maximum(0, 1 - margin)
        # loss = 0.5||w||^2 + C * sum(hinge)
        loss = 0.5*np.sum(w**2) + self.C*np.sum(hinge)
        return loss

    def decision_function(self, X):
        """
        返回 (n_samples, K) 的打分矩阵
        """
        n_samples = X.shape[0]
        K = len(self.ovr_models)
        scores = np.zeros((n_samples, K))
        for k, (w, b, cls) in enumerate(self.ovr_models):
            scores[:, k] = X.dot(w) + b
        return scores

    def predict(self, X):
        """
        多分类预测：选打分最高的类别
        """
        scores = self.decision_function(X)
        idx_max = np.argmax(scores, axis=1)
        preds = []
        for i in range(len(idx_max)):
            k = idx_max[i]
            preds.append(self.ovr_models[k][2])
        return np.array(preds)

# ================== 下面是独立运行的 main =====================

def load_data(file_path):
    """
    加载带表头的CSV：
    - 跳过第一行(表头)
    - 最后一列是标签
    - 其余列是数值特征
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    # 跳过第一行表头
    rows = rows[1:]
    data = np.array(rows)

    X = data[:, :-1].astype(float)
    y_raw = data[:, -1]
    # 构建标签映射
    classes = np.unique(y_raw)
    label_map = {}
    for i, cls in enumerate(classes):
        label_map[cls] = i
    y = np.array([label_map[v] for v in y_raw], dtype=int)
    return X, y

def split_data(X, y, test_ratio=0.2):
    """
    自动拆分数据为 train/test
    """
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
    # ============可修改的默认变量===========
    train_file = "../preProcess/train.csv"  # 训练数据文件（带表头）
    test_file = None                    # 如果为None则自动拆分
    lr = 0.01
    epochs = 100
    C = 1.0
    # =====================================

    print("=== 线性SVM 多分类示例 ===")
    print("假设训练文件为:", train_file)
    print("若 test_file=None，则自动拆分训练数据的一部分做测试。")

    X_all, y_all = load_data(train_file)
    if test_file is None:
        print("未指定测试集文件，自动拆分 80%训练, 20%测试...")
        X_train, y_train, X_test, y_test = split_data(X_all, y_all, test_ratio=0.2)
    else:
        print("使用独立的测试集文件:", test_file)
        X_train, y_train = X_all, y_all
        X_test, y_test = load_data(test_file)

    print(f"训练集大小: {len(y_train)}，测试集大小: {len(y_test)}")

    model = SVMClassifier(lr=lr, epochs=epochs, C=C)
    print("开始训练SVMClassifier...")
    model.fit(X_train, y_train)
    print("训练结束。")

    print("在测试集上进行评估...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {acc:.4f}")

if __name__ == '__main__':
    main()
