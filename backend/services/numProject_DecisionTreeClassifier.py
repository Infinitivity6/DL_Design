#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
numProject_DecisionTreeClassifier.py

说明：
- 多分类决策树，使用基尼系数，不做剪枝。
- 自动跳过CSV第一行(表头)。
- 若没有test_file，则自动拆分80%训练、20%测试。
"""

import numpy as np
import csv

class DecisionTreeClassifier:
    """
    多分类决策树，基尼系数
    """
    def __init__(self, max_depth=5, min_samples_split=2):
        """
        参数：
        max_depth: 决策树最大深度
        min_samples_split: 节点最少样本数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def fit(self, X, y):
        self.tree_ = self._build_tree(X, y, depth=0)

    def predict(self, X):
        preds = []
        for x in X:
            preds.append(self._predict_one(x, self.tree_))
        return np.array(preds)

    def _gini(self, y):
        """
        多分类基尼系数
        """
        m = len(y)
        if m==0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        p = counts/m
        return 1 - np.sum(p**2)

    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        # 终止条件
        if depth>=self.max_depth or n_samples<self.min_samples_split or self._gini(y)==0:
            return {"leaf": True, "label": self._majority_class(y)}

        feat, thresh, gain = self._best_split(X, y)
        if gain==0:
            return {"leaf": True, "label": self._majority_class(y)}

        left_mask = (X[:, feat]<=thresh)
        right_mask = ~left_mask
        node = {
            "leaf": False,
            "feat": feat,
            "thresh": thresh,
            "left": self._build_tree(X[left_mask], y[left_mask], depth+1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth+1)
        }
        return node

    def _best_split(self, X, y):
        m, n = X.shape
        base_gini = self._gini(y)
        best_gain = 0
        best_feat = None
        best_thresh = None

        for feat in range(n):
            values = np.unique(X[:, feat])
            if len(values)==1:
                continue
            # 取相邻值中点做候选阈值
            thresholds = (values[:-1]+values[1:])/2
            for t in thresholds:
                left_mask = (X[:, feat]<=t)
                right_mask = ~left_mask
                y_left, y_right = y[left_mask], y[right_mask]
                w_left = len(y_left)/m
                w_right = len(y_right)/m
                g_left = self._gini(y_left)
                g_right = self._gini(y_right)
                g_split = w_left*g_left + w_right*g_right
                gain = base_gini - g_split
                if gain>best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = t

        return best_feat, best_thresh, best_gain

    def _majority_class(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["label"]
        if x[node["feat"]] <= node["thresh"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

# ========== 下面是独立运行的 main ==========

def load_data(file_path):
    """
    加载带表头CSV：跳过第一行
    最后一列是标签
    其余列都是数值特征
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))
    # 跳过表头
    rows = reader[1:]
    data = np.array(rows)

    X = data[:, :-1].astype(float)
    y_raw = data[:, -1]
    # 标签映射
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
    # 可修改的默认变量
    train_file = "../preProcess/electricity.csv"
    test_file = None
    max_depth = 5
    min_samples_split = 2

    print("=== 多分类决策树示例 ===")
    print("训练文件:", train_file)
    print("若 test_file=None，则自动拆分 80%训练, 20%测试。")

    X_all, y_all = load_data(train_file)
    if test_file is None:
        print("未指定测试集 => 自动拆分 80/20.")
        X_train, y_train, X_test, y_test = split_data(X_all, y_all, test_ratio=0.2)
    else:
        print("使用独立测试集:", test_file)
        X_train, y_train = X_all, y_all
        X_test, y_test = load_data(test_file)

    print(f"训练集大小: {len(y_train)}, 测试集大小: {len(y_test)}")

    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    print("开始训练 DecisionTreeClassifier...")
    model.fit(X_train, y_train)
    print("训练结束。")

    print("在测试集上评估...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {acc:.4f}")

if __name__=='__main__':
    main()
