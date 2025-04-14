# services/models/classification/Image/ResNet18.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import time
import numpy as np
from PIL import Image

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 在CustomImageDataset类中需要确保正确使用文件夹名称作为类别
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, categories, transform=None):
        """
        自定义图像数据集类
        
        Args:
            root_dir: 数据根目录
            categories: 类别列表（文件夹名称）
            transform: 数据预处理变换
        """
        self.root_dir = root_dir
        self.categories = categories
        self.transform = transform
        self.samples = []
        self.targets = []
        
        # 为每个类别分配一个索引
        self.class_to_idx = {category: idx for idx, category in enumerate(categories)}
        
        print(f"类别映射: {self.class_to_idx}")
        
        # 收集数据集中的所有图像
        for category in categories:
            category_path = os.path.join(root_dir, category)
            
            print(f"扫描类别目录: {category_path}")
            if not os.path.exists(category_path):
                print(f"警告: 类别目录不存在: {category_path}")
                continue
                
            image_count = 0
            for img_name in os.listdir(category_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(category_path, img_name)
                    self.samples.append(img_path)
                    self.targets.append(self.class_to_idx[category])
                    image_count += 1
            
            print(f"类别 '{category}' 加载了 {image_count} 张图像")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        
        # 读取图像
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个固定大小的黑色图像作为替代
            img = Image.new('RGB', (224, 224))
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        
        return img, target

def train_resnet18(data_dir, categories, epochs=50, batch_size=16, learning_rate=0.001, eval_metric="accuracy", task_id=None, task_manager=None):
    """
    训练 ResNet18 模型
    
    Args:
        data_dir: 数据目录
        categories: 类别列表
        epochs: 训练轮数
        batch_size: 批处理大小
        learning_rate: 学习率
        eval_metric: 评估指标
        task_id: 任务ID
        task_manager: 任务管理器实例
        
    Returns:
        dict: 包含训练结果的字典
    """
    # 创建日志文件
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_filename = f"traininglogs/{task_id}_{current_time}_resnet18.log" if task_id else f"traininglogs/{current_time}_resnet18.log"
    
    # 打开日志文件
    log_file = open(log_filename, 'w')
    
    # 记录训练模型和超参数
    log_file.write(f"Model: ResNet18\n")
    log_file.write(f"Epochs: {epochs}\n")
    log_file.write(f"Batch Size: {batch_size}\n")
    log_file.write(f"Learning Rate: {learning_rate}\n")
    log_file.write(f"Evaluation Metric: {eval_metric}\n")
    log_file.write(f"Categories: {categories}\n")
    
    try:
        # 定义数据预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整大小为 224x224
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        ])
        
        # 创建数据集
        dataset = CustomImageDataset(data_dir, categories, transform=transform)
        
        # 检查数据集大小
        if len(dataset) == 0:
            log_file.write("Error: 数据集为空！\n")
            log_file.close()
            return {
                "status": "failed",
                "message": "数据集为空，请确保每个类别目录中有图像文件"
            }
        
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 加载预训练的 ResNet18 模型
        model = models.resnet18(pretrained=True)
        
        # 修改最后一层，适应我们的类别数
        num_classes = len(categories)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # 移动模型到设备
        model = model.to(device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练日志
        logs = []
        
        # 开始训练
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                # 统计损失
                running_loss += loss.item() * inputs.size(0)
                
                # 记录预测结果
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # 计算epoch损失
            epoch_loss = running_loss / len(dataset)
            
            # 计算评估指标
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
            
            # 记录日志
            log_entry = {"epoch": epoch, "loss": round(epoch_loss, 4), "metric": round(epoch_metric, 4)}
            logs.append(log_entry)
            
            # 写入日志文件
            log_file.write(f"Epoch {epoch}: Loss={epoch_loss:.4f}, {eval_metric.capitalize()}={epoch_metric:.4f}\n")
            print(f"[ResNet18] Epoch {epoch}: Loss={epoch_loss:.4f}, {eval_metric.capitalize()}={epoch_metric:.4f}")
            
            # 更新任务状态
            if task_manager and task_id:
                task_manager.update_task(task_id, {'training_logs': logs})
        
        # 关闭日志文件
        log_file.close()
        
        # 保存模型(可选)
        # model_save_path = os.path.join(data_dir, f"resnet18_{task_id}.pth")
        # torch.save(model.state_dict(), model_save_path)
        
        # 返回训练结果
        return {
            "model": "ResNet18",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "eval_metric": eval_metric,
            "final_metric": logs[-1]["metric"],
            "training_logs": logs,
            "final_accuracy": logs[-1]["metric"],
            "status": "completed",
            "message": "ResNet18训练完成",
            "categories": categories
        }
        
    except Exception as e:
        # 记录异常
        log_file.write(f"训练过程中发生错误: {str(e)}\n")
        log_file.close()
        
        print(f"[ResNet18] Error: {str(e)}")
        
        # 返回错误信息
        return {
            "model": "ResNet18",
            "status": "failed",
            "message": f"训练过程中发生错误: {str(e)}",
            "categories": categories
        }