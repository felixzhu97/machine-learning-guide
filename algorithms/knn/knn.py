"""
K-近邻算法实现
"""
import numpy as np
from collections import Counter

class KNN:
    """
    K-近邻分类器
    """
    
    def __init__(self, k=3):
        """
        初始化K-近邻分类器
        
        参数:
        k: 邻居数量
        """
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        训练模型（实际上是存储训练数据）
        
        参数:
        X: 训练特征数据
        y: 训练标签数据
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """
        预测
        
        参数:
        X: 待预测的特征数据
        
        返回:
        predictions: 预测结果
        """
        predictions = []
        for x in X:
            # 计算距离
            distances = self._calculate_distances(x)
            # 找到最近的k个邻居
            k_nearest_indices = np.argsort(distances)[:self.k]
            # 获取这k个邻居的标签
            k_nearest_labels = self.y_train[k_nearest_indices]
            # 投票决定预测结果
            prediction = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(prediction)
        return np.array(predictions)
    
    def _calculate_distances(self, x):
        """
        计算欧几里得距离
        
        参数:
        x: 单个样本
        
        返回:
        distances: 距离数组
        """
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        return distances
    
    def predict_proba(self, X):
        """
        预测概率
        
        参数:
        X: 待预测的特征数据
        
        返回:
        probabilities: 预测概率
        """
        probabilities = []
        for x in X:
            # 计算距离
            distances = self._calculate_distances(x)
            # 找到最近的k个邻居
            k_nearest_indices = np.argsort(distances)[:self.k]
            # 获取这k个邻居的标签
            k_nearest_labels = self.y_train[k_nearest_indices]
            # 计算各类别的概率
            label_counts = Counter(k_nearest_labels)
            total_neighbors = len(k_nearest_labels)
            proba = {label: count / total_neighbors for label, count in label_counts.items()}
            probabilities.append(proba)
        return probabilities

def normalize_data(data):
    """
    归一化数据
    
    参数:
    data: 输入数据
    
    返回:
    normalized_data: 归一化后的数据
    ranges: 数据范围
    """
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    ranges = max_vals - min_vals
    normalized_data = (data - min_vals) / ranges
    return normalized_data, ranges, min_vals

def calculate_accuracy(y_true, y_pred):
    """
    计算准确率
    
    参数:
    y_true: 真实标签
    y_pred: 预测标签
    
    返回:
    accuracy: 准确率
    """
    return np.mean(y_true == y_pred) 