"""
决策树算法实现
"""
import numpy as np
import pandas as pd
from collections import Counter
import math

class DecisionTree:
    """
    决策树分类器
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, criterion='entropy'):
        """
        初始化决策树
        
        参数:
        max_depth: 最大深度
        min_samples_split: 最小分割样本数
        criterion: 分割标准 ('entropy' 或 'gini')
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None
        self.feature_names = None
        self.class_names = None
    
    def fit(self, X, y, feature_names=None, class_names=None):
        """
        训练决策树
        
        参数:
        X: 特征数据
        y: 标签数据
        feature_names: 特征名称
        class_names: 类别名称
        """
        self.feature_names = feature_names
        self.class_names = class_names
        self.tree = self._build_tree(X, y, depth=0)
    
    def predict(self, X):
        """
        预测
        
        参数:
        X: 待预测的特征数据
        
        返回:
        predictions: 预测结果
        """
        return np.array([self._predict_sample(x, self.tree) for x in X])
    
    def _build_tree(self, X, y, depth):
        """
        构建决策树
        
        参数:
        X: 特征数据
        y: 标签数据
        depth: 当前深度
        
        返回:
        tree: 树节点
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            # 创建叶节点
            most_common_class = Counter(y).most_common(1)[0][0]
            return {'class': most_common_class, 'samples': n_samples}
        
        # 寻找最佳分割
        best_split = self._find_best_split(X, y)
        
        if best_split is None:
            # 无法分割，创建叶节点
            most_common_class = Counter(y).most_common(1)[0][0]
            return {'class': most_common_class, 'samples': n_samples}
        
        # 根据最佳分割创建子树
        feature_idx, threshold = best_split
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree,
            'samples': n_samples,
            'impurity': self._calculate_impurity(y)
        }
    
    def _find_best_split(self, X, y):
        """
        寻找最佳分割
        
        参数:
        X: 特征数据
        y: 标签数据
        
        返回:
        best_split: 最佳分割 (feature_idx, threshold)
        """
        n_samples, n_features = X.shape
        
        if n_samples <= 1:
            return None
        
        # 计算父节点的不纯度
        parent_impurity = self._calculate_impurity(y)
        
        best_gain = 0
        best_split = None
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # 计算信息增益
                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                gain = parent_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, threshold)
        
        return best_split
    
    def _calculate_impurity(self, y):
        """
        计算不纯度
        
        参数:
        y: 标签数据
        
        返回:
        impurity: 不纯度
        """
        if len(y) == 0:
            return 0
        
        class_counts = Counter(y)
        n_samples = len(y)
        
        if self.criterion == 'entropy':
            # 信息熵
            entropy = 0
            for count in class_counts.values():
                p = count / n_samples
                if p > 0:
                    entropy -= p * math.log2(p)
            return entropy
        
        elif self.criterion == 'gini':
            # 基尼不纯度
            gini = 1
            for count in class_counts.values():
                p = count / n_samples
                gini -= p * p
            return gini
    
    def _predict_sample(self, x, tree):
        """
        预测单个样本
        
        参数:
        x: 样本特征
        tree: 决策树节点
        
        返回:
        prediction: 预测结果
        """
        if 'class' in tree:
            return tree['class']
        
        if x[tree['feature_idx']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def print_tree(self, tree=None, depth=0):
        """
        打印决策树
        
        参数:
        tree: 树节点
        depth: 深度
        """
        if tree is None:
            tree = self.tree
        
        if 'class' in tree:
            class_name = tree['class']
            if self.class_names is not None:
                class_name = self.class_names[class_name]
            print(f"{'  ' * depth}预测: {class_name} (样本数: {tree['samples']})")
        else:
            feature_name = f"特征{tree['feature_idx']}"
            if self.feature_names is not None:
                feature_name = self.feature_names[tree['feature_idx']]
            
            print(f"{'  ' * depth}{feature_name} <= {tree['threshold']:.3f}? (样本数: {tree['samples']}, 不纯度: {tree['impurity']:.3f})")
            print(f"{'  ' * depth}├─ 是:")
            self.print_tree(tree['left'], depth + 1)
            print(f"{'  ' * depth}└─ 否:")
            self.print_tree(tree['right'], depth + 1)
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        返回:
        importance: 特征重要性
        """
        if self.tree is None:
            return None
        
        n_features = len(self.feature_names) if self.feature_names else self._get_max_feature_idx(self.tree) + 1
        importance = np.zeros(n_features)
        
        self._calculate_feature_importance(self.tree, importance)
        
        # 归一化
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance
    
    def _get_max_feature_idx(self, tree):
        """
        获取最大特征索引
        """
        if 'class' in tree:
            return -1
        
        max_idx = tree['feature_idx']
        left_max = self._get_max_feature_idx(tree['left'])
        right_max = self._get_max_feature_idx(tree['right'])
        
        return max(max_idx, left_max, right_max)
    
    def _calculate_feature_importance(self, tree, importance):
        """
        计算特征重要性
        """
        if 'class' in tree:
            return
        
        # 计算该节点的重要性增益
        left_samples = tree['left']['samples']
        right_samples = tree['right']['samples']
        total_samples = tree['samples']
        
        left_impurity = tree['left'].get('impurity', 0)
        right_impurity = tree['right'].get('impurity', 0)
        
        weighted_impurity = (left_samples * left_impurity + right_samples * right_impurity) / total_samples
        importance_gain = tree['impurity'] - weighted_impurity
        
        importance[tree['feature_idx']] += importance_gain * total_samples
        
        # 递归计算子树
        self._calculate_feature_importance(tree['left'], importance)
        self._calculate_feature_importance(tree['right'], importance)

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