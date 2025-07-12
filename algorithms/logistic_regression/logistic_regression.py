"""
逻辑回归算法实现
"""
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    逻辑回归分类器
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=100, tolerance=1e-6):
        """
        初始化逻辑回归
        
        参数:
        learning_rate: 学习率
        max_iterations: 最大迭代次数
        tolerance: 收敛容差
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.cost_history = []
        
    def sigmoid(self, z):
        """
        Sigmoid激活函数
        
        参数:
        z: 线性组合结果
        
        返回:
        sigmoid值
        """
        # 防止溢出
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def fit(self, X, y):
        """
        训练逻辑回归模型
        
        参数:
        X: 特征数据
        y: 标签数据 (0或1)
        """
        # 添加偏置项
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        n_features = X_with_bias.shape[1]
        
        # 初始化权重
        self.weights = np.zeros(n_features)
        
        # 梯度上升
        for i in range(self.max_iterations):
            # 计算预测值
            z = np.dot(X_with_bias, self.weights)
            predictions = self.sigmoid(z)
            
            # 计算误差
            error = y - predictions
            
            # 计算梯度
            gradient = np.dot(X_with_bias.T, error)
            
            # 更新权重
            old_weights = self.weights.copy()
            self.weights += self.learning_rate * gradient
            
            # 计算代价函数
            cost = self._compute_cost(y, predictions)
            self.cost_history.append(cost)
            
            # 检查收敛
            if np.sum(np.abs(self.weights - old_weights)) < self.tolerance:
                print(f"在第 {i+1} 次迭代后收敛")
                break
    
    def _compute_cost(self, y_true, y_pred):
        """
        计算逻辑回归代价函数
        
        参数:
        y_true: 真实标签
        y_pred: 预测概率
        
        返回:
        cost: 代价值
        """
        # 防止log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def predict_proba(self, X):
        """
        预测概率
        
        参数:
        X: 特征数据
        
        返回:
        probabilities: 预测概率
        """
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        z = np.dot(X_with_bias, self.weights)
        return self.sigmoid(z)
    
    def predict(self, X):
        """
        预测类别
        
        参数:
        X: 特征数据
        
        返回:
        predictions: 预测类别 (0或1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def plot_cost_history(self):
        """
        绘制代价函数变化
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('逻辑回归代价函数收敛过程')
        plt.xlabel('迭代次数')
        plt.ylabel('代价值')
        plt.grid(True, alpha=0.3)
        plt.show()

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