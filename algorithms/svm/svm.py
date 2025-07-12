"""
支持向量机（SVM）算法实现
包括线性SVM和核SVM
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

class SVM:
    """
    支持向量机分类器
    使用序列最小优化（SMO）算法
    """
    
    def __init__(self, C=1.0, kernel='linear', gamma=1.0, degree=3, coef0=0.0, tolerance=1e-3, max_iterations=1000):
        """
        初始化SVM
        
        参数:
        C: 正则化参数
        kernel: 核函数类型 ('linear', 'rbf', 'poly', 'sigmoid')
        gamma: RBF核参数
        degree: 多项式核次数
        coef0: 核函数常数项
        tolerance: 收敛容差
        max_iterations: 最大迭代次数
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
        # 模型参数
        self.alphas = None
        self.b = 0.0
        self.X = None
        self.y = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        
        # 核矩阵缓存
        self.kernel_cache = {}
        
    def _kernel_function(self, xi, xj):
        """
        核函数计算
        
        参数:
        xi, xj: 输入向量
        
        返回:
        核函数值
        """
        if self.kernel == 'linear':
            return np.dot(xi, xj)
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.sum((xi - xj) ** 2))
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(xi, xj) + self.coef0) ** self.degree
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(xi, xj) + self.coef0)
        else:
            raise ValueError(f"未知的核函数类型: {self.kernel}")
    
    def _compute_kernel_matrix(self, X1, X2=None):
        """
        计算核矩阵
        
        参数:
        X1: 第一组样本
        X2: 第二组样本，如果为None则使用X1
        
        返回:
        核矩阵
        """
        if X2 is None:
            X2 = X1
        
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel_function(X1[i], X2[j])
        
        return K
    
    def _compute_error(self, i):
        """
        计算第i个样本的预测误差
        
        参数:
        i: 样本索引
        
        返回:
        预测误差
        """
        prediction = self._predict_single(self.X[i])
        return prediction - self.y[i]
    
    def _predict_single(self, x):
        """
        预测单个样本
        
        参数:
        x: 输入样本
        
        返回:
        预测值
        """
        if self.alphas is None:
            return 0
        
        result = 0
        for i in range(len(self.X)):
            if self.alphas[i] > 0:
                result += self.alphas[i] * self.y[i] * self._kernel_function(self.X[i], x)
        
        return result + self.b
    
    def _select_j(self, i):
        """
        选择第二个优化变量
        使用启发式方法选择使|Ei - Ej|最大的j
        
        参数:
        i: 第一个变量索引
        
        返回:
        第二个变量索引
        """
        valid_alphas = [idx for idx in range(len(self.alphas)) 
                       if self.alphas[idx] > 0 and self.alphas[idx] < self.C and idx != i]
        
        if len(valid_alphas) > 1:
            Ei = self._compute_error(i)
            max_delta = 0
            j = -1
            
            for idx in valid_alphas:
                Ej = self._compute_error(idx)
                delta = abs(Ei - Ej)
                if delta > max_delta:
                    max_delta = delta
                    j = idx
            
            if j != -1:
                return j
        
        # 随机选择
        j = i
        while j == i:
            j = random.randint(0, len(self.alphas) - 1)
        return j
    
    def _clip_alpha(self, alpha, L, H):
        """
        裁剪alpha值到可行域
        
        参数:
        alpha: alpha值
        L: 下界
        H: 上界
        
        返回:
        裁剪后的alpha值
        """
        if alpha > H:
            return H
        elif alpha < L:
            return L
        else:
            return alpha
    
    def fit(self, X, y):
        """
        训练SVM模型
        
        参数:
        X: 训练特征
        y: 训练标签 (必须是-1或1)
        """
        # 数据预处理
        self.X = np.array(X)
        self.y = np.array(y)
        
        # 确保标签是-1和1
        unique_labels = np.unique(self.y)
        if len(unique_labels) != 2:
            raise ValueError("SVM只支持二分类问题")
        
        if not all(label in [-1, 1] for label in unique_labels):
            # 转换标签
            label_map = {unique_labels[0]: -1, unique_labels[1]: 1}
            self.y = np.array([label_map[label] for label in self.y])
            self.label_map = label_map
            self.inverse_label_map = {v: k for k, v in label_map.items()}
        
        n_samples = len(X)
        self.alphas = np.zeros(n_samples)
        self.b = 0.0
        
        # SMO算法主循环
        num_changed_alphas = 0
        examine_all = True
        iteration = 0
        
        while (num_changed_alphas > 0 or examine_all) and iteration < self.max_iterations:
            num_changed_alphas = 0
            
            if examine_all:
                # 检查所有样本
                for i in range(n_samples):
                    if self._examine_example(i):
                        num_changed_alphas += 1
            else:
                # 只检查边界样本
                for i in range(n_samples):
                    if 0 < self.alphas[i] < self.C:
                        if self._examine_example(i):
                            num_changed_alphas += 1
            
            if examine_all:
                examine_all = False
            elif num_changed_alphas == 0:
                examine_all = True
            
            iteration += 1
        
        # 提取支持向量
        support_vector_indices = np.where(self.alphas > 1e-5)[0]
        self.support_vectors = self.X[support_vector_indices]
        self.support_vector_labels = self.y[support_vector_indices]
        self.support_vector_alphas = self.alphas[support_vector_indices]
        
        print(f"训练完成，迭代次数: {iteration}")
        print(f"支持向量数量: {len(self.support_vectors)}")
        print(f"支持向量比例: {len(self.support_vectors)/n_samples:.2%}")
        
        return self
    
    def _examine_example(self, i):
        """
        检查样本i是否违反KKT条件
        
        参数:
        i: 样本索引
        
        返回:
        是否进行了优化
        """
        y_i = self.y[i]
        alpha_i = self.alphas[i]
        E_i = self._compute_error(i)
        r_i = E_i * y_i
        
        # 检查KKT条件
        if (r_i < -self.tolerance and alpha_i < self.C) or \
           (r_i > self.tolerance and alpha_i > 0):
            
            # 选择第二个变量
            j = self._select_j(i)
            
            return self._take_step(i, j)
        
        return False
    
    def _take_step(self, i, j):
        """
        优化alpha_i和alpha_j
        
        参数:
        i, j: 要优化的变量索引
        
        返回:
        是否进行了优化
        """
        if i == j:
            return False
        
        alpha_i_old = self.alphas[i]
        alpha_j_old = self.alphas[j]
        y_i, y_j = self.y[i], self.y[j]
        
        # 计算边界
        if y_i != y_j:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            L = max(0, alpha_i_old + alpha_j_old - self.C)
            H = min(self.C, alpha_i_old + alpha_j_old)
        
        if L == H:
            return False
        
        # 计算eta
        K_ii = self._kernel_function(self.X[i], self.X[i])
        K_jj = self._kernel_function(self.X[j], self.X[j])
        K_ij = self._kernel_function(self.X[i], self.X[j])
        eta = K_ii + K_jj - 2 * K_ij
        
        if eta <= 0:
            return False
        
        # 计算新的alpha_j
        E_i = self._compute_error(i)
        E_j = self._compute_error(j)
        alpha_j_new = alpha_j_old + y_j * (E_i - E_j) / eta
        alpha_j_new = self._clip_alpha(alpha_j_new, L, H)
        
        if abs(alpha_j_new - alpha_j_old) < 1e-5:
            return False
        
        # 计算新的alpha_i
        alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)
        
        # 更新阈值b
        b_i = self.b - E_i - y_i * (alpha_i_new - alpha_i_old) * K_ii - \
              y_j * (alpha_j_new - alpha_j_old) * K_ij
        b_j = self.b - E_j - y_i * (alpha_i_new - alpha_i_old) * K_ij - \
              y_j * (alpha_j_new - alpha_j_old) * K_jj
        
        if 0 < alpha_i_new < self.C:
            self.b = b_i
        elif 0 < alpha_j_new < self.C:
            self.b = b_j
        else:
            self.b = (b_i + b_j) / 2
        
        # 更新alphas
        self.alphas[i] = alpha_i_new
        self.alphas[j] = alpha_j_new
        
        return True
    
    def predict(self, X):
        """
        预测样本标签
        
        参数:
        X: 测试样本
        
        返回:
        预测标签
        """
        predictions = []
        
        for x in X:
            prediction = self._predict_single(x)
            label = 1 if prediction >= 0 else -1
            
            # 转换回原始标签
            if hasattr(self, 'inverse_label_map'):
                label = self.inverse_label_map[label]
            
            predictions.append(label)
        
        return np.array(predictions)
    
    def decision_function(self, X):
        """
        计算决策函数值
        
        参数:
        X: 测试样本
        
        返回:
        决策函数值
        """
        return np.array([self._predict_single(x) for x in X])
    
    def plot_decision_boundary(self, X, y, title="SVM决策边界"):
        """
        绘制决策边界（只适用于2D数据）
        
        参数:
        X: 特征数据
        y: 标签数据
        title: 图标题
        """
        if X.shape[1] != 2:
            print("只能绘制2D数据的决策边界")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 创建网格
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # 预测网格点
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.decision_function(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
        
        # 绘制决策边界线
        plt.contour(xx, yy, Z, levels=[0], colors='black', linestyles='--', linewidths=2)
        
        # 绘制数据点
        colors = ['red', 'blue']
        unique_labels = np.unique(y)
        for i, label in enumerate(unique_labels):
            mask = y == label
            plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                       label=f'类别 {label}', alpha=0.7, s=50)
        
        # 绘制支持向量
        if self.support_vectors is not None:
            plt.scatter(self.support_vectors[:, 0], self.support_vectors[:, 1], 
                       s=100, facecolors='none', edgecolors='black', linewidths=2,
                       label='支持向量')
        
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title(title)
        plt.legend()
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