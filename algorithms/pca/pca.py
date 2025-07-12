"""
主成分分析（PCA）算法实现
"""
import numpy as np
import matplotlib.pyplot as plt

class PCA:
    """
    主成分分析算法
    """
    
    def __init__(self, n_components=None, whiten=False):
        """
        初始化PCA
        
        参数:
        n_components: 主成分数量，如果为None则保留所有成分
        whiten: 是否白化数据
        """
        self.n_components = n_components
        self.whiten = whiten
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_features_ = None
        
    def fit(self, X):
        """
        拟合PCA模型
        
        参数:
        X: 输入数据
        """
        # 中心化数据
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        self.n_features_ = X.shape[1]
        
        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered.T)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 存储主成分
        if self.n_components is None:
            self.n_components = len(eigenvalues)
        
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        """
        将数据转换到主成分空间
        
        参数:
        X: 输入数据
        
        返回:
        X_transformed: 转换后的数据
        """
        # 中心化
        X_centered = X - self.mean_
        
        # 投影到主成分空间
        X_transformed = np.dot(X_centered, self.components_.T)
        
        # 白化处理
        if self.whiten:
            X_transformed = X_transformed / np.sqrt(self.explained_variance_)
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        拟合模型并转换数据
        
        参数:
        X: 输入数据
        
        返回:
        X_transformed: 转换后的数据
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        将主成分空间的数据转换回原始空间
        
        参数:
        X_transformed: 主成分空间的数据
        
        返回:
        X_reconstructed: 重构的原始数据
        """
        # 逆白化
        if self.whiten:
            X_transformed = X_transformed * np.sqrt(self.explained_variance_)
        
        # 投影回原始空间
        X_reconstructed = np.dot(X_transformed, self.components_) + self.mean_
        
        return X_reconstructed
    
    def explained_variance_cumsum(self):
        """
        计算累积解释方差比
        
        返回:
        cumsum: 累积解释方差比
        """
        return np.cumsum(self.explained_variance_ratio_)
    
    def plot_explained_variance(self):
        """
        绘制解释方差图
        """
        plt.figure(figsize=(12, 5))
        
        # 子图1：各主成分的解释方差比
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(self.explained_variance_ratio_) + 1), 
                self.explained_variance_ratio_, alpha=0.7)
        plt.xlabel('主成分')
        plt.ylabel('解释方差比')
        plt.title('各主成分的解释方差比')
        plt.grid(True, alpha=0.3)
        
        # 子图2：累积解释方差比
        plt.subplot(1, 2, 2)
        cumsum = self.explained_variance_cumsum()
        plt.plot(range(1, len(cumsum) + 1), cumsum, 'ro-')
        plt.axhline(y=0.95, color='red', linestyle='--', label='95%')
        plt.axhline(y=0.90, color='orange', linestyle='--', label='90%')
        plt.xlabel('主成分数量')
        plt.ylabel('累积解释方差比')
        plt.title('累积解释方差比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_components_for_variance(self, variance_threshold=0.95):
        """
        获取达到指定方差解释比例所需的主成分数量
        
        参数:
        variance_threshold: 方差解释比例阈值
        
        返回:
        n_components: 主成分数量
        """
        cumsum = self.explained_variance_cumsum()
        n_components = np.argmax(cumsum >= variance_threshold) + 1
        return n_components

def plot_2d_pca(X_original, X_pca, labels=None, title="PCA降维结果"):
    """
    绘制2D PCA结果
    
    参数:
    X_original: 原始数据
    X_pca: PCA降维后的数据
    labels: 数据标签
    title: 图表标题
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始数据（如果是高维，只显示前两维）
    if X_original.shape[1] >= 2:
        if labels is not None:
            scatter1 = ax1.scatter(X_original[:, 0], X_original[:, 1], c=labels, cmap='viridis')
            plt.colorbar(scatter1, ax=ax1)
        else:
            ax1.scatter(X_original[:, 0], X_original[:, 1], alpha=0.6)
        ax1.set_xlabel('原始特征1')
        ax1.set_ylabel('原始特征2')
        ax1.set_title('原始数据')
    
    # PCA降维后的数据
    if labels is not None:
        scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
        plt.colorbar(scatter2, ax=ax2)
    else:
        ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    ax2.set_xlabel('第一主成分')
    ax2.set_ylabel('第二主成分')
    ax2.set_title('PCA降维后数据')
    
    plt.tight_layout()
    plt.show()

def reconstruction_error(X_original, X_reconstructed):
    """
    计算重构误差
    
    参数:
    X_original: 原始数据
    X_reconstructed: 重构数据
    
    返回:
    error: 重构误差
    """
    return np.mean(np.sum((X_original - X_reconstructed) ** 2, axis=1)) 