"""
K-means聚类算法实现
"""
import numpy as np
import matplotlib.pyplot as plt
import random

class KMeans:
    """
    K-means聚类算法
    """
    
    def __init__(self, k=3, max_iterations=100, tolerance=1e-4, init_method='random'):
        """
        初始化K-means
        
        参数:
        k: 聚类数量
        max_iterations: 最大迭代次数
        tolerance: 收敛容差
        init_method: 初始化方法 ('random' 或 'kmeans++')
        """
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.init_method = init_method
        self.centroids = None
        self.labels = None
        self.inertia_history = []
        
    def fit(self, X):
        """
        训练K-means模型
        
        参数:
        X: 特征数据
        """
        n_samples, n_features = X.shape
        
        # 初始化聚类中心
        if self.init_method == 'kmeans++':
            self.centroids = self._kmeans_plus_plus_init(X)
        else:
            self.centroids = self._random_init(X)
        
        # 初始化标签
        self.labels = np.zeros(n_samples)
        
        for iteration in range(self.max_iterations):
            # 分配样本到最近的聚类中心
            old_labels = self.labels.copy()
            self.labels = self._assign_clusters(X)
            
            # 更新聚类中心
            old_centroids = self.centroids.copy()
            self._update_centroids(X)
            
            # 计算惯性（WCSS）
            inertia = self._calculate_inertia(X)
            self.inertia_history.append(inertia)
            
            # 检查收敛
            centroid_shift = np.sum(np.abs(self.centroids - old_centroids))
            if centroid_shift < self.tolerance:
                print(f"在第 {iteration + 1} 次迭代后收敛")
                break
        
        return self
    
    def _random_init(self, X):
        """
        随机初始化聚类中心
        
        参数:
        X: 特征数据
        
        返回:
        centroids: 初始聚类中心
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))
        
        for i in range(n_features):
            min_val = X[:, i].min()
            max_val = X[:, i].max()
            centroids[:, i] = np.random.uniform(min_val, max_val, self.k)
        
        return centroids
    
    def _kmeans_plus_plus_init(self, X):
        """
        K-means++初始化
        
        参数:
        X: 特征数据
        
        返回:
        centroids: 初始聚类中心
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))
        
        # 随机选择第一个中心
        centroids[0] = X[np.random.randint(n_samples)]
        
        # 选择其余的中心
        for i in range(1, self.k):
            # 计算每个点到最近中心的距离
            distances = np.array([min([np.sum((x - c)**2) for c in centroids[:i]]) for x in X])
            
            # 基于距离的概率选择下一个中心
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids[i] = X[j]
                    break
        
        return centroids
    
    def _assign_clusters(self, X):
        """
        分配样本到最近的聚类中心
        
        参数:
        X: 特征数据
        
        返回:
        labels: 聚类标签
        """
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X):
        """
        更新聚类中心
        
        参数:
        X: 特征数据
        """
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                self.centroids[i] = cluster_points.mean(axis=0)
    
    def _calculate_inertia(self, X):
        """
        计算惯性（簇内平方和）
        
        参数:
        X: 特征数据
        
        返回:
        inertia: 惯性值
        """
        inertia = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i])**2)
        return inertia
    
    def predict(self, X):
        """
        预测新数据的聚类标签
        
        参数:
        X: 特征数据
        
        返回:
        labels: 聚类标签
        """
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def plot_inertia_history(self):
        """
        绘制惯性变化历史
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.inertia_history, 'bo-')
        plt.title('K-means收敛过程')
        plt.xlabel('迭代次数')
        plt.ylabel('惯性 (WCSS)')
        plt.grid(True, alpha=0.3)
        plt.show()

def elbow_method(X, k_range=(1, 11)):
    """
    肘部法则选择最佳K值
    
    参数:
    X: 特征数据
    k_range: K值范围
    
    返回:
    inertias: 不同K值对应的惯性
    """
    inertias = []
    k_values = range(k_range[0], k_range[1])
    
    for k in k_values:
        kmeans = KMeans(k=k, max_iterations=100)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_history[-1])
    
    # 绘制肘部图
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, 'bo-')
    plt.title('肘部法则选择最佳K值')
    plt.xlabel('K值')
    plt.ylabel('惯性 (WCSS)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return k_values, inertias

def silhouette_score(X, labels):
    """
    计算轮廓系数
    
    参数:
    X: 特征数据
    labels: 聚类标签
    
    返回:
    score: 轮廓系数
    """
    n_samples = len(X)
    silhouettes = []
    
    for i in range(n_samples):
        # 计算簇内距离
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) > 1:
            a = np.mean([np.sum((X[i] - point)**2)**0.5 for point in same_cluster if not np.array_equal(X[i], point)])
        else:
            a = 0
        
        # 计算到其他簇的最小距离
        b = float('inf')
        for cluster_id in np.unique(labels):
            if cluster_id != labels[i]:
                other_cluster = X[labels == cluster_id]
                if len(other_cluster) > 0:
                    avg_dist = np.mean([np.sum((X[i] - point)**2)**0.5 for point in other_cluster])
                    b = min(b, avg_dist)
        
        # 计算轮廓系数
        if max(a, b) > 0:
            silhouettes.append((b - a) / max(a, b))
        else:
            silhouettes.append(0)
    
    return np.mean(silhouettes) 