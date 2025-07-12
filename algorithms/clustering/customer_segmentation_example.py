"""
客户细分案例
使用K-means聚类算法进行客户群体分析
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from kmeans import KMeans, elbow_method, silhouette_score
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.visualization import plot_clustering_result

class CustomerSegmentationSystem:
    """
    客户细分系统
    """
    
    def __init__(self):
        self.kmeans = None
        self.scaler = None
        self.feature_names = ['年收入', '年消费', '年龄', '家庭成员数', '消费频率', '客户满意度']
        self.X = None
        self.customer_data = None
        
    def create_sample_data(self, n_samples=1000):
        """
        创建客户数据样本
        
        参数:
        n_samples: 样本数量
        """
        np.random.seed(42)
        
        # 定义客户群体
        # 群体1：高收入高消费客户 (25%)
        n1 = int(n_samples * 0.25)
        income1 = np.random.normal(80000, 15000, n1)
        spending1 = np.random.normal(60000, 10000, n1)
        age1 = np.random.normal(40, 8, n1)
        family1 = np.random.poisson(2, n1) + 1
        frequency1 = np.random.normal(25, 5, n1)
        satisfaction1 = np.random.normal(4.2, 0.5, n1)
        
        # 群体2：中等收入中等消费客户 (40%)
        n2 = int(n_samples * 0.4)
        income2 = np.random.normal(50000, 10000, n2)
        spending2 = np.random.normal(30000, 8000, n2)
        age2 = np.random.normal(35, 10, n2)
        family2 = np.random.poisson(1.5, n2) + 1
        frequency2 = np.random.normal(15, 4, n2)
        satisfaction2 = np.random.normal(3.5, 0.6, n2)
        
        # 群体3：低收入低消费客户 (25%)
        n3 = int(n_samples * 0.25)
        income3 = np.random.normal(30000, 8000, n3)
        spending3 = np.random.normal(15000, 5000, n3)
        age3 = np.random.normal(28, 8, n3)
        family3 = np.random.poisson(1, n3) + 1
        frequency3 = np.random.normal(8, 3, n3)
        satisfaction3 = np.random.normal(3.0, 0.7, n3)
        
        # 群体4：年轻高消费客户 (10%)
        n4 = n_samples - n1 - n2 - n3
        income4 = np.random.normal(45000, 8000, n4)
        spending4 = np.random.normal(40000, 6000, n4)
        age4 = np.random.normal(25, 3, n4)
        family4 = np.random.poisson(0.5, n4) + 1
        frequency4 = np.random.normal(20, 4, n4)
        satisfaction4 = np.random.normal(3.8, 0.4, n4)
        
        # 组合所有特征
        self.X = np.column_stack([
            np.concatenate([income1, income2, income3, income4]),
            np.concatenate([spending1, spending2, spending3, spending4]),
            np.concatenate([age1, age2, age3, age4]),
            np.concatenate([family1, family2, family3, family4]),
            np.concatenate([frequency1, frequency2, frequency3, frequency4]),
            np.concatenate([satisfaction1, satisfaction2, satisfaction3, satisfaction4])
        ])
        
        # 确保所有值都是正数且合理
        self.X[:, 0] = np.clip(self.X[:, 0], 20000, 150000)  # 年收入
        self.X[:, 1] = np.clip(self.X[:, 1], 5000, 100000)   # 年消费
        self.X[:, 2] = np.clip(self.X[:, 2], 18, 70)         # 年龄
        self.X[:, 3] = np.clip(self.X[:, 3], 1, 8)           # 家庭成员数
        self.X[:, 4] = np.clip(self.X[:, 4], 1, 50)          # 消费频率
        self.X[:, 5] = np.clip(self.X[:, 5], 1, 5)           # 客户满意度
        
        # 创建真实标签（用于验证）
        self.true_labels = np.concatenate([
            np.zeros(n1),      # 高收入高消费
            np.ones(n2),       # 中等收入中等消费
            np.full(n3, 2),    # 低收入低消费
            np.full(n4, 3)     # 年轻高消费
        ])
        
        # 打乱数据
        indices = np.random.permutation(n_samples)
        self.X = self.X[indices]
        self.true_labels = self.true_labels[indices]
        
        # 创建DataFrame便于分析
        self.customer_data = pd.DataFrame(self.X, columns=self.feature_names)
        
        print(f"生成了 {n_samples} 个客户样本")
        print(f"特征: {self.feature_names}")
        print(f"数据范围:")
        for i, feature in enumerate(self.feature_names):
            print(f"  {feature}: {self.X[:, i].min():.1f} - {self.X[:, i].max():.1f}")
        
        return self.X, self.customer_data
    
    def preprocess_data(self):
        """
        数据预处理
        """
        # 标准化数据
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        print("数据标准化完成")
        
        return self.X_scaled
    
    def find_optimal_k(self, k_range=(2, 11)):
        """
        寻找最佳K值
        
        参数:
        k_range: K值范围
        """
        print("=== 使用肘部法则寻找最佳K值 ===")
        k_values, inertias = elbow_method(self.X_scaled, k_range)
        
        # 计算不同K值的轮廓系数
        print("\n=== 计算轮廓系数 ===")
        silhouette_scores = []
        
        for k in range(2, k_range[1]):
            kmeans = KMeans(k=k, init_method='kmeans++')
            kmeans.fit(self.X_scaled)
            score = silhouette_score(self.X_scaled, kmeans.labels)
            silhouette_scores.append(score)
            print(f"K = {k}: 轮廓系数 = {score:.4f}")
        
        # 绘制轮廓系数图
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, k_range[1]), silhouette_scores, 'ro-')
        plt.title('轮廓系数评估最佳K值')
        plt.xlabel('K值')
        plt.ylabel('轮廓系数')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 推荐最佳K值
        best_k = range(2, k_range[1])[np.argmax(silhouette_scores)]
        print(f"\n推荐的最佳K值: {best_k}")
        
        return best_k
    
    def train_model(self, k=4, init_method='kmeans++'):
        """
        训练K-means模型
        
        参数:
        k: 聚类数量
        init_method: 初始化方法
        """
        self.kmeans = KMeans(k=k, init_method=init_method, max_iterations=100)
        self.kmeans.fit(self.X_scaled)
        
        # 计算评估指标
        inertia = self.kmeans.inertia_history[-1]
        silhouette = silhouette_score(self.X_scaled, self.kmeans.labels)
        
        print(f"K-means聚类完成")
        print(f"聚类数量: {k}")
        print(f"最终惯性: {inertia:.2f}")
        print(f"轮廓系数: {silhouette:.4f}")
        
        # 绘制收敛过程
        self.kmeans.plot_inertia_history()
        
        return self.kmeans
    
    def analyze_clusters(self):
        """
        分析聚类结果
        """
        if self.kmeans is None:
            print("请先训练模型")
            return
        
        # 将聚类标签添加到数据中
        self.customer_data['cluster'] = self.kmeans.labels
        
        print("\n=== 聚类结果分析 ===")
        
        # 每个聚类的统计信息
        cluster_stats = self.customer_data.groupby('cluster').agg({
            '年收入': ['mean', 'std', 'count'],
            '年消费': ['mean', 'std'],
            '年龄': ['mean', 'std'],
            '家庭成员数': ['mean', 'std'],
            '消费频率': ['mean', 'std'],
            '客户满意度': ['mean', 'std']
        }).round(2)
        
        print(cluster_stats)
        
        # 可视化聚类结果
        self._visualize_clusters()
        
        # 聚类特征描述
        self._describe_clusters()
        
        return cluster_stats
    
    def _visualize_clusters(self):
        """
        可视化聚类结果
        """
        # 选择两个主要特征进行2D可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        feature_pairs = [
            (0, 1, '年收入', '年消费'),
            (0, 2, '年收入', '年龄'),
            (1, 2, '年消费', '年龄'),
            (1, 4, '年消费', '消费频率'),
            (2, 5, '年龄', '客户满意度'),
            (3, 4, '家庭成员数', '消费频率')
        ]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for idx, (i, j, xlabel, ylabel) in enumerate(feature_pairs):
            ax = axes[idx // 3, idx % 3]
            
            for cluster_id in range(self.kmeans.k):
                cluster_mask = self.kmeans.labels == cluster_id
                ax.scatter(self.X[cluster_mask, i], self.X[cluster_mask, j], 
                          c=colors[cluster_id], label=f'聚类{cluster_id}', alpha=0.6)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{xlabel} vs {ylabel}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _describe_clusters(self):
        """
        描述各个聚类的特征
        """
        print("\n=== 聚类特征描述 ===")
        
        cluster_descriptions = {
            0: "高价值客户",
            1: "标准客户", 
            2: "价格敏感客户",
            3: "潜力客户"
        }
        
        for cluster_id in range(self.kmeans.k):
            cluster_data = self.customer_data[self.customer_data['cluster'] == cluster_id]
            n_customers = len(cluster_data)
            
            avg_income = cluster_data['年收入'].mean()
            avg_spending = cluster_data['年消费'].mean()
            avg_age = cluster_data['年龄'].mean()
            avg_satisfaction = cluster_data['客户满意度'].mean()
            
            print(f"\n聚类 {cluster_id} ({n_customers} 个客户):")
            print(f"  平均年收入: ¥{avg_income:,.0f}")
            print(f"  平均年消费: ¥{avg_spending:,.0f}")
            print(f"  平均年龄: {avg_age:.1f} 岁")
            print(f"  平均满意度: {avg_satisfaction:.2f}/5.0")
            
            # 客户群体特征判断
            if avg_income > 60000 and avg_spending > 40000:
                description = "高价值客户群 - 高收入高消费，是公司的核心客户"
            elif avg_age < 30 and avg_spending > 30000:
                description = "年轻潜力客户群 - 年轻且消费能力强，未来价值高"
            elif avg_income < 40000:
                description = "价格敏感客户群 - 收入较低，对价格敏感"
            else:
                description = "标准客户群 - 中等收入和消费水平"
            
            print(f"  群体特征: {description}")
    
    def predict_customer_cluster(self, income, spending, age, family_size, frequency, satisfaction):
        """
        预测新客户的聚类
        
        参数:
        income: 年收入
        spending: 年消费
        age: 年龄
        family_size: 家庭成员数
        frequency: 消费频率
        satisfaction: 客户满意度
        
        返回:
        cluster: 聚类标签
        """
        if self.kmeans is None:
            print("请先训练模型")
            return None
        
        # 准备特征
        features = np.array([[income, spending, age, family_size, frequency, satisfaction]])
        features_scaled = self.scaler.transform(features)
        
        # 预测
        cluster = self.kmeans.predict(features_scaled)[0]
        
        print(f"客户特征:")
        print(f"  年收入: ¥{income:,}")
        print(f"  年消费: ¥{spending:,}")
        print(f"  年龄: {age} 岁")
        print(f"  家庭成员数: {family_size}")
        print(f"  消费频率: {frequency} 次/年")
        print(f"  客户满意度: {satisfaction}/5.0")
        print(f"预测聚类: {cluster}")
        
        return cluster
    
    def marketing_recommendations(self):
        """
        基于聚类结果提供营销建议
        """
        if self.kmeans is None:
            print("请先训练模型")
            return
        
        print("\n=== 营销策略建议 ===")
        
        for cluster_id in range(self.kmeans.k):
            cluster_data = self.customer_data[self.customer_data['cluster'] == cluster_id]
            
            avg_income = cluster_data['年收入'].mean()
            avg_spending = cluster_data['年消费'].mean()
            avg_satisfaction = cluster_data['客户满意度'].mean()
            
            print(f"\n聚类 {cluster_id} 营销策略:")
            
            if avg_income > 60000 and avg_spending > 40000:
                print("  - 提供高端产品和VIP服务")
                print("  - 个性化推荐和专属优惠")
                print("  - 重点维护客户关系")
            elif avg_spending > 30000 and cluster_data['年龄'].mean() < 35:
                print("  - 推广时尚和科技产品")
                print("  - 社交媒体营销")
                print("  - 会员成长计划")
            elif avg_income < 40000:
                print("  - 强调性价比和优惠活动")
                print("  - 分期付款和促销")
                print("  - 基础产品推荐")
            else:
                print("  - 平衡的产品组合")
                print("  - 定期优惠和会员福利")
                print("  - 提升客户满意度")

def main():
    """
    主函数
    """
    print("=== 客户细分分析系统 ===")
    
    # 创建客户细分系统
    customer_system = CustomerSegmentationSystem()
    
    # 生成样本数据
    X, customer_data = customer_system.create_sample_data(1000)
    
    # 数据预处理
    X_scaled = customer_system.preprocess_data()
    
    # 寻找最佳K值
    best_k = customer_system.find_optimal_k()
    
    # 训练K-means模型
    print(f"\n=== 使用K={best_k}训练K-means模型 ===")
    kmeans_model = customer_system.train_model(k=best_k)
    
    # 分析聚类结果
    cluster_stats = customer_system.analyze_clusters()
    
    # 营销策略建议
    customer_system.marketing_recommendations()
    
    # 预测新客户
    print("\n=== 新客户聚类预测 ===")
    
    # 示例客户
    test_customers = [
        (75000, 55000, 42, 3, 24, 4.5),  # 高收入高消费客户
        (35000, 18000, 28, 2, 10, 3.2),  # 低收入低消费客户
        (48000, 35000, 26, 1, 18, 3.8),  # 年轻潜力客户
        (52000, 28000, 38, 4, 12, 3.5),  # 标准客户
    ]
    
    descriptions = [
        "高管客户",
        "学生客户", 
        "白领客户",
        "家庭客户"
    ]
    
    for customer, desc in zip(test_customers, descriptions):
        print(f"\n{desc}:")
        customer_system.predict_customer_cluster(*customer)

if __name__ == "__main__":
    main() 