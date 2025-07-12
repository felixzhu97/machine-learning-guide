"""
约会网站推荐系统案例
使用K-近邻算法根据用户特征预测喜好程度
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from knn import KNN, normalize_data, calculate_accuracy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.visualization import plot_2d_scatter, plot_decision_boundary

class DatingRecommendationSystem:
    """
    约会网站推荐系统
    """
    
    def __init__(self):
        self.knn = None
        self.feature_names = ['每年获得的飞行常客里程数', '玩视频游戏所耗时间百分比', '每周消费的冰淇淋公升数']
        self.class_names = ['不喜欢', '魅力一般', '极具魅力']
        self.data = None
        self.labels = None
        
    def create_sample_data(self, n_samples=1000):
        """
        创建样本数据
        
        参数:
        n_samples: 样本数量
        """
        np.random.seed(42)
        
        # 生成三个特征
        # 特征1：每年获得的飞行常客里程数 (0-100000)
        feature1 = np.random.uniform(0, 100000, n_samples)
        
        # 特征2：玩视频游戏所耗时间百分比 (0-100)
        feature2 = np.random.uniform(0, 100, n_samples)
        
        # 特征3：每周消费的冰淇淋公升数 (0-10)
        feature3 = np.random.uniform(0, 10, n_samples)
        
        # 组合特征
        self.data = np.column_stack([feature1, feature2, feature3])
        
        # 根据规则生成标签
        self.labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # 规则：高里程数 + 低游戏时间 + 适中冰淇淋消费 = 极具魅力
            if feature1[i] > 50000 and feature2[i] < 30 and 2 < feature3[i] < 6:
                self.labels[i] = 2  # 极具魅力
            # 规则：中等里程数 + 中等游戏时间 = 魅力一般
            elif 20000 < feature1[i] < 60000 and 20 < feature2[i] < 70:
                self.labels[i] = 1  # 魅力一般
            # 其他情况：不喜欢
            else:
                self.labels[i] = 0  # 不喜欢
                
        print(f"生成了 {n_samples} 个样本")
        print(f"标签分布: {np.bincount(self.labels)}")
    
    def load_data_from_file(self, file_path):
        """
        从文件加载数据
        
        参数:
        file_path: 文件路径
        """
        try:
            data = pd.read_csv(file_path, delimiter='\t', header=None)
            self.data = data.iloc[:, :-1].values
            self.labels = data.iloc[:, -1].values
            print(f"成功加载数据: {self.data.shape}")
        except Exception as e:
            print(f"加载数据失败: {e}")
            self.create_sample_data()
    
    def preprocess_data(self):
        """
        数据预处理
        """
        # 归一化数据
        self.data, self.ranges, self.min_vals = normalize_data(self.data)
        print("数据归一化完成")
    
    def train_model(self, k=5, test_size=0.2):
        """
        训练模型
        
        参数:
        k: 邻居数量
        test_size: 测试集比例
        """
        # 拆分数据
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=42
        )
        
        # 创建并训练kNN模型
        self.knn = KNN(k=k)
        self.knn.fit(X_train, y_train)
        
        # 预测
        y_pred = self.knn.predict(X_test)
        
        # 计算准确率
        accuracy = calculate_accuracy(y_test, y_pred)
        
        print(f"K = {k}, 测试集准确率: {accuracy:.4f}")
        
        return X_train, X_test, y_train, y_test, y_pred, accuracy
    
    def find_best_k(self, k_range=(1, 21)):
        """
        寻找最佳K值
        
        参数:
        k_range: K值范围
        """
        accuracies = []
        k_values = range(k_range[0], k_range[1])
        
        # 拆分数据
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.labels, test_size=0.2, random_state=42
        )
        
        for k in k_values:
            knn = KNN(k=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = calculate_accuracy(y_test, y_pred)
            accuracies.append(accuracy)
            print(f"K = {k:2d}, 准确率: {accuracy:.4f}")
        
        # 找到最佳K值
        best_k = k_values[np.argmax(accuracies)]
        best_accuracy = max(accuracies)
        
        print(f"\n最佳K值: {best_k}, 最佳准确率: {best_accuracy:.4f}")
        
        # 绘制K值与准确率的关系
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracies, 'bo-')
        plt.axvline(x=best_k, color='red', linestyle='--', label=f'最佳K={best_k}')
        plt.xlabel('K值')
        plt.ylabel('准确率')
        plt.title('K值选择：准确率随K值变化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return best_k, best_accuracy
    
    def predict_person(self, features):
        """
        预测单个用户
        
        参数:
        features: 用户特征 [里程数, 游戏时间, 冰淇淋消费]
        
        返回:
        prediction: 预测结果
        probability: 预测概率
        """
        if self.knn is None:
            print("请先训练模型")
            return None, None
        
        # 归一化特征
        normalized_features = (np.array(features) - self.min_vals) / self.ranges
        
        # 预测
        prediction = self.knn.predict([normalized_features])[0]
        probabilities = self.knn.predict_proba([normalized_features])[0]
        
        print(f"用户特征: {features}")
        print(f"预测结果: {self.class_names[prediction]}")
        print(f"各类别概率: {probabilities}")
        
        return prediction, probabilities
    
    def visualize_data(self):
        """
        可视化数据
        """
        if self.data is None:
            print("请先加载数据")
            return
        
        # 绘制特征分布
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 特征1 vs 特征2
        axes[0, 0].scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis')
        axes[0, 0].set_xlabel(self.feature_names[0])
        axes[0, 0].set_ylabel(self.feature_names[1])
        axes[0, 0].set_title('特征1 vs 特征2')
        
        # 特征1 vs 特征3
        axes[0, 1].scatter(self.data[:, 0], self.data[:, 2], c=self.labels, cmap='viridis')
        axes[0, 1].set_xlabel(self.feature_names[0])
        axes[0, 1].set_ylabel(self.feature_names[2])
        axes[0, 1].set_title('特征1 vs 特征3')
        
        # 特征2 vs 特征3
        axes[1, 0].scatter(self.data[:, 1], self.data[:, 2], c=self.labels, cmap='viridis')
        axes[1, 0].set_xlabel(self.feature_names[1])
        axes[1, 0].set_ylabel(self.feature_names[2])
        axes[1, 0].set_title('特征2 vs 特征3')
        
        # 类别分布
        axes[1, 1].hist(self.labels, bins=3, alpha=0.7, color='skyblue')
        axes[1, 1].set_xlabel('类别')
        axes[1, 1].set_ylabel('数量')
        axes[1, 1].set_title('类别分布')
        axes[1, 1].set_xticks([0, 1, 2])
        axes[1, 1].set_xticklabels(self.class_names)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    主函数
    """
    print("=== 约会网站推荐系统 ===")
    
    # 创建推荐系统
    dating_system = DatingRecommendationSystem()
    
    # 创建样本数据
    dating_system.create_sample_data(1000)
    
    # 可视化数据
    dating_system.visualize_data()
    
    # 数据预处理
    dating_system.preprocess_data()
    
    # 寻找最佳K值
    print("\n=== 寻找最佳K值 ===")
    best_k, best_accuracy = dating_system.find_best_k()
    
    # 使用最佳K值训练模型
    print(f"\n=== 使用最佳K值 {best_k} 训练模型 ===")
    X_train, X_test, y_train, y_test, y_pred, accuracy = dating_system.train_model(k=best_k)
    
    # 预测几个示例用户
    print("\n=== 预测示例用户 ===")
    
    # 用户1：高里程数，低游戏时间，适中冰淇淋消费
    user1_features = [75000, 20, 4]
    dating_system.predict_person(user1_features)
    
    print()
    
    # 用户2：低里程数，高游戏时间，高冰淇淋消费
    user2_features = [10000, 80, 8]
    dating_system.predict_person(user2_features)
    
    print()
    
    # 用户3：中等里程数，中等游戏时间，中等冰淇淋消费
    user3_features = [40000, 50, 3]
    dating_system.predict_person(user3_features)

if __name__ == "__main__":
    main() 