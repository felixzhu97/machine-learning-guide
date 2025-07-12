"""
弱分类器组合案例
使用AdaBoost将多个弱分类器组合成强分类器
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from adaboost import AdaBoost, calculate_accuracy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.data_utils import plot_confusion_matrix

class WeakClassifierEnsemble:
    """
    弱分类器集成系统
    """
    
    def __init__(self):
        self.adaboost = None
        self.scaler = None
        self.X = None
        self.y = None
        
    def create_synthetic_dataset(self, dataset_type='classification', n_samples=1000):
        """
        创建合成数据集
        
        参数:
        dataset_type: 数据集类型 ('classification', 'moons', 'circles')
        n_samples: 样本数量
        """
        np.random.seed(42)
        
        if dataset_type == 'classification':
            self.X, self.y = make_classification(
                n_samples=n_samples,
                n_features=2,
                n_redundant=0,
                n_informative=2,
                n_clusters_per_class=1,
                random_state=42
            )
            dataset_name = "线性可分数据集"
            
        elif dataset_type == 'moons':
            self.X, self.y = make_moons(
                n_samples=n_samples,
                noise=0.3,
                random_state=42
            )
            dataset_name = "月牙形数据集"
            
        elif dataset_type == 'circles':
            self.X, self.y = make_circles(
                n_samples=n_samples,
                noise=0.2,
                factor=0.5,
                random_state=42
            )
            dataset_name = "同心圆数据集"
            
        else:
            raise ValueError("不支持的数据集类型")
        
        # 转换标签为-1和1
        self.y = np.where(self.y == 0, -1, 1)
        
        print(f"=== {dataset_name} ===")
        print(f"数据形状: {self.X.shape}")
        print(f"类别分布: {np.bincount(self.y == 1)}")
        
        return self.X, self.y
    
    def visualize_dataset(self):
        """
        可视化数据集
        """
        plt.figure(figsize=(10, 8))
        
        # 绘制数据点
        colors = ['red', 'blue']
        labels = ['类别 -1', '类别 +1']
        
        for i, label_val in enumerate([-1, 1]):
            mask = self.y == label_val
            plt.scatter(self.X[mask, 0], self.X[mask, 1], 
                       c=colors[i], label=labels[i], alpha=0.7, s=50)
        
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('数据集可视化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def preprocess_data(self, test_size=0.2, standardize=True):
        """
        数据预处理
        
        参数:
        test_size: 测试集比例
        standardize: 是否标准化
        """
        # 分割数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        if standardize:
            # 标准化特征
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
        else:
            self.X_train_scaled = self.X_train
            self.X_test_scaled = self.X_test
        
        print(f"训练集大小: {self.X_train_scaled.shape}")
        print(f"测试集大小: {self.X_test_scaled.shape}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_adaboost(self, n_estimators=50):
        """
        训练AdaBoost模型
        
        参数:
        n_estimators: 弱分类器数量
        """
        print(f"\n=== 训练AdaBoost (使用{n_estimators}个弱分类器) ===")
        
        self.adaboost = AdaBoost(n_estimators=n_estimators)
        self.adaboost.fit(self.X_train_scaled, self.y_train)
        
        # 评估性能
        y_pred_train = self.adaboost.predict(self.X_train_scaled)
        y_pred_test = self.adaboost.predict(self.X_test_scaled)
        
        train_accuracy = calculate_accuracy(self.y_train, y_pred_train)
        test_accuracy = calculate_accuracy(self.y_test, y_pred_test)
        
        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")
        
        return self.adaboost
    
    def analyze_weak_classifiers(self):
        """
        分析弱分类器
        """
        if self.adaboost is None:
            print("请先训练模型")
            return
        
        print(f"\n=== 弱分类器分析 ===")
        print(f"总共使用了 {len(self.adaboost.estimators)} 个弱分类器")
        
        # 显示前10个分类器的详细信息
        for i, (estimator, weight, error) in enumerate(zip(
            self.adaboost.estimators[:10], 
            self.adaboost.estimator_weights[:10],
            self.adaboost.estimator_errors[:10]
        )):
            print(f"分类器 {i+1}:")
            print(f"  特征索引: {estimator.feature_index}")
            print(f"  阈值: {estimator.threshold:.4f}")
            print(f"  分裂方式: {estimator.inequality}")
            print(f"  错误率: {error:.4f}")
            print(f"  权重: {weight:.4f}")
            print()
        
        # 绘制特征重要性
        self.adaboost.plot_feature_importance()
    
    def plot_decision_boundary(self, title="AdaBoost决策边界"):
        """
        绘制决策边界
        
        参数:
        title: 图标题
        """
        if self.adaboost is None:
            print("请先训练模型")
            return
        
        plt.figure(figsize=(15, 5))
        
        # 创建网格
        h = 0.02
        x_min, x_max = self.X_train_scaled[:, 0].min() - 1, self.X_train_scaled[:, 0].max() + 1
        y_min, y_max = self.X_train_scaled[:, 1].min() - 1, self.X_train_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # 预测网格点
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.adaboost.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 子图1：决策边界
        plt.subplot(1, 3, 1)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')
        
        # 绘制训练数据
        colors = ['red', 'blue']
        for i, label_val in enumerate([-1, 1]):
            mask = self.y_train == label_val
            plt.scatter(self.X_train_scaled[mask, 0], self.X_train_scaled[mask, 1], 
                       c=colors[i], alpha=0.7, s=30, edgecolors='black')
        
        plt.title('训练集决策边界')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        
        # 子图2：概率预测
        plt.subplot(1, 3, 2)
        Z_proba = self.adaboost.predict_proba(grid_points)
        Z_proba = Z_proba.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z_proba, levels=20, alpha=0.8, cmap='viridis')
        plt.colorbar(label='预测概率')
        
        # 绘制测试数据
        for i, label_val in enumerate([-1, 1]):
            mask = self.y_test == label_val
            plt.scatter(self.X_test_scaled[mask, 0], self.X_test_scaled[mask, 1], 
                       c=colors[i], alpha=0.7, s=30, edgecolors='black')
        
        plt.title('测试集概率预测')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        
        # 子图3：分类器权重分布
        plt.subplot(1, 3, 3)
        weights = self.adaboost.estimator_weights
        plt.hist(weights, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('分类器权重')
        plt.ylabel('频率')
        plt.title('分类器权重分布')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_n_estimators(self, n_estimators_list=[1, 5, 10, 20, 50, 100]):
        """
        比较不同弱分类器数量的效果
        
        参数:
        n_estimators_list: 要比较的分类器数量列表
        """
        print(f"\n=== 比较不同弱分类器数量的效果 ===")
        
        train_accuracies = []
        test_accuracies = []
        
        for n_estimators in n_estimators_list:
            print(f"训练 {n_estimators} 个弱分类器...")
            
            # 训练模型
            ada = AdaBoost(n_estimators=n_estimators)
            ada.fit(self.X_train_scaled, self.y_train)
            
            # 评估
            y_pred_train = ada.predict(self.X_train_scaled)
            y_pred_test = ada.predict(self.X_test_scaled)
            
            train_acc = calculate_accuracy(self.y_train, y_pred_train)
            test_acc = calculate_accuracy(self.y_test, y_pred_test)
            
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            print(f"  训练准确率: {train_acc:.4f}")
            print(f"  测试准确率: {test_acc:.4f}")
        
        # 可视化结果
        plt.figure(figsize=(12, 5))
        
        # 准确率曲线
        plt.subplot(1, 2, 1)
        plt.plot(n_estimators_list, train_accuracies, 'o-', label='训练集', linewidth=2)
        plt.plot(n_estimators_list, test_accuracies, 's-', label='测试集', linewidth=2)
        plt.xlabel('弱分类器数量')
        plt.ylabel('准确率')
        plt.title('准确率 vs 弱分类器数量')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 过拟合分析
        plt.subplot(1, 2, 2)
        overfitting = np.array(train_accuracies) - np.array(test_accuracies)
        plt.plot(n_estimators_list, overfitting, 'ro-', linewidth=2)
        plt.xlabel('弱分类器数量')
        plt.ylabel('过拟合程度 (训练-测试)')
        plt.title('过拟合分析')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return train_accuracies, test_accuracies
    
    def learning_curve_analysis(self):
        """
        学习曲线分析
        """
        if self.adaboost is None:
            print("请先训练模型")
            return
        
        print(f"\n=== AdaBoost学习曲线分析 ===")
        
        # 绘制学习曲线
        errors = self.adaboost.plot_learning_curve(self.X_train_scaled, self.y_train)
        
        # 分析学习过程
        print(f"初始错误率: {errors[0]:.4f}")
        print(f"最终错误率: {errors[-1]:.4f}")
        print(f"错误率改善: {errors[0] - errors[-1]:.4f}")
        
        # 找到最佳分类器数量
        min_error_idx = np.argmin(errors)
        print(f"最低错误率在第 {min_error_idx + 1} 个分类器: {errors[min_error_idx]:.4f}")
        
        return errors
    
    def ensemble_vs_individual(self):
        """
        比较集成效果与单个分类器
        """
        if self.adaboost is None:
            print("请先训练模型")
            return
        
        print(f"\n=== 集成效果 vs 单个分类器 ===")
        
        # 单个弱分类器的性能
        individual_accuracies = []
        
        for i, estimator in enumerate(self.adaboost.estimators[:10]):  # 只看前10个
            y_pred = estimator.predict(self.X_test_scaled)
            
            # 转换标签进行比较
            y_test_compare = self.y_test
            
            accuracy = calculate_accuracy(y_test_compare, y_pred)
            individual_accuracies.append(accuracy)
            
            print(f"弱分类器 {i+1}: 准确率 = {accuracy:.4f}")
        
        # 集成分类器性能
        ensemble_pred = self.adaboost.predict(self.X_test_scaled)
        ensemble_accuracy = calculate_accuracy(self.y_test, ensemble_pred)
        
        print(f"\nAdaBoost集成: 准确率 = {ensemble_accuracy:.4f}")
        print(f"平均弱分类器准确率: {np.mean(individual_accuracies):.4f}")
        print(f"最佳弱分类器准确率: {np.max(individual_accuracies):.4f}")
        print(f"集成提升: {ensemble_accuracy - np.mean(individual_accuracies):.4f}")
        
        # 可视化比较
        plt.figure(figsize=(10, 6))
        
        x_pos = range(len(individual_accuracies))
        plt.bar(x_pos, individual_accuracies, alpha=0.7, label='单个弱分类器')
        plt.axhline(y=ensemble_accuracy, color='red', linestyle='-', 
                   linewidth=3, label=f'AdaBoost集成 ({ensemble_accuracy:.3f})')
        plt.axhline(y=np.mean(individual_accuracies), color='green', linestyle='--', 
                   linewidth=2, label=f'平均性能 ({np.mean(individual_accuracies):.3f})')
        
        plt.xlabel('弱分类器索引')
        plt.ylabel('测试准确率')
        plt.title('单个分类器 vs 集成分类器性能比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return individual_accuracies, ensemble_accuracy

def main():
    """
    主函数
    """
    print("=== AdaBoost弱分类器集成系统 ===")
    
    # 创建集成系统
    ensemble_system = WeakClassifierEnsemble()
    
    # 测试不同数据集
    datasets = ['classification', 'moons', 'circles']
    
    for dataset_type in datasets:
        print(f"\n{'='*60}")
        print(f"测试数据集: {dataset_type}")
        print(f"{'='*60}")
        
        # 生成数据
        X, y = ensemble_system.create_synthetic_dataset(dataset_type, 800)
        
        # 可视化数据
        ensemble_system.visualize_dataset()
        
        # 数据预处理
        X_train, X_test, y_train, y_test = ensemble_system.preprocess_data()
        
        # 训练AdaBoost
        adaboost_model = ensemble_system.train_adaboost(n_estimators=50)
        
        # 分析弱分类器
        ensemble_system.analyze_weak_classifiers()
        
        # 绘制决策边界
        ensemble_system.plot_decision_boundary(f"AdaBoost决策边界 - {dataset_type}")
        
        # 学习曲线分析
        ensemble_system.learning_curve_analysis()
        
        # 集成vs单个分类器比较
        ensemble_system.ensemble_vs_individual()
        
        if dataset_type == 'classification':  # 只对第一个数据集做详细分析
            # 比较不同分类器数量
            ensemble_system.compare_n_estimators([1, 5, 10, 20, 30, 50])
    
    print(f"\n{'='*60}")
    print("AdaBoost算法总结")
    print(f"{'='*60}")
    print("1. AdaBoost通过加权训练样本，迭代训练弱分类器")
    print("2. 每轮训练后，增加被错误分类样本的权重")
    print("3. 弱分类器的投票权重基于其准确率确定")
    print("4. 最终分类器是所有弱分类器的加权组合")
    print("5. AdaBoost能将弱分类器提升为强分类器")
    print("6. 对噪声和异常值较为敏感")

if __name__ == "__main__":
    main() 