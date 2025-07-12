"""
疝气病症预测案例
使用逻辑回归算法预测马的疝气病症是否致命
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logistic_regression import LogisticRegression, calculate_accuracy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.visualization import plot_2d_scatter, plot_decision_boundary
from utils.data_utils import plot_confusion_matrix

class ColicPredictionSystem:
    """
    疝气病症预测系统
    """
    
    def __init__(self):
        self.logistic_model = None
        self.scaler = None
        self.feature_names = ['体温', '脉搏', '呼吸频率', '体重', '年龄', '疼痛程度', '肠音', '腹部触诊']
        self.X = None
        self.y = None
        
    def create_sample_data(self, n_samples=1000):
        """
        创建疝气病症样本数据
        
        参数:
        n_samples: 样本数量
        """
        np.random.seed(42)
        
        # 生成特征
        # 正常马的特征分布
        normal_temp = np.random.normal(37.5, 0.5, n_samples//2)  # 体温 (摄氏度)
        normal_pulse = np.random.normal(40, 5, n_samples//2)     # 脉搏 (次/分钟)
        normal_resp = np.random.normal(12, 2, n_samples//2)      # 呼吸频率 (次/分钟)
        normal_weight = np.random.normal(500, 50, n_samples//2)  # 体重 (公斤)
        normal_age = np.random.uniform(2, 15, n_samples//2)      # 年龄 (年)
        normal_pain = np.random.uniform(0, 2, n_samples//2)      # 疼痛程度 (0-5)
        normal_sound = np.random.uniform(3, 5, n_samples//2)     # 肠音 (0-5)
        normal_palp = np.random.uniform(0, 1, n_samples//2)      # 腹部触诊 (0-3)
        
        # 患病马的特征分布 (特征值异常)
        sick_temp = np.random.normal(38.5, 1.0, n_samples//2)   # 体温升高
        sick_pulse = np.random.normal(60, 10, n_samples//2)      # 脉搏加快
        sick_resp = np.random.normal(20, 5, n_samples//2)        # 呼吸急促
        sick_weight = np.random.normal(480, 60, n_samples//2)    # 体重可能下降
        sick_age = np.random.uniform(5, 20, n_samples//2)        # 年龄偏大
        sick_pain = np.random.uniform(3, 5, n_samples//2)        # 疼痛严重
        sick_sound = np.random.uniform(0, 2, n_samples//2)       # 肠音减弱
        sick_palp = np.random.uniform(2, 3, n_samples//2)        # 腹部触诊异常
        
        # 组合特征
        features = np.column_stack([
            np.concatenate([normal_temp, sick_temp]),
            np.concatenate([normal_pulse, sick_pulse]),
            np.concatenate([normal_resp, sick_resp]),
            np.concatenate([normal_weight, sick_weight]),
            np.concatenate([normal_age, sick_age]),
            np.concatenate([normal_pain, sick_pain]),
            np.concatenate([normal_sound, sick_sound]),
            np.concatenate([normal_palp, sick_palp])
        ])
        
        # 生成标签
        labels = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
        
        # 打乱数据
        indices = np.random.permutation(n_samples)
        self.X = features[indices]
        self.y = labels[indices]
        
        print(f"生成了 {n_samples} 个疝气病症样本")
        print(f"正常马: {np.sum(self.y == 0)} 匹")
        print(f"患病马: {np.sum(self.y == 1)} 匹")
        print(f"特征: {self.feature_names}")
        
        return self.X, self.y
    
    def preprocess_data(self):
        """
        数据预处理
        """
        # 特征标准化
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        print("数据标准化完成")
        
        return self.X_scaled
    
    def train_model(self, test_size=0.2, learning_rate=0.01, max_iterations=1000):
        """
        训练模型
        
        参数:
        test_size: 测试集比例
        learning_rate: 学习率
        max_iterations: 最大迭代次数
        """
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=42
        )
        
        # 创建并训练逻辑回归模型
        self.logistic_model = LogisticRegression(
            learning_rate=learning_rate,
            max_iterations=max_iterations
        )
        self.logistic_model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = self.logistic_model.predict(X_train)
        y_pred_test = self.logistic_model.predict(X_test)
        
        # 计算准确率
        train_accuracy = calculate_accuracy(y_train, y_pred_train)
        test_accuracy = calculate_accuracy(y_test, y_pred_test)
        
        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")
        
        # 绘制代价函数变化
        self.logistic_model.plot_cost_history()
        
        return X_train, X_test, y_train, y_test, y_pred_test, test_accuracy
    
    def predict_horse_condition(self, temp, pulse, resp, weight, age, pain, sound, palp):
        """
        预测马的疝气病症
        
        参数:
        temp: 体温
        pulse: 脉搏
        resp: 呼吸频率
        weight: 体重
        age: 年龄
        pain: 疼痛程度
        sound: 肠音
        palp: 腹部触诊
        
        返回:
        prediction: 预测结果
        probability: 预测概率
        """
        if self.logistic_model is None:
            print("请先训练模型")
            return None, None
        
        # 准备特征
        features = np.array([[temp, pulse, resp, weight, age, pain, sound, palp]])
        features_scaled = self.scaler.transform(features)
        
        # 预测
        prediction = self.logistic_model.predict(features_scaled)[0]
        probability = self.logistic_model.predict_proba(features_scaled)[0]
        
        result = "患疝气病症" if prediction == 1 else "正常"
        
        print(f"马匹特征:")
        for i, (name, value) in enumerate(zip(self.feature_names, features[0])):
            print(f"  {name}: {value}")
        print(f"预测结果: {result}")
        print(f"患病概率: {probability:.4f}")
        
        return prediction, probability
    
    def analyze_feature_importance(self):
        """
        分析特征重要性
        """
        if self.logistic_model is None:
            print("请先训练模型")
            return
        
        # 获取权重 (除去偏置项)
        weights = self.logistic_model.weights[1:]
        
        print("\n=== 特征重要性分析 ===")
        for i, (feature, weight) in enumerate(zip(self.feature_names, weights)):
            print(f"{feature}: {weight:.4f}")
        
        # 可视化特征重要性
        plt.figure(figsize=(12, 6))
        colors = ['red' if w < 0 else 'blue' for w in weights]
        plt.bar(self.feature_names, weights, color=colors, alpha=0.7)
        plt.title('特征重要性 (逻辑回归权重)')
        plt.xlabel('特征')
        plt.ylabel('权重')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return weights
    
    def visualize_data_distribution(self):
        """
        可视化数据分布
        """
        if self.X is None:
            print("请先生成数据")
            return
        
        # 选择几个重要特征进行可视化
        key_features = [0, 1, 2, 5]  # 体温、脉搏、呼吸频率、疼痛程度
        key_names = [self.feature_names[i] for i in key_features]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (feature_idx, name) in enumerate(zip(key_features, key_names)):
            ax = axes[i]
            
            # 正常马的特征分布
            normal_data = self.X[self.y == 0, feature_idx]
            sick_data = self.X[self.y == 1, feature_idx]
            
            ax.hist(normal_data, alpha=0.7, label='正常马', bins=20, color='blue')
            ax.hist(sick_data, alpha=0.7, label='患病马', bins=20, color='red')
            ax.set_xlabel(name)
            ax.set_ylabel('频率')
            ax.set_title(f'{name}分布')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def compare_learning_rates(self):
        """
        比较不同学习率的效果
        """
        print("\n=== 比较不同学习率 ===")
        
        learning_rates = [0.001, 0.01, 0.1, 0.5]
        accuracies = []
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )
        
        plt.figure(figsize=(15, 5))
        
        for i, lr in enumerate(learning_rates):
            # 训练模型
            model = LogisticRegression(learning_rate=lr, max_iterations=1000)
            model.fit(X_train, y_train)
            
            # 计算准确率
            y_pred = model.predict(X_test)
            accuracy = calculate_accuracy(y_test, y_pred)
            accuracies.append(accuracy)
            
            print(f"学习率 {lr}: 准确率 = {accuracy:.4f}")
            
            # 绘制代价函数变化
            plt.subplot(1, 4, i+1)
            plt.plot(model.cost_history)
            plt.title(f'学习率 = {lr}\n准确率 = {accuracy:.4f}')
            plt.xlabel('迭代次数')
            plt.ylabel('代价值')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 找到最佳学习率
        best_lr = learning_rates[np.argmax(accuracies)]
        print(f"\n最佳学习率: {best_lr}")
        
        return best_lr
    
    def test_various_cases(self):
        """
        测试各种情况
        """
        print("\n=== 测试各种疝气病症案例 ===")
        
        test_cases = [
            # 体温, 脉搏, 呼吸, 体重, 年龄, 疼痛, 肠音, 触诊
            (37.2, 38, 12, 520, 5, 1, 4, 0),   # 正常马
            (38.8, 65, 22, 460, 12, 4, 1, 3),  # 患病马
            (37.8, 45, 15, 500, 8, 2, 3, 1),   # 轻微异常
            (39.2, 80, 25, 420, 15, 5, 0, 3),  # 严重患病
            (37.0, 35, 10, 550, 3, 0, 5, 0),   # 健康马
        ]
        
        descriptions = [
            "健康成年马",
            "疑似疝气病症马",
            "轻微不适马",
            "严重疝气病症马",
            "健康幼马"
        ]
        
        for case, desc in zip(test_cases, descriptions):
            print(f"\n{desc}:")
            self.predict_horse_condition(*case)

def main():
    """
    主函数
    """
    print("=== 疝气病症预测系统 ===")
    
    # 创建疝气病症预测系统
    colic_system = ColicPredictionSystem()
    
    # 生成样本数据
    X, y = colic_system.create_sample_data(1000)
    
    # 可视化数据分布
    colic_system.visualize_data_distribution()
    
    # 数据预处理
    X_scaled = colic_system.preprocess_data()
    
    # 比较不同学习率
    best_lr = colic_system.compare_learning_rates()
    
    # 使用最佳学习率训练模型
    print(f"\n=== 使用最佳学习率 {best_lr} 训练模型 ===")
    X_train, X_test, y_train, y_test, y_pred_test, test_accuracy = colic_system.train_model(
        learning_rate=best_lr
    )
    
    # 分析特征重要性
    colic_system.analyze_feature_importance()
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_test, y_pred_test, class_names=['正常', '患疝气病症'])
    
    # 测试各种情况
    colic_system.test_various_cases()
    
    print(f"\n=== 模型总结 ===")
    print(f"逻辑回归是一种线性分类算法，适用于二分类问题")
    print(f"通过Sigmoid函数将线性组合映射到[0,1]区间")
    print(f"本案例最终测试准确率: {test_accuracy:.4f}")

if __name__ == "__main__":
    main() 