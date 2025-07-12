"""
手写数字识别案例
使用支持向量机进行手写数字分类
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from svm import SVM, calculate_accuracy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.data_utils import plot_confusion_matrix

class HandwrittenDigitsRecognition:
    """
    手写数字识别系统
    """
    
    def __init__(self):
        self.svm_models = {}
        self.scaler = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """
        加载手写数字数据集
        """
        digits = load_digits()
        self.X = digits.data
        self.y = digits.target
        
        print("=== 手写数字数据集信息 ===")
        print(f"数据形状: {self.X.shape}")
        print(f"图像尺寸: 8x8")
        print(f"类别数量: {len(np.unique(self.y))}")
        print(f"各类别样本数:")
        for digit in range(10):
            count = np.sum(self.y == digit)
            print(f"  数字{digit}: {count}个样本")
        
        return self.X, self.y
    
    def visualize_samples(self, n_samples=20):
        """
        可视化数据样本
        
        参数:
        n_samples: 显示的样本数量
        """
        fig, axes = plt.subplots(2, 10, figsize=(15, 6))
        
        for digit in range(10):
            # 找到该数字的样本
            digit_indices = np.where(self.y == digit)[0]
            
            # 随机选择两个样本
            selected_indices = np.random.choice(digit_indices, 2, replace=False)
            
            for i, idx in enumerate(selected_indices):
                ax = axes[i, digit]
                image = self.X[idx].reshape(8, 8)
                ax.imshow(image, cmap='gray')
                ax.set_title(f'数字 {digit}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.suptitle('手写数字样本展示', fontsize=16, y=1.02)
        plt.show()
    
    def preprocess_data(self, test_size=0.2):
        """
        数据预处理
        
        参数:
        test_size: 测试集比例
        """
        # 分割数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # 标准化特征
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")
        print("数据标准化完成")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_binary_classifiers(self, digit_pairs=[(0, 1), (2, 3), (4, 5)]):
        """
        训练二分类器
        
        参数:
        digit_pairs: 要分类的数字对
        """
        print("\n=== 训练二分类SVM ===")
        
        for digit1, digit2 in digit_pairs:
            print(f"\n训练数字{digit1} vs 数字{digit2}分类器...")
            
            # 选择相关数据
            mask = (self.y_train == digit1) | (self.y_train == digit2)
            X_binary = self.X_train_scaled[mask]
            y_binary = self.y_train[mask]
            
            # 转换标签为-1和1
            y_binary = np.where(y_binary == digit1, -1, 1)
            
            # 训练不同核函数的SVM
            kernels = ['linear', 'rbf']
            
            for kernel in kernels:
                print(f"  训练{kernel}核SVM...")
                
                if kernel == 'linear':
                    svm = SVM(C=1.0, kernel='linear', tolerance=1e-3, max_iterations=200)
                else:  # RBF
                    svm = SVM(C=1.0, kernel='rbf', gamma=0.01, tolerance=1e-3, max_iterations=200)
                
                svm.fit(X_binary, y_binary)
                
                # 评估性能
                y_pred = svm.predict(X_binary)
                accuracy = calculate_accuracy(y_binary, y_pred)
                
                print(f"    {kernel}核SVM训练准确率: {accuracy:.4f}")
                print(f"    支持向量数量: {len(svm.support_vectors)}")
                
                # 保存模型
                model_key = f"{digit1}_vs_{digit2}_{kernel}"
                self.svm_models[model_key] = svm
    
    def compare_kernels(self, digit1=0, digit2=1):
        """
        比较不同核函数的效果
        
        参数:
        digit1, digit2: 要比较的数字
        """
        print(f"\n=== 比较不同核函数 (数字{digit1} vs 数字{digit2}) ===")
        
        # 选择数据
        mask_train = (self.y_train == digit1) | (self.y_train == digit2)
        mask_test = (self.y_test == digit1) | (self.y_test == digit2)
        
        X_train_binary = self.X_train_scaled[mask_train]
        y_train_binary = self.y_train[mask_train]
        X_test_binary = self.X_test_scaled[mask_test]
        y_test_binary = self.y_test[mask_test]
        
        # 转换标签
        y_train_binary = np.where(y_train_binary == digit1, -1, 1)
        y_test_binary = np.where(y_test_binary == digit1, -1, 1)
        
        kernels_configs = [
            ('linear', {'C': 1.0, 'kernel': 'linear'}),
            ('rbf', {'C': 1.0, 'kernel': 'rbf', 'gamma': 0.01}),
            ('poly', {'C': 1.0, 'kernel': 'poly', 'degree': 3, 'gamma': 0.01}),
        ]
        
        results = {}
        
        fig, axes = plt.subplots(1, len(kernels_configs), figsize=(15, 5))
        
        for i, (kernel_name, config) in enumerate(kernels_configs):
            print(f"\n测试{kernel_name}核:")
            
            # 训练模型
            svm = SVM(**config, tolerance=1e-3, max_iterations=200)
            svm.fit(X_train_binary, y_train_binary)
            
            # 预测
            y_pred_train = svm.predict(X_train_binary)
            y_pred_test = svm.predict(X_test_binary)
            
            # 计算准确率
            train_acc = calculate_accuracy(y_train_binary, y_pred_train)
            test_acc = calculate_accuracy(y_test_binary, y_pred_test)
            
            results[kernel_name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'n_support_vectors': len(svm.support_vectors),
                'support_vector_ratio': len(svm.support_vectors) / len(X_train_binary)
            }
            
            print(f"  训练准确率: {train_acc:.4f}")
            print(f"  测试准确率: {test_acc:.4f}")
            print(f"  支持向量数量: {len(svm.support_vectors)}")
            print(f"  支持向量比例: {len(svm.support_vectors) / len(X_train_binary):.2%}")
            
            # 可视化（使用PCA降维到2D）
            if X_train_binary.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X_train_binary)
                
                # 重新训练2D SVM用于可视化
                svm_2d = SVM(**config, tolerance=1e-3, max_iterations=100)
                svm_2d.fit(X_2d, y_train_binary)
                
                ax = axes[i]
                self._plot_2d_decision_boundary(svm_2d, X_2d, y_train_binary, ax, 
                                              f'{kernel_name.upper()}核', digit1, digit2)
        
        plt.tight_layout()
        plt.show()
        
        # 结果汇总
        print(f"\n=== 核函数比较结果汇总 ===")
        results_df = pd.DataFrame(results).T
        print(results_df.round(4))
        
        return results
    
    def _plot_2d_decision_boundary(self, svm, X, y, ax, title, digit1, digit2):
        """
        绘制2D决策边界
        """
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.decision_function(grid_points)
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
        ax.contour(xx, yy, Z, levels=[0], colors='black', linestyles='--', linewidths=2)
        
        # 绘制数据点
        colors = ['red', 'blue']
        labels = [f'数字{digit1}', f'数字{digit2}']
        for i, label_val in enumerate([-1, 1]):
            mask = y == label_val
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                      label=labels[i], alpha=0.7, s=30)
        
        # 绘制支持向量
        if svm.support_vectors is not None:
            ax.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                      s=100, facecolors='none', edgecolors='black', linewidths=2)
        
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def parameter_tuning(self, digit1=0, digit2=1):
        """
        参数调优
        
        参数:
        digit1, digit2: 要调优的数字对
        """
        print(f"\n=== SVM参数调优 (数字{digit1} vs 数字{digit2}) ===")
        
        # 选择数据
        mask_train = (self.y_train == digit1) | (self.y_train == digit2)
        X_train_binary = self.X_train_scaled[mask_train]
        y_train_binary = self.y_train[mask_train]
        y_train_binary = np.where(y_train_binary == digit1, -1, 1)
        
        # 分割训练集和验证集
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train_binary, y_train_binary, test_size=0.2, random_state=42
        )
        
        # 调优C参数
        C_values = [0.1, 1.0, 10.0, 100.0]
        gamma_values = [0.001, 0.01, 0.1, 1.0]
        
        print("调优RBF核参数...")
        
        best_score = 0
        best_params = {}
        results = []
        
        for C in C_values:
            for gamma in gamma_values:
                print(f"  测试 C={C}, gamma={gamma}")
                
                svm = SVM(C=C, kernel='rbf', gamma=gamma, tolerance=1e-3, max_iterations=100)
                svm.fit(X_train_sub, y_train_sub)
                
                y_pred = svm.predict(X_val)
                accuracy = calculate_accuracy(y_val, y_pred)
                
                results.append({
                    'C': C,
                    'gamma': gamma,
                    'accuracy': accuracy,
                    'n_support_vectors': len(svm.support_vectors)
                })
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_params = {'C': C, 'gamma': gamma}
                
                print(f"    验证准确率: {accuracy:.4f}")
        
        print(f"\n最佳参数: {best_params}")
        print(f"最佳验证准确率: {best_score:.4f}")
        
        # 可视化参数调优结果
        self._plot_parameter_grid(results, C_values, gamma_values)
        
        return best_params, results
    
    def _plot_parameter_grid(self, results, C_values, gamma_values):
        """
        绘制参数网格搜索结果
        """
        # 创建准确率矩阵
        accuracy_matrix = np.zeros((len(C_values), len(gamma_values)))
        support_vector_matrix = np.zeros((len(C_values), len(gamma_values)))
        
        for result in results:
            i = C_values.index(result['C'])
            j = gamma_values.index(result['gamma'])
            accuracy_matrix[i, j] = result['accuracy']
            support_vector_matrix[i, j] = result['n_support_vectors']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准确率热力图
        im1 = ax1.imshow(accuracy_matrix, cmap='viridis', aspect='auto')
        ax1.set_xticks(range(len(gamma_values)))
        ax1.set_yticks(range(len(C_values)))
        ax1.set_xticklabels(gamma_values)
        ax1.set_yticklabels(C_values)
        ax1.set_xlabel('Gamma')
        ax1.set_ylabel('C')
        ax1.set_title('验证准确率')
        
        # 添加数值标注
        for i in range(len(C_values)):
            for j in range(len(gamma_values)):
                text = ax1.text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                               ha="center", va="center", color="white")
        
        plt.colorbar(im1, ax=ax1)
        
        # 支持向量数量热力图
        im2 = ax2.imshow(support_vector_matrix, cmap='plasma', aspect='auto')
        ax2.set_xticks(range(len(gamma_values)))
        ax2.set_yticks(range(len(C_values)))
        ax2.set_xticklabels(gamma_values)
        ax2.set_yticklabels(C_values)
        ax2.set_xlabel('Gamma')
        ax2.set_ylabel('C')
        ax2.set_title('支持向量数量')
        
        # 添加数值标注
        for i in range(len(C_values)):
            for j in range(len(gamma_values)):
                text = ax2.text(j, i, f'{int(support_vector_matrix[i, j])}',
                               ha="center", va="center", color="white")
        
        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        plt.show()
    
    def test_digit_recognition(self, test_indices=None):
        """
        测试数字识别效果
        
        参数:
        test_indices: 要测试的样本索引
        """
        if test_indices is None:
            test_indices = np.random.choice(len(self.X_test), 10, replace=False)
        
        print("\n=== 手写数字识别测试 ===")
        
        # 使用已训练的二分类器进行组合预测
        fig, axes = plt.subplots(2, 5, figsize=(15, 8))
        axes = axes.ravel()
        
        for i, idx in enumerate(test_indices):
            # 显示图像
            image = self.X_test[idx].reshape(8, 8)
            axes[i].imshow(image, cmap='gray')
            
            true_label = self.y_test[idx]
            
            # 简单投票机制（仅使用可用的分类器）
            votes = {}
            for model_key in self.svm_models:
                if 'linear' in model_key:  # 只使用线性核
                    digit1, digit2 = model_key.split('_')[0], model_key.split('_')[2]
                    if digit1.isdigit() and digit2.isdigit():
                        digit1, digit2 = int(digit1), int(digit2)
                        
                        if true_label in [digit1, digit2]:
                            svm = self.svm_models[model_key]
                            pred = svm.predict([self.X_test_scaled[idx]])[0]
                            
                            # 转换预测结果
                            predicted_digit = digit1 if pred == -1 else digit2
                            votes[predicted_digit] = votes.get(predicted_digit, 0) + 1
            
            # 获取得票最多的数字
            if votes:
                predicted_label = max(votes, key=votes.get)
                confidence = votes[predicted_label] / sum(votes.values())
            else:
                predicted_label = "?"
                confidence = 0
            
            # 设置标题
            color = 'green' if str(predicted_label) == str(true_label) else 'red'
            axes[i].set_title(f'真实: {true_label}\n预测: {predicted_label}\n置信度: {confidence:.2f}', 
                             color=color)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle('手写数字识别结果', fontsize=16, y=1.02)
        plt.show()

def main():
    """
    主函数
    """
    print("=== 手写数字识别系统 (SVM) ===")
    
    # 创建识别系统
    recognizer = HandwrittenDigitsRecognition()
    
    # 加载数据
    X, y = recognizer.load_data()
    
    # 可视化样本
    recognizer.visualize_samples()
    
    # 数据预处理
    X_train, X_test, y_train, y_test = recognizer.preprocess_data()
    
    # 训练二分类器
    digit_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    recognizer.train_binary_classifiers(digit_pairs)
    
    # 比较不同核函数
    recognizer.compare_kernels(0, 1)
    
    # 参数调优
    best_params, tuning_results = recognizer.parameter_tuning(0, 1)
    
    # 测试识别效果
    recognizer.test_digit_recognition()
    
    print(f"\n=== SVM算法总结 ===")
    print("1. SVM通过寻找最大间隔超平面来进行分类")
    print("2. 支持向量是距离决策边界最近的样本点")
    print("3. 核函数可以处理非线性可分的数据")
    print("4. 参数C控制对误分类的惩罚程度")
    print("5. RBF核的gamma参数控制决策边界的复杂度")

if __name__ == "__main__":
    main() 