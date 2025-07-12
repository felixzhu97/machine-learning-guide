"""
数据降维案例
使用主成分分析（PCA）进行数据降维和可视化
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris, make_blobs
from sklearn.preprocessing import StandardScaler
from pca import PCA, plot_2d_pca, reconstruction_error
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class DimensionalityReductionSystem:
    """
    数据降维系统
    """
    
    def __init__(self):
        self.pca = None
        self.scaler = None
        self.X = None
        self.y = None
        self.feature_names = None
        
    def load_iris_dataset(self):
        """
        加载鸢尾花数据集
        """
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        print("=== 鸢尾花数据集 ===")
        print(f"数据形状: {self.X.shape}")
        print(f"特征名称: {self.feature_names}")
        print(f"类别名称: {self.target_names}")
        
        return self.X, self.y
    
    def load_digits_dataset(self):
        """
        加载手写数字数据集
        """
        digits = load_digits()
        self.X = digits.data
        self.y = digits.target
        self.feature_names = [f"像素{i}" for i in range(self.X.shape[1])]
        self.target_names = [str(i) for i in range(10)]
        
        print("=== 手写数字数据集 ===")
        print(f"数据形状: {self.X.shape}")
        print(f"图像尺寸: 8x8")
        print(f"类别数量: {len(self.target_names)}")
        
        return self.X, self.y
    
    def create_synthetic_dataset(self, n_samples=1000, n_features=10, n_centers=3):
        """
        创建合成数据集
        
        参数:
        n_samples: 样本数量
        n_features: 特征数量
        n_centers: 聚类中心数量
        """
        self.X, self.y = make_blobs(
            n_samples=n_samples,
            centers=n_centers,
            n_features=n_features,
            random_state=42,
            cluster_std=2.0
        )
        
        self.feature_names = [f"特征{i+1}" for i in range(n_features)]
        self.target_names = [f"类别{i}" for i in range(n_centers)]
        
        print("=== 合成数据集 ===")
        print(f"数据形状: {self.X.shape}")
        print(f"聚类中心数量: {n_centers}")
        
        return self.X, self.y
    
    def preprocess_data(self, standardize=True):
        """
        数据预处理
        
        参数:
        standardize: 是否标准化
        """
        if standardize:
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.X)
            print("数据标准化完成")
        else:
            self.X_scaled = self.X.copy()
        
        return self.X_scaled
    
    def fit_pca(self, n_components=None):
        """
        拟合PCA模型
        
        参数:
        n_components: 主成分数量
        """
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.X_scaled)
        
        print(f"\n=== PCA分析结果 ===")
        print(f"原始特征数量: {self.X_scaled.shape[1]}")
        print(f"主成分数量: {self.pca.n_components}")
        print(f"前5个主成分解释方差比: {self.pca.explained_variance_ratio_[:5]}")
        print(f"总解释方差比: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        
        return self.pca
    
    def analyze_components(self):
        """
        分析主成分
        """
        if self.pca is None:
            print("请先拟合PCA模型")
            return
        
        # 绘制解释方差图
        self.pca.plot_explained_variance()
        
        # 分析不同方差阈值所需的主成分数量
        thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
        print(f"\n=== 不同方差解释比例所需的主成分数量 ===")
        
        for threshold in thresholds:
            n_comp = self.pca.get_components_for_variance(threshold)
            print(f"解释 {threshold*100}% 方差需要 {n_comp} 个主成分")
    
    def visualize_2d_projection(self):
        """
        可视化2D投影结果
        """
        if self.pca is None:
            print("请先拟合PCA模型")
            return
        
        # 降维到2D
        pca_2d = PCA(n_components=2)
        X_2d = pca_2d.fit_transform(self.X_scaled)
        
        print(f"\n=== 2D投影结果 ===")
        print(f"前两个主成分解释方差比: {pca_2d.explained_variance_ratio_}")
        print(f"累积解释方差比: {np.sum(pca_2d.explained_variance_ratio_):.4f}")
        
        # 绘制2D投影
        plot_2d_pca(self.X_scaled, X_2d, self.y, "PCA 2D投影")
        
        return X_2d
    
    def compression_analysis(self, target_variance=0.95):
        """
        数据压缩分析
        
        参数:
        target_variance: 目标方差解释比例
        """
        # 获取达到目标方差所需的主成分数量
        n_components = self.pca.get_components_for_variance(target_variance)
        
        # 使用指定数量的主成分进行压缩
        pca_compressed = PCA(n_components=n_components)
        X_compressed = pca_compressed.fit_transform(self.X_scaled)
        X_reconstructed = pca_compressed.inverse_transform(X_compressed)
        
        # 计算压缩比和重构误差
        original_size = self.X_scaled.size
        compressed_size = X_compressed.size
        compression_ratio = compressed_size / original_size
        error = reconstruction_error(self.X_scaled, X_reconstructed)
        
        print(f"\n=== 数据压缩分析 ===")
        print(f"目标方差解释比例: {target_variance*100}%")
        print(f"使用主成分数量: {n_components}/{self.X_scaled.shape[1]}")
        print(f"实际解释方差比例: {np.sum(pca_compressed.explained_variance_ratio_)*100:.2f}%")
        print(f"压缩比: {compression_ratio:.4f}")
        print(f"重构误差: {error:.6f}")
        
        return X_compressed, X_reconstructed, compression_ratio, error
    
    def feature_contribution_analysis(self):
        """
        特征贡献度分析
        """
        if self.pca is None:
            print("请先拟合PCA模型")
            return
        
        print(f"\n=== 特征贡献度分析 ===")
        
        # 分析前几个主成分中各特征的贡献
        n_top_components = min(3, self.pca.n_components)
        
        for i in range(n_top_components):
            component = self.pca.components_[i]
            
            # 获取贡献度最大的几个特征
            top_indices = np.argsort(np.abs(component))[-5:][::-1]
            
            print(f"\n第{i+1}主成分 (解释方差比: {self.pca.explained_variance_ratio_[i]:.4f}):")
            print("  主要贡献特征:")
            
            for idx in top_indices:
                if self.feature_names and idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                else:
                    feature_name = f"特征{idx}"
                print(f"    {feature_name}: {component[idx]:.4f}")
    
    def visualize_digits_reconstruction(self, n_components_list=[2, 5, 10, 20, 40]):
        """
        可视化手写数字重构结果
        
        参数:
        n_components_list: 不同主成分数量列表
        """
        if self.X.shape[1] != 64:  # 不是8x8的图像数据
            print("此功能仅适用于手写数字数据集")
            return
        
        # 选择几个数字进行展示
        sample_indices = [0, 10, 20, 30, 40]
        n_samples = len(sample_indices)
        n_components_count = len(n_components_list)
        
        fig, axes = plt.subplots(n_samples, n_components_count + 1, 
                                figsize=(3 * (n_components_count + 1), 3 * n_samples))
        
        for i, sample_idx in enumerate(sample_indices):
            # 原始图像
            original_image = self.X[sample_idx].reshape(8, 8)
            axes[i, 0].imshow(original_image, cmap='gray')
            axes[i, 0].set_title(f'原始 (数字{self.y[sample_idx]})')
            axes[i, 0].axis('off')
            
            # 不同主成分数量的重构结果
            for j, n_comp in enumerate(n_components_list):
                pca_temp = PCA(n_components=n_comp)
                X_transformed = pca_temp.fit_transform(self.X_scaled)
                X_reconstructed = pca_temp.inverse_transform(X_transformed)
                
                # 反标准化
                if self.scaler:
                    X_reconstructed = self.scaler.inverse_transform(X_reconstructed)
                
                reconstructed_image = X_reconstructed[sample_idx].reshape(8, 8)
                axes[i, j + 1].imshow(reconstructed_image, cmap='gray')
                axes[i, j + 1].set_title(f'{n_comp}个主成分')
                axes[i, j + 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def compare_datasets(self):
        """
        比较不同数据集的PCA效果
        """
        datasets = [
            ("鸢尾花", self.load_iris_dataset),
            ("合成数据", lambda: self.create_synthetic_dataset(1000, 20, 4))
        ]
        
        results = {}
        
        for name, load_func in datasets:
            print(f"\n{'='*50}")
            print(f"分析数据集: {name}")
            print(f"{'='*50}")
            
            # 加载数据
            X, y = load_func()
            
            # 预处理
            X_scaled = self.preprocess_data()
            
            # PCA分析
            pca = self.fit_pca()
            
            # 分析主成分
            self.analyze_components()
            
            # 2D可视化
            X_2d = self.visualize_2d_projection()
            
            # 压缩分析
            _, _, compression_ratio, error = self.compression_analysis(0.95)
            
            results[name] = {
                'n_features': X.shape[1],
                'n_components_95': pca.get_components_for_variance(0.95),
                'compression_ratio': compression_ratio,
                'reconstruction_error': error
            }
        
        # 汇总比较结果
        print(f"\n{'='*60}")
        print("数据集比较汇总")
        print(f"{'='*60}")
        
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  原始特征数: {result['n_features']}")
            print(f"  95%方差主成分数: {result['n_components_95']}")
            print(f"  压缩比: {result['compression_ratio']:.4f}")
            print(f"  重构误差: {result['reconstruction_error']:.6f}")

def main():
    """
    主函数
    """
    print("=== 主成分分析（PCA）数据降维系统 ===")
    
    # 创建降维系统
    dr_system = DimensionalityReductionSystem()
    
    # 案例1：鸢尾花数据集分析
    print("\n" + "="*60)
    print("案例1：鸢尾花数据集PCA分析")
    print("="*60)
    
    dr_system.load_iris_dataset()
    dr_system.preprocess_data()
    dr_system.fit_pca()
    dr_system.analyze_components()
    dr_system.visualize_2d_projection()
    dr_system.compression_analysis(0.95)
    dr_system.feature_contribution_analysis()
    
    # 案例2：手写数字数据集分析
    print("\n" + "="*60)
    print("案例2：手写数字数据集PCA分析")
    print("="*60)
    
    dr_system.load_digits_dataset()
    dr_system.preprocess_data()
    dr_system.fit_pca()
    dr_system.analyze_components()
    dr_system.visualize_2d_projection()
    dr_system.compression_analysis(0.95)
    dr_system.visualize_digits_reconstruction()
    
    # 案例3：高维合成数据分析
    print("\n" + "="*60)
    print("案例3：高维合成数据PCA分析")
    print("="*60)
    
    dr_system.create_synthetic_dataset(1000, 50, 5)
    dr_system.preprocess_data()
    dr_system.fit_pca()
    dr_system.analyze_components()
    dr_system.visualize_2d_projection()
    dr_system.compression_analysis(0.90)
    
    print(f"\n{'='*60}")
    print("PCA分析总结")
    print(f"{'='*60}")
    print("1. PCA是一种无监督降维技术，通过线性变换找到数据的主要变化方向")
    print("2. 主成分按解释方差的大小排序，前几个主成分通常包含了数据的主要信息")
    print("3. 可以根据累积解释方差比例选择合适的主成分数量")
    print("4. PCA在数据可视化、噪声降低、数据压缩等方面有重要应用")
    print("5. 注意PCA假设数据的主要变化是线性的，对于非线性结构可能效果有限")

if __name__ == "__main__":
    main() 