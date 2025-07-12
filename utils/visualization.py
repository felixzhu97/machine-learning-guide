"""
可视化工具函数
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_2d_scatter(X, y, title="数据分布", xlabel="特征1", ylabel="特征2", save_path=None):
    """
    绘制2D散点图
    
    参数:
    X: 特征数据 (n_samples, 2)
    y: 标签数据
    title: 图表标题
    xlabel: X轴标签
    ylabel: Y轴标签
    save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    # 获取唯一标签
    unique_labels = np.unique(y)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[i % len(colors)], 
                   label=f'类别 {label}',
                   alpha=0.7, s=60)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_decision_boundary(X, y, classifier, title="决策边界", save_path=None):
    """
    绘制决策边界
    
    参数:
    X: 特征数据 (n_samples, 2)
    y: 标签数据
    classifier: 分类器对象
    title: 图表标题
    save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测网格点
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # 绘制数据点
    unique_labels = np.unique(y)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[i % len(colors)], 
                   label=f'类别 {label}',
                   edgecolors='black', s=60)
    
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_learning_curve(train_scores, val_scores, title="学习曲线"):
    """
    绘制学习曲线
    
    参数:
    train_scores: 训练分数
    val_scores: 验证分数
    title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_scores, 'o-', label='训练集', color='blue')
    plt.plot(val_scores, 'o-', label='验证集', color='red')
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_feature_importance(feature_names, importance_scores, title="特征重要性"):
    """
    绘制特征重要性图
    
    参数:
    feature_names: 特征名称
    importance_scores: 重要性分数
    title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(importance_scores)[::-1]
    
    plt.bar(range(len(feature_names)), 
            importance_scores[sorted_idx], 
            color='skyblue')
    plt.xticks(range(len(feature_names)), 
               [feature_names[i] for i in sorted_idx], 
               rotation=45)
    plt.xlabel('特征')
    plt.ylabel('重要性')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_scores, title="ROC曲线"):
    """
    绘制ROC曲线
    
    参数:
    y_true: 真实标签
    y_scores: 预测分数
    title: 图表标题
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_tree_structure(tree, feature_names=None, class_names=None, title="决策树结构"):
    """
    绘制决策树结构
    
    参数:
    tree: 决策树对象
    feature_names: 特征名称
    class_names: 类别名称
    title: 图表标题
    """
    from sklearn import tree
    
    plt.figure(figsize=(20, 10))
    tree.plot_tree(tree, 
                   feature_names=feature_names,
                   class_names=class_names,
                   filled=True, 
                   rounded=True,
                   fontsize=10)
    plt.title(title, fontsize=16)
    plt.show()

def plot_clustering_result(X, labels, centers=None, title="聚类结果"):
    """
    绘制聚类结果
    
    参数:
    X: 特征数据
    labels: 聚类标签
    centers: 聚类中心
    title: 图表标题
    """
    plt.figure(figsize=(10, 8))
    
    # 获取唯一标签
    unique_labels = np.unique(labels)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[i % len(colors)], 
                   label=f'簇 {label}',
                   alpha=0.7, s=60)
    
    # 绘制聚类中心
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], 
                   c='black', marker='x', s=200, 
                   linewidths=3, label='聚类中心')
    
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show() 