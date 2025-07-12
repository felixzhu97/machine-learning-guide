"""
数据处理工具函数
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path, delimiter=',', header=None):
    """
    加载数据文件
    
    参数:
    file_path: 文件路径
    delimiter: 分隔符
    header: 头部行数
    
    返回:
    data: 数据数组
    """
    try:
        data = pd.read_csv(file_path, delimiter=delimiter, header=header)
        return data
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def normalize_data(data):
    """
    数据归一化
    
    参数:
    data: 输入数据
    
    返回:
    normalized_data: 归一化后的数据
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def split_data(X, y, test_size=0.2, random_state=42):
    """
    拆分训练集和测试集
    
    参数:
    X: 特征数据
    y: 标签数据
    test_size: 测试集比例
    random_state: 随机种子
    
    返回:
    X_train, X_test, y_train, y_test: 训练集和测试集
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def encode_labels(labels):
    """
    编码标签
    
    参数:
    labels: 标签数据
    
    返回:
    encoded_labels: 编码后的标签
    encoder: 编码器
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder

def plot_data_distribution(data, title="数据分布"):
    """
    绘制数据分布图
    
    参数:
    data: 数据
    title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    if isinstance(data, pd.DataFrame):
        for column in data.columns:
            plt.hist(data[column], alpha=0.7, label=column)
    else:
        plt.hist(data, alpha=0.7)
    plt.title(title)
    plt.xlabel("值")
    plt.ylabel("频率")
    plt.legend()
    plt.show()

def plot_correlation_matrix(data, title="相关性矩阵"):
    """
    绘制相关性矩阵
    
    参数:
    data: 数据
    title: 图表标题
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    plt.show()

def create_sample_data(n_samples=100, n_features=2, n_classes=2, random_state=42):
    """
    创建样本数据
    
    参数:
    n_samples: 样本数量
    n_features: 特征数量
    n_classes: 类别数量
    random_state: 随机种子
    
    返回:
    X: 特征数据
    y: 标签数据
    """
    np.random.seed(random_state)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y

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

def calculate_confusion_matrix(y_true, y_pred):
    """
    计算混淆矩阵
    
    参数:
    y_true: 真实标签
    y_pred: 预测标签
    
    返回:
    confusion_matrix: 混淆矩阵
    """
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    绘制混淆矩阵
    
    参数:
    y_true: 真实标签
    y_pred: 预测标签
    class_names: 类别名称
    """
    cm = calculate_confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.show() 