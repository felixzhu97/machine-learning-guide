"""
朴素贝叶斯算法实现
"""
import numpy as np
from collections import Counter
import re
import math

class NaiveBayes:
    """
    朴素贝叶斯分类器
    """
    
    def __init__(self, alpha=1.0):
        """
        初始化朴素贝叶斯分类器
        
        参数:
        alpha: 拉普拉斯平滑参数
        """
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}
        self.feature_probs = {}
        self.vocabulary = None
        
    def fit(self, X, y):
        """
        训练朴素贝叶斯模型
        
        参数:
        X: 特征数据 (文档-词频矩阵)
        y: 标签数据
        """
        self.classes = np.unique(y)
        n_samples = len(y)
        
        # 计算类别先验概率
        for class_label in self.classes:
            class_count = np.sum(y == class_label)
            self.class_priors[class_label] = class_count / n_samples
        
        # 计算特征在各类别下的概率
        n_features = X.shape[1]
        
        for class_label in self.classes:
            # 获取该类别的所有样本
            class_mask = y == class_label
            class_samples = X[class_mask]
            
            # 计算该类别下每个特征的概率
            feature_counts = np.sum(class_samples, axis=0)
            total_count = np.sum(feature_counts)
            
            # 使用拉普拉斯平滑
            smoothed_probs = (feature_counts + self.alpha) / (total_count + self.alpha * n_features)
            self.feature_probs[class_label] = smoothed_probs
    
    def predict(self, X):
        """
        预测
        
        参数:
        X: 待预测的特征数据
        
        返回:
        predictions: 预测结果
        """
        predictions = []
        
        for sample in X:
            class_scores = {}
            
            # 对每个类别计算分数
            for class_label in self.classes:
                # 先验概率（对数）
                score = math.log(self.class_priors[class_label])
                
                # 似然概率（对数）
                feature_probs = self.feature_probs[class_label]
                
                for i, feature_count in enumerate(sample):
                    if feature_count > 0:  # 只考虑出现的特征
                        score += feature_count * math.log(feature_probs[i])
                
                class_scores[class_label] = score
            
            # 选择得分最高的类别
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        预测概率
        
        参数:
        X: 待预测的特征数据
        
        返回:
        probabilities: 预测概率
        """
        probabilities = []
        
        for sample in X:
            class_scores = {}
            
            # 对每个类别计算分数
            for class_label in self.classes:
                # 先验概率（对数）
                score = math.log(self.class_priors[class_label])
                
                # 似然概率（对数）
                feature_probs = self.feature_probs[class_label]
                
                for i, feature_count in enumerate(sample):
                    if feature_count > 0:  # 只考虑出现的特征
                        score += feature_count * math.log(feature_probs[i])
                
                class_scores[class_label] = score
            
            # 转换为概率
            max_score = max(class_scores.values())
            exp_scores = {k: math.exp(v - max_score) for k, v in class_scores.items()}
            total_exp = sum(exp_scores.values())
            
            proba = {k: v / total_exp for k, v in exp_scores.items()}
            probabilities.append(proba)
        
        return probabilities

class TextProcessor:
    """
    文本处理器
    """
    
    def __init__(self):
        self.vocabulary = None
        self.word_to_idx = {}
        
    def preprocess_text(self, text):
        """
        预处理文本
        
        参数:
        text: 原始文本
        
        返回:
        processed_text: 处理后的文本
        """
        # 转换为小写
        text = text.lower()
        
        # 移除标点符号和数字
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 分词
        words = text.split()
        
        # 移除停用词（简单版本）
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        words = [word for word in words if word not in stop_words and len(word) > 1]
        
        return words
    
    def build_vocabulary(self, documents, min_freq=2):
        """
        构建词汇表
        
        参数:
        documents: 文档列表
        min_freq: 最小词频
        
        返回:
        vocabulary: 词汇表
        """
        word_counts = Counter()
        
        for doc in documents:
            words = self.preprocess_text(doc)
            word_counts.update(words)
        
        # 只保留频率大于等于min_freq的词
        self.vocabulary = [word for word, count in word_counts.items() if count >= min_freq]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        
        return self.vocabulary
    
    def documents_to_matrix(self, documents):
        """
        将文档转换为词频矩阵
        
        参数:
        documents: 文档列表
        
        返回:
        matrix: 词频矩阵
        """
        if self.vocabulary is None:
            raise ValueError("请先构建词汇表")
        
        matrix = np.zeros((len(documents), len(self.vocabulary)))
        
        for i, doc in enumerate(documents):
            words = self.preprocess_text(doc)
            word_counts = Counter(words)
            
            for word, count in word_counts.items():
                if word in self.word_to_idx:
                    matrix[i, self.word_to_idx[word]] = count
        
        return matrix

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

def calculate_precision_recall(y_true, y_pred, positive_class=1):
    """
    计算精确率和召回率
    
    参数:
    y_true: 真实标签
    y_pred: 预测标签
    positive_class: 正类标签
    
    返回:
    precision: 精确率
    recall: 召回率
    """
    true_positives = np.sum((y_true == positive_class) & (y_pred == positive_class))
    false_positives = np.sum((y_true != positive_class) & (y_pred == positive_class))
    false_negatives = np.sum((y_true == positive_class) & (y_pred != positive_class))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall 