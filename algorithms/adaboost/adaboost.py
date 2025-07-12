"""
AdaBoost算法实现
包括决策树桩作为弱分类器
"""
import numpy as np
import matplotlib.pyplot as plt

class DecisionStump:
    """
    决策树桩 - 简单的单层决策树
    """
    
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.inequality = None  # 'less' or 'greater'
        self.alpha = None
        
    def fit(self, X, y, weights):
        """
        训练决策树桩
        
        参数:
        X: 特征数据
        y: 标签 (必须是-1或1)
        weights: 样本权重
        
        返回:
        min_error: 最小错误率
        """
        n_samples, n_features = X.shape
        min_error = float('inf')
        best_prediction = None
        
        # 遍历每个特征
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # 遍历每个可能的阈值
            for threshold in unique_values:
                # 尝试两种分裂方式
                for inequality in ['less', 'greater']:
                    # 预测
                    predictions = np.ones(n_samples)
                    if inequality == 'less':
                        predictions[feature_values <= threshold] = -1
                    else:
                        predictions[feature_values > threshold] = -1
                    
                    # 计算加权错误率
                    misclassified = predictions != y
                    error = np.sum(weights[misclassified])
                    
                    # 保存最佳分类器
                    if error < min_error:
                        min_error = error
                        self.feature_index = feature_idx
                        self.threshold = threshold
                        self.inequality = inequality
                        best_prediction = predictions.copy()
        
        return min_error, best_prediction
    
    def predict(self, X):
        """
        预测
        
        参数:
        X: 特征数据
        
        返回:
        predictions: 预测结果
        """
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        
        feature_values = X[:, self.feature_index]
        
        if self.inequality == 'less':
            predictions[feature_values <= self.threshold] = -1
        else:
            predictions[feature_values > self.threshold] = -1
            
        return predictions

class AdaBoost:
    """
    AdaBoost分类器
    """
    
    def __init__(self, n_estimators=50, max_error_rate=0.5):
        """
        初始化AdaBoost
        
        参数:
        n_estimators: 弱分类器数量
        max_error_rate: 最大错误率阈值
        """
        self.n_estimators = n_estimators
        self.max_error_rate = max_error_rate
        self.estimators = []
        self.estimator_weights = []
        self.estimator_errors = []
        
    def fit(self, X, y):
        """
        训练AdaBoost模型
        
        参数:
        X: 训练特征
        y: 训练标签 (必须是-1或1)
        """
        X = np.array(X)
        y = np.array(y)
        
        # 确保标签是-1和1
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("AdaBoost只支持二分类问题")
        
        if not all(label in [-1, 1] for label in unique_labels):
            # 转换标签
            label_map = {unique_labels[0]: -1, unique_labels[1]: 1}
            y = np.array([label_map[label] for label in y])
            self.label_map = label_map
            self.inverse_label_map = {v: k for k, v in label_map.items()}
        
        n_samples = len(X)
        
        # 初始化样本权重
        weights = np.ones(n_samples) / n_samples
        
        self.estimators = []
        self.estimator_weights = []
        self.estimator_errors = []
        
        for i in range(self.n_estimators):
            # 创建并训练弱分类器
            stump = DecisionStump()
            error, predictions = stump.fit(X, y, weights)
            
            # 如果错误率太高，停止训练
            if error >= self.max_error_rate:
                if len(self.estimators) == 0:
                    # 如果第一个分类器就失败，使用随机分类器
                    self.estimators.append(stump)
                    self.estimator_weights.append(1.0)
                    self.estimator_errors.append(error)
                break
            
            # 计算分类器权重
            alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))
            stump.alpha = alpha
            
            # 保存分类器
            self.estimators.append(stump)
            self.estimator_weights.append(alpha)
            self.estimator_errors.append(error)
            
            # 更新样本权重
            weights = weights * np.exp(-alpha * y * predictions)
            weights = weights / np.sum(weights)  # 归一化
            
            print(f"第{i+1}个分类器: 错误率={error:.4f}, 权重={alpha:.4f}")
        
        print(f"训练完成，共使用{len(self.estimators)}个弱分类器")
        return self
    
    def predict(self, X):
        """
        预测
        
        参数:
        X: 测试特征
        
        返回:
        predictions: 预测标签
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # 加权投票
        ensemble_predictions = np.zeros(n_samples)
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            predictions = estimator.predict(X)
            ensemble_predictions += weight * predictions
        
        # 最终预测
        final_predictions = np.sign(ensemble_predictions)
        final_predictions[final_predictions == 0] = 1  # 处理零值
        
        # 转换回原始标签
        if hasattr(self, 'inverse_label_map'):
            final_predictions = np.array([self.inverse_label_map[pred] for pred in final_predictions])
        
        return final_predictions
    
    def predict_proba(self, X):
        """
        预测概率
        
        参数:
        X: 测试特征
        
        返回:
        probabilities: 预测概率
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # 加权投票
        ensemble_predictions = np.zeros(n_samples)
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            predictions = estimator.predict(X)
            ensemble_predictions += weight * predictions
        
        # 转换为概率
        probabilities = 1 / (1 + np.exp(-2 * ensemble_predictions))
        return probabilities
    
    def staged_predict(self, X):
        """
        阶段性预测，返回每个阶段的预测结果
        
        参数:
        X: 测试特征
        
        返回:
        stage_predictions: 每个阶段的预测结果
        """
        X = np.array(X)
        n_samples = X.shape[0]
        stage_predictions = []
        
        ensemble_predictions = np.zeros(n_samples)
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            predictions = estimator.predict(X)
            ensemble_predictions += weight * predictions
            
            # 当前阶段的预测
            current_predictions = np.sign(ensemble_predictions)
            current_predictions[current_predictions == 0] = 1
            
            # 转换回原始标签
            if hasattr(self, 'inverse_label_map'):
                current_predictions = np.array([self.inverse_label_map[pred] for pred in current_predictions])
            
            stage_predictions.append(current_predictions.copy())
        
        return stage_predictions
    
    def plot_feature_importance(self):
        """
        绘制特征重要性
        """
        if not self.estimators:
            print("还没有训练模型")
            return
        
        # 计算每个特征被使用的频率和权重
        feature_usage = {}
        feature_weights = {}
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            feature_idx = estimator.feature_index
            
            if feature_idx not in feature_usage:
                feature_usage[feature_idx] = 0
                feature_weights[feature_idx] = 0
            
            feature_usage[feature_idx] += 1
            feature_weights[feature_idx] += weight
        
        features = list(feature_usage.keys())
        usage_counts = [feature_usage[f] for f in features]
        importance_weights = [feature_weights[f] for f in features]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 特征使用频率
        ax1.bar(features, usage_counts, alpha=0.7)
        ax1.set_xlabel('特征索引')
        ax1.set_ylabel('使用次数')
        ax1.set_title('特征使用频率')
        ax1.grid(True, alpha=0.3)
        
        # 特征重要性权重
        ax2.bar(features, importance_weights, alpha=0.7, color='orange')
        ax2.set_xlabel('特征索引')
        ax2.set_ylabel('累积权重')
        ax2.set_title('特征重要性权重')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return feature_usage, feature_weights
    
    def plot_learning_curve(self, X, y):
        """
        绘制学习曲线
        
        参数:
        X: 训练特征
        y: 训练标签
        """
        if not self.estimators:
            print("还没有训练模型")
            return
        
        # 获取阶段性预测
        stage_predictions = self.staged_predict(X)
        
        # 计算每个阶段的错误率
        errors = []
        for predictions in stage_predictions:
            # 转换标签用于比较
            if hasattr(self, 'label_map'):
                y_compare = np.array([self.label_map[label] for label in y])
                pred_compare = np.array([self.label_map[pred] for pred in predictions])
            else:
                y_compare = y
                pred_compare = predictions
            
            error = np.mean(y_compare != pred_compare)
            errors.append(error)
        
        # 绘制学习曲线
        plt.figure(figsize=(12, 8))
        
        # 子图1：错误率曲线
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(errors) + 1), errors, 'b-', linewidth=2)
        plt.xlabel('弱分类器数量')
        plt.ylabel('训练错误率')
        plt.title('AdaBoost学习曲线')
        plt.grid(True, alpha=0.3)
        
        # 子图2：分类器权重
        plt.subplot(2, 2, 2)
        plt.bar(range(1, len(self.estimator_weights) + 1), self.estimator_weights, alpha=0.7)
        plt.xlabel('弱分类器索引')
        plt.ylabel('分类器权重')
        plt.title('弱分类器权重分布')
        plt.grid(True, alpha=0.3)
        
        # 子图3：分类器错误率
        plt.subplot(2, 2, 3)
        plt.bar(range(1, len(self.estimator_errors) + 1), self.estimator_errors, alpha=0.7, color='red')
        plt.xlabel('弱分类器索引')
        plt.ylabel('分类器错误率')
        plt.title('弱分类器错误率')
        plt.axhline(y=0.5, color='black', linestyle='--', label='随机分类线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图4：累积权重
        cumulative_weights = np.cumsum(self.estimator_weights)
        plt.subplot(2, 2, 4)
        plt.plot(range(1, len(cumulative_weights) + 1), cumulative_weights, 'g-', linewidth=2)
        plt.xlabel('弱分类器数量')
        plt.ylabel('累积权重')
        plt.title('累积分类器权重')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return errors

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