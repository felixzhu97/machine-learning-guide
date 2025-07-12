"""
垃圾邮件分类案例
使用朴素贝叶斯算法分类垃圾邮件
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from naive_bayes import NaiveBayes, TextProcessor, calculate_accuracy, calculate_precision_recall
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.visualization import plot_2d_scatter
from utils.data_utils import plot_confusion_matrix

class SpamFilterSystem:
    """
    垃圾邮件过滤系统
    """
    
    def __init__(self):
        self.naive_bayes = None
        self.text_processor = TextProcessor()
        self.emails = None
        self.labels = None
        self.vocabulary = None
        self.X_matrix = None
        
    def create_sample_data(self):
        """
        创建样本邮件数据
        """
        # 正常邮件样本
        normal_emails = [
            "Hi John, how are you doing today? I hope you're having a great day.",
            "Meeting scheduled for tomorrow at 3 PM in conference room A.",
            "Please review the attached document and send your feedback by Friday.",
            "Happy birthday! Hope you have a wonderful celebration.",
            "The project deadline has been extended to next Monday.",
            "Thank you for your help with the presentation yesterday.",
            "Let's grab lunch together this Friday if you're available.",
            "The weather forecast shows rain for the weekend.",
            "I enjoyed our conversation about the new product features.",
            "Please confirm your attendance for the team building event.",
            "The quarterly report is ready for your review.",
            "Congratulations on your promotion! Well deserved.",
            "The training session has been rescheduled to next week.",
            "I need your input on the budget proposal by tomorrow.",
            "The client meeting went very well yesterday.",
            "Please update the project status in the shared document.",
            "The new office space looks great after the renovation.",
            "Can you help me with the technical documentation?",
            "The conference registration deadline is approaching.",
            "I'll be working from home tomorrow due to a family event.",
        ]
        
        # 垃圾邮件样本
        spam_emails = [
            "URGENT! You have won $1,000,000! Click here to claim your prize NOW!",
            "FREE MONEY! No strings attached! Get rich quick scheme that actually works!",
            "VIAGRA CHEAP! Best prices guaranteed! No prescription needed!",
            "Make $5000 per week working from home! No experience required!",
            "CONGRATULATIONS! You are our lucky winner! Claim your free iPhone now!",
            "GET RICH QUICK! Investment opportunity of a lifetime! Act now!",
            "FREE LOAN APPROVED! Bad credit OK! Get money instantly!",
            "LOSE WEIGHT FAST! Miracle diet pill! No exercise needed!",
            "CASINO BONUS! Free $500 to play! No deposit required!",
            "CREDIT CARD DEBT RELIEF! We can help you eliminate all debt!",
            "WORK FROM HOME! Earn $3000 weekly! No skills required!",
            "CHEAP MEDICATION! Save up to 80% on prescriptions!",
            "LOTTERY WINNER! You have won millions! Contact us immediately!",
            "MAKE MONEY ONLINE! Easy profits! Start earning today!",
            "FREE TRIAL! Risk-free offer! Limited time only!",
            "REFINANCE NOW! Lowest rates ever! Save thousands!",
            "DIET PILLS! Lose 30 pounds in 30 days! Guaranteed results!",
            "INHERITANCE MONEY! You have inherited millions from unknown relative!",
            "INVESTMENT OPPORTUNITY! Double your money in 30 days!",
            "FREE GIFT! Claim your prize! No purchase necessary!",
        ]
        
        # 组合数据
        self.emails = normal_emails + spam_emails
        self.labels = np.array([0] * len(normal_emails) + [1] * len(spam_emails))
        
        print(f"创建了 {len(self.emails)} 封邮件")
        print(f"正常邮件: {len(normal_emails)} 封")
        print(f"垃圾邮件: {len(spam_emails)} 封")
        
        return self.emails, self.labels
    
    def preprocess_data(self, min_freq=2):
        """
        预处理数据
        
        参数:
        min_freq: 最小词频
        """
        # 构建词汇表
        self.vocabulary = self.text_processor.build_vocabulary(self.emails, min_freq)
        print(f"词汇表大小: {len(self.vocabulary)}")
        
        # 转换为词频矩阵
        self.X_matrix = self.text_processor.documents_to_matrix(self.emails)
        print(f"特征矩阵形状: {self.X_matrix.shape}")
        
        return self.X_matrix, self.vocabulary
    
    def train_model(self, test_size=0.2, alpha=1.0):
        """
        训练模型
        
        参数:
        test_size: 测试集比例
        alpha: 拉普拉斯平滑参数
        """
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_matrix, self.labels, test_size=test_size, random_state=42
        )
        
        # 创建并训练朴素贝叶斯模型
        self.naive_bayes = NaiveBayes(alpha=alpha)
        self.naive_bayes.fit(X_train, y_train)
        
        # 预测
        y_pred_train = self.naive_bayes.predict(X_train)
        y_pred_test = self.naive_bayes.predict(X_test)
        
        # 计算准确率
        train_accuracy = calculate_accuracy(y_train, y_pred_train)
        test_accuracy = calculate_accuracy(y_test, y_pred_test)
        
        # 计算精确率和召回率
        precision, recall = calculate_precision_recall(y_test, y_pred_test, positive_class=1)
        
        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")
        print(f"垃圾邮件检测精确率: {precision:.4f}")
        print(f"垃圾邮件检测召回率: {recall:.4f}")
        
        return X_train, X_test, y_train, y_test, y_pred_test, test_accuracy
    
    def predict_email(self, email_text):
        """
        预测单封邮件
        
        参数:
        email_text: 邮件文本
        
        返回:
        prediction: 预测结果
        probability: 预测概率
        """
        if self.naive_bayes is None:
            print("请先训练模型")
            return None, None
        
        # 转换邮件为特征向量
        email_matrix = self.text_processor.documents_to_matrix([email_text])
        
        # 预测
        prediction = self.naive_bayes.predict(email_matrix)[0]
        probabilities = self.naive_bayes.predict_proba(email_matrix)[0]
        
        result = "垃圾邮件" if prediction == 1 else "正常邮件"
        spam_prob = probabilities.get(1, 0)
        
        print(f"邮件内容: {email_text}")
        print(f"预测结果: {result}")
        print(f"垃圾邮件概率: {spam_prob:.4f}")
        
        return prediction, probabilities
    
    def analyze_word_importance(self, top_n=10):
        """
        分析词汇重要性
        
        参数:
        top_n: 显示前N个重要词汇
        """
        if self.naive_bayes is None:
            print("请先训练模型")
            return
        
        # 计算词汇在垃圾邮件和正常邮件中的概率比
        spam_probs = self.naive_bayes.feature_probs[1]  # 垃圾邮件类别
        normal_probs = self.naive_bayes.feature_probs[0]  # 正常邮件类别
        
        # 计算概率比（垃圾邮件概率 / 正常邮件概率）
        prob_ratios = spam_probs / normal_probs
        
        # 获取最能指示垃圾邮件的词汇
        spam_indicators = np.argsort(prob_ratios)[-top_n:]
        spam_words = [self.vocabulary[i] for i in spam_indicators]
        spam_ratios = prob_ratios[spam_indicators]
        
        # 获取最能指示正常邮件的词汇
        normal_indicators = np.argsort(prob_ratios)[:top_n]
        normal_words = [self.vocabulary[i] for i in normal_indicators]
        normal_ratios = prob_ratios[normal_indicators]
        
        print(f"\n=== 垃圾邮件指示词汇 (前{top_n}个) ===")
        for word, ratio in zip(spam_words, spam_ratios):
            print(f"{word}: {ratio:.4f}")
        
        print(f"\n=== 正常邮件指示词汇 (前{top_n}个) ===")
        for word, ratio in zip(normal_words, normal_ratios):
            print(f"{word}: {ratio:.4f}")
        
        # 可视化词汇重要性
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 垃圾邮件指示词汇
        ax1.barh(range(len(spam_words)), spam_ratios, color='red', alpha=0.7)
        ax1.set_yticks(range(len(spam_words)))
        ax1.set_yticklabels(spam_words)
        ax1.set_xlabel('概率比 (垃圾邮件/正常邮件)')
        ax1.set_title('垃圾邮件指示词汇')
        
        # 正常邮件指示词汇
        ax2.barh(range(len(normal_words)), normal_ratios, color='blue', alpha=0.7)
        ax2.set_yticks(range(len(normal_words)))
        ax2.set_yticklabels(normal_words)
        ax2.set_xlabel('概率比 (垃圾邮件/正常邮件)')
        ax2.set_title('正常邮件指示词汇')
        
        plt.tight_layout()
        plt.show()
    
    def test_various_emails(self):
        """
        测试各种邮件
        """
        print("\n=== 测试各种邮件 ===")
        
        test_emails = [
            "Congratulations! You have won a lottery prize of $100,000! Click here to claim now!",
            "Hi Sarah, can you please review the quarterly report and send me your feedback?",
            "FREE MONEY! No investment required! Get rich quick!",
            "The project meeting is scheduled for next Monday at 2 PM.",
            "URGENT! Your account will be closed unless you verify your information immediately!",
            "Thank you for your presentation yesterday. It was very informative.",
            "CHEAP VIAGRA! Best prices online! No prescription needed!",
            "Please confirm your attendance for the company holiday party.",
        ]
        
        for email in test_emails:
            print("\n" + "="*50)
            self.predict_email(email)
    
    def compare_smoothing_parameters(self):
        """
        比较不同平滑参数的效果
        """
        print("\n=== 比较不同平滑参数 ===")
        
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
        accuracies = []
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_matrix, self.labels, test_size=0.2, random_state=42
        )
        
        for alpha in alphas:
            nb = NaiveBayes(alpha=alpha)
            nb.fit(X_train, y_train)
            y_pred = nb.predict(X_test)
            accuracy = calculate_accuracy(y_test, y_pred)
            accuracies.append(accuracy)
            print(f"α = {alpha}: 准确率 = {accuracy:.4f}")
        
        # 可视化比较结果
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, accuracies, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('拉普拉斯平滑参数 α')
        plt.ylabel('测试集准确率')
        plt.title('不同平滑参数的模型性能')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        best_alpha = alphas[np.argmax(accuracies)]
        print(f"\n最佳平滑参数: α = {best_alpha}")
        
        return best_alpha

def main():
    """
    主函数
    """
    print("=== 垃圾邮件过滤系统 ===")
    
    # 创建垃圾邮件过滤系统
    spam_filter = SpamFilterSystem()
    
    # 创建样本数据
    emails, labels = spam_filter.create_sample_data()
    
    # 预处理数据
    X_matrix, vocabulary = spam_filter.preprocess_data(min_freq=1)
    
    # 比较不同平滑参数
    best_alpha = spam_filter.compare_smoothing_parameters()
    
    # 使用最佳参数训练模型
    print(f"\n=== 使用最佳平滑参数 α = {best_alpha} 训练模型 ===")
    X_train, X_test, y_train, y_test, y_pred_test, test_accuracy = spam_filter.train_model(alpha=best_alpha)
    
    # 分析词汇重要性
    spam_filter.analyze_word_importance(top_n=10)
    
    # 测试各种邮件
    spam_filter.test_various_emails()
    
    # 绘制混淆矩阵
    from utils.data_utils import plot_confusion_matrix
    plot_confusion_matrix(y_test, y_pred_test, class_names=['正常邮件', '垃圾邮件'])
    
    # 交互式测试
    print("\n=== 交互式测试 ===")
    print("输入邮件内容进行垃圾邮件检测:")
    
    # 示例邮件
    example_emails = [
        "Hello, this is a reminder about our meeting tomorrow at 10 AM.",
        "CONGRATULATIONS!!! You have won $1,000,000! Click here NOW!",
        "Please find the attached report for your review.",
        "FREE MONEY! No investment required! Get rich quick!",
        "The weather is nice today. How about going for a walk?",
    ]
    
    for email in example_emails:
        print(f"\n示例邮件: {email}")
        spam_filter.predict_email(email)

if __name__ == "__main__":
    main() 