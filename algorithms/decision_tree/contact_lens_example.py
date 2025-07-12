"""
隐形眼镜类型预测案例
使用决策树算法根据用户特征预测隐形眼镜类型
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decision_tree import DecisionTree, calculate_accuracy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.visualization import plot_2d_scatter
from utils.data_utils import encode_labels

class ContactLensPredictionSystem:
    """
    隐形眼镜类型预测系统
    """
    
    def __init__(self):
        self.decision_tree = None
        self.feature_names = ['年龄', '视力', '散光', '泪腺分泌']
        self.class_names = ['不适合', '软性', '硬性']
        self.data = None
        self.labels = None
        self.feature_encoders = {}
        self.label_encoder = None
        
    def create_sample_data(self):
        """
        创建隐形眼镜数据集
        """
        # 原始数据
        raw_data = [
            # 年龄, 视力, 散光, 泪腺分泌, 隐形眼镜类型
            ['青年', '近视', '无', '少', '不适合'],
            ['青年', '近视', '无', '多', '软性'],
            ['青年', '近视', '有', '少', '不适合'],
            ['青年', '近视', '有', '多', '硬性'],
            ['青年', '正常', '无', '少', '不适合'],
            ['青年', '正常', '无', '多', '软性'],
            ['青年', '正常', '有', '少', '不适合'],
            ['青年', '正常', '有', '多', '硬性'],
            ['中年', '近视', '无', '少', '不适合'],
            ['中年', '近视', '无', '多', '软性'],
            ['中年', '近视', '有', '少', '不适合'],
            ['中年', '近视', '有', '多', '硬性'],
            ['中年', '正常', '无', '少', '不适合'],
            ['中年', '正常', '无', '多', '软性'],
            ['中年', '正常', '有', '少', '不适合'],
            ['中年', '正常', '有', '多', '不适合'],
            ['老年', '近视', '无', '少', '不适合'],
            ['老年', '近视', '无', '多', '不适合'],
            ['老年', '近视', '有', '少', '不适合'],
            ['老年', '近视', '有', '多', '硬性'],
            ['老年', '正常', '无', '少', '不适合'],
            ['老年', '正常', '无', '多', '软性'],
            ['老年', '正常', '有', '少', '不适合'],
            ['老年', '正常', '有', '多', '不适合'],
        ]
        
        # 转换为DataFrame
        df = pd.DataFrame(raw_data, columns=self.feature_names + ['隐形眼镜类型'])
        
        # 编码特征
        self.data = np.zeros((len(df), len(self.feature_names)))
        
        # 年龄编码
        age_mapping = {'青年': 0, '中年': 1, '老年': 2}
        self.data[:, 0] = df['年龄'].map(age_mapping).values
        
        # 视力编码
        vision_mapping = {'近视': 0, '正常': 1}
        self.data[:, 1] = df['视力'].map(vision_mapping).values
        
        # 散光编码
        astigmatism_mapping = {'无': 0, '有': 1}
        self.data[:, 2] = df['散光'].map(astigmatism_mapping).values
        
        # 泪腺分泌编码
        tear_mapping = {'少': 0, '多': 1}
        self.data[:, 3] = df['泪腺分泌'].map(tear_mapping).values
        
        # 标签编码
        label_mapping = {'不适合': 0, '软性': 1, '硬性': 2}
        self.labels = df['隐形眼镜类型'].map(label_mapping).values
        
        # 保存编码映射
        self.feature_mappings = {
            '年龄': age_mapping,
            '视力': vision_mapping,
            '散光': astigmatism_mapping,
            '泪腺分泌': tear_mapping
        }
        self.label_mapping = label_mapping
        
        print(f"创建了 {len(self.data)} 个样本")
        print(f"特征数量: {len(self.feature_names)}")
        print(f"类别分布: {np.bincount(self.labels)}")
        
        return df
    
    def train_model(self, max_depth=10, criterion='entropy'):
        """
        训练决策树模型
        
        参数:
        max_depth: 最大深度
        criterion: 分割标准
        """
        self.decision_tree = DecisionTree(
            max_depth=max_depth,
            criterion=criterion
        )
        
        self.decision_tree.fit(
            self.data, 
            self.labels,
            feature_names=self.feature_names,
            class_names=self.class_names
        )
        
        # 计算训练准确率
        y_pred = self.decision_tree.predict(self.data)
        accuracy = calculate_accuracy(self.labels, y_pred)
        
        print(f"训练完成，准确率: {accuracy:.4f}")
        
        return accuracy
    
    def predict_person(self, age, vision, astigmatism, tear_production):
        """
        预测单个用户
        
        参数:
        age: 年龄 ('青年', '中年', '老年')
        vision: 视力 ('近视', '正常')
        astigmatism: 散光 ('无', '有')
        tear_production: 泪腺分泌 ('少', '多')
        
        返回:
        prediction: 预测结果
        """
        if self.decision_tree is None:
            print("请先训练模型")
            return None
        
        # 编码输入特征
        try:
            encoded_features = np.array([
                self.feature_mappings['年龄'][age],
                self.feature_mappings['视力'][vision],
                self.feature_mappings['散光'][astigmatism],
                self.feature_mappings['泪腺分泌'][tear_production]
            ]).reshape(1, -1)
            
            # 预测
            prediction = self.decision_tree.predict(encoded_features)[0]
            
            print(f"用户特征: 年龄={age}, 视力={vision}, 散光={astigmatism}, 泪腺分泌={tear_production}")
            print(f"预测结果: {self.class_names[prediction]}")
            
            return prediction
        except KeyError as e:
            print(f"无效的特征值: {e}")
            return None
    
    def print_decision_tree(self):
        """
        打印决策树结构
        """
        if self.decision_tree is None:
            print("请先训练模型")
            return
        
        print("\n=== 决策树结构 ===")
        self.decision_tree.print_tree()
    
    def analyze_feature_importance(self):
        """
        分析特征重要性
        """
        if self.decision_tree is None:
            print("请先训练模型")
            return
        
        importance = self.decision_tree.get_feature_importance()
        
        print("\n=== 特征重要性分析 ===")
        for i, (feature, imp) in enumerate(zip(self.feature_names, importance)):
            print(f"{feature}: {imp:.4f}")
        
        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        plt.bar(self.feature_names, importance, color='skyblue')
        plt.title('特征重要性')
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return importance
    
    def visualize_data(self, df):
        """
        可视化数据分布
        
        参数:
        df: 原始数据DataFrame
        """
        print("\n=== 数据分布可视化 ===")
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 各特征的分布
        for i, feature in enumerate(self.feature_names):
            ax = axes[i//2, i%2]
            
            # 统计各类别在该特征下的分布
            feature_class_counts = df.groupby([feature, '隐形眼镜类型']).size().unstack(fill_value=0)
            
            feature_class_counts.plot(kind='bar', ax=ax, stacked=True)
            ax.set_title(f'{feature}特征分布')
            ax.set_xlabel(feature)
            ax.set_ylabel('数量')
            ax.legend(title='隐形眼镜类型')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # 类别分布饼图
        plt.figure(figsize=(8, 6))
        class_counts = df['隐形眼镜类型'].value_counts()
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        plt.title('隐形眼镜类型分布')
        plt.show()
    
    def test_various_cases(self):
        """
        测试各种情况
        """
        print("\n=== 测试各种情况 ===")
        
        test_cases = [
            ('青年', '近视', '无', '多'),
            ('中年', '正常', '有', '多'),
            ('老年', '近视', '有', '少'),
            ('青年', '正常', '无', '少'),
            ('中年', '近视', '无', '多'),
            ('老年', '正常', '无', '多'),
        ]
        
        for age, vision, astigmatism, tear in test_cases:
            self.predict_person(age, vision, astigmatism, tear)
            print()
    
    def compare_criteria(self):
        """
        比较不同分割标准
        """
        print("\n=== 比较不同分割标准 ===")
        
        criteria = ['entropy', 'gini']
        results = {}
        
        for criterion in criteria:
            print(f"\n使用 {criterion} 标准:")
            
            # 创建并训练模型
            dt = DecisionTree(criterion=criterion)
            dt.fit(self.data, self.labels, 
                  feature_names=self.feature_names,
                  class_names=self.class_names)
            
            # 计算准确率
            y_pred = dt.predict(self.data)
            accuracy = calculate_accuracy(self.labels, y_pred)
            
            results[criterion] = accuracy
            print(f"准确率: {accuracy:.4f}")
        
        # 可视化比较结果
        plt.figure(figsize=(8, 6))
        plt.bar(results.keys(), results.values(), color=['blue', 'orange'])
        plt.title('不同分割标准的准确率比较')
        plt.xlabel('分割标准')
        plt.ylabel('准确率')
        plt.ylim(0, 1)
        for criterion, accuracy in results.items():
            plt.text(criterion, accuracy + 0.01, f'{accuracy:.4f}', 
                    ha='center', va='bottom')
        plt.show()
        
        return results

def main():
    """
    主函数
    """
    print("=== 隐形眼镜类型预测系统 ===")
    
    # 创建预测系统
    lens_system = ContactLensPredictionSystem()
    
    # 创建数据
    df = lens_system.create_sample_data()
    
    # 可视化数据
    lens_system.visualize_data(df)
    
    # 比较不同分割标准
    lens_system.compare_criteria()
    
    # 使用信息熵训练模型
    print("\n=== 使用信息熵训练模型 ===")
    lens_system.train_model(criterion='entropy')
    
    # 打印决策树
    lens_system.print_decision_tree()
    
    # 分析特征重要性
    lens_system.analyze_feature_importance()
    
    # 测试各种情况
    lens_system.test_various_cases()
    
    # 交互式预测
    print("\n=== 交互式预测 ===")
    print("输入用户特征进行预测:")
    print("年龄选项: 青年, 中年, 老年")
    print("视力选项: 近视, 正常")
    print("散光选项: 无, 有")
    print("泪腺分泌选项: 少, 多")
    
    # 示例预测
    example_cases = [
        ("青年用户，近视，无散光，泪腺分泌多", '青年', '近视', '无', '多'),
        ("中年用户，正常视力，有散光，泪腺分泌多", '中年', '正常', '有', '多'),
        ("老年用户，近视，有散光，泪腺分泌少", '老年', '近视', '有', '少'),
    ]
    
    for description, age, vision, astigmatism, tear in example_cases:
        print(f"\n{description}:")
        lens_system.predict_person(age, vision, astigmatism, tear)

if __name__ == "__main__":
    main() 