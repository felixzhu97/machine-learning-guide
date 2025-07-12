"""
购物篮分析案例
使用Apriori算法进行市场篮子分析，发现商品之间的关联规则
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from apriori import Apriori, create_sample_transactions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class MarketBasketAnalysis:
    """
    购物篮分析系统
    """
    
    def __init__(self):
        self.apriori = None
        self.transactions = None
        self.items_df = None
        
    def create_supermarket_data(self, n_transactions=1000):
        """
        创建超市购物数据
        
        参数:
        n_transactions: 事务数量
        """
        np.random.seed(42)
        
        # 定义商品及其基础购买概率
        items_info = {
            '牛奶': 0.6,
            '面包': 0.5,
            '鸡蛋': 0.4,
            '黄油': 0.3,
            '奶酪': 0.25,
            '酸奶': 0.35,
            '麦片': 0.2,
            '果汁': 0.3,
            '咖啡': 0.25,
            '茶叶': 0.2,
            '饼干': 0.15,
            '巧克力': 0.18,
            '水果': 0.4,
            '蔬菜': 0.45,
            '肉类': 0.35
        }
        
        # 定义关联规则（某些商品经常一起购买）
        associations = {
            '牛奶': ['麦片', '咖啡', '饼干'],
            '面包': ['黄油', '奶酪', '果汁'],
            '鸡蛋': ['黄油', '牛奶', '面包'],
            '咖啡': ['牛奶', '饼干'],
            '茶叶': ['饼干', '牛奶'],
            '巧克力': ['牛奶', '饼干'],
            '水果': ['酸奶', '麦片'],
            '蔬菜': ['肉类'],
            '肉类': ['蔬菜']
        }
        
        self.transactions = []
        
        for _ in range(n_transactions):
            transaction = []
            
            # 基于基础概率选择商品
            for item, prob in items_info.items():
                if np.random.random() < prob:
                    transaction.append(item)
            
            # 添加关联商品
            items_to_check = transaction.copy()
            for item in items_to_check:
                if item in associations:
                    for associated_item in associations[item]:
                        # 如果该商品不在购物车中，有一定概率添加
                        if associated_item not in transaction and np.random.random() < 0.4:
                            transaction.append(associated_item)
            
            # 确保每个事务至少有1个商品
            if not transaction:
                transaction = [np.random.choice(list(items_info.keys()))]
            
            self.transactions.append(transaction)
        
        print(f"生成了 {len(self.transactions)} 个购物事务")
        print(f"平均每个事务包含 {np.mean([len(t) for t in self.transactions]):.2f} 个商品")
        
        return self.transactions
    
    def analyze_data_distribution(self):
        """
        分析数据分布
        """
        if not self.transactions:
            print("请先生成数据")
            return
        
        # 计算商品频率
        all_items = [item for transaction in self.transactions for item in transaction]
        item_counts = Counter(all_items)
        
        # 计算事务长度分布
        transaction_lengths = [len(transaction) for transaction in self.transactions]
        
        # 创建数据框
        self.items_df = pd.DataFrame([
            {'商品': item, '购买次数': count, '支持度': count / len(self.transactions)}
            for item, count in item_counts.items()
        ]).sort_values('购买次数', ascending=False)
        
        print("=== 数据分布分析 ===")
        print(f"总事务数: {len(self.transactions)}")
        print(f"不同商品数: {len(item_counts)}")
        print(f"商品总购买次数: {len(all_items)}")
        
        print(f"\n最受欢迎的商品:")
        print(self.items_df.head(10).to_string(index=False))
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 商品购买频率
        ax1 = axes[0, 0]
        top_items = self.items_df.head(10)
        ax1.barh(top_items['商品'], top_items['购买次数'])
        ax1.set_title('商品购买频率 (Top 10)')
        ax1.set_xlabel('购买次数')
        
        # 事务长度分布
        ax2 = axes[0, 1]
        ax2.hist(transaction_lengths, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('事务长度分布')
        ax2.set_xlabel('事务中商品数量')
        ax2.set_ylabel('频率')
        
        # 支持度分布
        ax3 = axes[1, 0]
        ax3.hist(self.items_df['支持度'], bins=15, alpha=0.7, edgecolor='black')
        ax3.set_title('商品支持度分布')
        ax3.set_xlabel('支持度')
        ax3.set_ylabel('商品数量')
        
        # 累积支持度
        ax4 = axes[1, 1]
        sorted_support = self.items_df['支持度'].sort_values(ascending=False)
        cumulative_support = np.cumsum(sorted_support)
        ax4.plot(range(1, len(cumulative_support) + 1), cumulative_support)
        ax4.set_title('累积支持度曲线')
        ax4.set_xlabel('商品排名')
        ax4.set_ylabel('累积支持度')
        
        plt.tight_layout()
        plt.show()
        
        return self.items_df
    
    def run_apriori_analysis(self, min_support=0.1, min_confidence=0.6):
        """
        运行Apriori分析
        
        参数:
        min_support: 最小支持度
        min_confidence: 最小置信度
        """
        print(f"\n=== 运行Apriori算法 ===")
        print(f"最小支持度: {min_support}")
        print(f"最小置信度: {min_confidence}")
        
        # 创建Apriori实例
        self.apriori = Apriori(min_support=min_support, min_confidence=min_confidence)
        
        # 挖掘频繁项集
        frequent_itemsets, support_data = self.apriori.fit(self.transactions)
        
        # 生成关联规则
        association_rules = self.apriori.generate_rules(min_confidence)
        
        # 打印结果
        self.apriori.print_frequent_itemsets(max_display=8)
        self.apriori.print_association_rules(max_display=15)
        
        return frequent_itemsets, association_rules
    
    def parameter_sensitivity_analysis(self):
        """
        参数敏感性分析
        """
        print(f"\n=== 参数敏感性分析 ===")
        
        # 测试不同的支持度阈值
        support_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        support_results = []
        confidence_results = []
        
        # 支持度敏感性
        print("分析支持度敏感性...")
        for min_sup in support_thresholds:
            apriori_temp = Apriori(min_support=min_sup, min_confidence=0.6)
            frequent_itemsets, _ = apriori_temp.fit(self.transactions)
            rules = apriori_temp.generate_rules()
            
            total_itemsets = sum(len(itemsets) for itemsets in frequent_itemsets.values())
            
            support_results.append({
                'min_support': min_sup,
                'frequent_itemsets': total_itemsets,
                'association_rules': len(rules)
            })
            
            print(f"  支持度 {min_sup}: {total_itemsets} 个频繁项集, {len(rules)} 条规则")
        
        # 置信度敏感性
        print("\n分析置信度敏感性...")
        for min_conf in confidence_thresholds:
            apriori_temp = Apriori(min_support=0.1, min_confidence=min_conf)
            frequent_itemsets, _ = apriori_temp.fit(self.transactions)
            rules = apriori_temp.generate_rules()
            
            confidence_results.append({
                'min_confidence': min_conf,
                'association_rules': len(rules)
            })
            
            print(f"  置信度 {min_conf}: {len(rules)} 条规则")
        
        # 可视化结果
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 支持度 vs 频繁项集数量
        support_df = pd.DataFrame(support_results)
        ax1 = axes[0]
        ax1.plot(support_df['min_support'], support_df['frequent_itemsets'], 'o-', linewidth=2)
        ax1.set_xlabel('最小支持度')
        ax1.set_ylabel('频繁项集数量')
        ax1.set_title('支持度 vs 频繁项集数量')
        ax1.grid(True, alpha=0.3)
        
        # 支持度 vs 关联规则数量
        ax2 = axes[1]
        ax2.plot(support_df['min_support'], support_df['association_rules'], 's-', linewidth=2, color='red')
        ax2.set_xlabel('最小支持度')
        ax2.set_ylabel('关联规则数量')
        ax2.set_title('支持度 vs 关联规则数量')
        ax2.grid(True, alpha=0.3)
        
        # 置信度 vs 关联规则数量
        confidence_df = pd.DataFrame(confidence_results)
        ax3 = axes[2]
        ax3.plot(confidence_df['min_confidence'], confidence_df['association_rules'], '^-', linewidth=2, color='green')
        ax3.set_xlabel('最小置信度')
        ax3.set_ylabel('关联规则数量')
        ax3.set_title('置信度 vs 关联规则数量')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return support_results, confidence_results
    
    def visualize_association_network(self, top_rules=20):
        """
        可视化关联规则网络
        
        参数:
        top_rules: 显示的规则数量
        """
        if not self.apriori or not self.apriori.association_rules:
            print("请先运行Apriori分析")
            return
        
        import networkx as nx
        
        print(f"\n=== 关联规则网络可视化 ===")
        
        # 创建网络图
        G = nx.DiGraph()
        
        # 添加节点和边
        for rule in self.apriori.association_rules[:top_rules]:
            antecedent_str = ', '.join(sorted(list(rule['antecedent'])))
            consequent_str = ', '.join(sorted(list(rule['consequent'])))
            
            # 添加边，权重为置信度
            G.add_edge(antecedent_str, consequent_str, 
                      weight=rule['confidence'], 
                      support=rule['support'],
                      lift=rule['lift'])
        
        # 设置布局
        plt.figure(figsize=(15, 12))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 绘制节点
        node_sizes = [len(self.apriori.get_rules_for_item(node)) * 100 + 300 
                     for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.7)
        
        # 绘制边，粗细表示置信度
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, 
                              edge_color='gray', arrows=True, arrowsize=20)
        
        # 添加标签
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title(f'关联规则网络图 (Top {top_rules} 规则)', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return G
    
    def generate_recommendations(self, customer_basket):
        """
        基于购物篮生成推荐
        
        参数:
        customer_basket: 客户当前购物篮
        
        返回:
        recommendations: 推荐商品列表
        """
        if not self.apriori or not self.apriori.association_rules:
            print("请先运行Apriori分析")
            return []
        
        print(f"\n=== 基于购物篮的推荐 ===")
        print(f"当前购物篮: {customer_basket}")
        
        recommendations = {}
        
        # 寻找匹配的规则
        for rule in self.apriori.association_rules:
            # 检查前件是否被当前购物篮包含
            if rule['antecedent'].issubset(set(customer_basket)):
                for item in rule['consequent']:
                    if item not in customer_basket:
                        if item not in recommendations:
                            recommendations[item] = {
                                'confidence': 0,
                                'support': 0,
                                'lift': 0,
                                'rule_count': 0
                            }
                        
                        # 累积推荐分数
                        recommendations[item]['confidence'] = max(
                            recommendations[item]['confidence'], rule['confidence']
                        )
                        recommendations[item]['support'] += rule['support']
                        recommendations[item]['lift'] = max(
                            recommendations[item]['lift'], rule['lift']
                        )
                        recommendations[item]['rule_count'] += 1
        
        # 按综合分数排序
        for item in recommendations:
            score = (recommendations[item]['confidence'] * 0.4 + 
                    recommendations[item]['support'] * 0.3 + 
                    recommendations[item]['lift'] * 0.3)
            recommendations[item]['score'] = score
        
        # 排序并返回
        sorted_recommendations = sorted(recommendations.items(), 
                                      key=lambda x: x[1]['score'], reverse=True)
        
        print(f"\n推荐商品:")
        for i, (item, metrics) in enumerate(sorted_recommendations[:5], 1):
            print(f"{i}. {item}")
            print(f"   置信度: {metrics['confidence']:.3f}")
            print(f"   支持度: {metrics['support']:.3f}")
            print(f"   提升度: {metrics['lift']:.3f}")
            print(f"   综合分数: {metrics['score']:.3f}")
            print()
        
        return sorted_recommendations
    
    def market_insights(self):
        """
        市场洞察分析
        """
        if not self.apriori:
            print("请先运行Apriori分析")
            return
        
        print(f"\n=== 市场洞察分析 ===")
        
        # 统计信息
        stats = self.apriori.get_statistics()
        print(f"挖掘统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 最强关联规则
        if self.apriori.association_rules:
            print(f"\n最强关联规则 (按提升度排序):")
            rules_by_lift = sorted(self.apriori.association_rules, 
                                 key=lambda x: x['lift'], reverse=True)
            
            for i, rule in enumerate(rules_by_lift[:5], 1):
                antecedent_str = ', '.join(sorted(list(rule['antecedent'])))
                consequent_str = ', '.join(sorted(list(rule['consequent'])))
                print(f"{i}. {{{antecedent_str}}} => {{{consequent_str}}}")
                print(f"   提升度: {rule['lift']:.3f} (表示关联性是随机的{rule['lift']:.1f}倍)")
                print()
        
        # 商品关联度分析
        print(f"热门商品的关联分析:")
        if self.items_df is not None:
            top_items = self.items_df.head(5)['商品'].tolist()
            
            for item in top_items:
                antecedent_rules, consequent_rules = self.apriori.analyze_item_associations(item)

def main():
    """
    主函数
    """
    print("=== 购物篮分析系统 (Apriori算法) ===")
    
    # 创建分析系统
    market_analysis = MarketBasketAnalysis()
    
    # 生成超市数据
    transactions = market_analysis.create_supermarket_data(1500)
    
    # 分析数据分布
    items_df = market_analysis.analyze_data_distribution()
    
    # 运行Apriori分析
    frequent_itemsets, rules = market_analysis.run_apriori_analysis(
        min_support=0.1, min_confidence=0.6
    )
    
    # 参数敏感性分析
    support_results, confidence_results = market_analysis.parameter_sensitivity_analysis()
    
    # 可视化关联网络
    network = market_analysis.visualize_association_network(top_rules=15)
    
    # 推荐系统测试
    test_baskets = [
        ['牛奶', '面包'],
        ['咖啡', '牛奶'],
        ['水果', '酸奶'],
        ['肉类', '蔬菜']
    ]
    
    for basket in test_baskets:
        recommendations = market_analysis.generate_recommendations(basket)
    
    # 市场洞察
    market_analysis.market_insights()
    
    print(f"\n=== Apriori算法总结 ===")
    print("1. Apriori使用频繁项集挖掘关联规则")
    print("2. 支持度衡量项集出现的频率")
    print("3. 置信度衡量规则的可靠性")
    print("4. 提升度衡量规则的有效性")
    print("5. 可用于商品推荐、交叉销售等场景")

if __name__ == "__main__":
    main() 