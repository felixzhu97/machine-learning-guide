"""
Apriori算法实现
用于挖掘频繁项集和关联规则
"""
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict

class Apriori:
    """
    Apriori关联规则挖掘算法
    """
    
    def __init__(self, min_support=0.5, min_confidence=0.75):
        """
        初始化Apriori算法
        
        参数:
        min_support: 最小支持度阈值
        min_confidence: 最小置信度阈值
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
        self.association_rules = []
        self.transaction_list = None
        
    def _create_C1(self, transaction_list):
        """
        创建候选1-项集
        
        参数:
        transaction_list: 事务列表
        
        返回:
        C1: 候选1-项集
        """
        C1 = set()
        for transaction in transaction_list:
            for item in transaction:
                C1.add(frozenset([item]))
        return list(C1)
    
    def _scan_D(self, transaction_list, Ck, min_support):
        """
        扫描数据库，计算候选项集的支持度
        
        参数:
        transaction_list: 事务列表
        Ck: 候选k-项集
        min_support: 最小支持度
        
        返回:
        Lk: 频繁k-项集
        support_data: 支持度数据
        """
        ss_cnt = defaultdict(int)
        
        # 计算每个候选项集在事务中出现的次数
        for transaction in transaction_list:
            transaction_set = set(transaction)
            for candidate in Ck:
                if candidate.issubset(transaction_set):
                    ss_cnt[candidate] += 1
        
        num_items = len(transaction_list)
        Lk = []
        support_data = {}
        
        # 筛选满足最小支持度的项集
        for candidate in ss_cnt:
            support = ss_cnt[candidate] / num_items
            if support >= min_support:
                Lk.append(candidate)
            support_data[candidate] = support
        
        return Lk, support_data
    
    def _apriori_gen(self, Lk, k):
        """
        根据频繁(k-1)-项集生成候选k-项集
        
        参数:
        Lk: 频繁(k-1)-项集
        k: 项集大小
        
        返回:
        Ck: 候选k-项集
        """
        Ck = []
        len_Lk = len(Lk)
        
        # 连接步：合并两个只有一个项不同的(k-1)-项集
        for i in range(len_Lk):
            for j in range(i + 1, len_Lk):
                # 将frozenset转换为排序的列表进行比较
                L1 = list(Lk[i])
                L2 = list(Lk[j])
                L1.sort()
                L2.sort()
                
                # 如果前k-2个项相同，则可以合并
                if L1[:-1] == L2[:-1]:
                    candidate = Lk[i] | Lk[j]
                    # 剪枝步：检查所有(k-1)-子集是否都是频繁的
                    if self._has_infrequent_subset(candidate, Lk):
                        continue
                    Ck.append(candidate)
        
        return Ck
    
    def _has_infrequent_subset(self, candidate, Lk):
        """
        检查候选项集是否包含非频繁子集
        
        参数:
        candidate: 候选项集
        Lk: 频繁(k-1)-项集
        
        返回:
        bool: 是否包含非频繁子集
        """
        k = len(candidate)
        # 生成所有(k-1)-子集
        for subset in combinations(candidate, k - 1):
            subset_frozenset = frozenset(subset)
            if subset_frozenset not in Lk:
                return True
        return False
    
    def fit(self, transaction_list):
        """
        挖掘频繁项集
        
        参数:
        transaction_list: 事务列表，每个事务是一个项的列表
        
        返回:
        frequent_itemsets: 所有频繁项集
        support_data: 支持度数据
        """
        self.transaction_list = transaction_list
        support_data = {}
        
        # 生成候选1-项集
        C1 = self._create_C1(transaction_list)
        
        # 获取频繁1-项集
        L1, support_data_1 = self._scan_D(transaction_list, C1, self.min_support)
        support_data.update(support_data_1)
        
        # 存储频繁项集
        self.frequent_itemsets[1] = L1
        k = 2
        
        # 生成更大的频繁项集
        while len(self.frequent_itemsets[k-1]) > 0:
            # 生成候选k-项集
            Ck = self._apriori_gen(self.frequent_itemsets[k-1], k)
            
            if not Ck:
                break
            
            # 扫描数据库，获取频繁k-项集
            Lk, support_data_k = self._scan_D(transaction_list, Ck, self.min_support)
            support_data.update(support_data_k)
            
            if Lk:
                self.frequent_itemsets[k] = Lk
                k += 1
            else:
                break
        
        self.support_data = support_data
        
        print(f"频繁项集挖掘完成:")
        for k in self.frequent_itemsets:
            print(f"  {k}-项集: {len(self.frequent_itemsets[k])} 个")
        
        return self.frequent_itemsets, support_data
    
    def generate_rules(self, min_confidence=None):
        """
        生成关联规则
        
        参数:
        min_confidence: 最小置信度，如果为None则使用初始化时的值
        
        返回:
        association_rules: 关联规则列表
        """
        if min_confidence is None:
            min_confidence = self.min_confidence
        
        self.association_rules = []
        
        # 从2-项集开始生成规则
        for k in range(2, len(self.frequent_itemsets) + 1):
            if k not in self.frequent_itemsets:
                continue
                
            for itemset in self.frequent_itemsets[k]:
                # 生成所有可能的规则
                for i in range(1, len(itemset)):
                    # 生成所有大小为i的子集作为前件
                    for antecedent_tuple in combinations(itemset, i):
                        antecedent = frozenset(antecedent_tuple)
                        consequent = itemset - antecedent
                        
                        # 计算置信度
                        confidence = self.support_data[itemset] / self.support_data[antecedent]
                        
                        if confidence >= min_confidence:
                            # 计算提升度
                            lift = confidence / self.support_data[consequent]
                            
                            rule = {
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': self.support_data[itemset],
                                'confidence': confidence,
                                'lift': lift
                            }
                            
                            self.association_rules.append(rule)
        
        # 按置信度排序
        self.association_rules.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"生成了 {len(self.association_rules)} 条关联规则")
        
        return self.association_rules
    
    def print_frequent_itemsets(self, max_display=10):
        """
        打印频繁项集
        
        参数:
        max_display: 最大显示数量
        """
        print("\n=== 频繁项集 ===")
        
        for k in sorted(self.frequent_itemsets.keys()):
            print(f"\n{k}-项集 (共{len(self.frequent_itemsets[k])}个):")
            
            # 按支持度排序
            itemsets_with_support = [(itemset, self.support_data[itemset]) 
                                   for itemset in self.frequent_itemsets[k]]
            itemsets_with_support.sort(key=lambda x: x[1], reverse=True)
            
            for i, (itemset, support) in enumerate(itemsets_with_support[:max_display]):
                items_str = ', '.join(sorted(list(itemset)))
                print(f"  {{{items_str}}} : 支持度 = {support:.4f}")
            
            if len(itemsets_with_support) > max_display:
                print(f"  ... (还有 {len(itemsets_with_support) - max_display} 个)")
    
    def print_association_rules(self, max_display=20):
        """
        打印关联规则
        
        参数:
        max_display: 最大显示数量
        """
        if not self.association_rules:
            print("还没有生成关联规则")
            return
        
        print(f"\n=== 关联规则 (前{min(max_display, len(self.association_rules))}条) ===")
        
        for i, rule in enumerate(self.association_rules[:max_display]):
            antecedent_str = ', '.join(sorted(list(rule['antecedent'])))
            consequent_str = ', '.join(sorted(list(rule['consequent'])))
            
            print(f"\n规则 {i+1}:")
            print(f"  {{{antecedent_str}}} => {{{consequent_str}}}")
            print(f"  支持度: {rule['support']:.4f}")
            print(f"  置信度: {rule['confidence']:.4f}")
            print(f"  提升度: {rule['lift']:.4f}")
    
    def get_rules_for_item(self, item):
        """
        获取包含特定项的规则
        
        参数:
        item: 要查询的项
        
        返回:
        matching_rules: 包含该项的规则列表
        """
        matching_rules = []
        
        for rule in self.association_rules:
            if item in rule['antecedent'] or item in rule['consequent']:
                matching_rules.append(rule)
        
        return matching_rules
    
    def analyze_item_associations(self, item):
        """
        分析特定项的关联情况
        
        参数:
        item: 要分析的项
        """
        print(f"\n=== 项目 '{item}' 的关联分析 ===")
        
        # 作为前件的规则
        antecedent_rules = [rule for rule in self.association_rules 
                           if item in rule['antecedent']]
        
        # 作为后件的规则
        consequent_rules = [rule for rule in self.association_rules 
                           if item in rule['consequent']]
        
        print(f"\n'{item}' 作为前件的规则 (共{len(antecedent_rules)}条):")
        for rule in antecedent_rules[:5]:  # 显示前5条
            consequent_str = ', '.join(sorted(list(rule['consequent'])))
            print(f"  {item} => {{{consequent_str}}} (置信度: {rule['confidence']:.4f})")
        
        print(f"\n'{item}' 作为后件的规则 (共{len(consequent_rules)}条):")
        for rule in consequent_rules[:5]:  # 显示前5条
            antecedent_str = ', '.join(sorted(list(rule['antecedent'])))
            print(f"  {{{antecedent_str}}} => {item} (置信度: {rule['confidence']:.4f})")
        
        return antecedent_rules, consequent_rules
    
    def get_statistics(self):
        """
        获取挖掘统计信息
        
        返回:
        stats: 统计信息字典
        """
        stats = {
            'total_transactions': len(self.transaction_list) if self.transaction_list else 0,
            'total_items': len(set(item for transaction in self.transaction_list 
                                 for item in transaction)) if self.transaction_list else 0,
            'frequent_itemsets_count': sum(len(itemsets) for itemsets in self.frequent_itemsets.values()),
            'association_rules_count': len(self.association_rules),
            'max_itemset_size': max(self.frequent_itemsets.keys()) if self.frequent_itemsets else 0,
            'min_support': self.min_support,
            'min_confidence': self.min_confidence
        }
        
        return stats

def create_sample_transactions():
    """
    创建示例事务数据
    
    返回:
    transactions: 事务列表
    """
    transactions = [
        ['牛奶', '面包'],
        ['牛奶', '面包', '黄油'],
        ['牛奶', '黄油'],
        ['面包', '黄油'],
        ['牛奶', '面包', '黄油', '奶酪'],
        ['面包', '奶酪'],
        ['牛奶', '奶酪'],
        ['面包'],
        ['牛奶', '面包', '黄油', '奶酪', '鸡蛋'],
        ['牛奶', '鸡蛋']
    ]
    
    return transactions 