"""
运行所有机器学习案例
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_knn_example():
    """运行K-近邻算法案例"""
    print("="*60)
    print("运行 K-近邻算法案例：约会网站推荐系统")
    print("="*60)
    
    try:
        from algorithms.knn.dating_example import main as knn_main
        knn_main()
    except Exception as e:
        print(f"运行KNN案例时出错: {e}")
    
    print("\n" + "="*60)
    print("K-近邻算法案例完成")
    print("="*60)

def run_decision_tree_example():
    """运行决策树案例"""
    print("="*60)
    print("运行决策树案例：隐形眼镜类型预测")
    print("="*60)
    
    try:
        from algorithms.decision_tree.contact_lens_example import main as dt_main
        dt_main()
    except Exception as e:
        print(f"运行决策树案例时出错: {e}")
    
    print("\n" + "="*60)
    print("决策树案例完成")
    print("="*60)

def run_naive_bayes_example():
    """运行朴素贝叶斯案例"""
    print("="*60)
    print("运行朴素贝叶斯案例：垃圾邮件分类")
    print("="*60)
    
    try:
        from algorithms.naive_bayes.spam_filter_example import main as nb_main
        nb_main()
    except Exception as e:
        print(f"运行朴素贝叶斯案例时出错: {e}")
    
    print("\n" + "="*60)
    print("朴素贝叶斯案例完成")
    print("="*60)

def run_linear_regression_example():
    """运行线性回归案例"""
    print("="*60)
    print("运行线性回归案例：房价预测")
    print("="*60)
    
    try:
        from algorithms.linear_regression.linear_regression import main as lr_main
        lr_main()
    except Exception as e:
        print(f"运行线性回归案例时出错: {e}")
    
    print("\n" + "="*60)
    print("线性回归案例完成")
    print("="*60)

def run_recommendation_example():
    """运行推荐系统案例"""
    print("="*60)
    print("运行推荐系统案例：协同过滤电影推荐")
    print("="*60)
    
    try:
        from algorithms.recommendation.collaborative_filtering import main as rec_main
        rec_main()
    except Exception as e:
        print(f"运行推荐系统案例时出错: {e}")
    
    print("\n" + "="*60)
    print("推荐系统案例完成")
    print("="*60)

def run_logistic_regression_example():
    """运行逻辑回归案例"""
    print("="*60)
    print("运行逻辑回归案例：疝气病症预测")
    print("="*60)
    
    try:
        from algorithms.logistic_regression.colic_example import main as lr_main
        lr_main()
    except Exception as e:
        print(f"运行逻辑回归案例时出错: {e}")
    
    print("\n" + "="*60)
    print("逻辑回归案例完成")
    print("="*60)

def run_clustering_example():
    """运行聚类算法案例"""
    print("="*60)
    print("运行聚类算法案例：K-means客户细分")
    print("="*60)
    
    try:
        from algorithms.clustering.customer_segmentation_example import main as cluster_main
        cluster_main()
    except Exception as e:
        print(f"运行聚类算法案例时出错: {e}")
    
    print("\n" + "="*60)
    print("聚类算法案例完成")
    print("="*60)

def run_pca_example():
    """运行PCA降维案例"""
    print("="*60)
    print("运行PCA降维案例：主成分分析")
    print("="*60)
    
    try:
        from algorithms.pca.dimensionality_reduction_example import main as pca_main
        pca_main()
    except Exception as e:
        print(f"运行PCA案例时出错: {e}")
    
    print("\n" + "="*60)
    print("PCA案例完成")
    print("="*60)

def run_svm_example():
    """运行SVM案例"""
    print("="*60)
    print("运行SVM案例：手写数字识别")
    print("="*60)
    
    try:
        from algorithms.svm.handwritten_digits_example import main as svm_main
        svm_main()
    except Exception as e:
        print(f"运行SVM案例时出错: {e}")
    
    print("\n" + "="*60)
    print("SVM案例完成")
    print("="*60)

def run_adaboost_example():
    """运行AdaBoost案例"""
    print("="*60)
    print("运行AdaBoost案例：弱分类器集成")
    print("="*60)
    
    try:
        from algorithms.adaboost.weak_classifier_example import main as ada_main
        ada_main()
    except Exception as e:
        print(f"运行AdaBoost案例时出错: {e}")
    
    print("\n" + "="*60)
    print("AdaBoost案例完成")
    print("="*60)

def run_apriori_example():
    """运行Apriori案例"""
    print("="*60)
    print("运行Apriori案例：购物篮分析")
    print("="*60)
    
    try:
        from algorithms.apriori.market_basket_example import main as apriori_main
        apriori_main()
    except Exception as e:
        print(f"运行Apriori案例时出错: {e}")
    
    print("\n" + "="*60)
    print("Apriori案例完成")
    print("="*60)

def show_algorithm_summary():
    """显示算法总结"""
    print("\n" + "="*80)
    print("机器学习算法总结")
    print("="*80)
    
    algorithms = [
        {
            'name': 'K-近邻算法 (kNN)',
            'case': '约会网站推荐系统',
            'description': '基于距离的分类算法，适用于小数据集',
            'advantages': ['简单易懂', '无需训练过程', '适用于多分类'],
            'disadvantages': ['计算复杂度高', '对异常值敏感', '需要存储所有训练数据']
        },
        {
            'name': '决策树',
            'case': '隐形眼镜类型预测',
            'description': '基于树状结构的分类算法，规则清晰易解释',
            'advantages': ['可解释性强', '处理分类和连续特征', '不需要特征缩放'],
            'disadvantages': ['容易过拟合', '对噪声敏感', '可能产生偏向性']
        },
        {
            'name': '朴素贝叶斯',
            'case': '垃圾邮件分类',
            'description': '基于贝叶斯定理的概率分类算法',
            'advantages': ['训练速度快', '处理多分类问题', '对小样本效果好'],
            'disadvantages': ['特征独立性假设', '对输入数据敏感', '分类性能有限']
        },
        {
            'name': '线性回归',
            'case': '房价预测',
            'description': '预测连续数值的回归算法',
            'advantages': ['简单快速', '可解释性强', '无需调参'],
            'disadvantages': ['假设线性关系', '对异常值敏感', '特征选择重要']
        },
        {
            'name': '逻辑回归',
            'case': '疝气病症预测',
            'description': '基于Sigmoid函数的二分类算法',
            'advantages': ['输出概率解释', '不需要调参', '不容易过拟合'],
            'disadvantages': ['假设线性关系', '对异常值敏感', '需要特征缩放']
        },
        {
            'name': 'K-means聚类',
            'case': '客户细分分析',
            'description': '基于距离的无监督聚类算法',
            'advantages': ['简单高效', '适用于球形簇', '可扩展性好'],
            'disadvantages': ['需要预设K值', '对初始化敏感', '假设球形分布']
        },
        {
            'name': '主成分分析 (PCA)',
            'case': '数据降维和可视化',
            'description': '线性降维技术，保留主要变化方向',
            'advantages': ['降低维度', '去除噪声', '数据压缩'],
            'disadvantages': ['线性假设', '主成分难解释', '信息丢失']
        },
        {
            'name': '支持向量机 (SVM)',
            'case': '手写数字识别',
            'description': '通过最大间隔超平面进行分类',
            'advantages': ['处理高维数据', '核函数处理非线性', '泛化能力强'],
            'disadvantages': ['参数敏感', '计算复杂度高', '难以解释']
        },
        {
            'name': 'AdaBoost',
            'case': '弱分类器集成',
            'description': '自适应提升算法，组合弱分类器',
            'advantages': ['提升弱分类器', '不易过拟合', '自动特征选择'],
            'disadvantages': ['对噪声敏感', '训练时间长', '参数调优复杂']
        },
        {
            'name': 'Apriori算法',
            'case': '购物篮分析',
            'description': '挖掘频繁项集和关联规则',
            'advantages': ['发现隐藏模式', '解释性强', '适用于推荐'],
            'disadvantages': ['计算复杂度高', '内存消耗大', '参数敏感']
        },
        {
            'name': '协同过滤',
            'case': '电影推荐系统',
            'description': '基于用户行为的推荐算法',
            'advantages': ['无需内容信息', '发现潜在兴趣', '个性化推荐'],
            'disadvantages': ['冷启动问题', '数据稀疏性', '计算复杂度高']
        }
    ]
    
    for i, algo in enumerate(algorithms, 1):
        print(f"\n{i}. {algo['name']}")
        print(f"   案例: {algo['case']}")
        print(f"   描述: {algo['description']}")
        print(f"   优点: {', '.join(algo['advantages'])}")
        print(f"   缺点: {', '.join(algo['disadvantages'])}")
    
    print("\n" + "="*80)
    print("学习建议")
    print("="*80)
    
    suggestions = [
        "1. 从简单算法开始：kNN → 决策树 → 朴素贝叶斯",
        "2. 理解每种算法的适用场景和局限性",
        "3. 动手实践，修改参数观察效果变化",
        "4. 学会数据预处理和特征工程",
        "5. 掌握模型评估方法和性能指标",
        "6. 了解算法的数学原理和推导过程",
        "7. 学习使用scikit-learn等成熟库",
        "8. 实践更多真实世界的问题"
    ]
    
    for suggestion in suggestions:
        print(f"   {suggestion}")

def main():
    """主函数"""
    print("欢迎使用机器学习实战案例集！")
    print("本项目包含多个经典机器学习算法的完整实现和案例")
    
    while True:
        print("\n请选择要运行的案例:")
        print("1. K-近邻算法 (约会网站推荐)")
        print("2. 决策树 (隐形眼镜类型预测)")
        print("3. 朴素贝叶斯 (垃圾邮件分类)")
        print("4. 线性回归 (房价预测)")
        print("5. 逻辑回归 (疝气病症预测)")
        print("6. K-means聚类 (客户细分)")
        print("7. PCA降维 (数据降维)")
        print("8. SVM (手写数字识别)")
        print("9. AdaBoost (弱分类器集成)")
        print("10. Apriori (购物篮分析)")
        print("11. 推荐系统 (协同过滤)")
        print("12. 运行所有案例")
        print("13. 显示算法总结")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-13): ").strip()
        
        if choice == '0':
            print("感谢使用！")
            break
        elif choice == '1':
            run_knn_example()
        elif choice == '2':
            run_decision_tree_example()
        elif choice == '3':
            run_naive_bayes_example()
        elif choice == '4':
            run_linear_regression_example()
        elif choice == '5':
            run_logistic_regression_example()
        elif choice == '6':
            run_clustering_example()
        elif choice == '7':
            run_pca_example()
        elif choice == '8':
            run_svm_example()
        elif choice == '9':
            run_adaboost_example()
        elif choice == '10':
            run_apriori_example()
        elif choice == '11':
            run_recommendation_example()
        elif choice == '12':
            print("开始运行所有案例...")
            run_knn_example()
            run_decision_tree_example()
            run_naive_bayes_example()
            run_linear_regression_example()
            run_logistic_regression_example()
            run_clustering_example()
            run_pca_example()
            run_svm_example()
            run_adaboost_example()
            run_apriori_example()
            run_recommendation_example()
            print("\n所有案例运行完成！")
        elif choice == '13':
            show_algorithm_summary()
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main() 