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
        print("5. 推荐系统 (协同过滤)")
        print("6. 运行所有案例")
        print("7. 显示算法总结")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-7): ").strip()
        
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
            run_recommendation_example()
        elif choice == '6':
            print("开始运行所有案例...")
            run_knn_example()
            run_decision_tree_example()
            run_naive_bayes_example()
            run_linear_regression_example()
            run_recommendation_example()
            print("\n所有案例运行完成！")
        elif choice == '7':
            show_algorithm_summary()
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main() 