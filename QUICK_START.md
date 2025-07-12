# 机器学习实战 - 快速入门指南

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行示例

```bash
# 运行所有案例的交互式菜单
python examples/run_all_examples.py

# 或者单独运行特定案例
python algorithms/knn/dating_example.py
python algorithms/decision_tree/contact_lens_example.py
python algorithms/naive_bayes/spam_filter_example.py
python algorithms/linear_regression/linear_regression.py
python algorithms/logistic_regression/colic_example.py
python algorithms/clustering/customer_segmentation_example.py
python algorithms/pca/dimensionality_reduction_example.py
python algorithms/recommendation/collaborative_filtering.py
```

## 📚 学习路径

### 初学者路径

1. **K-近邻算法** - 理解分类的基本概念
2. **决策树** - 学习特征选择和树结构
3. **朴素贝叶斯** - 掌握概率分类方法

### 进阶路径

4. **线性回归** - 进入回归分析领域
5. **逻辑回归** - 学习概率分类方法
6. **K-means 聚类** - 无监督学习入门
7. **主成分分析** - 数据降维和可视化
8. **推荐系统** - 综合应用多种技术

## 🎯 案例介绍

### 1. K-近邻算法 (kNN)

- **案例**: 约会网站推荐系统
- **学习目标**:
  - 理解距离度量
  - 学习 K 值选择
  - 掌握特征归一化
- **运行**: `python algorithms/knn/dating_example.py`

### 2. 决策树

- **案例**: 隐形眼镜类型预测
- **学习目标**:
  - 理解信息熵和基尼不纯度
  - 学习特征重要性分析
  - 掌握决策树可视化
- **运行**: `python algorithms/decision_tree/contact_lens_example.py`

### 3. 朴素贝叶斯

- **案例**: 垃圾邮件分类
- **学习目标**:
  - 理解贝叶斯定理
  - 学习文本预处理
  - 掌握拉普拉斯平滑
- **运行**: `python algorithms/naive_bayes/spam_filter_example.py`

### 4. 线性回归

- **案例**: 房价预测
- **学习目标**:
  - 理解梯度下降
  - 学习特征缩放
  - 掌握回归评估指标
- **运行**: `python algorithms/linear_regression/linear_regression.py`

### 5. 逻辑回归

- **案例**: 疝气病症预测
- **学习目标**:
  - 理解 Sigmoid 函数
  - 学习梯度上升算法
  - 掌握二分类评估指标
- **运行**: `python algorithms/logistic_regression/colic_example.py`

### 6. K-means 聚类

- **案例**: 客户细分分析
- **学习目标**:
  - 理解无监督学习
  - 学习聚类评估方法
  - 掌握肘部法则选择 K 值
- **运行**: `python algorithms/clustering/customer_segmentation_example.py`

### 7. 主成分分析（PCA）

- **案例**: 数据降维和可视化
- **学习目标**:
  - 理解线性降维原理
  - 学习特征值分解
  - 掌握方差解释和数据压缩
- **运行**: `python algorithms/pca/dimensionality_reduction_example.py`

### 8. 推荐系统

- **案例**: 协同过滤电影推荐
- **学习目标**:
  - 理解协同过滤原理
  - 学习相似度计算
  - 掌握推荐系统评估
- **运行**: `python algorithms/recommendation/collaborative_filtering.py`

## 🛠️ 项目结构

```
machine-learning-guide/
├── README.md                    # 项目说明
├── QUICK_START.md              # 快速入门指南
├── requirements.txt            # 依赖包
├── utils/                      # 工具函数
│   ├── data_utils.py          # 数据处理工具
│   └── visualization.py       # 可视化工具
├── algorithms/                 # 算法实现
│   ├── knn/                   # K-近邻算法
│   ├── decision_tree/         # 决策树
│   ├── naive_bayes/           # 朴素贝叶斯
│   ├── linear_regression/     # 线性回归
│   ├── logistic_regression/   # 逻辑回归
│   ├── clustering/            # 聚类算法
│   ├── pca/                   # 主成分分析
│   └── recommendation/        # 推荐系统
└── examples/                  # 使用示例
    └── run_all_examples.py    # 运行所有案例
```

## 🔧 自定义使用

### 使用自己的数据

每个算法都支持自定义数据，例如：

```python
# K-近邻算法
from algorithms.knn.knn import KNN
knn = KNN(k=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# 决策树
from algorithms.decision_tree.decision_tree import DecisionTree
dt = DecisionTree(max_depth=5)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
```

### 修改参数

```python
# 调整K值
knn = KNN(k=3)  # 改变邻居数量

# 调整决策树深度
dt = DecisionTree(max_depth=10, criterion='gini')

# 调整朴素贝叶斯平滑参数
nb = NaiveBayes(alpha=0.5)

# 调整线性回归学习率
lr = LinearRegression(learning_rate=0.001, max_iterations=2000)
```

## 📊 性能对比

| 算法       | 训练速度 | 预测速度 | 内存使用 | 可解释性 |
| ---------- | -------- | -------- | -------- | -------- |
| kNN        | 快       | 慢       | 高       | 中       |
| 决策树     | 中       | 快       | 低       | 高       |
| 朴素贝叶斯 | 快       | 快       | 低       | 中       |
| 线性回归   | 中       | 快       | 低       | 高       |
| 逻辑回归   | 中       | 快       | 低       | 高       |
| K-means    | 中       | 快       | 中       | 中       |
| PCA        | 中       | 快       | 低       | 低       |
| 协同过滤   | 中       | 中       | 中       | 中       |

## 🎨 可视化功能

所有案例都包含丰富的可视化功能：

- **数据分布图**: 理解数据特征
- **决策边界**: 观察分类效果
- **学习曲线**: 监控训练过程
- **特征重要性**: 分析特征贡献
- **混淆矩阵**: 评估分类性能
- **相似度矩阵**: 理解推荐原理

## 🔍 常见问题

### Q: 如何选择合适的 K 值？

A: 使用交叉验证，通常选择奇数，避免平票情况。

### Q: 决策树如何避免过拟合？

A: 设置最大深度、最小样本数等参数，或使用剪枝技术。

### Q: 朴素贝叶斯的独立性假设现实吗？

A: 虽然假设强，但在实际应用中效果往往不错。

### Q: 线性回归适用于什么场景？

A: 特征与目标变量存在线性关系的回归问题。

### Q: 协同过滤如何处理冷启动问题？

A: 结合内容过滤，或使用混合推荐系统。

## 🎓 进阶学习

完成这些案例后，建议继续学习：

1. **支持向量机 (SVM)**
2. **神经网络**
3. **集成学习** (随机森林、梯度提升)
4. **无监督学习** (聚类、降维)
5. **深度学习** (CNN、RNN)

## 💡 实践建议

1. **动手实践**: 修改代码，观察效果变化
2. **理解原理**: 不只是使用，要理解算法背后的数学
3. **数据预处理**: 数据质量决定模型效果
4. **特征工程**: 好的特征比复杂的算法更重要
5. **模型评估**: 学会正确评估模型性能
6. **业务理解**: 算法要解决实际问题

## 🤝 贡献

欢迎提交问题和改进建议！

- 报告 Bug: 在 Issues 中描述问题
- 新增功能: 提交 Pull Request
- 文档改进: 优化文档和注释

## 📖 参考资料

- 《机器学习实战》- Peter Harrington
- 《统计学习方法》- 李航
- [scikit-learn 官方文档](https://scikit-learn.org/)
- [机器学习课程](https://www.coursera.org/learn/machine-learning)

开始您的机器学习之旅吧！🚀
