"""
线性回归算法实现
"""
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    线性回归算法
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        """
        初始化线性回归
        
        参数:
        learning_rate: 学习率
        max_iterations: 最大迭代次数
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X, y):
        """
        训练线性回归模型
        
        参数:
        X: 特征数据
        y: 目标值
        """
        n_samples, n_features = X.shape
        
        # 初始化权重和偏置
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.max_iterations):
            # 前向传播
            y_pred = self.predict(X)
            
            # 计算成本
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # 计算梯度
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        """
        预测
        
        参数:
        X: 特征数据
        
        返回:
        predictions: 预测结果
        """
        return np.dot(X, self.weights) + self.bias
    
    def _compute_cost(self, y_true, y_pred):
        """
        计算成本函数 (均方误差)
        
        参数:
        y_true: 真实值
        y_pred: 预测值
        
        返回:
        cost: 成本
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def score(self, X, y):
        """
        计算R²分数
        
        参数:
        X: 特征数据
        y: 真实值
        
        返回:
        r2_score: R²分数
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def plot_cost_history(self):
        """
        绘制成本历史
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('成本函数收敛过程')
        plt.xlabel('迭代次数')
        plt.ylabel('成本')
        plt.grid(True, alpha=0.3)
        plt.show()

class HousePricePrediction:
    """
    房价预测系统
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = ['房屋面积', '卧室数量', '卫生间数量', '房屋年龄', '距离市中心距离']
        self.X = None
        self.y = None
        
    def create_sample_data(self, n_samples=1000):
        """
        创建房价样本数据
        
        参数:
        n_samples: 样本数量
        """
        np.random.seed(42)
        
        # 生成特征
        area = np.random.uniform(50, 300, n_samples)  # 房屋面积 (平方米)
        bedrooms = np.random.randint(1, 6, n_samples)  # 卧室数量
        bathrooms = np.random.randint(1, 4, n_samples)  # 卫生间数量
        age = np.random.uniform(0, 50, n_samples)  # 房屋年龄
        distance = np.random.uniform(1, 50, n_samples)  # 距离市中心距离 (公里)
        
        # 组合特征
        self.X = np.column_stack([area, bedrooms, bathrooms, age, distance])
        
        # 生成目标值 (房价)
        # 房价 = 基础价格 + 面积系数 * 面积 + 卧室系数 * 卧室数 + ... + 噪声
        base_price = 50000
        area_coef = 3000
        bedroom_coef = 15000
        bathroom_coef = 10000
        age_coef = -800
        distance_coef = -2000
        
        self.y = (base_price + 
                 area_coef * area + 
                 bedroom_coef * bedrooms + 
                 bathroom_coef * bathrooms + 
                 age_coef * age + 
                 distance_coef * distance + 
                 np.random.normal(0, 20000, n_samples))
        
        print(f"生成了 {n_samples} 个房价样本")
        print(f"特征: {self.feature_names}")
        print(f"房价范围: {self.y.min():.2f} - {self.y.max():.2f}")
        
        return self.X, self.y
    
    def train_model(self, test_size=0.2):
        """
        训练模型
        
        参数:
        test_size: 测试集比例
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        # 特征缩放
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练模型
        self.model = LinearRegression(learning_rate=0.01, max_iterations=1000)
        self.model.fit(X_train_scaled, y_train)
        
        # 评估模型
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"训练集 R² 分数: {train_score:.4f}")
        print(f"测试集 R² 分数: {test_score:.4f}")
        
        # 绘制成本历史
        self.model.plot_cost_history()
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def predict_house_price(self, area, bedrooms, bathrooms, age, distance, scaler):
        """
        预测房价
        
        参数:
        area: 房屋面积
        bedrooms: 卧室数量
        bathrooms: 卫生间数量
        age: 房屋年龄
        distance: 距离市中心距离
        scaler: 特征缩放器
        """
        if self.model is None:
            print("请先训练模型")
            return None
        
        # 准备特征
        features = np.array([[area, bedrooms, bathrooms, age, distance]])
        features_scaled = scaler.transform(features)
        
        # 预测
        predicted_price = self.model.predict(features_scaled)[0]
        
        print(f"房屋特征:")
        print(f"  面积: {area} 平方米")
        print(f"  卧室: {bedrooms} 个")
        print(f"  卫生间: {bathrooms} 个")
        print(f"  房龄: {age} 年")
        print(f"  距离市中心: {distance} 公里")
        print(f"预测房价: ¥{predicted_price:,.2f}")
        
        return predicted_price

def main():
    """
    主函数
    """
    print("=== 房价预测系统 ===")
    
    # 创建房价预测系统
    house_system = HousePricePrediction()
    
    # 创建样本数据
    X, y = house_system.create_sample_data(1000)
    
    # 训练模型
    X_train, X_test, y_train, y_test, scaler = house_system.train_model()
    
    # 预测示例
    print("\n=== 房价预测示例 ===")
    
    # 示例1: 大房子
    house_system.predict_house_price(150, 3, 2, 5, 10, scaler)
    
    print()
    
    # 示例2: 小房子
    house_system.predict_house_price(80, 2, 1, 20, 25, scaler)
    
    print()
    
    # 示例3: 豪宅
    house_system.predict_house_price(250, 4, 3, 2, 5, scaler)

if __name__ == "__main__":
    main() 