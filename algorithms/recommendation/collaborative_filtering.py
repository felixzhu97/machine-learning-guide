"""
协同过滤推荐系统实现
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    """
    协同过滤推荐系统
    """
    
    def __init__(self, method='user_based'):
        """
        初始化协同过滤
        
        参数:
        method: 推荐方法 ('user_based' 或 'item_based')
        """
        self.method = method
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.users = None
        self.items = None
        
    def fit(self, user_item_matrix):
        """
        训练推荐模型
        
        参数:
        user_item_matrix: 用户-物品评分矩阵
        """
        self.user_item_matrix = user_item_matrix
        self.users = user_item_matrix.index
        self.items = user_item_matrix.columns
        
        if self.method == 'user_based':
            # 计算用户之间的余弦相似度
            self.similarity_matrix = cosine_similarity(user_item_matrix.fillna(0))
            self.similarity_matrix = pd.DataFrame(
                self.similarity_matrix, 
                index=self.users, 
                columns=self.users
            )
        else:  # item_based
            # 计算物品之间的余弦相似度
            self.similarity_matrix = cosine_similarity(user_item_matrix.fillna(0).T)
            self.similarity_matrix = pd.DataFrame(
                self.similarity_matrix, 
                index=self.items, 
                columns=self.items
            )
    
    def predict_rating(self, user, item, k=5):
        """
        预测用户对物品的评分
        
        参数:
        user: 用户ID
        item: 物品ID
        k: 考虑的相似用户/物品数量
        
        返回:
        predicted_rating: 预测评分
        """
        if self.method == 'user_based':
            return self._predict_user_based(user, item, k)
        else:
            return self._predict_item_based(user, item, k)
    
    def _predict_user_based(self, user, item, k):
        """
        基于用户的协同过滤预测
        """
        # 获取与目标用户最相似的k个用户
        user_similarities = self.similarity_matrix[user].sort_values(ascending=False)
        similar_users = user_similarities.index[1:k+1]  # 排除自己
        
        # 计算加权平均评分
        weighted_ratings = 0
        similarity_sum = 0
        
        for similar_user in similar_users:
            if not pd.isna(self.user_item_matrix.loc[similar_user, item]):
                rating = self.user_item_matrix.loc[similar_user, item]
                similarity = user_similarities[similar_user]
                
                weighted_ratings += rating * similarity
                similarity_sum += similarity
        
        if similarity_sum == 0:
            return self.user_item_matrix.mean().mean()  # 返回全局平均分
        
        return weighted_ratings / similarity_sum
    
    def _predict_item_based(self, user, item, k):
        """
        基于物品的协同过滤预测
        """
        # 获取与目标物品最相似的k个物品
        item_similarities = self.similarity_matrix[item].sort_values(ascending=False)
        similar_items = item_similarities.index[1:k+1]  # 排除自己
        
        # 计算加权平均评分
        weighted_ratings = 0
        similarity_sum = 0
        
        for similar_item in similar_items:
            if not pd.isna(self.user_item_matrix.loc[user, similar_item]):
                rating = self.user_item_matrix.loc[user, similar_item]
                similarity = item_similarities[similar_item]
                
                weighted_ratings += rating * similarity
                similarity_sum += similarity
        
        if similarity_sum == 0:
            return self.user_item_matrix.mean().mean()  # 返回全局平均分
        
        return weighted_ratings / similarity_sum
    
    def recommend_items(self, user, n_recommendations=5):
        """
        为用户推荐物品
        
        参数:
        user: 用户ID
        n_recommendations: 推荐物品数量
        
        返回:
        recommendations: 推荐物品列表
        """
        # 获取用户未评分的物品
        user_ratings = self.user_item_matrix.loc[user]
        unrated_items = user_ratings[user_ratings.isna()].index
        
        # 预测每个未评分物品的评分
        predictions = []
        for item in unrated_items:
            predicted_rating = self.predict_rating(user, item)
            predictions.append((item, predicted_rating))
        
        # 按预测评分排序
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前n个推荐
        recommendations = predictions[:n_recommendations]
        
        print(f"为用户 {user} 推荐的物品:")
        for item, rating in recommendations:
            print(f"  {item}: 预测评分 {rating:.2f}")
        
        return recommendations

class MovieRecommendationSystem:
    """
    电影推荐系统
    """
    
    def __init__(self):
        self.cf_user = None
        self.cf_item = None
        self.ratings_matrix = None
        self.movies = ['阿凡达', '泰坦尼克号', '星球大战', '盗梦空间', '肖申克的救赎', 
                     '阿甘正传', '教父', '低俗小说', '指环王', '黑客帝国']
        self.users = [f'用户{i}' for i in range(1, 11)]
        
    def create_sample_data(self):
        """
        创建电影评分样本数据
        """
        np.random.seed(42)
        
        # 创建用户-电影评分矩阵
        ratings = np.random.randint(1, 6, (len(self.users), len(self.movies)))
        
        # 随机设置一些评分为缺失值
        missing_rate = 0.3
        missing_mask = np.random.random((len(self.users), len(self.movies))) < missing_rate
        ratings = ratings.astype(float)
        ratings[missing_mask] = np.nan
        
        # 创建DataFrame
        self.ratings_matrix = pd.DataFrame(
            ratings, 
            index=self.users, 
            columns=self.movies
        )
        
        print("电影评分矩阵:")
        print(self.ratings_matrix)
        
        return self.ratings_matrix
    
    def train_models(self):
        """
        训练推荐模型
        """
        # 基于用户的协同过滤
        self.cf_user = CollaborativeFiltering(method='user_based')
        self.cf_user.fit(self.ratings_matrix)
        
        # 基于物品的协同过滤
        self.cf_item = CollaborativeFiltering(method='item_based')
        self.cf_item.fit(self.ratings_matrix)
        
        print("推荐模型训练完成")
    
    def visualize_similarity(self):
        """
        可视化相似度矩阵
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 用户相似度矩阵
        im1 = ax1.imshow(self.cf_user.similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax1.set_title('用户相似度矩阵')
        ax1.set_xlabel('用户')
        ax1.set_ylabel('用户')
        ax1.set_xticks(range(len(self.users)))
        ax1.set_yticks(range(len(self.users)))
        ax1.set_xticklabels(self.users, rotation=45)
        ax1.set_yticklabels(self.users)
        plt.colorbar(im1, ax=ax1)
        
        # 物品相似度矩阵
        im2 = ax2.imshow(self.cf_item.similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax2.set_title('电影相似度矩阵')
        ax2.set_xlabel('电影')
        ax2.set_ylabel('电影')
        ax2.set_xticks(range(len(self.movies)))
        ax2.set_yticks(range(len(self.movies)))
        ax2.set_xticklabels(self.movies, rotation=45)
        ax2.set_yticklabels(self.movies)
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.show()
    
    def compare_methods(self):
        """
        比较不同推荐方法
        """
        print("\n=== 比较推荐方法 ===")
        
        # 选择一个有缺失评分的用户
        test_user = '用户1'
        
        print(f"为 {test_user} 推荐电影:")
        print(f"{test_user} 的当前评分:")
        user_ratings = self.ratings_matrix.loc[test_user]
        for movie, rating in user_ratings.items():
            if not pd.isna(rating):
                print(f"  {movie}: {rating}")
        
        print("\n基于用户的协同过滤推荐:")
        user_recommendations = self.cf_user.recommend_items(test_user, n_recommendations=3)
        
        print("\n基于物品的协同过滤推荐:")
        item_recommendations = self.cf_item.recommend_items(test_user, n_recommendations=3)
        
        return user_recommendations, item_recommendations
    
    def evaluate_prediction_accuracy(self):
        """
        评估预测准确性
        """
        print("\n=== 评估预测准确性 ===")
        
        # 选择一些已知评分作为测试集
        test_cases = []
        for user in self.users:
            for movie in self.movies:
                if not pd.isna(self.ratings_matrix.loc[user, movie]):
                    test_cases.append((user, movie, self.ratings_matrix.loc[user, movie]))
        
        # 随机选择一些测试案例
        np.random.shuffle(test_cases)
        test_cases = test_cases[:20]  # 选择20个测试案例
        
        user_mae = 0
        item_mae = 0
        
        for user, movie, true_rating in test_cases:
            # 暂时隐藏真实评分
            original_rating = self.ratings_matrix.loc[user, movie]
            self.ratings_matrix.loc[user, movie] = np.nan
            
            # 预测评分
            user_pred = self.cf_user.predict_rating(user, movie)
            item_pred = self.cf_item.predict_rating(user, movie)
            
            # 计算误差
            user_mae += abs(true_rating - user_pred)
            item_mae += abs(true_rating - item_pred)
            
            # 恢复真实评分
            self.ratings_matrix.loc[user, movie] = original_rating
        
        user_mae /= len(test_cases)
        item_mae /= len(test_cases)
        
        print(f"基于用户的协同过滤 MAE: {user_mae:.3f}")
        print(f"基于物品的协同过滤 MAE: {item_mae:.3f}")
        
        return user_mae, item_mae

def main():
    """
    主函数
    """
    print("=== 电影推荐系统 ===")
    
    # 创建电影推荐系统
    movie_system = MovieRecommendationSystem()
    
    # 创建样本数据
    ratings_matrix = movie_system.create_sample_data()
    
    # 训练模型
    movie_system.train_models()
    
    # 可视化相似度矩阵
    movie_system.visualize_similarity()
    
    # 比较推荐方法
    user_recs, item_recs = movie_system.compare_methods()
    
    # 评估预测准确性
    user_mae, item_mae = movie_system.evaluate_prediction_accuracy()
    
    print(f"\n=== 推荐系统总结 ===")
    print(f"基于用户的协同过滤更适合发现新的兴趣点")
    print(f"基于物品的协同过滤更稳定，适合个性化推荐")
    print(f"在本次测试中，MAE更低的方法是: {'基于用户' if user_mae < item_mae else '基于物品'}")

if __name__ == "__main__":
    main() 