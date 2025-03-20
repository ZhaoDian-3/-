from sklearn.linear_model import LinearRegression
import numpy as np

# 假设 X 是自变量矩阵，Y 是因变量向量
X = np.array([[1.1, 2], [3, 4], [5, 6],[6.8,7.9]])
Y = np.array([1, 2, 3,4.2])

# 创建线性回归模型实例
model = LinearRegression()

# 拟合模型
model.fit(X, Y)
r_squared = model.score(X, Y)

n = len(Y)
p=X.shape[1]
r_squared_adjusted = 1 - (1 - r_squared) * ((n - 1) / (n - p - 1))
# 获取模型系数和截距
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print('R_squared_adjused:',r_squared_adjusted)