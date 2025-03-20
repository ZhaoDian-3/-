import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy.optimize import minimize
def objective(X, model, poly):
    X = np.array(X).reshape(1, -1)  # Reshape for a single sample prediction
    X_poly = poly.transform(X)  # Transform to polynomial features
    return -model.predict(X_poly)[0]  # Return the negative of the prediction

a = pd.read_excel(r"C:\Users\Lenovo\PycharmProjects\数学建模\2021B\data.xlsx")
a = a.to_numpy()
X=[]
Y=[]
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# 创建线性回归模型实例
model = LinearRegression()
model.fit(X_poly, Y)

# 拟合模型
Y_pred = model.predict(X_poly)
mse = mean_squared_error(Y, Y_pred)
r_squared = model.score(X_poly, Y)

print(f"Mean Squared Error: {mse}")  # 计算均方误差（MSE）
print(f"R-squared: {r_squared}")  # 计算决定系数（R-squared）
print(f"Coefficients: {model.coef_}")  # 打印模型的系数和截距
print(f"Intercept: {model.intercept_}")  # 打印生成的所有特征名称
print(f"Polynomial Features: {poly.get_feature_names_out(input_features=['x1', 'x2', 'x3', 'x4', 'T'])}")

bounds =[]
initial_guess = np.mean(X, axis=0)  # Start with the mean of the input data
result = minimize(objective, initial_guess, args=(model, poly), method='L-BFGS-B',
                          bounds=bounds)

optimal_X = result.x
max_prediction = -result.fun

print(f"features: {optimal_X}")
print(f"Maximum predicted value: {max_prediction}")