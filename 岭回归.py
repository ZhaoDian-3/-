import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
# 生成一些示例数据
# np.random.seed(42)
# X = np.random.randn(100, 3)
# y = X @ np.array([1.5, -2.0, 1.0]) + np.random.randn(100) * 0.5
a = pd.read_excel(r"C:\Users\Lenovo\PycharmProjects\数学建模\2021B\data.xlsx")   #数据地址
a = a.to_numpy()
X=a[:,0:5]  #输入特征
y=a[:,5]    #输出
y=np.reshape(y,(114,1))
#归一化
X=(X-np.amin(X,axis=0))/(np.amax(X,axis=0)-np.amin(X,axis=0))
y=(y-np.amin(y))/(np.amax(y)-np.amin(y))
poly = PolynomialFeatures(degree=2, include_bias=False)  # degree=2 用于生成二次交互项
X_poly = poly.fit_transform(X)  # 将原特征扩展为包括交互项的新特征矩阵

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# 定义岭回归模型并进行训练
ridge = Ridge(alpha=1.0)  # alpha即为正则化参数λ
ridge.fit(X_train, y_train)

# 进行预测
y_pred = ridge.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')

n=len(y)
p=X.shape[1]
r2 = r2_score(y_test, y_pred)
r2=1 - (1 - r2) * ((n - 1) / (n - p - 1))
print('R_squared_adjused:',r2)

# 动态生成特征名称列表
feature_names = [f'x{i+1}' for i in range(p)]
feature_names = poly.get_feature_names_out(input_features=feature_names)  # 特征名称
coefficients = ridge.coef_
print("Feature Names and Coefficients:")
for name, coef in zip(feature_names, coefficients.flatten()):
    print(f'{name}: {coef:.4f}')

# 打印回归系数
# print('Coefficients:', ridge.coef_)


