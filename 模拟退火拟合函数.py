import numpy as np
from scipy.optimize import dual_annealing

# 定义目标函数（例如，二次函数的残差平方和）
def objective_function(params, x, y):
    a, b, c = params
    y_pred = a * x**2 + b * x + c
    return np.sum((y - y_pred)**2)

# 模拟一些数据点
np.random.seed(0)
x_data = np.linspace(-10, 10, 100)
y_data = 3 * x_data**2 + 2 * x_data + 1 + np.random.normal(0, 10, size=x_data.shape)

# 设置参数的边界
bounds = [(-10, 10), (-10, 10), (-10, 10)]

# 执行模拟退火优化
result = dual_annealing(objective_function, bounds, args=(x_data, y_data))

# 输出结果
print(f"拟合参数: a = {result.x[0]}, b = {result.x[1]}, c = {result.x[2]}")
print(f"最小化的目标函数值: {result.fun}")
print(result)
