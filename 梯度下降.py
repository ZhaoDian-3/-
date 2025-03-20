import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数和梯度
def f(x):
    return x**2 + 4*x + 4
def grad_f(x):
    return 2*x + 4

# 梯度下降算法
def gradient_descent(initial_x, learning_rate, epochs):
    x = initial_x
    history = [x]
    for _ in range(epochs):
        grad = grad_f(x)
        x = x - learning_rate * grad
        history.append(x)
    return x, history

# 参数设置
initial_x = 0.0
learning_rate = 0.1
epochs = 20

# 执行梯度下降
optimal_x, history = gradient_descent(initial_x, learning_rate, epochs)

print(f"最优解 x = {optimal_x}")
print(f"最小值 f(x) = {f(optimal_x)}")

# 绘制收敛过程
x_vals = np.linspace(-5, 1, 100)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals, label='f(x)')
plt.scatter(history, [f(x) for x in history], color='red', label='Iterations')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
