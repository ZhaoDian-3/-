import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
x_data = np.linspace(0, 4, 50)
y_data = func(x_data, 2.5, 1.3, 0.5) + 0.2 * np.random.normal(size=len(x_data))
popt, pcov = curve_fit(func, x_data, y_data)
plt.figure()
plt.plot(x_data, y_data, 'b-', label='data')
plt.plot(x_data, func(x_data, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
