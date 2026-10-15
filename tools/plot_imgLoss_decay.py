import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 定义指数函数模型
def exponential_model(p, a, b):
    return a * np.exp(b * p)

# 定义线性模型
def linear_model(p, m, c):
    return m * p + c

# 模拟已知的数据点
p_values = np.array([1.0, 0.0166])
loss_values = np.array([0.003, 0.03])

# 使用 curve_fit 拟合指数函数模型，得到参数 a 和 b
params_exp, covariance_exp = curve_fit(exponential_model, p_values, loss_values)

# 使用 curve_fit 拟合线性函数模型，得到参数 m 和 c
params_linear, covariance_linear = curve_fit(linear_model, p_values, loss_values)

# 提取指数函数模型参数
a_exp, b_exp = params_exp

# 提取线性函数模型参数
m_linear, c_linear = params_linear

# 构建指数函数模型
def exponential_loss_function(p):
    return a_exp * np.exp(b_exp * p)

# 构建线性函数模型
def linear_loss_function(p):
    return m_linear * p + c_linear

# 测试不同 p 值对应的 loss
test_p_values = np.linspace(0.0166, 1, 100)
test_loss_values_exp = exponential_loss_function(test_p_values)
test_loss_values_linear = linear_loss_function(test_p_values)

print('a_exp: ', exponential_loss_function(0.55))
print('b_exp: ', linear_loss_function(0.55))

# 可视化拟合结果
plt.scatter(p_values, loss_values, label='a')
plt.plot(test_p_values, test_loss_values_exp, label='b')
plt.plot(test_p_values, test_loss_values_linear, label='v')
plt.xlabel('p')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss_with_linear.png')
