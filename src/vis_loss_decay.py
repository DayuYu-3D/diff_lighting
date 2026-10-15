import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# 构建指数函数模型
def exponential_loss_function(p, min_loss, max_loss):
    def exponential_model(p, a, b):
        return a * np.exp(b * p)

    p_values = np.array([1.0, 0.0166])
    loss_values = np.array([min_loss, max_loss])
    # 使用 curve_fit 拟合指数函数模型，得到参数 a 和 b
    params_exp, covariance_exp = curve_fit(exponential_model, p_values, loss_values)
    a_exp, b_exp = params_exp
    return a_exp * np.exp(b_exp * p)


def vis_loss(pecent, type='exp', min_loss=0.003, max_loss=0.03):
    if type == 'exp':
        return exponential_loss_function(pecent, min_loss, max_loss)
    else:
        raise ValueError("Invalid decay_type function")
    