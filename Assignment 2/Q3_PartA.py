import numpy as np
import matplotlib.pyplot as plt

def NewtonRaphson(f, df, x0, k_max, eps_x, eps_f):
    x = x0
    conv = False
    err_arr = []

    for k in range(k_max):
        fx = f(x) 
        dx = -fx / df(x) 
        max_err = abs(dx)  
        res = abs(fx)  
        err_arr.append(max_err)

        print(f'Iteration {k + 1}: err={max_err:.4e}, res={res:.4e}')

        if max_err < eps_x and res < eps_f:
            conv = True
            break
            
        x = x + dx

    if (conv == False):
        print(f'No convergence after {k_max} interations')

    return x, err_arr, res, conv

def f(x):
    return 4*x**4 - 8*x**3 + 7*x**2 - 3*x + 0.5

def df(x):
    return 16 * x**3 - 24 * x**2 + 14 * x - 3

def f2(x):
    return np.exp(x)**(-x) + np.cos(x+1) - 1

def df2(x):
    return -np.exp(-x) - np.sin(x+1)

x0 = 1
k_max = 100
eps_x = 10e-12
eps_f = 1.0

x_nr, err_nr, res_nr, conv_nr = NewtonRaphson(f, df, x0, k_max, eps_x, eps_f)
x_nr2, err_nr2, res_nr2, conv_nr2 = NewtonRaphson(f2, df2, x0, k_max, eps_x, eps_f)

plt.semilogy(np.arange(len(err_nr)), err_nr, label='Newton-Raphson')
plt.semilogy(np.arange(len(err_nr2)), err_nr2, label='Newton-Raphson2')

plt.xlabel('Iteration')
plt.ylabel('Approximate Error')
plt.title('Approximate Error vs. Iteration')
plt.legend()

plt.show()
