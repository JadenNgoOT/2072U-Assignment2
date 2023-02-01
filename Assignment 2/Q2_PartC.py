import numpy as np
import matplotlib.pyplot as plt

def EulerChebyshev(f, df, ddf, x0, k_max, eps_x, eps_f):
    x = x0
    conv = False  
    err_arr = []

    for k in range(k_max):
        fx = f(x)  
        df_x = df(x)
        ddf_x = ddf(x)
        x_new = x - (2 * fx * df_x) / (2 * (df_x ** 2) - fx * ddf_x)
        max_err = abs(x_new - x)  
        res = abs(fx)
        err_arr.append(max_err)

        print(f'Iteration {k + 1}: err={max_err:.4e}, res={res:.4e}')

        if max_err < eps_x and res < eps_f: 
            conv = True
            break
            
        x = x_new

    if (conv == False):
       print(f'No convergence after {k_max} interations')

    return x, err_arr, res, conv

def NewtonRaphson(f, df, x0, k_max, eps_x, eps_f):
    # Input: initial point, max number of iterations, tolerance for error and residual

    x = x0
    conv = False  # flag for convergence
    err_arr = []

    for k in range(k_max):
        fx = f(x)  # current function value
        dx = -fx / df(x)  # update step
        max_err = abs(dx)  # current error estimate
        res = abs(fx)  # current residual
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
    return np.exp(-x) + np.cos(x + 1) - 1

def df(x):
    return -np.exp(-x) - np.sin(x + 1)

def ddf(x):
    return np.exp(-x) - np.cos(x + 1)

x0 = 1.0  
k_max = 100  
eps_x = 10e-14  
eps_f = 1e-8  


x_nr, err_nr, res_nr, conv_nr = NewtonRaphson(f, df, x0, k_max, eps_x, eps_f)
x_ec, err_ec, res_ec, conv = EulerChebyshev(f, df, ddf, x0, k_max, eps_x, eps_f)

plt.semilogy(np.arange(len(err_nr)), err_nr, label='Newton-Raphson')
plt.semilogy(np.arange(len(err_ec)), err_ec, label='EulerChebyshev')

plt.xlabel('Iteration')
plt.ylabel('Approximate Error')
plt.title('Approximate Error vs. Iteration')
plt.legend()

plt.show()
