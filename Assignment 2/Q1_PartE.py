import numpy as np

def NewtonRaphson(f, df, x0, k_max, eps_x, eps_f):
    x = x0
    conv = False  

    for k in range(k_max):
        fx = f(x)  
        dx = -fx / df(x)  
        max_err = abs(dx)  
        res = abs(fx)  

        print(f'Iteration {k + 1}: err={max_err:.4e}, res={res:.4e}')

        if max_err < eps_x and res < eps_f:
            conv = True
            break
        x = x + dx

    if (conv == False):
        print(f'No convergence after {k_max} interations')

    return x, max_err, res, conv

def f(x):
    return np.exp(x)**(-x) + np.cos(x+1) - 1

def df(x):
    return -np.exp(-x) - np.sin(x+1)

x0 = 1
k_max = 100
eps_x = 1.0e-4
eps_f = 1.0

xstar, err, res, conv = NewtonRaphson(f, df, x0, k_max, eps_x, eps_f)

if (conv):
    print(f'\n x* = {xstar}')
