import numpy as np

def EulerChebyshev(f, df, ddf, x0, k_max, eps_x, eps_f):
    x = x0
    conv = False

    for k in range(k_max):
        fx = f(x)
        dfx = df(x)
        ddfx = ddf(x)
        x_new = x - (fx/dfx) - (ddfx/(2*fx)) * ((fx/dfx)**2)
        max_err = abs(x_new - x)
        res = abs(fx) 

        print(f'Iteration {k + 1}: err={max_err:.4e}, res={res:.4e}')

        if max_err < eps_x and res < eps_f:
            conv = True
            break
            
        x = x_new

    if (conv == False):
        print(f'No convergence after {k_max} interations')

    return x, max_err, res, conv

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

xstar, err, res, conv = EulerChebyshev(f, df, ddf, x0, k_max, eps_x, eps_f)

if (conv):
    print(f'\n x* = {xstar}')
