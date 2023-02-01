import numpy as np

def EulerChebyshev(f, df, ddf, x0, kMax, eps_x, eps_f):
    # Input: initial point, max number of iterations, tolerance for error and residual

    x = x0
    conv = False  # flag for convergence

    for k in range(kMax):
        fx = f(x)  # current function value
        df_x = df(x)
        ddf_x = ddf(x)
        x_new = x - (2 * fx * df_x) / (2 * (df_x ** 2) - fx * ddf_x)# update step
        err = abs(x_new - x)  # current error estimate
        res = abs(fx)  # current residual

        print(f'Iteration {k + 1}: err={err:.4e}, res={res:.4e}')

        if err < eps_x and res < eps_f:  # If converged ...
            print("Converged!")
            conv = True
            break
        x = x_new

    if (conv == False):
        print("No convergence!")

    return x, err, res, conv

def f(x):
    return np.exp(-x) + np.cos(x + 1) - 1

def df(x):
    return -np.exp(-x) - np.sin(x + 1)

def ddf(x):
    return np.exp(-x) - np.cos(x + 1)


#  assign parameter values
x0 = 1.0  # initial guess
kMax = 100  # max number of iterations
eps_x = 1e-8  # tolerance for error
eps_f = 1e-8  # tolerance for residual

#  call NewtonRaphson
x, err, res, conv = EulerChebyshev(f, df, ddf, x0, kMax, eps_x, eps_f)

#  print out solution
print("x= " + str(x))