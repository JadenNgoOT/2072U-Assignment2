import numpy as np

def NewtonRaphson(f, df, x0, kMax, eps_x, eps_f):
    # Input: initial point, max number of iterations, tolerance for error and residual

    x = x0
    conv = False  # flag for convergence

    for k in range(kMax):
        fx = f(x)  # current function value
        dx = -fx / df(x)  # update step
        err = abs(dx)  # current error estimate
        res = abs(fx)  # current residual

        print(f'Iteration {k + 1}: err={err:.4e}, res={res:.4e}')

        if err < eps_x and res < eps_f:  # If converged ...
            print("Converged!")
            conv = True
            break
        x = x + dx

    if (conv == False):
        print("No convergence!")

    return x, err, res, conv

def f(x):
    return np.exp(x)**(-x) + np.cos(x+1) - 1


#  put function definition of df here (derivative of f)
def df(x):
    return -np.exp(-x) - np.sin(x+1)


#  assign parameter values
x0 = 1
k_max = 100
eps_x = 1.0e-4
eps_f = 1.0

#  call NewtonRaphson
x, err, res, conv = NewtonRaphson(f, df, x0, k_max, eps_x, eps_f)

#  print out solution
print("x= " + str(x))