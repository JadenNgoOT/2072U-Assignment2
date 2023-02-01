import numpy as np
import matplotlib.pyplot as plt

def EulerChebyshev(f, df, ddf, x0, kMax, eps_x, eps_f):
    # Input: initial point, max number of iterations, tolerance for error and residual

    x = x0
    conv = False  # flag for convergence
    errs = []

    for k in range(kMax):
        fx = f(x)  # current function value
        df_x = df(x)
        ddf_x = ddf(x)
        x_new = x - (2 * fx * df_x) / (2 * (df_x ** 2) - fx * ddf_x)# update step
        err = abs(x_new - x)  # current error estimate
        res = abs(fx)  # current residual
        errs.append(err)

        print(f'Iteration {k + 1}: err={err:.4e}, res={res:.4e}')

        if err < eps_x and res < eps_f:  # If converged ...
            print("Converged!")
            conv = True
            break
        x = x_new

    if (conv == False):
        print("No convergence!")

    return x, errs, res, conv

def NewtonRaphson(f, df, x0, kMax, eps_x, eps_f):
    # Input: initial point, max number of iterations, tolerance for error and residual

    x = x0
    conv = False  # flag for convergence
    errs = []

    for k in range(kMax):
        fx = f(x)  # current function value
        dx = -fx / df(x)  # update step
        err = abs(dx)  # current error estimate
        res = abs(fx)  # current residual
        errs.append(err)

        print(f'Iteration {k + 1}: err={err:.4e}, res={res:.4e}')

        if err < eps_x and res < eps_f:  # If converged ...
            print("Converged!")
            conv = True
            break
        x = x + dx

    if (conv == False):
        print("No convergence!")

    return x, errs, res, conv

def f(x):
    return np.exp(-x) + np.cos(x + 1) - 1

def df(x):
    return -np.exp(-x) - np.sin(x + 1)

def ddf(x):
    return np.exp(-x) - np.cos(x + 1)


#  assign parameter values
x0 = 1.0  # initial guess
k_max = 100  # max number of iterations
eps_x = 10e-14  # tolerance for error
eps_f = 1e-8  # tolerance for residual

#  print out solution
print("x= " + str(x))

x_nr, err_nr, res_nr, conv_nr = NewtonRaphson(f, df, x0, k_max, eps_x, eps_f)
x, err_ec, res_ec, conv = EulerChebyshev(f, df, ddf, x0, k_max, eps_x, eps_f)

plt.semilogy(np.arange(len(err_nr)), err_nr, label='Newton-Raphson')
plt.semilogy(np.arange(len(err_ec)), err_ec, label='EulerChebyshev')

plt.xlabel('Iteration')
plt.ylabel('Approximate Error')
plt.title('Approximate Error vs. Iteration')
plt.legend()

plt.show()