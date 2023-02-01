import numpy as np
import matplotlib.pyplot as plt

def bisect(f, a0, b0, k_max, eps_x, eps_f):
    conv = False  # flag for convergence, default is "not converged"

    if (f(a0) * f(b0) > 0):  # check to see whether we can guarantee convergence via IVT condition
        print('Error. f(a0) f(b0) > 0: Starting condition not satisfied.')
        return None, None, conv  # abort and print message if we can't guarantee convergence

    a = a0
    b = b0
    max_errs = []

    for k in range(k_max):  # loop over at most k_max bisection steps
        c = (a + b) / 2.0  # find the current midpoint
        f_mid = f(c)  # compute the function value at the midpoint
        f_left = f(a)  # compute the function value at the current left boundary
        if (f_mid * f_left > 0):  # if they have the same sign...
            a = c  # update the left boundary, otherwise...
        else:
            b = c  # update the right boundary
        max_err = abs(b - a)  # compute the maximal error and the residual
        res = abs(f_mid)
        max_errs.append(max_err)

        print(
            f'iteration {k + 1}, err={max_err:.4e} and res={res:.4e}')  # Since k starts at 0, I added 1 to k to make the first iteration 1

        if (max_err < eps_x) and (res < eps_f):  # if both are less than their tolerance, stop iterations
            conv = True  # set the convergence flag to "converged"
            break

    if (conv == False):  # print warning if the iterations did not converge
        print(f'No convergence after {k_max} interations')

    return c, max_errs, conv  # return the approximate solution, maximal error and convergence flag

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
    return np.exp(x)**(-x) + np.cos(x+1) - 1


#  put function definition of df here (derivative of f)
def df(x):
    return -np.exp(-x) - np.sin(x+1)

x0 = 1
k_max = 100
eps_x = 10e-12
eps_f = 1.0
a0=0
b0=2

x_bis, err_bis, conv_bis = bisect(f, 0, 2, k_max, eps_x, eps_f)
x_nr, err_nr, res_nr, conv_nr = NewtonRaphson(f, df, x0, k_max, eps_x, eps_f)

plt.semilogy(np.arange(len(err_bis)), err_bis, label='Bisection')
plt.semilogy(np.arange(len(err_nr)), err_nr, label='Newton-Raphson')

plt.xlabel('Iteration')
plt.ylabel('Approximate Error')
plt.title('Approximate Error vs. Iteration')
plt.legend()

plt.show()