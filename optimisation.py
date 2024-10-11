from pylab import *
from typing import Callable
from datetime import datetime


'''
Quadratic test function: f(x) = 1/2 * x^T * A * x + b^T x
Rosenbrock's test function: f(x) = 100 * (y - x**2)**2 + (1 - x)**2
'''


def main():
    start = datetime.now()
    # 1.a) quasi-Newton DFP, exact line search
    error = 10**-6
    A = np.array([[1, 0, -1, 3], [0, 2, 1, 0], [-1, 1, 6, -1], [3, 0, -1, 10]])
    b = np.array([[2], [18], [-19], [-5]])
    x0 = np.array([[0], [0], [0], [0]])
    f = lambda x: (
        0.5 * np.linalg.multi_dot([np.transpose(x), A, x])
        + np.dot(np.transpose(b), x)
    )
    g = lambda x: np.dot(A, x) + b
    C0 = np.identity(np.size(x0))
    g0 = g(x0)
    d0 = -np.dot(C0, g0)
    alpha0 = (
        -np.dot(np.transpose(d0), g0)
        / np.linalg.multi_dot([np.transpose(d0), A, d0])
    )
    x = x0 + alpha0 * d0
    k = 1
        
    while np.linalg.norm(g(x)) >= error:
        gx = g(x)
        delta_x = x - x0
        delta_g = gx - g(x0)
        C = (
            C0 + np.dot(delta_x, np.transpose(delta_x))
            / np.dot(np.transpose(delta_x), delta_g)
            - np.linalg.multi_dot([C0, delta_g, np.transpose(delta_g), C0])
            / np.linalg.multi_dot([np.transpose(delta_g), C0, delta_g])
        )
        C0 = np.copy(C)
        d = -np.dot(C, gx)
        alpha = (
            -np.dot(np.transpose(d), gx)
            / np.linalg.multi_dot([np.transpose(d), A, d])
        )
        x0 = np.copy(x)
        x += alpha * d
        k += 1
    
    print(datetime.now() - start)
    print("Quasi-Newton DFP:")
    print(f"k = {k}, x = {x.flatten()}, ||g|| = {np.linalg.norm(g(x))}")
    
    start = datetime.now()
    # 1.b) steepest descent, exact line search
    x = np.array([[0.], [0.], [0.], [0.]])
    gx = g(x)
    dx = -gx
    alpha = (
        -np.dot(np.transpose(dx), gx)
        / np.linalg.multi_dot([np.transpose(dx), A, dx])
    )
    x += alpha * dx
    k = 1
    
    while np.linalg.norm(g(x)) >= error:
        gx = g(x)
        dx = -gx
        alpha = (
            -np.dot(np.transpose(dx), gx)
            / np.linalg.multi_dot([np.transpose(dx), A, dx])
        )
        x += alpha * dx
        k += 1
        
    print(datetime.now() - start)
    print("Steepest descent:")
    print(f"k = {k}, x = {x.flatten()}, ||g|| = {np.linalg.norm(g(x))}")
        
    start = datetime.now()
    # (2) quasi-Newton DFP, Newton's root finding method; Rosenbrock's function
    error = 10 ** -4
    x = np.array([[-1.], [2.]])
    x_list = [np.copy(x)]
    f = lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    g = lambda x: np.array([
        2 * (200 * x[0]**3 - 200 * x[0] * x[1] + x[0] - 1),
        200 * (x[1] - x[0]**2)
    ])
    C0 = np.identity(np.size(x))
    g0 = g(x)
    d = -np.dot(C0, g0)
    df_tilde = lambda a, x, d: (
        200 * (d[1] - 2 * d[0] * (x[0] + a * d[0]))
        * ((x[1] + a * d[1]) - (x[0] + a * d[0])**2)
        - 2 * d[0]*(1 - (x[0] + a * d[0]))
    )
    d2f_tilde = lambda a, x, d: (
        -400 * (d[0]**2) * ((x[1] + a * d[1]) - (x[0] + a * d[0])**2)
        + 200 * (d[1] - 2 * d[0] * (x[0] + a * d[0]))
        * (d[1] - 2 * d[0] * (x[0] + a * d[0])) + 2 * d[0]**2
    )
    x += newton(x, d, df_tilde, d2f_tilde, 1 / np.linalg.norm(d), error**2) * d
    x_list.append(np.copy(x))

    while np.linalg.norm(g(x)) >= error:
        gx = g(x)
        delta_x = x - x_list[-2]
        delta_g = gx - g(x_list[-2])
        C = (
            C0 + np.dot(delta_x, np.transpose(delta_x))
            / np.dot(np.transpose(delta_x), delta_g)
            - np.linalg.multi_dot([C0, delta_g, np.transpose(delta_g), C0])
            / np.linalg.multi_dot([np.transpose(delta_g), C0, delta_g])
        )
        C0 = np.copy(C)
        d = -np.dot(C, gx)
        x += newton(
            x,
            d,
            df_tilde,
            d2f_tilde,
            1 / np.linalg.norm(d),
            error**2
        ) * d
        x_list.append(np.copy(x))
    
    print(datetime.now() - start)
    print("Rosenbrock's function:")
    print(
        f"k = {len(x_list) - 1},"
        f"x = {x.flatten()},"
        f"||g|| = {np.linalg.norm(g(x))}"
    )
    xvals, yvals = np.split(np.array([i.flatten() for i in x_list]), 2, axis=1)
    plt.plot(xvals, yvals, 'ro-', linewidth=1.0)
    plt.text(xvals[0], yvals[0], r"$x^{(0)}$")
    plt.text(xvals[-1], yvals[-1], fr"$x^{{({len(xvals)-1})}}$")
    plt.title(r"$f(x_1,x_2)=100(x_2-x_1^2)^2+(1-x_1)^2$")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    X, Y = np.meshgrid(
        np.arange(xvals.min() - .5, xvals.max() + .5, .01),
        np.arange(yvals.min() - .5, yvals.max() + .5, .01)
    )
    Z = f([X, Y])
    plt.contour(X, Y, Z, 100)
    plt.show()
    
    
def newton(
    x: np.ndarray,
    d: np.ndarray,
    df_tilde: callable,
    d2f_tilde: callable,
    a: np.ndarray,
    error: float
) -> float:
    a -= df_tilde(a, x, d) / d2f_tilde(a, x, d)
    while abs(df_tilde(a, x, d)) > error:
        a -= df_tilde(a, x, d) / d2f_tilde(a, x, d)
    return a
    
if __name__ == "__main__":
    main()
