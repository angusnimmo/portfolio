from pylab import *
from typing import Callable
from scipy.misc import derivative


def main():
    def fa(t: float, y: float) -> float:
        return -y
    
    def fb(t: float, y: float) -> float:
        return -sin(t)
    
    def fc(t: float, y: float) -> float:
        return -y * sin(y)
    
    FEa = [forward_euler(fa, 0, 5, 1, 20 * 2**m) for m in range(6)]
    CNa = [crank_nicolson(fa, 0, 5, 1, 20 * 2**m) for m in range(6)]
    FEb = [forward_euler(fb, 0, 7 * pi / 2, 1, 20 * 2**m) for m in range(6)]
    CNb = [crank_nicolson(fb, 0, 7 * pi / 2, 1, 20 * 2**m) for m in range(6)]
    FEc = [forward_euler(fc, 0, 5, 1, 20 * 2**m) for m in range(6)]
    CNc = [crank_nicolson(fc, 0, 5, 1, 20 * 2**m) for m in range(6)]
    
    # a) y(t) = exp(-t)
    for m in range(6):
        plt.plot(
            FEa[m][0],
            abs(FEa[m][1] - exp(-FEa[m][0])),
            label=fr'FE, $\Delta t = {5/(20*2**m)}$'
        )
        plt.plot(
            CNa[m][0],
            abs(CNa[m][1] - exp(-CNa[m][0])),
            label=fr'CN, $\Delta t = {5/(20*2**m)}$'
        )
    plt.title(
        fr'Forward Euler and Crank Nicolson Testing for $y(t) = e^{{-t}}$'
    )
    plt.xlabel(fr'$t$')
    plt.ylabel(fr'$|y(t_{{N}})-u_{{N}}|$')
    plt.legend()
    plt.show()
    
    # b) y(t) = cos(t)
    for m in range(6):
        plt.plot(
            FEb[m][0],
            abs(FEb[m][1] - cos(FEb[m][0])),
            label=fr'FE, $\Delta t = {5/(20*2**m)}$'
        )
        plt.plot(
            CNb[m][0],
            abs(CNb[m][1] - cos(CNb[m][0])),
            label=fr'CN, $\Delta t = {5/(20*2**m)}$'
        )
    plt.title(fr'Forward Euler and Crank Nicolson Testing for $y(t) = cos(t)$')
    plt.xlabel(fr'$t$')
    plt.ylabel(fr'$|y(t_{{N}})-u_{{N}}|$')
    plt.legend()
    plt.show()
    
    # Forward Euler conditional stability
    for N in {1, 2, 3, 4, 6, 12}:
        a, b = forward_euler(fa, 0, 12, 1, N)
        plt.plot(a, abs(b - exp(-a)), label=fr'FE, $\Delta t = {12/N}$')
    plt.title(fr'Forward Euler Instability for $y(t) = e^{{-t}}$')
    plt.xlabel(fr'$t$')
    plt.ylabel(fr'$|y(t_{{N}})-u_{{N}}|$')
    plt.legend()
    plt.show()
    
    # c) dy/dx = -y * sin(y)
    for m in range(6):
        plt.plot(FEc[m][0], FEc[m][1], label=fr'FE, $\Delta t = {5/(20*2**m)}$')
        plt.plot(CNc[m][0], CNc[m][1], label=fr'CN, $\Delta t = {5/(20*2**m)}$')
    plt.title(
        fr'Forward Euler and Crank-Nicolson Solutions to the ODE:'
        fr'$\frac{{dy}}{{dt}} = -ysin(y)$'
    )
    plt.xlabel(fr'$t$')
    plt.ylabel(fr'$u_{{n}}$')
    plt.legend()
    plt.show()


def forward_euler(f: callable, t0: float, T: float, y0: float, N: int) -> tuple:
    delta_t = (T - t0) / N
    t = np.linspace(t0, T, N)
    u = np.zeros(N)
    u[0] = y0
    for n in range(N - 1):
        u[n + 1] = u[n] + delta_t * f(t[n], u[n])
    return (t, u)
    
    
def newton_method(g: callable, x0: float, error: float) -> float:
    z = x0
    e = abs(g(z))
    while e > error:
        z = z - g(z) / derivative(g, z)
        e = g(z)
    return z
    
    
def crank_nicolson(
    f: callable,
    t0: float,
    T: float,
    y0: float,
    N: int
) -> tuple:
    delta_t = (T - t0) / N
    t = np.linspace(t0, T, N)
    u = np.zeros(N)
    u[0] = y0
    for n in range(N - 1):
        def g(z):
            return u[n] + (delta_t * (f(t[n], u[n]) + f(t[n + 1], z)) / 2) - z
            
        u[n + 1] = newton_method(g, u[n] + delta_t * f(t[n], u[n]), 10e-8)
    return (t, u)
    
    
if __name__ == '__main__':
    main()
