from pylab import *
from scipy.integrate import solve_ivp
from typing import Callable
from random import uniform
from scipy.spatial.distance import pdist


def main():
    # (1) Hamiltonian of a single particle in a double well
    T = [0, 15]
    y0 = [-2, 0]
    dt = 1e-2
    # solve_ivp solution
    sol = solve_ivp(odes_1p, T, y0, method='RK45')
    t = sol.t
    q, p = sol.y
    h = H_1p(q, p)
    plt.plot(t, h, label='solve_ivp')
    # leapfrog solution
    tl, ql, pl = leapfrog(f_1p, T, y0, dt)
    hl = H_1p(ql, pl)
    plt.plot(tl, hl, label='leapfrog')
    # velocity_verlet solution
    tv, qv, pv = velocity_verlet(f_1p, T, y0, dt)
    hv = H_1p(qv, pv)
    plt.plot(tv, hv, label='velocity_verlet')
    # axes labels and title
    plt.xlabel('$t$')
    plt.ylabel('$H(t)$')
    plt.title('Hamiltonian of a single particle in a double well')
    plt.legend()
    plt.show()
    # (2) Hamiltonoian of a 1d 'm'-particle chain
    m = 10
    T = [0, 25]
    y0 = [i + .075 * uniform(-1, 1) for i in range(m)] + [0] * m
    dt = 1e-2
    # solve_ivp positions
    sol = solve_ivp(odes_m, T, y0, method='RK45')
    t = sol.t
    q, p = np.split(sol.y, 2)
    
    for i in range(m):
        plt.plot(t, q[i], label=fr'$q_{i}$')
    
    plt.xlabel('$t$')
    plt.ylabel('$q_i(t)$')
    plt.title(fr'Particle positions in a {m} particle 1D-chain, solve_ivp')
    plt.legend()
    plt.show()
    # solve_ivp hamiltonian
    h = [H_m(q[:,i], p[:,i]) for i in range(len(t))]
    plt.plot(t, h)
    plt.xlabel('$t$')
    plt.ylabel('$H(t)$')
    plt.title(fr'Hamiltonian of a {m} particle 1D-chain, solve_ivp')
    plt.show()
    # velocity_verlet positions
    tv, qv, pv = velocity_verlet(f_m, T, y0, dt)
    
    for i in range(m):
        plt.plot(tv, qv[:,i], label=fr'$q^v_{i}$')
    
    plt.xlabel('$t$')
    plt.ylabel('$q_i(t)$')
    plt.title(
        fr'Particle positions in a {m} particle 1D-chain,'
        fr'VV with $\Delta t = {dt}$'
    )
    plt.legend()
    plt.show()
    # velocity_verlet hamiltonian
    hv = [H_m(qv[i], pv[i]) for i in range(len(tv))]
    plt.plot(tv, hv, label='velocity_verlet')
    plt.xlabel('$t$')
    plt.ylabel('$H(t)$')
    plt.title(
        fr'Hamiltonian of a {m} particle 1D-chain, VV with $\Delta t = {dt}$'
    )
    plt.legend()
    plt.show()
    
    
def leapfrog(f: callable, T: list, y0: list, dt: float) -> list:
    N = int((T[1] - T[0]) / dt)
    m = len(y0) // 2
    t = np.linspace(T[0], T[1], N + 1)
    q, p = np.zeros((len(t), m)), np.zeros((len(t), m))
    q[0], p[0] = np.split(np.array(y0), 2)
    
    for i in range(N):
        p[i + 1] = p[i] + dt * f(q[i])
        q[i + 1] = q[i] + dt * p[i + 1]
    
    return [t, q, p]
    
    
def velocity_verlet(f: callable, t0: list, y0: list, dt: float) -> list:
    N = int((t0[1] - t0[0]) / dt)
    m = len(y0) // 2
    t = np.linspace(t0[0], t0[1], N + 1)
    q, p = np.zeros((len(t), m)), np.zeros((len(t), m))
    q[0], p[0] = np.split(np.array(y0), 2)
    a = f(q[0])
    
    for i in range(N):
        q[i + 1] = q[i] + dt * p[i] + .5 * (dt ** 2) * a
        an = f(q[i + 1])
        p[i + 1] = p[i] + .5 * dt * (a + an)
        a = an
    
    return [t, q, p]
    
    
def u_1p(q: float) -> float:
    return (q**2 / 2) * (q**2 / 2 - 1)
    
    
def f_1p(q: float) -> float:
    return q * (1 - q**2)
    

def H_1p(q: float, p: float):
    return p**2 / 2 + u_1p(q)


def odes_1p(t: float, x: list) -> list:
    q, p = x
    
    return [p, f_1p(q)]

    
def u_m(x: float) -> float:
    return (x ** -6) * (-2 + x**-6)


def f_m(q: list) -> float:
    def f(qi: float) -> float:
        x = qi - q
        x = x[x != 0]
        return (12 * x**-7 * (x**-6 - 1)).sum()
    
    return np.array([f(qi) for qi in q])
    
    
def H_m(q: list, p: list) -> float:
    vu_m = np.vectorize(u_m)
    
    return .5 * (np.square(p).sum() + vu_m(pdist(np.expand_dims(q, 1))).sum())
    
    
def odes_m(t: float, x: list) -> list:
    q, p = np.split(np.array(x), 2)
    
    return list(np.concatenate([p, f_m(q)]))
    
    
if __name__ == '__main__':
    main()
