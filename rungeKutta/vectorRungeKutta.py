from pylab import *
from typing import Callable


def main():
    def f(t: float, y: list) -> list:
        return [-2 * y[0] + sin(y[1]) + exp(-t) * sin(t), cos(y[0]) - 4 * y[1]]

    y0 = (0.5, 1)
    t0 = 0
    T = 2
    N = 25
    t, u = vector_RK(f, t0, T, y0, N, *RK_methods['SSPRK3'])
    plt.plot(u.transpose()[0], u.transpose()[1])
    plt.xlabel(rf'$u^{{(1)}}(t)$')
    plt.ylabel(rf'$u^{{(2)}}(t)$')
    plt.show()
    
    # Adaptive
    error = 10**-4
    t, u = adaptive_vector_RK(f, t0, T, y0, error, *adaptive_methods['RKDP'])
    plt.plot(u.transpose()[0], u.transpose()[1])
    plt.xlabel(rf'$u^{{(1)}}(t)$')
    plt.ylabel(rf'$u^{{(2)}}(t)$')
    plt.title(rf'{len(t)} steps')
    plt.show()
    

# key: (s, c, b, A)
RK_methods = {
    'Forward Euler': (1, [0], [1], [[0]]),
    'Explicit midpoint': (2, [0, 1/2], [0, 1], [[0, 0], [1/2, 0]]),
    'Heun\'s': (2, [0, 1], [1/2, 1/2], [[0, 0], [1, 0]]),
    'Kutta\'s third-order': (
        3,
        [0, 1/2, 1],
        [1/6, 2/3, 1/6],
        [[0, 0, 0], [1/2, 0, 0], [-1, 2, 0]]
    ),
    'SSPRK3': (
        3,
        [0, 1, 1/2],
        [1/6, 1/6, 2/3],
        [[0, 0, 0], [1, 0, 0], [1/4, 1/4, 0]]
    )
}


# key (s_plus, c_tilda, b_tilda, A_tilda, b, p)
adaptive_methods = {
    'RKDP': (
        7,
        [0, 1/5, 3/10, 4/5, 8/9, 1, 1],
        [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40],
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        ],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
        4
    )
}
    
    
def vector_RK(
    f: callable,
    t0: float,
    T: float,
    y0: list,
    N: int,
    s: int,
    c: list,
    b: list,
    A: list
) -> tuple:
    m = len(y0)
    delta_t = (T - t0) / N
    t = np.linspace(t0, T, N)
    u = np.zeros((N, m))
    u[0] = y0
    K = np.zeros((s, m))
    
    for n in range(N - 1):
        for i in range(s):
            v = np.dot(A, K)
            K[i] = f(t[n] + c[i] * delta_t, u[n] + v[i] * delta_t)
        
        u[n + 1] = u[n] + np.dot(b, K) * delta_t
    
    return (t, u)


def adaptive_vector_RK(
    f: callable,
    t0: float,
    T: float,
    y0: float,
    error: float,
    s_plus: int,
    c_tilda: list,
    b_tilda: list,
    A_tilda: list,
    b: list,
    p: int
) -> tuple:   
    m = len(y0)
    t = [t0]
    u = [y0]
    # When might this not be valid or is this valid for all valid errors?
    delta_t = error
    K_tilda = np.zeros((s_plus, m))
    
    while t[-1] < T:
        for i in range(s_plus):
            v = np.dot(A_tilda, K_tilda)
            K_tilda[i] = f(t[-1] + c_tilda[i] * delta_t, u[-1] + v[i] * delta_t)
        
        tau = max(
            abs(np.dot(np.array(b_tilda) - np.array(b), K_tilda[:,i]))
            for i in range(m)
        )
        q = (error / tau) ** (1 / p)
        delta_t = min(q * delta_t, T - t[-1])
        t.append(t[-1] + delta_t)
        
        for i in range(s_plus):
            K_tilda[i] = f(t[-1] + c_tilda[i] * delta_t, u[-1] + v[i] * delta_t)
        
        u.append(u[-1] + np.dot(b, K_tilda * delta_t))
    
    return (np.array(t), np.array(u))


if __name__ == '__main__':
    main()
