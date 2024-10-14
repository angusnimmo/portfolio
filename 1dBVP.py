from pylab import *
from typing import Callable


def main():
    def f(x: float) -> float:
        return cos(2 * pi * x) - sin(pi * x) / 2
    
    def phi(x: float) -> float:
        return (
            1 + 2 * pi * x - cos(2 * pi * x) + 2 * sin(pi * x)
        ) / (4 * pi**2)
    
    x, u = BVP1d(f, [0, 1], 25, [[1, 0, 0], [0, 1, 0]])
    v_phi = np.vectorize(phi)
    y = v_phi(x)
    plt.plot(x, u, 'x', label='$u(x)$')
    plt.plot(x, y, label='$y(x)$')
    plt.xlabel('$x$')
    plt.legend()
    plt.show()


# c = [[alpha_a, beta_a, gamma_a], [alpha_b, beta_b, gamma_b]]
def BVP1d(f: callable, ab: list, N: int, bc: list) -> list:
    h = (ab[1] - ab[0]) / N
    Aa = (h * bc[0][0] + 2 * bc[0][1]) / (2 * bc[0][1] - h * bc[0][0])
    Ab = (2 * bc[1][1] - h * bc[1][0]) / (h * bc[1][0] + 2 * bc[1][1])
    Ba = 2 * h * bc[0][2] / (2 * bc[0][1] - h * bc[0][0])
    Bb = -2 * h * bc[1][2] / (h * bc[1][0] + 2 * bc[1][1])
    L = np.zeros((N, N))
    
    for i in range(N - 1):
        L[i][i] = -2
        L[i][i+1] = 1
        L[i+1][i] = 1
    
    L[0][0] += Aa
    L[N - 1][N - 1] = Ab - 2
    B = np.zeros(N)
    B[0] = Ba
    B[-1] = Bb
    x = np.linspace(ab[0] + h / 2, ab[1] - h / 2, N)
    vf = np.vectorize(f)
    S = B + vf(x) * h**2
    phi = np.linalg.solve(L, S)
    
    return [x, phi]
    

if __name__ == '__main__':
    main()
