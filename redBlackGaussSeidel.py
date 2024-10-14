from pylab import *
from scipy.integrate import solve_ivp
from typing import Callable


def main():
    # (1) 2d Poisson Equation, RBGS
    def f(x: float, y: float) -> list:
        return np.zeros(x.shape)
    
    x, y, u, r = RBGS(f, 1, 128, np.random.uniform(-1, 1, (128, 128)), 1000)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, u, cmap=cm.viridis)
    fig.colorbar(surf, shrink=.5, label='$u(x,y)$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(
    r'$\frac{\partial^2 \phi}{\partial x^2}'
    r' + \frac{\partial^2 \phi}{\partial y^2} = 0, (x,y) \in [0,L]^2$'
    + '\n' + r'BC: $\phi(0,y) = \phi(L,y), \phi(x,0) = \phi(x,L)$'
    + '\n' r'IC: $\phi_{i,j}^{(0)} = U[-1,1]$'
    )
    plt.show()
    plt.semilogy(np.linspace(1, 1000, 1000), r)
    plt.xlabel('Number of iterations')
    plt.ylabel('Residual')
    plt.show()
    # (2) Diffusion Equation
    T = [0, .25]
    D = 1
    ab = [0, 1]
    bc = [[1, 0, 1], [0, 1, 0]]
    N = 25
    h = (ab[1] - ab[0]) / N
    x = np.linspace(ab[0] + h / 2, ab[1] - h / 2, N)
    y0 = (x**2 - 1)**2
    sol = solve_ivp(diffusion_ode, T, y0, args=(D, ab, bc))
    plt.imshow(sol.y, origin='lower', extent=[0, .25, 0, 1], aspect=.25)
    plt.colorbar(label=r'$y(x,t)$')
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('1D Diffusion')
    plt.show()
    
    
def RBGS(rhs: callable, L: float, N: int, u0: list, M: int) -> list:
    h = L / N
    s = np.linspace(h / 2, L - h / 2, N)
    x, y = np.meshgrid(s, s, indexing='ij')
    f = rhs(x, y)
    u = u0
    r = np.zeros(M)
    
    for n in range(M):
        tmp = 0
        
        for i in range(N):
            ip = (i + 1) % N
            im = (i - 1) % N
            
            for j in range(N):
                jp = (j + 1) % N
                jm = (j - 1) % N
                cross = u[ip, j] + u[i, jp] + u[im, j] + u[i, jm]
                
                if n % 2 == (i + j) % 2:
                    u[i, j] = (cross - f[i, j] * h**2) / 4
                
                tmp += ((cross - 4 * u[i,j]) / h**2 - f[i,j])**2
        
        r[n] = (tmp / N**2)**.5
    
    return [x, y, u, r]


def diffusion_ode(t: float, y: list, D: float, ab: list, bc: list) -> list:
    N = len(y)
    h = (ab[1] - ab[0]) / N
    Aa = (h * bc[0][0] + 2 * bc[0][1]) / (2 * bc[0][1] - h * bc[0][0])
    Ab = (2 * bc[1][1] - h * bc[1][0]) / (h * bc[1][0] + 2 * bc[1][1])
    Ba = 2 * h * bc[0][2] / (2 * bc[0][1] - h * bc[0][0])
    Bb = -2 * h * bc[1][2] / (h * bc[1][0] + 2 * bc[1][1])
    L = np.zeros((N, N))
    
    for i in range(N - 1):
        L[i][i] = -2
        L[i][i + 1] = 1
        L[i + 1][i] = 1
    
    L[0][0] += Aa
    L[N - 1][N - 1] = Ab - 2
    B = np.zeros(N)
    B[0] = Ba
    B[-1] = Bb
    
    return D * (np.dot(L, y) - B) / h**2

    
if __name__ == '__main__':
    main()
