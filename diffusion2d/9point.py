import matplotlib.animation as animation
from pylab import *
from typing import Callable
from scipy.integrate import solve_ivp
from pickle import dump, load
from datetime import datetime


def main():
    # Intial conditions
    a = 1 / 2
    N = 64
    h = 1 / N
    T = [0, 1 / 32]
    s = np.linspace(h / 2, 1 - h / 2, N)
    x, y = np.meshgrid(s, s, indexing='ij')
    u0 = exp(-32 * ((x - 1 / 2)**2 + (y - 1 / 2)**2))
    
    # (i) M = 1, linear
    with open('iii_sol.pkl', 'rb') as file:
        sol = load(file)
    
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ims = []
    
    for i in range(len(sol.t)):
        im = ax.imshow(sol.y[:,i].reshape((N, N)), origin='lower')
        
        if i == 0:
            ax.imshow(sol.y[:,i].reshape((N, N)), origin='lower')
        
        ims.append([im])
    
    ani = animation.ArtistAnimation(
        fig, ims, interval=50, blit=True, repeat_delay=1000
    )
    plt.show()
    
    # (ii) M = [phi(1 - phi)]^2, nonlinear
    with open('iv_sol.pkl', 'rb') as file:
        sol = load(file)
    
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ims = []
    
    for i in range(len(sol.t)):
        im = ax.imshow(sol.y[:,i].reshape((N, N)), origin='lower')
        
        if i == 0:
            ax.imshow(sol.y[:,i].reshape((N, N)), origin='lower')
        
        ims.append([im])
    
    ani = animation.ArtistAnimation(
        fig, ims, interval=50, blit=True, repeat_delay=1000
    )
    plt.show()
    

def diffusion_9point(
    t: float, y: list, M: callable, h: float, a: float
) -> list:
    b = a * (1 + 5**.5) / 4
    b1 = (1 - b)**2
    b2 = b * (1 - b)
    b3 = b**2
    C = (5**.5 - 1) / 2
    N = int(len(y)**.5)
    u = np.reshape(y, (N, N))
    F = np.zeros((N, N))
    
    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N
        
        for j in range(N):
            jp = (j + 1) % N
            jm = (j - 1) % N
            u1 = (1 - a) * u[i, j] + a * u[ip, j]
            u2 = (1 - a) * u[i, j] + a * u[i, jp]
            u3 = (1 - a) * u[i, j] + a * u[im, j]
            u4 = (1 - a) * u[i, j] + a * u[i, jm]
            u5 = b1 * u[i, j] + b2 * (u[ip, j] + u[i, jp]) + b3 * u[ip, jp]
            u6 = b1 * u[i, j] + b2 * (u[im, j] + u[i, jp]) + b3 * u[im, jp]
            u7 = b1 * u[i, j] + b2 * (u[im, j] + u[i, jm]) + b3 * u[im, jm]
            u8 = b1 * u[i, j] + b2 * (u[ip, j] + u[i, jm]) + b3 * u[ip, jm]
            M1 = M(u1)
            M2 = M(u2)
            M3 = M(u3)
            M4 = M(u4)
            M5 = M(u5)
            M6 = M(u6)
            M7 = M(u7)
            M8 = M(u8)
            F[i, j] = sum([
                (2 * (M1 + M3) + (M1 - M3) / a) * u[ip, j],
                (2 * (M2 + M4) + (M2 - M4) / a) * u[i, jp],
                (2 * (M1 + M3) + (M3 - M1) / a) * u[im, j],
                (2 * (M2 + M4) + (M4 - M2) / a) * u[i, jm],
                (M5 + M7 + C * (M5 - M7) / a) / 2 * u[ip, jp],
                (M6 + M8 + C * (M6 - M8) / a) / 2 * u[im, jp],
                (M5 + M7 - C * (M5 - M7) / a) / 2 * u[im, jm],
                (M6 + M8 - C * (M6 - M8) / a) / 2 * u[ip, jm],
                -(4 * (M1 + M2 + M3 + M4) + M5 + M6 + M7 + M8) * u[i,j],
            ]) / (6 * h**2)
    
    return F.flatten()
    
    
def iii_sol():
    start = datetime.now()
    # Intial conditions
    a = 3 / 4
    N = 64
    h = 1 / N
    T = [0, 1 / 32]
    s = np.linspace(h / 2, 1 - h / 2, N)
    x, y = np.meshgrid(s, s, indexing='ij')
    u0 = exp(-32 * ((x - 1 / 2)**2 + (y - 1 / 2)**2))
    
    # (i) M = 1, linear
    def M(phi: float) -> float:
            return 1.0
    
    sol = solve_ivp(
        diffusion_9point, T, u0.flatten(), args=(M, h, a), method='RK23'
    )
    
    with open('iii_sol.pkl', 'wb') as file:
        dump(sol, file)
    
    print(datetime.now() - start)
        
        
def iv_sol():
    start = datetime.now()
    # Intial conditions
    a = 3 / 4
    N = 64
    h = 1 / N
    T = [0, 1 / 32]
    s = np.linspace(h / 2, 1 - h / 2, N)
    x, y = np.meshgrid(s, s, indexing='ij')
    u0 = exp(-32 * ((x - 1 / 2)**2 + (y - 1 / 2)**2))
    
    # (ii) M = [phi(1 - phi)]^2, nonlinear
    def M(phi: float) -> float:
        return (phi * (1 - phi))**2
    
    sol = solve_ivp(
        diffusion_9point, T, u0.flatten(), args=(M, h, a), method='RK23'
    )
    
    with open('iv_sol.pkl', 'wb') as file:
        dump(sol, file)
    
    print(datetime.now() - start)


if __name__ == '__main__':
    main()
