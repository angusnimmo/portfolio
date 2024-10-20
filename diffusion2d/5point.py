import matplotlib.animation as animation
from pylab import *
from typing import Callable
from scipy.integrate import solve_ivp
from pickle import dump, load
from datetime import datetime


def main():
    # Intial conditions
    N = 64
    h = 1 / N
    T = [0, 1 / 32]
    s = np.linspace(h / 2, 1 - h / 2, N)
    x, y = np.meshgrid(s, s, indexing='ij')
    u0 = exp(-32 * ((x - 1 / 2)**2 + (y - 1 / 2)**2))
    
    # (i) M = 1, linear
    with open('i_sol.pkl', 'rb') as file:
        sol = load(file)
    
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title(
        r'$\frac{\partial \phi}{\partial t}'
        r'= \nabla \cdot [M(\phi) \nabla \phi], M = 1$'
    )
    ims = []
    
    for i in range(len(sol.t)):
        im = ax.imshow(
            sol.y[:,i].reshape((N, N)), origin='lower', extent=[0, 1, 0, 1]
        )
        
        if i == 0:
            cb = fig.colorbar(im, label='$u(x,y)$')
            ax.imshow(
                sol.y[:,i].reshape((N, N)), origin='lower', extent=[0, 1, 0, 1]
            )
        
        ims.append([im])
    
    ani = animation.ArtistAnimation(
        fig, ims, interval=50, blit=True, repeat_delay=1000
    )
    plt.show()
    
    # (ii) M = [phi(1 - phi)]^2, nonlinear
    with open('ii_sol.pkl', 'rb') as file:
        sol = load(file)
    
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title(
        r'$\frac{\partial \phi}{\partial t}'
        r'= \nabla \cdot [M(\phi) \nabla \phi], M = [\phi (1 - \phi)]^2$'
    )
    ims = []
    
    for i in range(len(sol.t)):
        im = ax.imshow(
            sol.y[:,i].reshape((N, N)), origin='lower', extent=[0, 1, 0, 1]
        )
        
        if i == 0:
            cb = fig.colorbar(im, label='$u(x,y)$')
            ax.imshow(
                sol.y[:,i].reshape((N, N)), origin='lower', extent=[0, 1, 0, 1]
            )
        
        ims.append([im])
    
    ani = animation.ArtistAnimation(
        fig, ims, interval=50, blit=True, repeat_delay=1000
    )
    plt.show()


def diffusion_2d(t: float, y: list, M: callable, h: float) -> list:
    N = int(len(y)**.5)
    Y = np.reshape(y, (N, N))
    F = np.zeros((N, N))
    
    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N
        
        for j in range(N):
            jp = (j + 1) % N
            jm = (j - 1) % N
            uc = Y[i, j]
            ue = Y[ip, j]
            un = Y[i, jp]
            uw = Y[im, j]
            us = Y[i, jm]
            F[i, j] = (
                M((ue + uc) / 2) * (ue - uc)
                + M((uw + uc) / 2) * (uw - uc)
                + M((un + uc) / 2) * (un - uc)
                + M((us + uc) / 2) * (us - uc)
            ) / h**2
    
    return F.flatten()
    
    
def i_sol():
    start_time = datetime.now()
    # Intial conditions
    N = 64
    h = 1 / N
    T = [0, 1 / 32]
    s = np.linspace(h / 2, 1 - h / 2, N)
    x, y = np.meshgrid(s, s, indexing='ij')
    u0 = exp(-32 * ((x - 1 / 2)**2 + (y - 1 / 2)**2))
    
    # (i) M = 1, linear
    def M(phi: float) -> float:
        return 1.0
        
    sol = solve_ivp(diffusion_2d, T, u0.flatten(), args=(M, h))
    
    with open('i_sol.pkl', 'wb') as file:
        dump(sol, file)
        
    print(datetime.now() - start_time)
    
    
def ii_sol():
    start_time = datetime.now()
    # Intial conditions
    N = 64
    h = 1 / N
    T = [0, 1 / 32]
    s = np.linspace(h / 2, 1 - h / 2, N)
    x, y = np.meshgrid(s, s, indexing='ij')
    u0 = exp(-32 * ((x - 1 / 2)**2 + (y - 1 / 2)**2))
    
    # (ii) M = [phi(1 - phi)]^2, nonlinear
    def M(phi: float) -> float:
        return (phi * (1 - phi))**2
        
    sol = solve_ivp(diffusion_2d, T, u0.flatten(), args=(M, h))
    
    with open('ii_sol.pkl', 'wb') as file:
        dump(sol, file)
        
    print(datetime.now() - start_time)
    
    
if __name__ == '__main__':
    main()
