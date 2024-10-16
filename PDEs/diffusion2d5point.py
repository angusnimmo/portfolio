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
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, u0, cmap=cm.viridis)
    #cb = fig.colorbar(surf, shrink=.5, label='$u(x,y)$')
    #ax.set_xlabel('$x$')
    #ax.set_ylabel('$y$')
    #ax.set_title(r'2-dimensional nonlinear diffusion, $M = 1$')
    fig.subplots_adjust(bottom=.25)
    axtime = fig.add_axes([.25, .1, .65, .03])
    time_slider = Slider(
        ax=axtime,
        label='Time',
        valmin=sol.t[0],
        valmax=sol.t[-1],
        valinit=0,
    )
    
    def update(val):
        t = min(
            range(len(sol.t)),
            key=lambda i: abs(sol.t[i] - time_slider.val)
        )
        ax.clear()
        surf = ax.plot_surface(
            x, y, sol.y[:,t].reshape((N, N)), cmap=cm.viridis
        )
    
    time_slider.on_changed(update)
    #ax.view_init(elev=90, azim=-90)
    plt.show()
    
    # (ii) M = [phi(1 - phi)]^2, nonlinear
    with open('ii_sol.pkl', 'rb') as file:
        sol = load(file)
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, u0, cmap=cm.viridis)
    #cb = fig.colorbar(surf, shrink=.5, label='$u(x,y)$')
    #ax.set_xlabel('$x$')
    #ax.set_ylabel('$y$')
    #ax.set_title(r'2-dimensional nonlinear diffusion, $M = [\phi(1-\phi)]^2$')
    fig.subplots_adjust(bottom=.25)
    axtime = fig.add_axes([.25, .1, .65, .03])
    time_slider = Slider(
        ax=axtime,
        label='Time',
        valmin=sol.t[0],
        valmax=sol.t[-1],
        valinit=0,
    )
    
    def update(val):
        t = min(
            range(len(sol.t)),
            key=lambda i: abs(sol.t[i] - time_slider.val)
        )
        ax.clear()
        surf = ax.plot_surface(
            x, y, sol.y[:,t].reshape((N, N)), cmap=cm.viridis
        )
    
    time_slider.on_changed(update)
    #ax.view_init(elev=90, azim=-90)
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
