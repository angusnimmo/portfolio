from pylab import *
from scipy.integrate import solve_ivp
from pickle import dump, load
from datetime import datetime


def main():
    N = 64
    h = 1 / N
    s = np.linspace(h / 2, 1 - h / 2, N)
    x, y = np.meshgrid(s, s, indexing='ij')
    
    with open('snapshot.pkl', 'rb') as file:
        snapshot = load(file)
        
    rho, gx, gy = np.split(snapshot[0], 3)

    # Contour plot
    cp = plt.contourf(rho.reshape((N, N)), origin='lower', extent=[0, 1, 0, 1])
    cb = plt.colorbar(cp, label=r'$\rho_0(x, y)$')

    plt.axis('scaled')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(
        r'ICs: $\mathbf{v}_0(x, y) = \mathbf{0}$, '
        r'$\rho_0(x, y) = 1 + \phi_0(x, y) + \langle \phi_0(x, y) \rangle$,'
        '\n'
        r'where $\phi_0(x, y) = 0.2 \exp\{-32[(x - 1 / 2)^2 - (y - 1 / 2)^2]\}$'
    )
    plt.show()
    
    rho, gx, gy = np.split(snapshot[-1], 3)
    u = (gx / rho).reshape((N, N))
    v = (gy / rho).reshape((N, N))

    # Contour plot
    cp = plt.contourf(rho.reshape((N, N)), origin='lower', extent=[0, 1, 0, 1])
    cb = plt.colorbar(cp, label=r'$\rho(t = 0.5, x, y)$')
    
    # Velocity field
    quiv = plt.quiver(
        x[::3, ::3],
        y[::3, ::3],
        u[::3, ::3],
        v[::3, ::3],
        color='lime',
        label=r'$\mathbf{v}(t = 0.5, x, y)$'
    )
    plt.axis('scaled')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(
        r'$\partial \rho / \partial t = \nabla \cdot (-\mathbf{g})$, '
        r'$\partial \mathbf{g} / \partial t = \nabla \cdot '
        r'[-\rho \mathbf{I} + \nabla (\mathbf{g} / \rho)]$,'
        '\n'
        r'at $t = 0.5$ where $\mathbf{g}(t, x, y)'
        r' = \rho(t, x, y) \mathbf{v}(t, x, y)$'
    )
    plt.legend()
    plt.show()


def compressible_stokes(t: float, y: list, h: float) -> list:
    rho, gx, gy = np.split(y, 3)
    N = int(len(rho)**.5)
    rho = rho.reshape((N, N))
    gx = gx.reshape((N, N))
    gy = gy.reshape((N, N))
    drhodt = np.zeros((N, N))
    dgxdt = np.zeros((N, N))
    dgydt = np.zeros((N, N))
    u = gx / rho
    v = gy / rho
    
    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N
        
        for j in range(N):
            jp = (j + 1) % N
            jm = (j - 1) % N
            drhodt[i, j] = -(
                gx[ip, j] - gx[im, j] + gy[i, jp] - gy[i, jm]
            ) / (2 * h)
            dgxdt[i, j] = -(rho[ip, j] - rho[im, j]) / (2 * h) + (
                u[ip, j] + u[im, j] + u[i, jp] + u[i, jm] - 4 * u[i, j]
            ) / h**2
            dgydt[i, j] = -(rho[i, jp] - rho[i, jm]) / (2 * h) + (
                v[ip, j] + v[im, j] + v[i, jp] + v[i, jm] - 4 * v[i, j]
            ) / h**2
    
    return np.concatenate((drhodt.flatten(), dgxdt.flatten(), dgydt.flatten()))
    
    
def v_sol():
    start = datetime.now()
    # Initial conditions
    N = 64
    h = 1 / N
    T = [0, 1 / 2]
    s = np.linspace(h / 2, 1 - h / 2, N)
    x, y = np.meshgrid(s, s, indexing='ij')
    phi0 = .2 * exp(-32 * ((x - 1 / 2)**2 + (y - 1 / 2)**2))
    rho0 = 1 + phi0 - np.mean(phi0)
    sol = solve_ivp(
        compressible_stokes,
        T,
        np.concatenate((rho0.flatten(), np.zeros(N**2), np.zeros(N**2))),
        args=(h,)
    )
    
    with open('v_sol.pkl', 'wb') as file:
        dump(sol, file)
    
    print(datetime.now() - start)


if __name__ == '__main__':
    main()
