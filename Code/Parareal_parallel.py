import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def f(u,t): # Define du/dt = f(u,t)
    return -u

def sol(t):
    return np.exp(-t)

def solver(u0, t0, t1, dt): # Solving via Euler method
    u = u0
    t = t0
    while t < t1:
        u = u + dt * f(u, t)
        t += dt
    return u

T = 5.0
N = 46 # Parareal iterations
m = 100  # Subintervals number
times = np.linspace(0, T, m+1)

dt_coarse = 0.5
dt_fine = 0.01

# Initialisation
u = np.zeros((N, m+1))
u[:, 0] = 1.0

counts = [m // size + (1 if i < m % size else 0) for i in range(size)]
starts = [sum(counts[:i]) for i in range(size)]
local_m = counts[rank]
start = starts[rank]
end = start + local_m

# Initialisation with coarse solver
for k in range(m):
    u[0, k+1] = solver(u[0, k], times[k], times[k+1], dt_coarse)

# Parareal
for n in range(0, N-1):
    F_local = np.zeros(local_m)

    for i, k in enumerate(range(start, end)):
        F_local[i] = solver(u[n, k], times[k], times[k+1], dt_fine)

    F_all = np.zeros(m)
    comm.Allgatherv(F_local, [F_all, counts, starts, MPI.DOUBLE])

    for k in range(m):
        G_new = solver(u[n+1, k], times[k], times[k+1], dt_coarse)
        G_old = solver(u[n, k], times[k], times[k+1], dt_coarse)

        u[n+1, k+1] = G_new + (F_all[k] - G_old)

if rank == 0 :
    # Exact solution
    t_exact = np.linspace(0, T, 200)
    u_exact = sol(t_exact)
    
    # Plot
    p = 5 # Plot only p iterations
    
    plt.figure()
    for n in [list(range(N))[int(i * (N - 1) / (p - 1))] for i in range(p)]:
        plt.plot(times, u[n], label=f"{n} iteration")
    
    plt.plot(t_exact, u_exact, '--', label="Exact")
    plt.legend()
    plt.title("Parareal method convergence")
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.show()
    
    plt.figure()
    for n in [list(range(N))[int(i * (N - 1) / (p - 1))] for i in range(p)]:
        plt.plot(times, np.abs(u[n]-sol(times)), label=f"{n} iteration")
    
    plt.legend()
    plt.title("Parareal method error")
    plt.xlabel("t")
    plt.ylabel("Error")
    plt.show()
    
    plt.figure()
    
    err = []
    for n in range(N):
        err.append(np.linalg.norm(u[n]-sol(times)))
    
    plt.plot(err)
    plt.title("Parareal method error")
    plt.xlabel("n")
    plt.ylabel("Error")
    plt.yscale('log')
    plt.show()