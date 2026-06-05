import numpy as np
import matplotlib.pyplot as plt

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

# Initialisation with coarse solver
for k in range(m):
    u[0, k+1] = solver(u[0, k], times[k], times[k+1], dt_coarse)

# Parareal
for n in range(0, N-1):
    for k in range(m):
        G_new = solver(u[n+1, k], times[k], times[k+1], dt_coarse)
        G_old = solver(u[n, k], times[k], times[k+1], dt_coarse)
        F_old = solver(u[n, k], times[k], times[k+1], dt_fine)

        u[n+1, k+1] = G_new + (F_old - G_old)

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