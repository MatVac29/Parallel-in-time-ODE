import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_u = pd.read_csv(r'C:\Users\artus\Desktop\Mathurin\Trinity\Dissertation\parareal_omp_results.txt', 
                   sep=" ", header = None)

u = np.array(df_u)

def sol(t):
    return np.exp(-t)


T = 5.0
N = 46 # Parareal iterations
m = 100  # Subintervals number
times = np.linspace(0, T, m+1)

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