import numpy as np
import matplotlib.pyplot as plt


NU = 0.05
L = 2.0
T = 1.0
NT = 400                # number of time steps
N_SUB_PTS = 60           # number of spatial grid points per subdomain
DELTA = 0.2              # overlap half-width
NX_REF = 200              # resolution of the reference solve


def u_init(x):
    return np.sin(np.pi * x / L) + 0.5 * np.sin(3 * np.pi * x / L)


def solve_heat_dirichlet(x_nodes, u0_vals, left_bc_vals, right_bc_vals, nu, dt, Nt):
    Nx = len(x_nodes)
    dx = x_nodes[1] - x_nodes[0]
    r = nu * dt / dx**2

    n_int = Nx - 2
    main = (1 + 2 * r) * np.ones(n_int)
    off = -r * np.ones(n_int - 1)
    A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

    U = np.zeros((Nt + 1, Nx))
    U[0] = u0_vals
    U[:, 0] = left_bc_vals
    U[:, -1] = right_bc_vals

    for n in range(Nt):
        b = U[n, 1:-1].copy()
        b[0] += r * U[n + 1, 0]
        b[-1] += r * U[n + 1, -1]
        U[n + 1, 1:-1] = np.linalg.solve(A, b)

    return U


def base_breakpoints(N):
    return np.linspace(0.0, L, N + 1)


def subdomain_extent(i, N, b, delta):
    x_left = 0.0 if i == 0 else b[i] - delta
    x_right = L if i == N - 1 else b[i + 1] + delta
    return x_left, x_right


def build_subdomains(N, delta, n_pts):
    b = base_breakpoints(N)
    grids, inits = [], []
    for i in range(N):
        xl, xr = subdomain_extent(i, N, b, delta)
        x_i = np.linspace(xl, xr, n_pts)
        grids.append(x_i)
        inits.append(u_init(x_i))
    return grids, inits, b



def schwarz_wr_heat(N=2, n_iter=6, delta=DELTA, n_sub=N_SUB_PTS):
    t = np.linspace(0.0, T, NT + 1)
    dt = t[1] - t[0]

    grids, inits, b = build_subdomains(N, delta, n_sub)

    left_bc = [np.zeros(NT + 1) for _ in range(N)]
    right_bc = [np.zeros(NT + 1) for _ in range(N)]

    errors = []
    U_all = [None] * N

    for k in range(n_iter):
        for i in range(N):
            U_all[i] = solve_heat_dirichlet(
                grids[i], inits[i], left_bc[i], right_bc[i], NU, dt, NT
            )

        new_left_bc = [arr.copy() for arr in left_bc]
        new_right_bc = [arr.copy() for arr in right_bc]
        err = 0.0

        for i in range(N):
            if i > 0:
                idx = np.argmin(np.abs(grids[i] - (b[i] + delta)))
                val = U_all[i][:, idx]
                err = max(err, np.max(np.abs(val - right_bc[i - 1])))
                new_right_bc[i - 1] = val

            if i < N - 1:
                idx = np.argmin(np.abs(grids[i] - (b[i + 1] - delta)))
                val = U_all[i][:, idx]
                err = max(err, np.max(np.abs(val - left_bc[i + 1])))
                new_left_bc[i + 1] = val

        left_bc, right_bc = new_left_bc, new_right_bc
        errors.append(err)

    return grids, U_all, errors, t, b


def solve_reference_heat():
    x = np.linspace(0, L, NX_REF)
    t = np.linspace(0, T, NT + 1)
    dt = t[1] - t[0]
    zero = np.zeros(NT + 1)
    U = solve_heat_dirichlet(x, u_init(x), zero, zero, NU, dt, NT)
    return x, t, U


x_ref, t_ref, U_ref = solve_reference_heat()

for N_demo in (2, 4):
    print(f"\n=== Schwarz Waveform Relaxation, N = {N_demo} subdomains ===")
    n_iter = 8
    grids, U_all, errors, t, b = schwarz_wr_heat(N=N_demo, n_iter=n_iter)
    print("errors per iteration:", ["%.2e" % e for e in errors])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].semilogy(range(1, n_iter + 1), errors, "o-", color="darkorange")
    axes[0].set_xlabel("iteration k")
    axes[0].set_ylabel("max interface change")
    axes[0].set_title(f"SWR convergence (N={N_demo} subdomains)")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(x_ref, U_ref[-1], "k-", lw=2, label="reference (t=T)")
    colors = plt.cm.tab10(np.linspace(0, 1, N_demo))
    for i in range(N_demo):
        axes[1].plot(grids[i], U_all[i][-1], "--", color=colors[i],
                     label=f"subdomain {i}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x,T)")
    axes[1].set_title(f"Final solution by subdomain (N={N_demo})")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    
    plt.show()