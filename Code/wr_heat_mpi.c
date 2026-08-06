#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define NU     0.05
#define LDOM   2.0
#define TFIN   1.0
#define NT     400
#define NSUB   60      /* number of spatial grid points PER subdomain */
#define DELTA  0.15    /* overlap half-width */
#define NITER  10

static double u_init(double x) {
    return sin(M_PI * x / LDOM) + 0.5 * sin(3.0 * M_PI * x / LDOM);
}

/*
 * Solve u_t = nu u_xx on x_nodes (Nx points) x [0,T] (Nt+1 points) with
 * implicit Euler.
 */
static void solve_heat_dirichlet(const double *x_nodes, int Nx,
                                  double dt, int Nt,
                                  const double *u0_vals,
                                  const double *left_bc, const double *right_bc,
                                  double nu, double *U /* (Nt+1) x Nx */)
{
    double dx = x_nodes[1] - x_nodes[0];
    double r = nu * dt / (dx * dx);
    int n_int = Nx - 2;

    double *c_prime = (double *) malloc(n_int * sizeof(double));
    double *rhs      = (double *) malloc(n_int * sizeof(double));

    double a = -r, b = 1.0 + 2.0 * r, c = -r;
    c_prime[0] = c / b;
    for (int i = 1; i < n_int; i++) {
        double m = b - a * c_prime[i - 1];
        c_prime[i] = c / m;
    }

    for (int j = 0; j < Nx; j++) U[0 * Nx + j] = u0_vals[j];
    for (int n = 0; n <= Nt; n++) {
        U[n * Nx + 0]      = left_bc[n];
        U[n * Nx + Nx - 1] = right_bc[n];
    }

    for (int n = 0; n < Nt; n++) {
        for (int i = 0; i < n_int; i++) rhs[i] = U[n * Nx + (i + 1)];
        rhs[0]         += r * U[(n + 1) * Nx + 0];
        rhs[n_int - 1] += r * U[(n + 1) * Nx + Nx - 1];

        rhs[0] = rhs[0] / b;
        for (int i = 1; i < n_int; i++) {
            double m = b - a * c_prime[i - 1];
            rhs[i] = (rhs[i] - a * rhs[i - 1]) / m;
        }
        for (int i = n_int - 2; i >= 0; i--) {
            rhs[i] -= c_prime[i] * rhs[i + 1];
        }
        for (int i = 0; i < n_int; i++) {
            U[(n + 1) * Nx + (i + 1)] = rhs[i];
        }
    }

    free(c_prime);
    free(rhs);
}

static void subdomain_bounds(int rank, int N, double L, double delta,
                              double *x_left, double *x_right,
                              double *breakpoints /* length N+1, caller-allocated */)
{
    for (int i = 0; i <= N; i++) breakpoints[i] = L * i / (double) N;

    *x_left  = (rank == 0)     ? 0.0 : breakpoints[rank] - delta;
    *x_right = (rank == N - 1) ? L   : breakpoints[rank + 1] + delta;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, N;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &N);   /* number of subdomains = number of MPI ranks */

    if (N < 2) {
        if (rank == 0)
            fprintf(stderr, "This program needs at least 2 processes "
                             "(mpirun -n N ./wr_heat_mpi_N, N >= 2)\n");
        MPI_Finalize();
        return 1;
    }

    double dt = TFIN / NT;
    double *t = (double *) malloc((NT + 1) * sizeof(double));
    for (int n = 0; n <= NT; n++) t[n] = n * dt;

    double *breakpoints = (double *) malloc((N + 1) * sizeof(double));
    double x_left, x_right;
    subdomain_bounds(rank, N, LDOM, DELTA, &x_left, &x_right, breakpoints);

    if (rank == 0) {
        double base_width = LDOM / N;
        if (base_width <= 2.0 * DELTA)
            fprintf(stderr, "[warning] delta=%.3f is large relative to the "
                             "subdomain width L/N=%.3f -- neighbouring "
                             "overlaps may collide.\n", DELTA, base_width);
    }

    double *x_local = (double *) malloc(NSUB * sizeof(double));
    for (int i = 0; i < NSUB; i++)
        x_local[i] = x_left + (x_right - x_left) * i / (NSUB - 1);

    double *u0_local = (double *) malloc(NSUB * sizeof(double));
    for (int i = 0; i < NSUB; i++) u0_local[i] = u_init(x_local[i]);

    int left_neighbor  = (rank > 0)     ? rank - 1 : MPI_PROC_NULL;
    int right_neighbor = (rank < N - 1) ? rank + 1 : MPI_PROC_NULL;

    double x_send_to_left  = breakpoints[rank] + DELTA;       /* used if rank>0   */
    double x_send_to_right = breakpoints[rank + 1] - DELTA;   /* used if rank<N-1 */

    int idx_send_left = 0, idx_send_right = 0;
    {
        double best_l = 1e30, best_r = 1e30;
        for (int i = 0; i < NSUB; i++) {
            double dl = fabs(x_local[i] - x_send_to_left);
            double dr = fabs(x_local[i] - x_send_to_right);
            if (dl < best_l) { best_l = dl; idx_send_left = i; }
            if (dr < best_r) { best_r = dr; idx_send_right = i; }
        }
    }

    /* Boundary conditions */
    double *left_bc  = (double *) calloc(NT + 1, sizeof(double));
    double *right_bc = (double *) calloc(NT + 1, sizeof(double));
    double *send_to_left   = (double *) malloc((NT + 1) * sizeof(double));
    double *send_to_right  = (double *) malloc((NT + 1) * sizeof(double));
    double *recv_from_left  = (double *) malloc((NT + 1) * sizeof(double));
    double *recv_from_right = (double *) malloc((NT + 1) * sizeof(double));

    double *U_local = (double *) malloc((NT + 1) * NSUB * sizeof(double));

    for (int k = 0; k < NITER; k++) {
        solve_heat_dirichlet(x_local, NSUB, dt, NT, u0_local, left_bc, right_bc, NU, U_local);

        for (int n = 0; n <= NT; n++) {
            send_to_left[n]  = U_local[n * NSUB + idx_send_left];
            send_to_right[n] = U_local[n * NSUB + idx_send_right];
        }

        /* Exchange with left neighbour */
        MPI_Sendrecv(send_to_left, NT + 1, MPI_DOUBLE, left_neighbor, 10,
                     recv_from_left, NT + 1, MPI_DOUBLE, left_neighbor, 11,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Exchange with right neighbour */
        MPI_Sendrecv(send_to_right, NT + 1, MPI_DOUBLE, right_neighbor, 11,
                     recv_from_right, NT + 1, MPI_DOUBLE, right_neighbor, 10,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double local_err = 0.0;
        if (left_neighbor != MPI_PROC_NULL) {
            for (int n = 0; n <= NT; n++) {
                double d = fabs(recv_from_left[n] - left_bc[n]);
                if (d > local_err) local_err = d;
            }
        }
        if (right_neighbor != MPI_PROC_NULL) {
            for (int n = 0; n <= NT; n++) {
                double d = fabs(recv_from_right[n] - right_bc[n]);
                if (d > local_err) local_err = d;
            }
        }

        double global_err;
        MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (left_neighbor != MPI_PROC_NULL)
            memcpy(left_bc, recv_from_left, (NT + 1) * sizeof(double));
        if (right_neighbor != MPI_PROC_NULL)
            memcpy(right_bc, recv_from_right, (NT + 1) * sizeof(double));

        if (rank == 0)
            printf("iteration %d: max interface error (all %d subdomains) = %.3e\n",
                   k + 1, N, global_err);
    }

    /* write the final profile u(x,T) of each subdomain to a file */
    char fname[64];
    snprintf(fname, sizeof(fname), "wr_heat_mpi_N%d_rank%d.txt", N, rank);
    FILE *f = fopen(fname, "w");
    for (int i = 0; i < NSUB; i++)
        fprintf(f, "%.10f %.10f\n", x_local[i], U_local[NT * NSUB + i]);
    fclose(f);

    free(t); free(breakpoints); free(x_local); free(u0_local);
    free(left_bc); free(right_bc);
    free(send_to_left); free(send_to_right);
    free(recv_from_left); free(recv_from_right);
    free(U_local);

    MPI_Finalize();
    return 0;
}
