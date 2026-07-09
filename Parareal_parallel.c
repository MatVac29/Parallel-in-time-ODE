#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

double f(double u, double t) {
    return -u;
}

double solver(double u0, double t0, double t1, double dt) {
    double u = u0;
    double t = t0;
    while (t < t1) {
        u = u + dt * f(u, t);
        t += dt;
    }
    return u;
}

double sol(double t) {
    return exp(-t);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double T = 5.0;
    int N = 46;
    int m = 100;

    double dt_coarse = 0.5;
    double dt_fine = 0.01;

    double* times = malloc((m+1) * sizeof(double));
    for (int i = 0; i <= m; i++)
        times[i] = T * i / m;

    // Allocation u[N][m+1]
    double** u = malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        u[i] = calloc(m+1, sizeof(double));
        u[i][0] = 1.0;
    }

    int* counts = malloc(size * sizeof(int));
    int* starts = malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        counts[i] = m / size + (i < m % size ? 1 : 0);
    }

    starts[0] = 0;
    for (int i = 1; i < size; i++)
        starts[i] = starts[i-1] + counts[i-1];

    int local_m = counts[rank];
    int start = starts[rank];
    int end = start + local_m;

    // Initialisation
    for (int k = 0; k < m; k++) {
        u[0][k+1] = solver(u[0][k], times[k], times[k+1], dt_coarse);
    }

    double* F_local = malloc(local_m * sizeof(double));
    double* F_all = malloc(m * sizeof(double));

    // Parareal
    for (int n = 0; n < N-1; n++) {

        for (int i = 0; i < local_m; i++) {
            int k = start + i;
            F_local[i] = solver(u[n][k], times[k], times[k+1], dt_fine);
        }

        MPI_Allgatherv(F_local, local_m, MPI_DOUBLE,
                       F_all, counts, starts, MPI_DOUBLE,
                       MPI_COMM_WORLD);

        for (int k = 0; k < m; k++) {
            double G_new = solver(u[n+1][k], times[k], times[k+1], dt_coarse);
            double G_old = solver(u[n][k], times[k], times[k+1], dt_coarse);

            u[n+1][k+1] = G_new + (F_all[k] - G_old);
        }
    }

    if (rank == 0) {
        FILE *file = fopen("parareal_mpi_results.txt", "w");

        if (file == NULL) {
            printf("Error opening file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int n = 0; n < N; n++) {
            for (int k = 0; k <= m; k++) {
                fprintf(file, "%lf ", u[n][k]);
            }
            fprintf(file, "\n");
        }

        fclose(file);
    }

    MPI_Finalize();
    return 0;
}