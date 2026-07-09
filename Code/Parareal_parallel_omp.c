#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

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

int main() {
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

    // Initialisation coarse
    for (int k = 0; k < m; k++) {
        u[0][k+1] = solver(u[0][k], times[k], times[k+1], dt_coarse);
    }

    double* F = malloc(m * sizeof(double));

    // Parareal
    for (int n = 0; n < N-1; n++) {

        #pragma omp parallel for
        for (int k = 0; k < m; k++) {
            F[k] = solver(u[n][k], times[k], times[k+1], dt_fine);
        }

        for (int k = 0; k < m; k++) {
            double G_new = solver(u[n+1][k], times[k], times[k+1], dt_coarse);
            double G_old = solver(u[n][k], times[k], times[k+1], dt_coarse);

            u[n+1][k+1] = G_new + (F[k] - G_old);
        }
    }

    FILE *file = fopen("parareal_omp_results.txt", "w");

    if (file == NULL) {
        printf("Error opening file\n");
        return 1;
    }

    for (int n = 0; n < N; n++) {
        for (int k = 0; k <= m; k++) {
            fprintf(file, "%lf ", u[n][k]);
        }
        fprintf(file, "\n");
    }

    fclose(file);

    return 0;
}