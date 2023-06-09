/*
   Project 2: Sources MPI (21.2.3)  

Author: Dan Weiss

Solve the sources and sinks problem using SOR

Inputs: N grid points in y direction, omega, tolerance, max iterations
Output: File with solution on an N by 2N - 1 grid

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/*
   sor_iteration_red

   Update the red points on a grid of size N_x by N_y
   It is assumed that the first red point is at position (0, 1)

Inputs: N_x, N_y, grid pointer, forcing pointer, dx, dy, omega, max residual, process rank)
Outputs: Maximum residual across the updated points

*/
double sor_iteration_red(int N_x, int N_y, double (*grid)[N_x], double (*forcing)[N_x], double dx, double dy, double omega, double max_resid, int start_row) {

    // variable to hold residual for an iteration
    double r;

    // loop variables
    int i;
    int j;

    for (i = start_row; i < N_y; ++i) {
        for (j = i%2; j < N_x; j += 2) {
            if (j > 0 & j < N_x - 1) {
                r = 0.25 * (grid[i][j + 1] + grid[i][j - 1] + grid[i + 1][j] + grid[i - 1][j] - 4 * grid[i][j]) - 0.25 * (dx * dx) * forcing[i][j]; 
            } else if (j == 0) {
                r = 0.25 * (grid[i][j + 1] + grid[i + 1][j] + grid[i - 1][j] - 3 * grid[i][j]) - 0.25 * (dx * dx) * forcing[i][j]; 
            } else {
                r = 0.25 * (grid[i][j - 1] + grid[i + 1][j] + grid[i - 1][j] - 3 * grid[i][j]) - 0.25 * (dx * dx) * forcing[i][j]; 
            }
            max_resid = fabs(r) > max_resid ? fabs(r) : max_resid;            
            grid[i][j] = grid[i][j] + omega * r;
        }
    }

    return max_resid;    

}

/*
   sor_iteration_black

   Update the black points on a grid of size N_x by N_y
   It is assumed that the first black point is at position (0, 0)

Inputs: N_x, N_y, grid pointer, forcing pointer, dx, dy, omega, max residual, process rank)
Outputs: Maximum residual across the updated points

*/
double sor_iteration_black(int N_x, int N_y, double (*grid)[N_x], double (*forcing)[N_x], double dx, double dy, double omega, double max_resid, int start_row) {

    // variable to hold residual for an iteration
    double r;

    // loop variables
    int i;
    int j;

    for (i = start_row; i < N_y; ++i) {
        for (j = (i+1)%2; j < N_x; j += 2) {
            if (j > 0 & j < (N_x - 1)) {
                r = 0.25 * (grid[i][j + 1] + grid[i][j - 1] + grid[i + 1][j] + grid[i - 1][j] - 4 * grid[i][j]) - 0.25 * (dx * dx) * forcing[i][j]; 
            } else if (j == 0) {
                r = 0.25 * (grid[i][j + 1] + grid[i + 1][j] + grid[i - 1][j] - 3 * grid[i][j]) - 0.25 * (dx * dx) * forcing[i][j]; 
            } else {
                r = 0.25 * (grid[i][j - 1] + grid[i + 1][j] + grid[i - 1][j] - 3 * grid[i][j]) - 0.25 * (dx * dx) * forcing[i][j]; 
            }
            max_resid = fabs(r) > max_resid ? fabs(r) : max_resid;            
            grid[i][j] = grid[i][j] + omega * r;
        }
    }

    return max_resid;    
}

int main(int argc, char* argv[])
{

    // initialize MPI system
    MPI_Init(&argc, &argv);

    // rank of current process 
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    // total number of processes 
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // start timing
    double starttime = MPI_Wtime();

    // initialize input arguments
    int argi = 0;
    int N = atoi(argv[++argi]);
    double omega = atof(argv[++argi]);
    double tau = atof(argv[++argi]);
    int max_iter = atof(argv[++argi]);

    // print input arguments
    if (rank == 0) {
        printf("N Grid Points: %d\n", N);
        printf("Omega: %le\n", omega);
        printf("Tau: %le\n", tau);
        printf("Max Iterations: %d\n", max_iter);
    }

    // initialize discretization
    double dy = 2. / (N - 1); // going from -1 to 1 in y
    double dx = 2. / (N - 1); // going from -2 to 2 in x
    int N_y = N;
    int N_x = 2 * N - 1;

    // calculate y dimension of subgrid
    // ranks 0 and size - 1 have one less point than the middle subgrids
    int N_y_sub = N_y / size + (rank > 0 ? 1 : 0) + (rank < size - 1 ? 1 : 0);

    // loop variables
    int i;
    int j;

    // initalize initial guess equal to 0 at all points on each subgrid
    double (*grid)[N_x] = malloc(sizeof(*grid) * N_y_sub);

    for (i = 0; i < N_y_sub; ++i) {
        for (j = 0; j < N_x; ++j) {
            grid[i][j] = 0;        
        }
    }

    // initialize forcing term on each subgrid 
    double (*forcing)[N_x] = malloc(sizeof(*forcing) * N_y_sub);

    int row = rank * (N_y / size) - (rank > 0 ? 1 : 0);
    for (i = 0; i < N_y_sub; ++i) {
        for (j = 0; j < N_x; ++j) {
            forcing[i][j] = 10 * 100 / sqrt(M_PI) * (
                    exp(-100 * 100 * (((-2 + dx * j) - 1) * ((-2 + dx * j) - 1) + (-1 + dy * (i + row)) * (-1 + dy * (i + row)))) - 
                    exp(-100 * 100 * (((-2 + dx * j) + 1) * ((-2 + dx * j) + 1)+ (-1 + dy * (i + row)) * (-1 + dy * (i + row))))
                    );
        }
    }

    // initialize residual and iteration count
    double max_residual = tau + 1; // add on 1 so the loop will run for sure 
    int iteration = 0;

    // status variable used by the MPI communications
    MPI_Status status;

    // solve the equation 
    while (max_residual > tau & iteration <= max_iter) {

        // reset max residual on each iteration
        max_residual = 0;

        if (rank > 0 & rank < size - 1) {

            // update red
            max_residual = sor_iteration_red(N_x, N_y_sub - 2, grid + 1, forcing + 1, dx, dy, omega, max_residual, 0);

            // send left and receive from right
            MPI_Sendrecv(&(grid[1][0]), N_x, MPI_DOUBLE, rank - 1, 0, &(grid[N_y_sub - 1][0]), N_x, 
                    MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status); 

            // send right and receive from left
            MPI_Sendrecv(&(grid[N_y_sub - 2][0]), N_x, MPI_DOUBLE, rank + 1, 0, &(grid[0][0]), N_x, 
                    MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status); 

            // update black
            max_residual = sor_iteration_black(N_x, N_y_sub - 2, grid + 1, forcing + 1, dx, dy, omega, max_residual, 0);

            // send left and receive from right
            MPI_Sendrecv(&(grid[1][0]), N_x, MPI_DOUBLE, rank - 1, 0, &(grid[N_y_sub - 1][0]), N_x, 
                    MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status); 

            // send right and receive from left
            MPI_Sendrecv(&(grid[N_y_sub - 2][0]), N_x, MPI_DOUBLE, rank + 1, 0, &(grid[0][0]), N_x, 
                    MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);

            iteration++;

        } else if (rank == 0){

            // update red
            max_residual = sor_iteration_red(N_x, N_y_sub - 1, grid, forcing, dx, dy, omega, max_residual, 1);

            // receive from right
            MPI_Recv(&(grid[N_y_sub - 1][0]), N_x, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);

            // send right
            MPI_Send(&(grid[N_y_sub - 2][0]), N_x, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

            // update black
            max_residual = sor_iteration_black(N_x, N_y_sub - 1, grid, forcing, dx, dy, omega, max_residual, 1);

            // receive from right
            MPI_Recv(&(grid[N_y_sub - 1][0]), N_x, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);

            // send right
            MPI_Send(&(grid[N_y_sub - 2][0]), N_x, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

            iteration++;

        } else {

            // update red
            max_residual = sor_iteration_red(N_x, N_y_sub - 2, grid + 1, forcing + 1, dx, dy, omega, max_residual, 0);

            // send left 
            MPI_Send(&(grid[1][0]), N_x, MPI_DOUBLE, size - 2, 0, MPI_COMM_WORLD);

            // receive from left 
            MPI_Recv(&(grid[0][0]), N_x, MPI_DOUBLE, size - 2, 0, MPI_COMM_WORLD, &status);

            // update black
            max_residual = sor_iteration_black(N_x, N_y_sub - 2, grid + 1, forcing + 1, dx, dy, omega, max_residual, 0);

            // send left 
            MPI_Send(&(grid[1][0]), N_x, MPI_DOUBLE, size - 2, 0, MPI_COMM_WORLD);

            // receive from left 
            MPI_Recv(&(grid[0][0]), N_x, MPI_DOUBLE, size - 2, 0, MPI_COMM_WORLD, &status);

            iteration++;

        }

        // send max residual across all points to all ranks so that all
        // processes break the loop at the same time
        MPI_Allreduce(&max_residual, &max_residual, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); 

    }

    // initialize pointers for full grid to use in MPI_Gather
    double (*grid_full)[N_x] = NULL;

    // on rank 0, allocate space for the full grid
    if (rank == 0) {
        grid_full = malloc(N_y * sizeof(*grid_full));
    }

    // consolidate and print
    MPI_Gather(grid + (rank > 0 ? 1 : 0), N_x * (N_y / size), MPI_DOUBLE, grid_full, N_x * (N_y / size), MPI_DOUBLE, 0, MPI_COMM_WORLD);   

    if (rank == 0) {

        if (iteration <= max_iter)
            printf("%d iterations to converge\n", iteration);
        else {
            printf("Did not converge...max residual is currently %f\n", max_residual);
        }

        // open file 
        FILE *Sources = fopen("Sources.out", "w");

        // write to file
        fwrite(grid_full, sizeof(*grid_full), N_y, Sources); 

        // close file
        fclose(Sources);

    }

    // free memory
    free(forcing);
    free(grid);

    // print time elapsed
    if (rank == 0) {

        free(grid_full);
        printf("Time elapsed: %f seconds\n", MPI_Wtime() - starttime);

        if (iteration <= max_iter) {
            printf("Average time per iteration: %f seconds\n", (MPI_Wtime() - starttime) / iteration);
        }

    }

    // shut down MPI system 
    MPI_Finalize();

    return 0;

} 

