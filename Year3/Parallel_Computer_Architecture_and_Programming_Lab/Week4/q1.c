#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nop;
    int len = 50;
    char estr[50];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int err = MPI_Comm_size(MPI_COMM_WORLD, &nop);
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    MPI_Status stat;
    int fact = 1;
    int ans = 0;
    for(int i = 1; i <= rank+1; i++) {
        fact *= i;
    }
    printf("Rank %d:\t%d\n",rank, fact);
    MPI_Scan(&fact, &ans, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("Rank %d:\t%d\n",rank, ans);
    MPI_Error_string(err, estr, &len);
    printf("\nError: %s", estr);
    MPI_Finalize();
    exit(0);
}
