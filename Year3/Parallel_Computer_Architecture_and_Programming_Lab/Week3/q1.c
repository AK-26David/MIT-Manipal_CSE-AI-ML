#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, num;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int A[size],B[size];
    if(rank == 0) {
        printf("Enter %d values\n", size);
        for(int i=0; i<size; i++)
            scanf("%d", &A[i]);
    }
    MPI_Scatter(A,1,MPI_INT,&num,1,MPI_INT,0,MPI_COMM_WORLD);
    int fact = 1;
    for(int i=num;i>=1;i--)
        fact *= i;
    //MPI_Gather(&fact,1,MPI_INT,B,1,MPI_INT,0,MPI_COMM_WORLD);
    int result = 0;
    MPI_Reduce(&fact,&result,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    if(rank == 0)
        printf("The sum of all factorials is: %d\n", result);
    MPI_Finalize();
    return 0;
}