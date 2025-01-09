#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
	int rank;
	int a = 10;
	int b = 5;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank == 0)
		printf("Addition: %d\n", a+b);
	else if(rank == 1)
		printf("Subtraction: %d\n", a-b);	
	else if(rank == 2)
		printf("Multiplication: %d\n", a*b);
	else if(rank == 3)
		printf("Division: %.2f\n", (float)a/b);
	MPI_Finalize();
	return 0;
}