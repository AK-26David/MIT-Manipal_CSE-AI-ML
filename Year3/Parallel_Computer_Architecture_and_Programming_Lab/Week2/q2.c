#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc,char *argv[]) {
	int rank, size, i, num;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;
	if(rank == 0) {
		printf("Enter a number\n");
		scanf("%d", &num);
		for(i=1; i<size; i++) {
			MPI_Send(&num,1,MPI_INT,i,1,MPI_COMM_WORLD);
		}
	}
	else {
		MPI_Recv(&num,1,MPI_INT,0,1,MPI_COMM_WORLD,&status);
		printf("Recieved: %d\n", num);
	}
	MPI_Finalize();
	return 0;
}