#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc,char *argv[]) {
	int rank, size, num;
	int *arr;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;
	arr = (int *)malloc(size*sizeof(int));
	if(rank == 0) {
		printf("Enter array elements\n");
		for(int i=0; i < size; i++)
			scanf("%d", &arr[i]);
		int bufsize = sizeof(int) + MPI_BSEND_OVERHEAD;
		int *buf = malloc(bufsize);
		MPI_Buffer_attach(buf, bufsize);
		for(int i=1; i<size; i++)
			MPI_Bsend(&arr[i], 1, MPI_INT, i, 1, MPI_COMM_WORLD);
		MPI_Buffer_detach(&buf, &bufsize);
		free(buf);
	}
	else {
		MPI_Recv(&num,1,MPI_INT,0,1,MPI_COMM_WORLD,&status);
		if(rank % 2 == 0)
			printf("[%d]: %d\n", rank, (int)pow(num, 2));
		else
			printf("[%d]: %d\n", rank, (int)pow(num, 3));
	}
	MPI_Finalize();
	return 0;
}
