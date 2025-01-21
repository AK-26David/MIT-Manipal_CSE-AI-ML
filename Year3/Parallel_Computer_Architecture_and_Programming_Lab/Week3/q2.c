#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
	int rank, size, num, m;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if(rank == 0) {
		printf("Enter M: ");
		scanf("%d",&m);		
	}
	MPI_Bcast(&m,1,MPI_INT,0,MPI_COMM_WORLD);
	int A[size*m], C[m];
	if(rank == 0) {
		printf("Enter %d values\n", size*m);
		for(int i=0; i<size*m; i++)
			scanf("%d", &A[i]);
	}
	MPI_Scatter(A,m,MPI_INT,C,m,MPI_INT,0,MPI_COMM_WORLD);
	//MPI_Gather(&fact,1,MPI_INT,B,1,MPI_INT,0,MPI_COMM_WORLD);
	float avg=0;
	for(int i=0; i<m; i++)
		avg+=C[i];
	avg = avg/(float)m;
	float result = 0;
	MPI_Reduce(&avg,&result,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	if(rank == 0)
		printf("The average of all is: %f\n", result/size);
	MPI_Finalize();
	return 0;
}