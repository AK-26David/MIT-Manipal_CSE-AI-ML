#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
	int rank, size;
	char st[]="hello";
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(st[rank] >= 97)
		st[rank] = st[rank] - 32;
	else
		st[rank] = st[rank] + 32;
	printf("Rank: %d, %s\n", rank, st);
	MPI_Finalize();
	return 0;
}