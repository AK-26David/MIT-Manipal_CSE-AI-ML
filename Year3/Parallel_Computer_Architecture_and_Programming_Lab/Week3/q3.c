#include "mpi.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
	int rank, size;
	char word[100];
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if(rank == 0) {
		printf("Enter your string: ");
		scanf("%s", word);
	}
	int l = strlen(word);
	MPI_Bcast(&l,1,MPI_INT,0,MPI_COMM_WORLD);
	char subword[l/size];
	MPI_Scatter(word,l/size,MPI_CHAR,subword,l/size,MPI_CHAR,0,MPI_COMM_WORLD);
	int total = l/size;
	for(int i=0; i<l/size; i++) {
		if(subword[i] == 'a' || subword[i] == 'e' || subword[i] == 'i' || subword[i] == 'o' || subword[i] == 'u')
			total--;
	}
	int B[size];
	MPI_Gather(&total,1,MPI_INT,B,1,MPI_INT,0,MPI_COMM_WORLD);
	if(rank == 0) {
		int res = 0;
		for(int i=0; i<size; i++) {
			printf("The number of non vowels in %d is: %d\n", rank, B[i]);
			res += B[i];
		}
		printf("total: %d\n", res);
	}
	MPI_Finalize();
	return 0;
}