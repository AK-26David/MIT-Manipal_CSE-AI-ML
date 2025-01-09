#include "mpi.h"
#include <stdio.h>

int factorial(int n)
{
	if(n==1 || n==0)
		return 1;
	return n*factorial(n-1);
}

int fibo(int n)
{
	if(n==1)
		return 1;
	if (n==2)
		return 1;
	return fibo(n-1)+fibo(n-2);
}

int main(int argc, char *argv[]) 
{
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank%2==0)
		printf("Rank: %d, fac %d\n", rank,factorial(rank));
	else
		printf("Rank: %d, fib %d\n", rank,fibo(rank));
	MPI_Finalize();
	return 0;
}
