#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, N = 8;  // Example: Array size N must be divisible by number of processes
    int even_count = 0, odd_count = 0;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = N / size;  // Each process handles chunk_size elements
    int A[N], local_chunk[chunk_size], modified_chunk[chunk_size];

    if (rank == 0) {
        // Root process initializes the array
        printf("Enter %d elements of array: ", N);
        for (int i = 0; i < N; i++) {
            scanf("%d", &A[i]);
        }
    }

    // Scatter chunks of the array to all processes
    MPI_Scatter(A, chunk_size, MPI_INT, local_chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Replace even numbers with 1 and odd numbers with 0
    for (int i = 0; i < chunk_size; i++) {
        if (local_chunk[i] % 2 == 0) {
            modified_chunk[i] = 1;
            even_count++;
        } else {
            modified_chunk[i] = 0;
            odd_count++;
        }
    }

    // Gather the modified array in the root process
    MPI_Gather(modified_chunk, chunk_size, MPI_INT, A, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Reduce counts of even and odd numbers to root process
    int total_even, total_odd;
    MPI_Reduce(&even_count, &total_even, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&odd_count, &total_odd, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Print the modified array
        printf("Modified array: ");
        for (int i = 0; i < N; i++) {
            printf("%d ", A[i]);
        }
        printf("\nTotal Even Count: %d\nTotal Odd Count: %d\n", total_even, total_odd);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
