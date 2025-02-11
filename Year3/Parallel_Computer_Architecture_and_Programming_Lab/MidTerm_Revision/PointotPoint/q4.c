#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size, number;
    int *array = NULL;  // Array to store elements at root process
    MPI_Status status;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate buffer for MPI_Bsend
    int buf_size = size * sizeof(int) + MPI_BSEND_OVERHEAD;
    void *buffer = malloc(buf_size);
    MPI_Buffer_attach(buffer, buf_size);

    if (rank == 0) {
        // Root process reads N elements
        array = (int *)malloc(size * sizeof(int));
        printf("Enter %d elements: ", size);
        for (int i = 0; i < size; i++) {
            scanf("%d", &array[i]);
        }

        // Send one element to each process using Buffered Send
        for (int i = 0; i < size; i++) {
            MPI_Bsend(&array[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        // Free allocated memory
        free(array);
    }

    // Each process receives its assigned element
    MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    // Compute result based on process rank
    if (rank % 2 == 0) {
        printf("Process %d received %d, squared: %d\n", rank, number, number * number);
    } else {
        printf("Process %d received %d, cubed: %d\n", rank, number, number * number * number);
    }

    // Detach and free buffer
    MPI_Buffer_detach(&buffer, &buf_size);
    free(buffer);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
