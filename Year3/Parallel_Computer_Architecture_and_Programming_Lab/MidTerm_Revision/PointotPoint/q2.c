#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, data;
    
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Status status;

    if (rank == 0) { 
        // Master process (rank 0) sends a number to each slave process
        for (int i = 1; i < size; i++) {
            data = i * 10; // Assign a unique number for each process
            printf("Master (Process 0) sending %d to Process %d\n", data, i);
            MPI_Send(&data, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } 
    else { 
        // Slave processes receive data from master (rank 0)
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Slave (Process %d) received %d from Master\n", rank, data);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
