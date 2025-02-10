#include <stdio.h>
#include <mpi.h>
#include <ctype.h>
#include <string.h>

#define MAX_LEN 100  // Maximum word length

// Function to toggle case of each letter
void toggle_case(char *word) {
    for (int i = 0; word[i] != '\0'; i++) {
        if (islower(word[i])) {
            word[i] = toupper(word[i]);
        } else if (isupper(word[i])) {
            word[i] = tolower(word[i]);
        }
    }
}

int main(int argc, char *argv[]) {
    int rank;
    char word[MAX_LEN];

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        // Process 0 (Sender)
        printf("Enter a word: ");
        scanf("%s", word);

        // Send the word to process 1 using synchronous send
        MPI_Ssend(word, MAX_LEN, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent: %s\n", word);

        // Receive the toggled word from process 1
        MPI_Recv(word, MAX_LEN, MPI_CHAR, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0 received toggled word: %s\n", word);
    } 
    else if (rank == 1) {
        // Process 1 (Receiver)
        MPI_Recv(word, MAX_LEN, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received: %s\n", word);

        // Toggle case of received word
        toggle_case(word);

        // Send the toggled word back to process 0
        MPI_Ssend(word, MAX_LEN, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        printf("Process 1 sent back toggled word: %s\n", word);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
