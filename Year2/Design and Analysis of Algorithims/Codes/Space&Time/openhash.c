#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TABLE_SIZE 10

// Structure to represent a key-value pair
typedef struct {
    int key;
    char value[50];
} Entry;

// Structure to represent an open hash table
typedef struct {
    Entry* entries[TABLE_SIZE];
    int count;
} OpenHashTable;

// Function to initialize the open hash table
void initOpenHashTable(OpenHashTable* ht) {
    ht->count = 0;
    for (int i = 0; i < TABLE_SIZE; i++) {
        ht->entries[i] = NULL;
    }
}

// Hash function to calculate the hash value
int hash(int key) {
    return key % TABLE_SIZE;
}

// Function to insert a key-value pair into the open hash table
void insert(OpenHashTable* ht, int key, const char* value) {
    Entry* newEntry = (Entry*)malloc(sizeof(Entry));
    newEntry->key = key;
    strcpy(newEntry->value, value);

    int index = hash(key);

    // Linear probing to handle collisions
    while (ht->entries[index] != NULL) {
        index = (index + 1) % TABLE_SIZE; // Move to the next slot
    }

    ht->entries[index] = newEntry;
    ht->count++;
}

// Function to search for a key in the open hash table
int search(OpenHashTable* ht, int key, int* comparisons) {
    int index = hash(key);
    int original_index = index;
    int count = 0;

    while (ht->entries[index] != NULL && ht->entries[index]->key != key && count < TABLE_SIZE) {
        index = (index + 1) % TABLE_SIZE;
        count++;
    }

    *comparisons = count + 1; // Include the final comparison

    if (ht->entries[index] != NULL && ht->entries[index]->key == key) {
        return index;
    }

    return -1; // Key not found
}

// Function to print the open hash table
void printHashTable(OpenHashTable* ht) {
    printf("Open Hash Table:\n");
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (ht->entries[i] != NULL) {
            printf("Index %d: Key = %d, Value = %s\n", i, ht->entries[i]->key, ht->entries[i]->value);
        } else {
            printf("Index %d: Empty\n", i);
        }
    }
}

int main() {
    OpenHashTable ht;
    initOpenHashTable(&ht);

    int key, choice;
    char value[50];
    int comparisons_successful = 0;
    int comparisons_unsuccessful = 0;

    do {
        printf("\n1. Insert\n2. Search\n3. Print Hash Table\n4. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice) {
            case 1:
                printf("Enter key and value to insert: ");
                scanf("%d %s", &key, value);
                insert(&ht, key, value);
                break;

            case 2:
                printf("Enter key to search: ");
                scanf("%d", &key);

                int index = search(&ht, key, &comparisons_successful);
                if (index != -1) {
                    printf("Key found at index %d\n", index);
                    printf("Number of comparisons in successful search: %d\n", comparisons_successful);
                } else {
                    printf("Key not found\n");
                    printf("Number of comparisons in unsuccessful search: %d\n", TABLE_SIZE);
                }
                break;

            case 3:
                printHashTable(&ht);
                break;

            case 4:
                printf("Exiting...\n");
                break;

            default:
                printf("Invalid choice. Please try again.\n");
        }
    } while (choice != 4);

    // Free memory allocated for entries in the hash table
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (ht.entries[i] != NULL) {
            free(ht.entries[i]);
        }
    }

    return 0;
}
