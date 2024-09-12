#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TABLE_SIZE 10

// Structure to represent a key-value pair
typedef struct Entry {
    int key;
    char value[50];
    struct Entry* next;
} Entry;

// Structure to represent a closed hash table
typedef struct {
    Entry* buckets[TABLE_SIZE];
} ClosedHashTable;

// Function to initialize the closed hash table
void initClosedHashTable(ClosedHashTable* ht) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        ht->buckets[i] = NULL;
    }
}

// Hash function to calculate the hash value
int hash(int key) {
    return key % TABLE_SIZE;
}

// Function to insert a key-value pair into the closed hash table
void insert(ClosedHashTable* ht, int key, const char* value) {
    int index = hash(key);

    // Create a new entry
    Entry* newEntry = (Entry*)malloc(sizeof(Entry));
    newEntry->key = key;
    strcpy(newEntry->value, value);
    newEntry->next = NULL;

    // Insert into the appropriate bucket (linked list)
    if (ht->buckets[index] == NULL) {
        ht->buckets[index] = newEntry;
    } else {
        Entry* current = ht->buckets[index];
        while (current->next != NULL) {
            current = current->next;
        }
        current->next = newEntry;
    }
}

// Function to search for a key in the closed hash table
int search(ClosedHashTable* ht, int key, int* comparisons) {
    int index = hash(key);
    Entry* current = ht->buckets[index];
    *comparisons = 0;

    // Traverse the linked list at the bucket index
    while (current != NULL) {
        (*comparisons)++;
        if (current->key == key) {
            return index; // Key found
        }
        current = current->next;
    }

    return -1; // Key not found
}

// Function to print the closed hash table
void printClosedHashTable(ClosedHashTable* ht) {
    printf("Closed Hash Table:\n");
    for (int i = 0; i < TABLE_SIZE; i++) {
        printf("Bucket %d: ", i);
        Entry* current = ht->buckets[i];
        while (current != NULL) {
            printf("(%d, %s) -> ", current->key, current->value);
            current = current->next;
        }
        printf("NULL\n");
    }
}

int main() {
    ClosedHashTable ht;
    initClosedHashTable(&ht);

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
                    printf("Key found in bucket %d\n", index);
                    printf("Number of comparisons in successful search: %d\n", comparisons_successful);
                } else {
                    printf("Key not found\n");
                    printf("Number of comparisons in unsuccessful search: %d\n", comparisons_successful + 1);
                }
                break;

            case 3:
                printClosedHashTable(&ht);
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
        Entry* current = ht.buckets[i];
        while (current != NULL) {
            Entry* temp = current;
            current = current->next;
            free(temp);
        }
    }

    return 0;
}
