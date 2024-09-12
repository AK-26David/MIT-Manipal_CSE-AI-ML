#include <stdio.h>
int stack[100]; // Changed global variables to be inside functions//
int top = -1; // Initialize top to -1//
int n; // Moved n to a global variable//
// Function prototypes//
void push();
void pop();
void display();
int main() {
    printf("Enter the dimensions in the stack: ");
    scanf("%d", &n);
    int choice;
    while (1)
     { // Use a loop to keep the menu running//
        printf("\nChoose one of the following options:\n");
        printf("1. Push\n");
        printf("2. Pop\n");
        printf("3. Display\n");
        printf("4. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);
        switch (choice) {
            case 1:
                push();
                break;
            case 2:
                pop();
                break;
            case 3:
                display();
                break;
            case 4:
                printf("Exiting......\n");
                return 0; // Exit the program//
            default:
                printf("Invalid choice!! Please enter your choice again.\n");
                break;
        }
    }
    return 0;
}
void push() {
    int element;
    if (top == n - 1) { // Check if the stack is full
        printf("Stack is overflow!!!\n");
    } else {
        printf("Enter the value in the stack: ");
        scanf("%d", &element);
        top += 1;
        stack[top] = element;
    }
}
void pop() {
    if (top == -1) { // Check if the stack is empty
        printf("Stack is underflow!!!\n");
    } else {
        printf("Popped element: %d\n", stack[top]);
        top -= 1;
    }
}
void display() {
    if (top != -1) {
        printf("Stack elements:\n");
        for (int i = top; i >= 0; i--) {
            printf("%d\n", stack[i]);
        }
    } else {
        printf("Empty stack!!!!\n");
    }
}