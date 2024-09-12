#include <stdio.h>
#define MAX_SIZE 100
int size;
int front = -1;
int rear = -1;
int queue[MAX_SIZE];
void Enqueue(int element)
{
    if (rear == MAX_SIZE - 1)
    {
        printf("Queue is overflowing!!!!\n");
    }
    else
    {
        if (front == -1)
        {
            front = 0;
        }
        rear++;
        queue[rear] = element;
    }
}
int Dequeue()
{
    int dequeuedElement = -1; 
    if (front == -1)
    {
        printf("Queue is underflowing!!!!\n");
    }
    else
    {
        dequeuedElement = queue[front];
        if (front == rear)
        {
            front = rear = -1; 
        }
        else
        {
            front++; 
        }
    }
    return dequeuedElement;
}
void Display()
{
    if (front == -1)
    {
        printf("Queue is empty\n");
    }
    else
    {
        printf("Queue elements: ");
        for (int i = front; i <= rear; i++)
        {
            printf("%d ", queue[i]);
        }
        printf("\n");
    }
}

int main()
{
    printf("Enter the size of the queue: ");
    scanf("%d", &size);

    int choice, element;
    do
    {
        printf("\nQueue Operations:\n");
        printf("1. Enqueue\n");
        printf("2. Dequeue\n");
        printf("3. Display\n");
        printf("4. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice)
        {
        case 1:
            printf("Enter the element to enqueue: ");
            scanf("%d", &element);
            Enqueue(element);
            break;
        case 2:
            element = Dequeue();
            if (element != -1)
            {
                printf("Dequeued element: %d\n", element);
            }
            break;
        case 3:
            Display();
            break;
        case 4:
            printf("Exiting the program\n");
            break;
        default:
            printf("Invalid choice! Please try again.\n");
        }
    } while (choice != 4);

    return 0;
}