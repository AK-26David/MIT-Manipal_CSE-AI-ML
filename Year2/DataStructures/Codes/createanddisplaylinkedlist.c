#include<stdio.h>
#include<stdlib.h>

struct Node
{
    int data;
    struct Node* next;
};

struct Node* createNode(int data)
{
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    if (newNode == NULL)
    {
        printf("Memory Allocation failed");
    }
    else
    {
        newNode->data = data;
        newNode->next = NULL;
    }
    return newNode;
}

void insertNode(struct Node** head, int data)
{
    struct Node* newNode = createNode(data);
    if (*head == NULL)
    {
        *head = newNode;
    }
    else
    {
        struct Node* current = *head;
        while (current->next != NULL)
        {
            current = current->next;
        }
        current->next = newNode;
    }
}

void displayNode(struct Node* head)
{
    struct Node* current = head;
    while (current != NULL)
    {
        printf("%d ", current->data);
        current = current->next;
    }
}
void insertElement(struct Node** head, int data, int pos)
{
    struct Node* newNode=createNode(data);
    if(pos==1)
    {
        newNode->next=*head;
        *head=newNode;
    }
    else
    {
        struct Node* current=*head;
        for(int i=0;i<pos-2&&current->next!=NULL;i++)
        {
            current=current->next;
        }
        if(current!=NULL)
        {
        printf("Cannot insert element!!!");
        }
    }
}

void deleteElement(struct Node** head, int ele) {
    struct Node* current = *head;
    struct Node* prev = NULL;

    // Check if the element to be deleted is in the first node
    if (current != NULL && current->data == ele) {
        *head = current->next;
        free(current);
        return;
    }

    // Search for the element to be deleted
    while (current != NULL && current->data != ele) {
        prev = current;
        current = current->next;
    }

    // If the element is not present in the linked list
    if (current == NULL) {
        printf("Element %d not found in the linked list.\n", ele);
        return;
    }

    // Unlink the node from the linked list
    prev->next = current->next;
    free(current);
}

int main()
{
    struct Node* head = NULL;
    int n, data;
    printf("Enter the number of entries in the linked list:");
    scanf("%d", &n);
    for (int i = 0; i < n; i++)
    {
        printf("Enter data:");
        scanf("%d", &data);
        insertNode(&head, data);
    }
    printf("The linked list is:\n");
    displayNode(head);

    // Free the memory for the linked list nodes
    struct Node* current = head;
    struct Node* temp;
    while (current != NULL)
    {
        temp = current;
        current = current->next;
        free(temp);
    }

    return 0;
}
