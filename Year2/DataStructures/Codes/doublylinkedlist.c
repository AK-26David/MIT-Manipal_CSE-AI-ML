#include<stdio.h>
#include<stdlib.h>
struct Node
{
    int data;
    struct Node* next;
    struct Node* prev;
};
struct Node* createNode(int data)
{
    struct Node* nextNode=(struct node*)malloc(sizeof(struct Node));
    nextNode->data=data;
    nextNode->next=nextNode->prev=NULL;
}
void insertNode(struct Node** head,int data)
{
    struct Node* nextNode=createNode(data);
    if(*head==NULL)
    {
        *head=nextNode;
    }
    else
    {
        struct Node* current=*head;
        while(current->next!=NULL)
        {
            current=current->next;
        }
        current->next=nextNode;
        nextNode->prev=current;
    }
}
void displayNode(struct Node* head,int data)
{
    struct Node* current=head;
    while(current->next!=NULL)
    {
        printf("%d",current->data);
        current=current->next;
    }
}
void insertElement(struct Node** head, int data, int pos)
{
    struct Node* nextNode = (struct Node*)malloc(sizeof(struct Node));
    nextNode->data = data;

    if (*head == NULL)
    {
        // If the list is empty, insert at the beginning
        nextNode->next = *head;
        nextNode->prev = NULL;
        if (*head != NULL)
        {
            (*head)->prev = nextNode;
        }
        *head = nextNode;
    }
    else
    {
        // Insert at the specified position
        struct Node* current = *head;
        int currentPos = 1;

        while (currentPos < pos && current->next != NULL)
        {
            current = current->next;
            currentPos++;
        }

        nextNode->next = current->next;
        nextNode->prev = current;

        if (current->next != NULL)
        {
            current->next->prev = nextNode;
        }

        current->next = nextNode;
    }
}
