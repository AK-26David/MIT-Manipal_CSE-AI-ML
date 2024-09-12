#include <stdio.h>
#include <stdlib.h>
struct Stack
{
    int *data;
    int top;
    int size;
};
struct Stack *createStack(int size)
{
    struct Stack *stack = (struct Stack *)malloc(sizeof(struct Stack));
    if (stack == NULL)
    {
        printf("Memory allocation has failed!!");
        exit(1);
    }
    stack->size = size; 
    stack->top = -1;
    stack->data = (int *)malloc(size * sizeof(int));
    if (stack->data == NULL)
    {
        printf("Memory Allocation has failed!!!");
        exit(1);
    }
    return stack;
}
int isFull(struct Stack *stack)
{
    return stack->top == stack->size - 1;
}
int isEmpty(struct Stack *stack) 
{
    return stack->top == -1;
}
void push(struct Stack *stack, int element)
{
    if (isFull(stack))
    {
        printf("The Stack is full, and the element cannot be pushed!!!");
    }
    else
    {
        stack->data[++stack->top] = element;
    }
}
int pop(struct Stack *stack)
{
    if (isEmpty(stack))
    {
        printf("The stack is empty, and no element can be popped!!!");
        return -1; 
    }
    else
    {
        return stack->data[stack->top--];
    }
}
void display(struct Stack *stack)
{
    if (isEmpty(stack))
    {
        printf("There are no elements present in the stack!!");
    }
    else
    {
        printf("%d\n", stack->data[stack->top]);
    }
}
void freeStack(struct Stack *stack)
{
    free(stack->data);
    free(stack);
}
int main()
{
    int choice, element;
    struct Stack *stack = createStack(100);
    while (1)
    {
        printf("Choose one of the following:\n");
        printf("-------------------------------\n");
        printf("1. Push\n");
        printf("2. Pop\n");
        printf("3. Display\n");
        printf("4. Free\n");
        printf("5. Exit\n");
        scanf("%d", &choice);
        switch (choice)
        {
        case 1:
            printf("Enter the element to be pushed:");
            scanf("%d", &element);
            push(stack, element);
            break;
        case 2:
            element = pop(stack);
            if (element != -1)
            {
                printf("Element popped is %d", element);
            }
            break;
        case 3:
            display(stack);
            break;
        case 4:
            freeStack(stack);
            break;
        case 5:
            freeStack(stack);
            exit(0);
        default:
            printf("Incorrect Choice!!\n");
        }
    }
    return 0;
}