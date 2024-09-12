#include <stdio.h>

struct Student
{
    char name[20];
    int rno;
    int perc;
};

int main()
{
    int n;
    printf("Enter the number of inputs required: ");
    scanf("%d", &n);
    
    struct Student student[n];
    
    for (int i = 0; i < n; i++)
    {
        struct Student *ptr = &student[i]; // Pointer to the current student
        
        printf("Enter the name: ");
        scanf("%s", &ptr->name);
        
        printf("Enter roll number: ");
        scanf("%d", &ptr->rno);
        
        printf("Enter percentage in integer form only: ");
        scanf("%d", &ptr->perc);
    }

    for (int i = 0; i < n; i++)
    {
        struct Student *ptr = &student[i]; // Pointer to the current student
        
        printf("Name: %s\n", ptr->name);
        printf("Roll number: %d\n", ptr->rno);
        printf("Percentage: %d\n", ptr->perc);
    }

    return 0;
}
