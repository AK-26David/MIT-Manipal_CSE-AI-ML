#include <stdio.h>
#include <stdlib.h>

struct dateofjoining
{
    int day;
    int month;
    int year;
};

struct Employee
{
    char name[20];
    int employeeid;
    int salary;
    struct dateofjoining doj; 
};

int main()
{
    int n;
    printf("Enter the number of employees: ");
    scanf("%d", &n);
    
    struct Employee employee[n];

    for (int i = 0; i < n; i++)
    {
        struct Employee *ptr = &employee[i]; 
        printf("Enter details for employee %d\n", i + 1);
        printf("-----------------------------\n");
        printf("Enter name: ");
        scanf("%s", ptr->name);
        printf("Enter employee id: ");
        scanf("%d", &ptr->employeeid);
        printf("Enter salary: ");
        scanf("%d", &ptr->salary);
        printf("Enter the date of joining (day month year): ");
        scanf("%d%d%d", &ptr->doj.day, &ptr->doj.month, &ptr->doj.year);
    }
    //to display//
    for (int i = 0; i < n; i++)
    {
        struct Employee *ptr = &employee[i]; 
        printf("Details for employee %d\n", i + 1);
        printf("-----------------------------\n");
        printf("Name: %s ",ptr->name);
        printf("Employee id: %d ",ptr->employeeid);
        printf("Salary:%d ",ptr->salary);
        printf("Date of joining (day month year):%d-%d-%d ",ptr->doj.day, ptr->doj.month, ptr->doj.year);
    }
    return 0;
}
