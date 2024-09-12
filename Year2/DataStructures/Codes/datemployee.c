/*Create a structure ‘Date’ with date as numeric field, month as string field and the 
year of the type numeric. Create another structure ‘Employee’ with employee name 
of string type, salary of float type , employee id of numeric type and the date of join 
which is of structure type ‘Date’. Write a program in ‘C’ to create an array of N 
instance of such ‘Employee records’, to read values and display all its member fields.*/
#include <stdio.h>
struct Date
{
    int date;
    char month[7];
    int year;
};
struct Employee
{
    char name[20];
    float salary;
    int employeeid;
    struct Date dateofjoining;
};
int main()
{
    int n;
    printf("Enter the number of employees to be inputted:");
    scanf("%d", &n);
    struct Employee employee[n];
    for (int i = 0; i < n; i++)
    {
        printf("Enter details of employee %d\n", i + 1);
        printf("----------------------------\n");
        printf("Enter Name:");
        scanf("%s", employee[i].name);
        printf("Enter Salary:");
        scanf("%f", &employee[i].salary);
        printf("Enter Employee id:");
        scanf("%d", &employee[i].employeeid);
        printf("Enter date of joining (dd mm yyyy):");
        scanf("%d %s %d", &employee[i].dateofjoining.date, employee[i].dateofjoining.month, &employee[i].dateofjoining.year);
    }
    for (int i = 0; i < n; i++)
    {
        printf("Details of employee %d\n", i + 1);
        printf("----------------------------\n");
        printf("Name: %s\n", employee[i].name);
        printf("Salary: %.2f\n", employee[i].salary);
        printf("Employee id: %d\n", employee[i].employeeid);
        printf("Date of joining: %d %s %d\n", employee[i].dateofjoining.date, employee[i].dateofjoining.month, employee[i].dateofjoining.year);
    }
    return 0;
}
