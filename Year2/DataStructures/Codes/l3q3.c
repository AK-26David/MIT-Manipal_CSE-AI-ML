#include <stdio.h>

struct DateofBirth
{
    int day;
    int month;
    int year;
};

struct Address
{
    int housenumber;
    int zipcode;
    char state[30];
};

struct Employee
{
    char name[20];
    struct DateofBirth dob;
    struct Address address;
};

int main()
{
    int n;
    printf("Enter the number of employees:");
    scanf("%d", &n);
    struct Employee employee[n];

    for (int i = 0; i < n; i++)
    {
        struct Employee *ptr = &employee[i];
        printf("Enter details of employee %d\n", i + 1);
        printf("-------------------------------\n");
        printf("Enter the name of the employee:");
        scanf("%s", ptr->name);
        printf("Enter the date of birth of the employee (day month year):");
        scanf("%d %d %d", &ptr->dob.day, &ptr->dob.month, &ptr->dob.year);
        printf("Enter the zipcode of the employee:");
        scanf("%d", &ptr->address.zipcode);
        printf("Enter the house-number of the employee:");
        scanf("%d", &ptr->address.housenumber);
        printf("Enter the state of the employee:");
        scanf("%s", ptr->address.state);
    }

    // Display employee details
    printf("\nEmployee Details:\n");
    for (int i = 0; i < n; i++)
    {
        struct Employee *ptr = &employee[i];
        printf("Employee %d\n", i + 1);
        printf("Name: %s\n", ptr->name);
        printf("Date of Birth: %d/%d/%d\n", ptr->dob.day, ptr->dob.month, ptr->dob.year);
        printf("Zipcode: %d\n", ptr->address.zipcode);
        printf("House Number: %d\n", ptr->address.housenumber);
        printf("State: %s\n", ptr->address.state);
        printf("-------------------------------\n");
    }

    return 0;
}
