#include<stdio.h>

struct employeerecord {
    char name[20];
    int employeeid;
    char address[50];
    char email[30];
};

int main() {
    struct employeerecord s1, s2, s3, s4;

    // Input
    printf("Input details for Employee 1:\n");
    printf("Enter employee name: ");
    scanf("%s", s1.name);
    printf("Input employee id: ");
    scanf("%d", &s1.employeeid);
    printf("Enter employee address: ");
    scanf("%s", s1.address);
    printf("Enter employee email id: ");
    scanf("%s", s1.email);

    printf("Input details for Employee 2:\n");
    printf("Enter employee name: ");
    scanf("%s", s2.name);
    printf("Input employee id: ");
    scanf("%d", &s2.employeeid);
    printf("Enter employee address: ");
    scanf("%s", s2.address);
    printf("Enter employee email id: ");
    scanf("%s", s2.email);

    printf("Input details for Employee 3:\n");
    printf("Enter employee name: ");
    scanf("%s", s3.name);
    printf("Input employee id: ");
    scanf("%d", &s3.employeeid);
    printf("Enter employee address: ");
    scanf("%s", s3.address);
    printf("Enter employee email id: ");
    scanf("%s", s3.email);

    printf("Input details for Employee 4:\n");
    printf("Enter employee name: ");
    scanf("%s", s4.name);
    printf("Input employee id: ");
    scanf("%d", &s4.employeeid);
    printf("Enter employee address: ");
    scanf("%s", s4.address);
    printf("Enter employee email id: ");
    scanf("%s", s4.email);

    // Display
    printf("\nEntered details for Employee 1:\n");
    printf("Name: %s\n", s1.name);
    printf("Employee id: %d\n", s1.employeeid);
    printf("Employee address: %s\n", s1.address);
    printf("Employee email id: %s\n", s1.email);

    printf("\nEntered details for Employee 2:\n");
    printf("Name: %s\n", s2.name);
    printf("Employee id: %d\n", s2.employeeid);
    printf("Employee address: %s\n", s2.address);
    printf("Employee email id: %s\n", s2.email);

    printf("\nEntered details for Employee 3:\n");
    printf("Name: %s\n", s3.name);
    printf("Employee id: %d\n", s3.employeeid);
    printf("Employee address: %s\n", s3.address);
    printf("Employee email id: %s\n", s3.email);

    printf("\nEntered details for Employee 4:\n");
    printf("Name: %s\n", s4.name);
    printf("Employee id: %d\n", s4.employeeid);
    printf("Employee address: %s\n", s4.address);
    printf("Employee email id: %s\n", s4.email);

    return 0;
}
