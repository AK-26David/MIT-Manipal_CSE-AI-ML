#include<stdio.h>
struct dateofjoining
{
    int day;
    char month;
    int year;
};
struct Employee
{
    char name[20];
    int employeeid;
    int salary;
    int age;
    struct dateofjoining doj;
};
int salarypermonth(int salary)
{
    int sm= salary/12;
    return sm;
}
int main()
{
    int n;
    printf("Enter the number of inputs required:");
    scanf("%d",&n);
    struct Employee employee[n];
    for(int i=0;i<n;i++)
    {
        struct Employee *ptr=&employee[i];
        printf("Enter details for employee %d",i+1);
        printf("-------------------------------");
        printf("Enter name:");
        scanf("%s",&employee[i].name);
        printf("Enter age:");
        scanf("%d",&employee[i].age);
        printf("Enter employee id:");
        scanf("%d",&employee[i].employeeid);
        printf("Enter salary in integer form only:");
        scanf("%d",&employee[i].salary);
        printf("Enter date of joining:");
        scanf("%d%s%d",&employee[i].doj.day,&employee[i].doj.month,&employee[i].doj.year);
    }
    //To display//
    for(int i=0;i<n;i++)
    {
        int spm= salarypermonth(employee[i].salary);
        struct Employee *ptr= &employee[i];
        printf("Details of employee %d",i+1);
        printf("-----------------------");
        printf("Name:%s",ptr->name);
        printf("Age:%d",ptr->age);
        printf("Employee id:%d",ptr->employeeid);
        printf("Salary:%d",ptr->salary);
        printf("Date of joining:%d-%s-%d",employee[i].doj.day,employee[i].doj.month,employee[i].doj.year);
        printf("Salary per month:%d",spm);
    }
    return 0;
}