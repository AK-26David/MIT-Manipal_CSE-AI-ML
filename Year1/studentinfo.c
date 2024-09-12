#include<stdio.h>
struct Student_Information
{
    char name[20];
    int age;
    int cgpa;
};
int main()
{
    struct Student_Information s1;
    printf("Enter name:");
    scanf("%s",s1.name);
    gets(s1.name);
    printf("\nEnter age:");
    scanf("%d",&s1.age);
    printf("\nEnter cgpa:");
    scanf("%d",&s1.cgpa);
    printf("Information:");
    puts(s1.name);
    printf("%s%d%d",s1.name,s1.age,s1.cgpa);
    return 0;
}