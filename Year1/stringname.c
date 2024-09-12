#include<stdio.h>
int main()
{
    char name[80];
    printf("Enter the name:");
    gets(name);
    printf("Your name is:");
    puts(name);
    return 0;
}