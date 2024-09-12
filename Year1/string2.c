#include<stdio.h>
int main()
{
    const int max=80;
    char str[max];
    printf("Enter the string");
    gets(str);
    printf("The string is:");
    puts(str);
    return 0;
}