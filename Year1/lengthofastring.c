#include<stdio.h>
int main()
{
    char str[100];
    int i,count=0;
    printf("Enter the string:");
    gets(str);
    for(i=0;str[i]!='\0';i++)
    {
        count++;
    }
    printf("Length of string:%d",count);
    return 0;
}