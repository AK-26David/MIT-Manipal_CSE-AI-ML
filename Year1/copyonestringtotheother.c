#include<stdio.h>
int main()
{
    char str[100];
    char str2[100];
    int i;
    printf("Enter the string:");
    gets(str);
    for(i=0;str[i]!='\0';i++)
    {
        str2[i]=str[i];
    }
    str2[i]='\0';
    printf("%d",i);
    return 0;
}