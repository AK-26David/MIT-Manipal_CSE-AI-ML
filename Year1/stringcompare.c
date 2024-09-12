#include<stdio.h>
#include<string.h>
int main()
{
    char str[100],str2[100];
    int i,result;
    printf("Enter the first string:");
    gets(str);
    printf("Enter the second string:");
    gets(str2);
    result=strcmp(str,str2);
    if(result==0)
    {
    printf("Both strings are equal.");
    }
    if(result==-1)
    {
        printf("String 1 is greater than string 2");
    }
    else
    {
        printf("String 2 is greater than string 1");
    }
    return 0;
}