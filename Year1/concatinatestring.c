#include<stdio.h>
#include<string.h>
int main()
{
    char str[100],str2[100],i;
    printf("Enter the first string:");
    gets(str);
    printf("Enter the second string:");
    gets(str2);
    strcat(str,str2);
    printf("The new string is : %s",str);
    return 0;
}