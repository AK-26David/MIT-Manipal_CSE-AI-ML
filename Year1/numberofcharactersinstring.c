#include<stdio.h>
int main()
{
    const char max=80;
    char str[max];
    int i=0,count=0;
    printf("Enter the string:");
    gets(str);
    puts(str);
    while(str[i]!='\0')
    {
        count++;
        i++;
    }
    printf("Number of characters is %d",count);
    return 0;
}