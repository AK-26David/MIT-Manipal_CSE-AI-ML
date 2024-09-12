#include<stdio.h>
int main()
{
    char question[]="Please enter your first name";
    char greeting[]="Hello";
    char yourname[80];
    scanf("%s",yourname);
    printf("%s,%s\n",greeting,yourname);
    return 0;
}