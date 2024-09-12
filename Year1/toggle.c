#include<stdio.h>
#include<string.h>
void toggle(char *str);
int main()
{
    char str[100];
    int i;
    printf("Enter the string:");
    gets(str);
    toggle(str);
    printf("The string after toggle is : %s",str);
    return 0;
}
void toggle(char *str)
{
    int i;
    for(i=0;str[i]!='\0';i++)
    {
        if(str[i]>='a'&& str[i]<='z')
        {
            str[i]=str[i]-32;
        }
        else if(str[i]>='A'&&str[i]<='Z');
        {
            str[i]=str[i]+32;
        }
    }
} 
