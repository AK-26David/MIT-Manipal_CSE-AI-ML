#include<stdio.h>
int isArmstrong(int num)
{
    int temp,lastdigit,sum;
    temp=num;
    sum=0;
    while(temp!=0)
    {
        lastdigit=temp%10;
        sum+=lastdigit*lastdigit*lastdigit;
        temp/=10;
    }
    if(num==sum)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
int printArmstrong(int start,int end)
{
    while(start<=end)
    {
        if (isArmstrong(start))
        {
            printf("%d",start);
        }
        start++;
    }
    return 0;
}
int main()
{
    int start,end;
    printf("Enter the higher and lower limits:");
    scanf("%d%d",&start,&end);
    printf("All the Armstrong numbers are:");
    printArmstrong(start,end);
    return 0;
}