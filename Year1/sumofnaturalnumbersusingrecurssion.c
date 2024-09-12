#include<stdio.h>
void sum(int lowerlimit, int upperlimit);
int main()
{
    int lowerlimit,upperlimit;
    printf("Enter the lower and upperlimits respectively:");
    scanf("%d%d",&lowerlimit,&upperlimit);
    sum(lowerlimit,upperlimit);
    return 0;
}
void sum(int lowerlimit,int upperlimit)
{
    if(lowerlimit==upperlimit)
    {
        return lowerlimit;
    }
    else
    {
        return lowerlimit,sum(lowerlimit+1,upperlimit);
    }
}