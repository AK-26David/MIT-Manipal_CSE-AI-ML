#include<stdio.h>
void naturalnum(int lowerlimit,int upperlimit);
int main()
{
    int lowerlimit,upperlimit;
    printf("Enter the lowerlimit and upperlimit:");
    scanf("%d%d",&lowerlimit,&upperlimit);
    naturalnum(lowerlimit,upperlimit);
    return 0;
}
void naturalnum(int lowerlimit,int upperlimit)
{
    if(lowerlimit>upperlimit)
    {
        return 1;
    }
    printf("%d\t",lowerlimit);
    return (lowerlimit+1,upperlimit);
}