#include<stdio.h>
void printnaturalnumbers(int lowerlimit,int upperlimit)
{
    if(lowerlimit>upperlimit)
    {
        return;
    }
    printf("%d\t",lowerlimit);
    printnaturalnumbers(lowerlimit+1,upperlimit);
}
int main()
{
    int lowerlimit,upperlimit,i,n;
    printf("Enter the lower limit and the upper limit:");
    scanf("%d%d",&lowerlimit,&upperlimit);
    printnaturalnumbers(lowerlimit,upperlimit);
    return 0;
}