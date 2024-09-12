#include<stdio.h>
#include<math.h>
int main()
{
    int x,y,num;
    printf("Input a number and itss power:");
    scanf("%d%d",&x,&y);
    num=pow(x,y);
    printf("%d",num);
    return 0;
}