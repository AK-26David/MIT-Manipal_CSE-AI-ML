#include<stdio.h>
int main()
{
    int ll,ul;
    printf("Enter the lower limit and upper limit respectively:");
    scanf("%d%d",&ll,&ul);
    int nn= printnaturalnumbers(ll,ul);
    return 0;
}
int printnaturalnumbers(int ll, int ul)
{
    if(ll>ul)
    {
        return;
    }
    printf("%d\n",ll);
    printnaturalnumbers(ll+1,ul);
}