#include<stdio.h>
int main()
{
    int x=2;
    int y=3;
    int z= &x;
    int w=&y;
   int  sum = *z + *w;
   printf("%d",sum);
}