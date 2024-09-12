#include <stdio.h>
#include <math.h>
int sqr(int a)
{
    int s;
    s = pow(a,2);
    return s;
}
void main()
{
    printf("%d", sqr(3));
}