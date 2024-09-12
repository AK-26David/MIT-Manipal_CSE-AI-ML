#include<stdio.h>
void fact(int num)
{
    int i;
    long fact=1;
    for(i=1;i<=num;i++)
    {
        fact=fact*i;
    }
    return (fact);
}
long find_ncr(int n, int r) {
   long result;
   result = factorial(n)/(factorial(r)*factorial(n-r));
   return result;
}
int main()
{
    int n,r;
    printf("Enter a number n:");
    scanf("%d",&n);
    printf("Enter r:");
    scanf("%d",&r);
    find_ncr(n,r);
    printf("%d",result);
    return 0;
}