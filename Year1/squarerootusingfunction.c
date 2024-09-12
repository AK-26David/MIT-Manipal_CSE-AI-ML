#include<stdio.h>
#include<math.h>
int root(int a)
{
    int squareroot;
    squareroot=pow((a),0.5);
    printf("%d",squareroot);
    return squareroot;
}
int main()
{
    int a;
    printf("Enter the number:");
    scanf("%d",&a);
    root(a);
    return 0;
}