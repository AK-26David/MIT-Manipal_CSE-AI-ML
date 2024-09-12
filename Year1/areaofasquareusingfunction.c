#include<stdio.h>
int area(int length, int breadth)
{
    int area;
    area=length*breadth;
    printf("%d",area);
    return area;
}
int main()
{
    int length,breadth;
    printf("Enter the dimensions:");
    scanf("%d%d",&length,&breadth);
    area(length,breadth);
    return 0;
}
