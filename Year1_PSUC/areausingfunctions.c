#include<stdio.h>
int rect_area(int l,int w)
{
int area;
area=l*w;
return area;
}
int rect_perimeter(int l,int w)
{
int p;
p=2*(l+w);
return p;
}
int main()
{
int a,b,x,y;
printf("Enter Length of Rectangle : ");
scanf("%d",&a);
printf("\nEnter Width of Rectangle : ");
scanf("%d",&b);
x=rect_area(a,b); // calling function rect_area
y=rect_perimeter(a,b); // calling function perimeter
printf("\nArea of Rectangle = %d\n\nPerimeter of Rectangle = %d",x,y);
return 0;
}