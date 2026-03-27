#include<stdio.h>
 struct book
 {
    char title[20];
    char author[15];
    int pages;
    float price;
 };
int main()
{
    int i,j;
    struct book b[10];
    printf("Input the values:");
    for(i=0;i<3;i++)
    {
    scanf("%s%s%d%f",b[i].title,b[i].author,&b[i].pages,&b[i].price);
    }
    for(j=0;j<3;j++)
    {
        printf("%s\t%s\t%d\t%f\n",b[j].title,b[j].author,b[j].pages,b[j].price);
    }
    return 0;
}