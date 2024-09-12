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
    struct book b1;
    printf("Input the values:");
    scanf("%s%s%d%f",b1.title,b1.author,&b1.pages,&b1.price);
    gets(b1.title);
    gets(b1.author);
    printf("%s%s%d%f",b1.title,b1.author,b1.pages,b1.price);
    return 0;
}