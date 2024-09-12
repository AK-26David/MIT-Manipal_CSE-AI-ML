#include<stdio.h>
void towerofhanoi(int n, char a, char b, char c)
{
    if(n == 1)
    {
        printf("Move disc 1 from pole %c to pole %c.\n", a, c);
        return;
    }
    
    towerofhanoi(n - 1, a, b, c);
    printf("Move disc %d from pole %c to pole %c.\n", n, a, c);
    towerofhanoi(n - 1, b, a, c);
}
int main()
{
    int n;
    char a,b,c; // Initialize the pole names //
    printf("Enter the number of discs:");
    scanf("%d", &n);
    towerofhanoi(n, a, b, c);
    return 0;
}
