#include <stdio.h>

int euclids_GCD(int a, int b)
{
    int r;
    int opcount = 0;
    while (b != 0)
    {
        opcount++;
        r = a % b;
        a = b;
        b = r;
    }
    printf("The opcount is %d\n", opcount);
    return a; // Return the GCD, not the remainder 'r'
}

int main()
{
    int a, b;
    printf("Enter the two numbers whose GCD has to be calculated: ");
    scanf("%d%d", &a, &b);
    printf("GCD: %d\n", euclids_GCD(a, b));
    return 0;
}
