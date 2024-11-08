
#include <stdio.h>

int main()
{
    long long decimal, tempDecimal, binary;
    int rem, place = 1;

    binary = 0;

    /* Input decimal number from user */
    printf("Enter any decimal number: ");
    scanf("%lld", &decimal);
    tempDecimal = decimal;

    /* Decimal to binary conversion */
    while(tempDecimal > 0)
    {
        rem = tempDecimal % 2;

        binary = (rem * place) + binary;

        tempDecimal /= 2;

        place *= 10;
    }

    printf("Decimal number = %lld\n", decimal);
    printf("Binary number = %lld", binary);

    return 0;
}