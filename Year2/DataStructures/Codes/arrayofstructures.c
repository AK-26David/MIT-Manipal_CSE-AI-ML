#include <stdio.h>

typedef struct Cricket
{
    char name[50];
    int age;
    int testmatches;
    int runs;
} cricket;

int main()
{
    int n;
    printf("Enter the number of cricketers whose records you want to display: ");
    scanf("%d", &n);
    cricket cricketers[n];
    
    for (int i = 0; i < n; i++)
    {
        printf("\nEnter details of cricketer %d:\n", i + 1);
        printf("Name: ");
        scanf("%s", cricketers[i].name);
        
        printf("Age: ");
        scanf("%d", &cricketers[i].age);

        printf("Number of Test Matches: ");
        scanf("%d", &cricketers[i].testmatches);

        printf("Runs: ");
        scanf("%d", &cricketers[i].runs);
    }

    printf("\nBelow are the records displayed:\n");
    for (int i = 0; i < n; i++)
    {
        printf("\nCricketer %d:\n", i + 1);
        printf("Name: %s\n", cricketers[i].name);
        printf("Age: %d\n", cricketers[i].age);
        printf("Number of Test Matches: %d\n", cricketers[i].testmatches);
        printf("Runs: %d\n", cricketers[i].runs);
    }

    return 0;
}
