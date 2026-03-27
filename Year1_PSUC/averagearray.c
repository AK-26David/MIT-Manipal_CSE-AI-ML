#include<stdio.h>
int main()
{
    int weight[1000],n,i;
    float sum,average;
    printf("Enter the dimension of the array (total number of weights to be inputted between 0 to 1000):");
    scanf("%d",&n);
    printf("Enter the elements of the weight array(input the weights):");
    for(i=0;i<n;i++)
    scanf("%d",&weight[i]);
    for(i=0;i<n;i++)
    sum+=weight[i];
    average=sum/n;
    printf("Average weight of the elephants is %f",average);
    return 0;
}