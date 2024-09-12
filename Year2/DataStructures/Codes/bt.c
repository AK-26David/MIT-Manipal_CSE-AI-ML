#include<stdio.h>
#include<stdlib.h>
struct Node
{
    int data;
    struct Node* left;
    struct Node* right;
};
struct Node* createNode(int data)
{
    struct Node* newNode=(struct Node*)malloc(sizeof(struct Node));
    newNode->data=data;
    newNode->left=newNode->right=NULL;
    return newNode;
}
void iterative(struct Node** root, int data)
{
    struct Node* newNode=createNode(data);
    if(*root==NULL)
    {
        *root=newNode;
        return;
    }
    struct Node* current=*root;
    struct Node* parent=NULL;
    while(current!=NULL)              
    {
        parent=current;
        if(data<current->data)
        {
            current=current->left;
        }
        else
        {
            current=current->right;
        }
    }
    if(data<parent->data)
    {
        parent->left=newNode;
    }
    else
    {
        parent->right=newNode;
    }
}
void inOrdertraversal(struct Node* root)
{
    if(root!=NULL)
    {
        inOrdertraversal(root->left);
        printf("%d",root->data);
        inOrdertraversal(root->right);
    }
}
void preOrdertraversal(struct Node* root)
{
    if(root!=NULL)
    {
        printf("%d",root->data);
        preOrdertraversal(root->left);
        preOrdertraversal(root->right);
    }
}
void postOrdertraversal(struct Node* root)
{
    if(root!=NULL)
    {
        postOrdertraversal(root->left);
        postOrdertraversal(root->right);
        printf("%d",root->data);
    }
}
int main()
{
    struct Node* root=NULL;
    int n,data;
    printf("Enter the number of required entires:");
    scanf("%d",&n);
    for(int i=0;i<=n;i++)
    {
        printf("Enter data:");
        scanf("%d",&data);
        iterative(&root,data);
    }
    inOrdertraversal(root);
    preOrdertraversal(root);
    postOrdertraversal(root);
    return 0;
}