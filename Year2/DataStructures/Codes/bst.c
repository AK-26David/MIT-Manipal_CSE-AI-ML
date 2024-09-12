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
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode->data = data;
    newNode->left = newNode->right = NULL;
    return newNode;
}
struct Node* createBST(struct Node* root, int item)
{
    if (root == NULL)
    {
        return createNode(item);
    }
    if (item < root->data)
    {
        root->left = createBST(root->left, item);
    }
    else if (item > root->data)
    {
        root->right = createBST(root->right, item);
    }
    return root;
}
void inOrderTraversal(struct Node* root)
{
    if (root != NULL)
    {
        inOrderTraversal(root->left);
        printf("%d ", root->data);
        inOrderTraversal(root->right);
    }
}
void postOrderTraversal(struct Node* root)
{
    if(root!=NULL)
    {
        postOrderTraversal(root->left);
        postOrderTraversal(root->right);
        printf("%d",root->data); 
    }
}
void preOrderTraversal(struct Node* root)
{
    if(root!=NULL)
    {
        printf("%d",root->data);
        preOrderTraversal(root->left);
        preOrderTraversal(root->right);
    }
}
void freeTree(struct Node* root)
{
    if (root != NULL)
    {
        freeTree(root->left);
        freeTree(root->right);
        free(root);
    }
}
int main()
{
    struct Node* root = NULL;
    int data, n;
    printf("Enter the number of entries required:");
    scanf("%d", &n);
    for (int i = 0; i < n; i++) 
    {
        printf("Enter data:");
        scanf("%d", &data);
        root = createBST(root, data);
    }
    printf("The inorder traversal is: ");
    inOrderTraversal(root);
    printf("The postorder traversal of the tree is");
    postOrderTraversal(root);
    printf("The preorder traversal is :");
    preOrderTraversal(root);
    freeTree(root);
    return 0;
}
