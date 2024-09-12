#include <stdio.h>
#include <stdlib.h>

// Create a struct for a node
struct node {
  int vertex;
  struct node* next;
};

// Create a struct for a graph
struct Graph {
  int numVertices;
  int* visited;

  // We need int** to store a two dimensional array.
  // Similarly, we need struct node** to store an array of Linked lists
  struct node** adjLists;
};

// Create a node
struct node* createNode(int v) {
  struct node* newNode = (struct node*) malloc(sizeof(struct node));
  newNode->vertex = v;
  newNode->next = NULL;
  return newNode;
}

// Create a graph
struct Graph* createGraph(int numVertices) {
  struct Graph* graph = (struct Graph*) malloc(sizeof(struct Graph));
  graph->numVertices = numVertices;
  graph->visited = (int*) malloc(numVertices * sizeof(int));
  graph->adjLists = (struct node**) malloc(numVertices * sizeof(struct node*));

  int i;
  for (i = 0; i < numVertices; i++) {
    graph->adjLists[i] = NULL;
    graph->visited[i] = 0;
  }

  return graph;
}

// Add an edge to the graph
void addEdge(struct Graph* graph, int src, int dest) {
  struct node* newNode = createNode(dest);
  newNode->next = graph->adjLists[src];
  graph->adjLists[src] = newNode;

  newNode = createNode(src);
  newNode->next = graph->adjLists[dest];
  graph->adjLists[dest] = newNode;
}

// Perform DFS traversal on the graph
void DFS(struct Graph* graph, int vertex) {
  struct node* adjList = graph->adjLists[vertex];
  struct node* temp = adjList;

  graph->visited[vertex] = 1;
  printf("Visited %d \n", vertex);

  while (temp != NULL) {
    int connectedVertex = temp->vertex;

    if (graph->visited[connectedVertex] == 0) {
      DFS(graph, connectedVertex);
    }
    temp = temp->next;
  }
}

// Take user input and perform DFS traversal
int main() {
  int numVertices, numEdges, i, start, end;

  printf("Enter the number of vertices: ");
  scanf("%d", &numVertices);

  printf("Enter the number of edges: ");
  scanf("%d", &numEdges);

  struct Graph* graph = createGraph(numVertices);

  printf("Enter the edges (start end): ");
  for (i = 0; i < numEdges; i++) {
    scanf("%d %d", &start, &end);
    addEdge(graph, start, end);
  }

  printf("Enter the starting vertex for DFS traversal: ");
  scanf("%d", &start);

  printf("DFS Traversal Order: ");
  DFS(graph, start);

  return 0;
}