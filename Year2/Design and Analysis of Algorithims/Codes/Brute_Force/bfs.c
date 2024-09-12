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
  struct node** adjLists;
};

// Create a node
struct node* createNode(int v) {
  struct node* newNode = (struct node*)malloc(sizeof(struct node));
  newNode->vertex = v;
  newNode->next = NULL;
  return newNode;
}

// Create a graph
struct Graph* createGraph(int numVertices) {
  struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
  graph->numVertices = numVertices;
  graph->visited = (int*)malloc(numVertices * sizeof(int));
  graph->adjLists = (struct node**)malloc(numVertices * sizeof(struct node*));

  for (int i = 0; i < numVertices; i++) {
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

// Implement a queue using an array
struct Queue {
  int front, rear, size;
  unsigned capacity;
  int* array;
};

struct Queue* createQueue(unsigned capacity) {
  struct Queue* queue = (struct Queue*)malloc(sizeof(struct Queue));
  queue->capacity = capacity;
  queue->front = queue->size = 0;
  queue->rear = capacity - 1;
  queue->array = (int*)malloc(queue->capacity * sizeof(int));
  return queue;
}

int isFull(struct Queue* queue) {
  return (queue->size == queue->capacity);
}

int isEmpty(struct Queue* queue) {
  return (queue->size == 0);
}

void enqueue(struct Queue* queue, int item) {
  if (isFull(queue))
    return;
  queue->rear = (queue->rear + 1) % queue->capacity;
  queue->array[queue->rear] = item;
  queue->size += 1;
}

int dequeue(struct Queue* queue) {
  if (isEmpty(queue))
    return -1;
  int item = queue->array[queue->front];
  queue->front = (queue->front + 1) % queue->capacity;
  queue->size -= 1;
  return item;
}

// Perform BFS traversal on the graph using a queue
void BFS(struct Graph* graph, int startVertex) {
  // Create a queue for BFS
  struct Queue* queue = createQueue(graph->numVertices);

  // Enqueue the starting vertex and mark it as visited
  enqueue(queue, startVertex);
  graph->visited[startVertex] = 1;

  while (!isEmpty(queue)) {
    int currentVertex = dequeue(queue);
    printf("Visited %d \n", currentVertex);

    struct node* adjList = graph->adjLists[currentVertex];
    struct node* temp = adjList;

    while (temp != NULL) {
      int connectedVertex = temp->vertex;
      if (graph->visited[connectedVertex] == 0) {
        enqueue(queue, connectedVertex);
        graph->visited[connectedVertex] = 1;
      }
      temp = temp->next;
    }
  }

  // Free the memory used by the queue
  free(queue->array);
  free(queue);
}

// Take user input and perform BFS traversal
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

  printf("Enter the starting vertex for BFS traversal: ");
  scanf("%d", &start);

  printf("BFS Traversal Order: ");
  BFS(graph, start);

  // Free the memory used by the graph
  for (i = 0; i < graph->numVertices; i++) {
    struct node* temp = graph->adjLists[i];
    while (temp != NULL) {
      struct node* prev = temp;
      temp = temp->next;
      free(prev);
    }
  }
  free(graph->adjLists);
  free(graph->visited);
  free(graph);

  return 0;
}
