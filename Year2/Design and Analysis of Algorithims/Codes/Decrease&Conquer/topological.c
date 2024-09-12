#include <stdio.h>
#include <stdlib.h>

#define MAX 100

struct Graph {
    int vertices;
    int** matrix;
};

struct Graph* createGraph(int vertices) {
    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
    graph->vertices = vertices;

    graph->matrix = (int**)malloc(vertices * sizeof(int*));
    for (int i = 0; i < vertices; i++) {
        graph->matrix[i] = (int*)malloc(vertices * sizeof(int));
        for (int j = 0; j < vertices; j++) {
            graph->matrix[i][j] = 0;
        }
    }

    return graph;
}

void addEdge(struct Graph* graph, int src, int dest) {
    graph->matrix[src][dest] = 1;
}

void topologicalSort(struct Graph* graph) {
    int* inDegree = (int*)malloc(graph->vertices * sizeof(int));
    for (int i = 0; i < graph->vertices; i++) {
        inDegree[i] = 0;
        for (int j = 0; j < graph->vertices; j++) {
            inDegree[i] += graph->matrix[j][i];
        }
    }

    printf("Topological Sort: ");
    for (int count = 0; count < graph->vertices; count++) {
        int found = 0;
        for (int i = 0; i < graph->vertices; i++) {
            if (inDegree[i] == 0) {
                printf("%d ", i);
                inDegree[i] = -1;
                for (int j = 0; j < graph->vertices; j++) {
                    if (graph->matrix[i][j] == 1) {
                        inDegree[j]--;
                    }
                }
                found = 1;
                break;
            }
        }
        if (!found) {
            printf("\nGraph has a cycle! Topological sort not possible.");
            break;
        }
    }

    free(inDegree);
}

int main() {
    int vertices, edges;
    printf("Enter the number of vertices and edges: ");
    scanf("%d %d", &vertices, &edges);

    struct Graph* graph = createGraph(vertices);

    printf("Enter the edges (source destination):\n");
    for (int i = 0; i < edges; i++) {
        int src, dest;
        scanf("%d %d", &src, &dest);
        addEdge(graph, src, dest);
    }

    topologicalSort(graph);

    return 0;
}
