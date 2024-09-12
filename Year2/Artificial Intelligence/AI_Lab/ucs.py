def ucs(matrix, start, end):
    queue = [(0, start)]
    visited = set()
    while queue:
        (cost, vertex) = queue.pop(0)
        if vertex == end:
            return cost
        if vertex in visited:
            continue
        visited.add(vertex)
        for i in range(len(matrix)):
            if matrix[vertex][i]!= 0 and i not in visited:
                queue.append((cost + matrix[vertex][i], i))
                queue.sort(key=lambda x: x[0])
    return None

def create_adj_matrix(size):
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    return matrix

def input_graph_adj_matrix():
    n = int(input("Enter the number of vertices: "))
    matrix = create_adj_matrix(n)

    for i in range(n):
        print(f"Enter edges for vertex {i}:")
        edge_input = list(map(int, input().split()))
        for j in range(0, len(edge_input), 2):
            if j+1 < len(edge_input):
                matrix[i][edge_input[j]] = edge_input[j+1]
                matrix[edge_input[j]][i] = edge_input[j+1]

    return matrix

matrix = input_graph_adj_matrix()
print("Adjacency Matrix:")
for row in matrix:
    print(row)

print(ucs(matrix, 0, len(matrix)-1))