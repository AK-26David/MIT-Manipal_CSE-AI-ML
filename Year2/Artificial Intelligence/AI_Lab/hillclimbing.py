def hill_climbing(matrix, start, end, heuristic):
    current = start
    while current!= end:
        neighbors = []
        for i in range(len(matrix)):
            if matrix[current][i]!= 0 and i not in visited:
                neighbors.append((i, matrix[current][i] + heuristic(i, end)))
        if not neighbors:
            return None
        next = min(neighbors, key=lambda x: x[1])
        visited.add(current)
        current = next[0]
    return current

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

def manhattan_distance(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

matrix = input_graph_adj_matrix()
print("Adjacency Matrix:")
for row in matrix:
    print(row)

visited = set()
print(hill_climbing(matrix, (0, 0), (len(matrix)-1, len(matrix)-1), manhattan_distance))