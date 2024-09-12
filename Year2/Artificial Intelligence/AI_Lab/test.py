import heapq

def create_adj_matrix(size):
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    return matrix

def input_matrix():
    n= int(input("Enter the number of vertices:"))
    matrix=create_adj_matrix(n)
    for i in range(n):
        print(f"Enter the edge input for vertex {i}")
        edge_input=list(map(int , input().split()))
        for j in range(0, len(edge_input), 2):
            if j+1<len(edge_input):
                matrix[i][edge_input[j]]=1
                matrix[edge_input[j]][i]=1
    return matrix
        

def printMatrix():
    matrix=input_matrix()
    print("The adjacency matrix is :")
    for row in range(matrix):
        print(row)
        
def UCS(matrix,start,end):
    queue=[(0,start)]
    visited=set()
    while queue:
        (cost,vertex)=queue.pop(0)
        if vertex==0:
            return cost
        if vertex in visited:
            continue
        visited.add(vertex)
        for i in range(len(matrix)):
            if matrix[visited][i]==0 and i not in visited:
                queue.append(matrix[visited][i],i)
                queue.sort(key=lambda x:x[0])
    return None

def astar(matrix,start,end,heuristic):
    queue=[heuristic(start,end),(0,start)]
    visited=set()
    while queue:
        (_,cost,vertex)=heapq.heappop(queue)
        if vertex==0:
            return cost
        if vertex in visited:
            continue
        visited.add(vertex)
        for i in range(len(matrix)):
            if matrix[visited][i]==0 and i not in visited:
                heapq.heappush(queue,(cost+matrix[visited][i]+heuristic(i,end)+cost+matrix[visited][i]),i)
    return None


def bestfirst(matrix,start,end,heuristic):
    queue=[heuristic(start,end),(0,start)]
    visited=set()
    while queue:
        (_,vertex)=heapq.heappop(queue)
        if vertex==0:
            return True
        if vertex in visited:
            continue
        visited.add(vertex)
        for i in range(len(matrix)):
            if matrix[visited][i]==0 and i not in visited:
                heapq.heappush(queue,heuristic(i,end),i)
    return None


