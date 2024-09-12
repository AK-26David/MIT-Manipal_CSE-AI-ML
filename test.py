import heapq
def ucs(G,start,goal):
    frontier=[(0,start)]
    prev={start:None}
    cost_so_far={start:0}
    while frontier:
        current_cost,current=heapq.heappop(frontier)
        if current==goal:
            break
        for next_node, weight in G[current]:
            new_cost=cost_so_far[current]+weight
            if next_node not in prev or new_cost<cost_so_far:
                cost_so_far[next_node]=new_cost
                heapq.heappush(frontier,(new_cost,next_node))
                prev[next_node]=current