from priority_queue import PriorityQueue
from queue import Queue


def breath_first_search(graph, start, end):
    frontier = Queue()
    frontier.put(start)
    backtrack = {start: None}

    while not frontier.empty():
        curr = frontier.get()

        # Early stopping, if we hit the goal we don't have to search the rest of the graph
        if curr == end:
            break

        for next_node in graph.get_neighbors(curr):
            if next_node not in backtrack:
                frontier.put(next_node)
                backtrack[next_node] = curr
    # If we exhaust the frontier (the queue ran out without the break) without finding the goal it means it is unreach-able
    else:
        raise ValueError(f"Goal: {end} unreachable from Start: {start}")
    return backtrack


def dijkstra(graph, start, end):
    frontier = PriorityQueue()
    frontier.put((0, start))
    backtrack = {start: None}
    cost = {start: 0}

    while not frontier.empty():
        _, curr = frontier.get()

        # Early stopping, if we hit the goal we don't have to search the rest of the graph
        if curr == end:
            break

        for next_node in graph.get_neighbors(curr):
            new_cost = cost[curr] + graph.cost(curr, next_node)
            if next_node not in cost or new_cost < cost[next_node]:
                cost[next_node] = new_cost
                frontier.put((new_cost, next_node))
                backtrack[next_node] = curr
    # If we exhaust the frontier (the queue ran out without the break) without finding the goal it means it is unreach-able
    else:
        raise ValueError(f"Goal: {end} unreachable from Start: {start}")
    return backtrack


def manhattan_distance(start, end):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


def a_star(graph, start, end, heuristic):
    frontier = PriorityQueue()
    frontier.put((0, start))
    backtrack = {start: None}
    cost = {start: 0}

    while not frontier.empty():
        _, curr = frontier.get()

        # Early stopping, if we hit the goal we don't have to search the rest of the graph
        if curr == end:
            break

        for next_node in graph.get_neighbors(curr):
            new_cost = cost[curr] + graph.cost(curr, next_node)
            if next_node not in cost or new_cost < cost[next_node]:
                cost[next_node] = new_cost
                frontier.put((new_cost + heuristic(next_node, end), next_node))
                backtrack[next_node] = curr
    # If we exhaust the frontier (the queue ran out without the break) without finding the goal it means it is unreach-able
    else:
        raise ValueError(f"Goal: {end} unreachable from Start: {start}")
    return backtrack

def reconstruct_path(backtrack, end):
    curr = end
    path = []
    while backtrack[curr] is not None:
        path.append(curr)
        curr = backtrack[curr]
    path.append(curr)
    return path[::-1][1:-1]


def path_find(graph, start, end, algorithm="a_start"):
    if algorithm == "bfs":
        backs = breath_first_search(graph, start, end)
    elif algorithm == "dijkstra":
        backs = dijkstra(graph, start, end)
    else:
        backs = a_star(graph, start, end, manhattan_distance)
    return reconstruct_path(backs, end)
