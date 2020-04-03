from queue import Queue
from copy import deepcopy
from collections import defaultdict
from itertools import combinations, chain
from .priority_queue import PriorityQueue
from .graph import AdjacencyGraph


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
    return backtrack, None


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
    return backtrack, cost[end]


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
    return backtrack, cost[end]


def reconstruct_path(backtrack, end):
    curr = end
    path = []
    while backtrack[curr] is not None:
        path.append(curr)
        curr = backtrack[curr]
    path.append(curr)
    return path[::-1][1:-1]


def path_find(graph, start, end, algorithm="a-star"):
    if algorithm == "bfs":
        backs, cost = breath_first_search(graph, start, end)
    elif algorithm == "dijkstra":
        backs, cost = dijkstra(graph, start, end)
    else:
        backs, cost = a_star(graph, start, end, manhattan_distance)
    return reconstruct_path(backs, end), cost


def fuel_path_find(capacity, graph, start, end, stations, algorithm="a-star"):
    edges = []
    paths = defaultdict(lambda: defaultdict(dict))

    # Find paths to all stations from the start, from all stations to the end, and between all stations
    for src, tgt in combinations(chain([start, end], stations), 2):
        path, cost = path_find(graph, src, tgt, algorithm)
        cost = cost if cost is not None else len(path) + 1
        # The returned paths don't have the start/end so there is an extra moves to make
        if cost <= capacity:
            edges.append((src, tgt, cost))
            # Track the actual paths from point to point to reconstruct later
            paths[src][tgt]['path'] = path
            paths[tgt][src]['path'] = path
            paths[src][tgt]['cost'] = cost
            paths[tgt][src]['cost'] = cost

    meta_graph = AdjacencyGraph(edges)
    meta_path, meta_cost = path_find(meta_graph, start, end, algorithm)

    # Reconstruct the real path
    meta_path = list(chain([start], meta_path, [end]))
    path = []
    cost = 0
    for i in range(1, len(meta_path)):
        path.extend(deepcopy(paths[meta_path[i - 1]][meta_path[i]]['path']))
        cost += paths[meta_path[i - 1]][meta_path[i]]['cost']
    return path, cost
