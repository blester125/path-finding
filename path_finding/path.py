import math
from typing import Tuple, Dict, Optional, Callable, List, Sequence
from queue import Queue
from copy import deepcopy
from collections import defaultdict
from itertools import combinations, chain
from .priority_queue import PriorityQueue
from .graph import AdjacencyGraph, Graph, Node, Cost

# Back-track is the object used to remember the path we took to some node. It is
# a mapping from the current node to the last node we are at. When we reach the
# starting node, the value will be `None`
BackTrack = Dict[Node, Optional[Node]]

def breath_first_search(
        graph: Graph,
        start: Node,
        end: Node
) -> Tuple[BackTrack, Optional[Cost]]:
    """Find the goal via simple breath-first-search, no costs considered."""
    frontier = Queue()
    frontier.put(start)
    backtrack = {start: None}

    while not frontier.empty():
        # Get a node from the frontier of possible next nodes.
        curr = frontier.get()

        # Early stopping, if we hit the goal we don't have to search the rest of
        # the graph
        if curr == end:
            break

        # Expand our frontier by looking at the neighbors.
        for next_node in graph.get_neighbors(curr):
            # If we have already expanded a node (we have decided how to get to
            # it and it is thus in our backtrack) we skip adding it to the
            # frontier
            if next_node not in backtrack:
                # Add the node to the frontier so it will be expanded.
                frontier.put(next_node)
                # Record that the way we get to that next node is through this
                # one.
                backtrack[next_node] = curr
    # If we exhaust the frontier (the queue ran out of nodes without hitting the
    # break from reaching the end) without finding the goal, it means the goal
    # is unreachable
    else:
        raise ValueError(f"Goal: {end} unreachable from Start: {start}")
    return backtrack, None


def dijkstra(
        graph: Graph,
        start: Node,
        end: Node
) -> Tuple[BackTrack, Cost]:
    """Find the shortest path to the goal via dijkstra's algorithm."""
    # Our frontier is a priority queue that will hold (Cost, Node) pairs, The
    # priority is based on the Cost. This means that low cost nodes will be
    # expanded first as they are the most likely to be in the shortest path to
    # the goal.
    frontier = PriorityQueue()
    frontier.put((0, start))
    backtrack = {start: None}
    cost = {start: 0}

    while not frontier.empty():
        # Get a node from the frontier of possible next nodes.
        _, curr = frontier.get()

        # Early stopping, if we hit the goal we don't have to search the rest of
        # the graph
        if curr == end:
            break

        # Expand our frontier by looking at the neighbors.
        for next_node in graph.get_neighbors(curr):
            # Calculate the cost of getting to the next node. This is a dynamic
            # programming algorithm so our cost to the next node is the cost to
            # this node plus the cost from this node to the next.
            new_cost = cost[curr] + graph.cost(curr, next_node)
            # If we haven't expanded this node yet, or we found a cheaper path
            # to this node, add it to the frontier so it can be expanded.
            if next_node not in cost or new_cost < cost[next_node]:
                # Record the total cost to this node.
                cost[next_node] = new_cost
                # Add it to the frontier, with the associated cost, so that
                # lower cost nodes will be expanded first.
                frontier.put((new_cost, next_node))
                # Record that the way we got to this next node, with this cost,
                # is through this node.
                backtrack[next_node] = curr
    # If we exhaust the frontier (the queue ran out of nodes without hitting the
    # break from reaching the end) without finding the goal, it means the goal
    # is unreachable
    else:
        raise ValueError(f"Goal: {end} unreachable from Start: {start}")
    return backtrack, cost[end]


def manhattan_distance(start: Node, end: Node) -> Cost:
    """Find the manhattan (cab-driver) distance between two nodes."""
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


def euclidean_distance(start: Node, end: Node) -> Cost:
    """Fine the straight line distance between two nodes."""
    return math.sqrt(sum((s_i - e_i) ** 2 for s_i, e_i in zip(start, end)))


def a_star(
        graph: Graph,
        start: Node,
        end: Node,
        heuristic: Callable[[Node, Node], Cost] = euclidean_distance
) -> Tuple[BackTrack, Cost]:
    """Find the shortest path with the a-star algorithm.

    Args:
      graph: The graph we searching for paths in.
      start: Where our path starts.
      end: Where our path ends.
      heuristic: A function that estimates the cheapest path between nodes.
        For this heuristic to be admissible, it should never overestimate the
        distance.

    Returns:
      The backtracking table and the cost of the shortest path.
    """
    # Our frontier is a priority queue that will hold (Cost, Node) pairs, The
    # priority is based on the Cost. This means that low cost nodes will be
    # expanded first as they are the most likely to be in the shortest path to
    # the goal. The Cost now includes the heuristic cost, this should speed
    # things by expanding nodes that are more likely to be in the shortest path
    # earlier.
    frontier = PriorityQueue()
    frontier.put((0, start))
    backtrack = {start: None}
    cost = {start: 0}

    while not frontier.empty():
        # Get a node from the frontier of possible next nodes.
        _, curr = frontier.get()

        # Early stopping, if we hit the goal we don't have to search the rest of
        # the graph
        if curr == end:
            break

        # Expand our frontier by looking at the neighbors.
        for next_node in graph.get_neighbors(curr):
            # Calculate the cost of getting to the next node. This is a dynamic
            # programming algorithm so our cost to the next node is the cost to
            # this node plus the cost from this node to the next.
            new_cost = cost[curr] + graph.cost(curr, next_node)
            # If we haven't expanded this node yet, or we found a cheaper path
            # to this node, add it to the frontier so it can be expanded.
            if next_node not in cost or new_cost < cost[next_node]:
                # Record the total cost to this node.
                cost[next_node] = new_cost
                # Add it to the frontier, with the associated cost, and the
                # heuristic cost so that lower cost nodes will be expanded
                # first.
                frontier.put((new_cost + heuristic(next_node, end), next_node))
                # Record that the way we got to this next node, with this cost,
                # is through this node.
                backtrack[next_node] = curr
    # If we exhaust the frontier (the queue ran out of nodes without hitting the
    # break from reaching the end) without finding the goal, it means the goal
    # is unreachable
    else:
        raise ValueError(f"Goal: {end} unreachable from Start: {start}")
    return backtrack, cost[end]


def reconstruct_path(backtrack: BackTrack, end: Node) -> List[Node]:
    """Create a path from the back-track data structure."""
    curr = end
    path = []
    # While we haven't hit the starting node (which has backtrack value o
    # `None`).
    while backtrack[curr] is not None:
        # Add the current node to the path.
        path.append(curr)
        # Move to the node we moved to this node from.
        curr = backtrack[curr]
    # Add the final (starting) node.
    path.append(curr)
    # Reverse the path and remove the start and end nodes.
    return path[::-1][1:-1]


def path_find(
        graph: Graph,
        start: Node,
        end: Node,
        algorithm: str = "a-star"
) -> Tuple[List[Node], Cost]:
    """Find a path from start to end."""
    if algorithm == "bfs":
        backs, cost = breath_first_search(graph, start, end)
    elif algorithm == "dijkstra":
        backs, cost = dijkstra(graph, start, end)
    else:
        backs, cost = a_star(graph, start, end, euclidean_distance)
    # Convert the back-track into an actual path.
    return reconstruct_path(backs, end), cost


def fuel_path_find(
        capacity: int,
        graph: Graph,
        start: Node,
        end: Node,
        stations: Sequence[Node],
        algorithm: str = "a-star"
) -> Tuple[List[Node], Cost]:
    """Find a path from start to end with constraints on how far the agent can move.

    """
    # Create a meta graph where the start, end, and each station in a node,
    # and the edges are weighted based on the fuel used to travel between the
    # nodes. Any edge that requires more fuel than the capacity is removed.
    edges = []
    paths = defaultdict(lambda: defaultdict(dict))

    # Find paths to all stations from the start, from all stations to the end,
    # and between all stations.
    for src, tgt in combinations(chain([start, end], stations), 2):
        path, cost = path_find(graph, src, tgt, algorithm)
        # If we don't have a cost (for example, we used bfs) the cost is just
        # the path length, plus 1 move because the paths don't include the end
        # node.
        cost = cost if cost is not None else len(path) + 1
        # If we have enough fuel to make it along that edges.
        if cost <= capacity:
            # Include src->tgt with cost=cost for the meta graph.
            edges.append((src, tgt, cost))
            # Save the actual path (and cost) used to get from src to tgt so
            # that once we pick a path through the meta graph, which just
            # operates on nodes like src and tgt, we can reconstruct the actual
            # path between nodes.
            paths[src][tgt]['path'] = path
            paths[tgt][src]['path'] = path[::-1]
            paths[src][tgt]['cost'] = cost
            paths[tgt][src]['cost'] = cost

    # Create the metagragh which only includes valid edges.
    meta_graph = AdjacencyGraph(edges)
    # Find a path through the metagraph, this will tell us which
    # stations to path through.
    meta_path, meta_cost = path_find(meta_graph, start, end, algorithm)

    # Reconstruct the real path from the meta path.
    # Add the start and end paths for paths that we normally ignore.
    meta_path = list(chain([start], meta_path, [end]))
    path = []
    cost = 0
    for i in range(1, len(meta_path)):
        # Add the real path from `node - 1` -> `node` to the real path.
        # Note: The real path does not have the start and end (`node - 1` and
        # `node` respectively) in it. This means when we print the path symbols
        # like start, end, and stations will not be overwritten.
        path.extend(deepcopy(paths[meta_path[i - 1]][meta_path[i]]['path']))
        # Record the cost.
        cost += paths[meta_path[i - 1]][meta_path[i]]['cost']
    return path, cost


def test_manhattan_distance():
    start = [12, 56]
    end = [6, 50]
    assert manhattan_distance(start, end) == 12


def test_euclidean_distance():
    start = [0, 0]
    end = [3, 4]
    assert euclidean_distance(start, end) == 5


if __name__ == "__main__":
  test_manhattan_distance()
  test_euclidean_distance()
  print("Smoke Tests Pass!")
