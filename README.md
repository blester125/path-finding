# Path Finding

## Simple Grid Path Finding

This has the code for the problem of finding a path through a grid world

The program can be run with the following.

 * `python driver.py data/1.txt`

There is also a command line argument (`--algorithm`) that lets you pick which algorithm to use to find the path. The supported ones are:

 * Breath First Search `--algorithm bfs`
 * Dijkstra's algorithm `--algorithm dijkstra`
 * A\* Search with a Manhattan Distance heuristic `--algorithm a-star`

Data is assumed to be an ASCII grid with the following symbols:

 * `G` is the goal state
 * `.` is walk-able space
 * `#` is an impassable wall
 * `R` is the agent

## Grid Path Finding with Fuel Constraints

This has code for finding a path through a grid world where the agent can only move so many steps before refueling.

The program can be run with the following.

 * `python fuel_driver.py fuel_data/1.txt`

This uses the same algorithms to solve path finding. It does this by first path finding the agent to each station and
the goal state as well as between all the stations (and the goal points). Any path that takes too much full (is longer
than the agents capacity) is removed and a new meta graph is created that represents the distances between each point.
The path finding algorithm is then run to find the best path between these points and the legs in that path are
converted back to the actual low level paths.

The Data is assumed to be a file where the first row is the agent capacity and the following lines is the grid. There is
a new symbol added (`+`) representing the re-fueling stations.

## Graph Class

These search algorithms are built on a graph class that exposes 2 main methods. `.get_neighbors` and `cost`.
`.get_neighbors(node)` gives us all the neighbors for a node. This is used when expanding the frontier. The
method `.cost(src, dst)` gives the cost (or weight on the edge) between the `src` node and the `dst` node. This
is used for the meta graph search where the edges are different length paths. It could also be used in encode
terrain information like two nodes might be in the swap vs on pavement and would have a higher cost associated with them.

There are two main implementations here, one is a `GridGraph` which stores vertex information in a grid and the
other is a `AdjancenyGraph` which stores a collection of nodes, edges with wights. The `GridGraph` includes a
lot of extra methods for outputing ascii representations of the grid.
