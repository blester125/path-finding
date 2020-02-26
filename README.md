# Path Finding for Grids

This has the code for Brian Lester's Interview problem of finding a path through a grid world

The program can be run with the following.

 * `python driver.py "data/Programming Test A5 Data File.txt"`

There is also a command line argument (`--algorithm`) that lets you pick which algorithm to use to find the path. The supported ones are:

 * Breath First Search `--algorithm bfs`
 * Dijkstra's algorithm `--algorithm dijkstra`
 * A\* Search with a Manhattan Distance heuristic `--algorithm a-start`
