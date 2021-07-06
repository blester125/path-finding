import sys
import math
from copy import deepcopy
from collections import defaultdict
from typing import TextIO, Union, Dict, List, Tuple, Sequence
from file_or_name import file_or_name


IMPASSABLE = 0
GRID_MAP = {
    "#": IMPASSABLE,
    ".": 1,
    "R": 3,
    "G": 4,
    "O": 5,
    "+": 6,
}
Node = Tuple[int, int]
Cost = int
Grid = List[List[int]]
ASCIIGrid = List[List[str]]


@file_or_name
def from_file(f: Union[str, TextIO], grid_mapping: Dict[str, int]) -> Grid:
    grid = []
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        row = []
        for cell in line:
            # Default unknown cells to be walls
            row.append(grid_mapping.get(cell, 0))
        grid.append(row)
    if len(set(len(r) for r in grid)) != 1:
        raise ValueError("Found rows of different sizes, input is malformed")
    return grid


@file_or_name
def from_fuel_file(f: Union[str, TextIO], grid_mapping: Dict[str, int]) -> Tuple[int, Grid]:
    line = f.readline().rstrip("\n")
    try:
        capacity = int(line)
    except ValueError:
        raise ValueError("File is malformatted, first line was not an int, got {line}")
    grid = from_file(f, grid_mapping)
    return capacity, grid


class Graph:
    def get_neighbors(self, node: Node) -> List[Node]:
        pass

    def cost(self, start: Node, end: Node) -> Cost:
        pass


class AdjacencyGraph(Graph):
    def __init__(self, edges: List[Tuple[Node, Node, Cost]]):
        super().__init__()
        self.graph = defaultdict(dict)
        for edge in edges:
            src, tgt, weight = edge
            self.graph[src][tgt] = weight
            self.graph[tgt][src] = weight

    def get_neighbors(self, node: Node) -> List[Node]:
        return self.graph[node].keys()

    def cost(self, start: Node, end: Node) -> Cost:
        return self.graph[start].get(end, sys.maxsize)


class GridGraph(Graph):
    def __init__(self, grid: Grid, grid_mapping: Dict[str, int], no_diagonal: bool = False):
        super().__init__()
        self.grid = grid
        self.grid_mapping = grid_mapping
        self.char_mapping = {v: k for k, v in grid_mapping.items()}
        self.no_diagonal = no_diagonal

    @classmethod
    def from_file(cls, file_name: Union[str, TextIO], *, grid_mapping: Dict[str, int] = GRID_MAP, **kwargs):
        grid = from_file(file_name, grid_mapping)
        return cls(grid, grid_mapping, **kwargs)

    def to_string(self, grid: Grid) -> str:
        return "\n".join("".join([self.char_mapping[c] for c in r]) for r in grid)

    def _add_path(self, path: Sequence[Node]) -> Grid:
        grid = deepcopy(self.grid)
        for node in path:
            grid[node[0]][node[1]] = self.grid_mapping["O"]
        return grid

    def print_path(self, path: Sequence[Node]):
        grid = self._add_path(path)
        print(self.to_string(grid))

    def to_file_path(self, path: Sequence[Node], wf: str):
        grid = self._add_path(path)
        with open(wf, 'w') as wf:
            wf.write(self.to_string(grid) + "\n")

    def print_grid(self):
        print(self.to_string(self.grid))

    def to_file(self, wf: str):
        with open(wf, 'w') as wf:
            wf.write(self.to_string(self.grid) + "\n")

    def get_neighbors(self, node: Node) -> List[Node]:
        i, j = node
        above = i - 1
        below = i + 1
        left = j - 1
        right = j + 1
        neighbors = []
        if above >= 0:
            if self.grid[above][j] != 0:
                neighbors.append((above, j))
            if not self.no_diagonal:
                if left >= 0:
                    if self.grid[above][left] != 0:
                        neighbors.append((above, left))
                if right < len(self.grid[above]):
                    if self.grid[above][right] != 0:
                        neighbors.append((above, right))
        if below < len(self.grid):
            if self.grid[below][j] != 0:
                neighbors.append((below, j))
            if not self.no_diagonal:
                if left >= 0:
                    if self.grid[below][left] != 0:
                        neighbors.append((below, left))
                if right < len(self.grid[below]):
                    if self.grid[below][right] != 0:
                        neighbors.append((below, right))
        if left >= 0:
            if self.grid[i][left] != 0:
                neighbors.append((i, left))
        if right < len(self.grid[i]):
            if self.grid[i][right] != 0:
                neighbors.append((i, right))
        return neighbors

    def cost(self, start: Node, end: Node) -> Cost:
        # If we disallow diagonal movement, or our x or our y don't change then
        # we did a move up, down, left, right and our cost is 1
        if self.no_diagonal or start[0] == end[0] or start[1] == end[1]:
            return 1
        # Any diagonal move is √2 because a diagonal move is the hypotenuse, and
        # c² = a² + b and a = b = 1, so c² = 2 and c = √2
        else:
            return math.sqrt(2)

    def find(self, item: str) -> Node:
        """Given a symbol, find the first instance in the graph."""
        item = self.grid_mapping[item]
        for i, row in enumerate(self.grid):
            for j, col in enumerate(row):
                if col == item:
                    yield (i, j)


class FuelGridGraph(GridGraph):
    @classmethod
    def from_file(
            cls,
            file_name: Union[str, TextIO],
            *,
            grid_mapping: Dict[str, int] = GRID_MAP,
            **kwargs
    ):
        cap, grid = from_fuel_file(file_name, grid_mapping)
        return cap, cls(grid, grid_mapping, **kwargs)


def test_get_neighbors_stop_at_edges():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    graph = GridGraph(grid, {}, no_diagonal=False)
    assert (0, 1) in graph.get_neighbors((0, 0))
    assert (1, 0) in graph.get_neighbors((0, 0))
    assert (1, 1) in graph.get_neighbors((0, 0))
    assert len(graph.get_neighbors((0, 0))) == 3


def test_get_neighbors_stop_at_edges_no_diagonal():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    graph = GridGraph(grid, {}, no_diagonal=True)
    assert (0, 1) in graph.get_neighbors((0, 0))
    assert (1, 0) in graph.get_neighbors((0, 0))
    assert len(graph.get_neighbors((0, 0))) == 2

def test_get_neighbors_stops_at_walls():
    # TODO add tests
    pass


if __name__ == "__main__":
    test_get_neighbors_stop_at_edges()
    test_get_neighbors_stop_at_edges_no_diagonal()
    test_get_neighbors_stops_at_walls()
    print("Smoke Tests Pass!")
