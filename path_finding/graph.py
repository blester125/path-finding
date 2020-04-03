import sys
import math
from copy import deepcopy
from collections import defaultdict
from typing import TextIO, Union, Dict, List, Tuple
from file_or_name import file_or_name


GRID_MAP = {
    "#": 0,
    ".": 1,
    "R": 3,
    "G": 4,
    "O": 5,
    "+": 6,
}


@file_or_name
def from_file(f, grid_mapping: Dict[str, int]) -> List[List[int]]:
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
def from_fuel_file(f, grid_mapping: Dict[str, int]) -> Tuple[int, List[List[int]]]:
    line = f.readline().rstrip("\n")
    try:
        capacity = int(line)
    except ValueError:
        raise ValueError("File is malformatted, first line was not an int, got {line}")
    grid = from_file(f, grid_mapping)
    return capacity, grid


class Graph:
    def get_neighbors(self, loc) -> List:
        pass

    def cost(self, start, end) -> int:
        pass


class AdjacencyGraph(Graph):
    def __init__(self, edges: List[Tuple[Tuple[int, int], Tuple[int, int], int]]):
        self.graph = defaultdict(dict)
        for edge in edges:
            src, tgt, weight = edge
            self.graph[src][tgt] = weight
            self.graph[tgt][src] = weight

    def get_neighbors(self, loc) -> List:
        return self.graph[loc].keys()

    def cost(self, start, end) -> int:
        return self.graph[start].get(end, sys.maxsize)


class GridGraph(Graph):
    def __init__(self, grid, grid_mapping):
        self.grid = grid
        self.grid_mapping = grid_mapping
        self.char_mapping = {v: k for k, v in grid_mapping.items()}

    @classmethod
    def from_file(cls, file_name, grid_mapping=GRID_MAP):
        grid = from_file(file_name, grid_mapping)
        return cls(grid, grid_mapping)

    def to_string(self, grid):
        return "\n".join("".join([self.char_mapping[c] for c in r]) for r in grid)

    def _add_path(self, path):
        grid = deepcopy(self.grid)
        for loc in path:
            grid[loc[0]][loc[1]] = self.grid_mapping["O"]
        return grid

    def print_path(self, path):
        grid = self._add_path(path)
        print(self.to_string(grid))

    def to_file_path(self, path, wf):
        grid = self._add_path(path)
        with open(wf, 'w') as wf:
            wf.write(self.to_string(grid) + "\n")

    def print_grid(self):
        print(self.to_string(self.grid))

    def to_file(self, wf):
        with open(wf, 'w') as wf:
            wf.write(self.to_string(self.grid) + "\n")

    def get_neighbors(self, loc: Tuple[int, int]):
        i, j = loc
        above = i - 1
        below = i + 1
        left = j - 1
        right = j + 1
        neighbors = []
        if above >= 0:
            if self.grid[above][j] != 0:
                neighbors.append((above, j))
            if left >= 0:
                if self.grid[above][left] != 0:
                    neighbors.append((above, left))
            if right < len(self.grid[above]):
                if self.grid[above][right] != 0:
                    neighbors.append((above, right))
        if below < len(self.grid):
            if self.grid[below][j] != 0:
                neighbors.append((below, j))
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

    def cost(self, start, end):
        # If our x or our y don't change then we did a move up, down, left, right
        if start[0] == end[0] or start[1] == end[1]:
            return 1
        else:
            return math.sqrt(2)

    def find(self, item: str):
        item = self.grid_mapping[item]
        for i, row in enumerate(self.grid):
            for j, col in enumerate(row):
                if col == item:
                    yield (i, j)


class FuelGridGraph(GridGraph):
    @classmethod
    def from_file(cls, file_name, grid_mapping=GRID_MAP):
        cap, grid = from_fuel_file(file_name, grid_mapping)
        return cap, cls(grid, grid_mapping)



def test_get_neighbors_stop_at_edges():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    graph = Graph(grid, {})
    assert (0, 1) in graph.get_neighbors((0, 0))
    assert (1, 0) in graph.get_neighbors((0, 0))
    assert len(graph.get_neighbors((0, 0))) == 2
    # TODO add tests

def test_get_neighbors_stops_at_walls():
    # TODO add tests
    pass


if __name__ == "__main__":
    test_get_neighbors_stop_at_edges()
    test_get_neighbors_stops_at_walls()
    print("Smoke tests past")
