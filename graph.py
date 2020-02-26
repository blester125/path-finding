from copy import deepcopy
from typing import TextIO, Union, Dict, List, Tuple
from file_or_name import file_or_name


GRID_MAP = {
    "#": 0,
    ".": 1,
    "R": 3,
    "G": 4,
    "O": 5
}


def from_file(file_name, grid_mapping: Dict[str, int]) -> List[List[int]]:
    grid = []
    with open(file_name) as f:
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


class Graph:
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
        if below < len(self.grid):
            if self.grid[below][j] != 0:
                neighbors.append((below, j))
        if left >= 0:
            if self.grid[i][left] != 0:
                neighbors.append((i, left))
        if right < len(self.grid[i]):
            if self.grid[i][right] != 0:
                neighbors.append((i, right))
        return neighbors

    def cost(self, start, end):
        # This could let us add movement costs in the future based on terrains
        return 1

    def find(self, item: str):
        item = self.grid_mapping[item]
        for i, row in enumerate(self.grid):
            for j, col in enumerate(row):
                if col == item:
                    return (i, j)
        raise IndexError(f"Item {item} not found in grid")


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
    print("Somke tests past")
