import sys
import argparse
from path_finding.graph import FuelGridGraph
from path_finding.path import fuel_path_find


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--start_symbol", default="R")
    parser.add_argument("--end_symbol", default="G")
    parser.add_argument("--fuel_symbol", default="+")
    parser.add_argument("--algorithm", choices=("bfs", "dijkstra", "a-star"), default="a-start")
    args = parser.parse_args()

    capacity, graph = FuelGridGraph.from_file(args.data)
    start = next(graph.find(args.start_symbol))
    end = next(graph.find(args.end_symbol))
    stations = list(graph.find(args.fuel_symbol))

    try:
        path, cost = fuel_path_find(capacity, graph, start, end, stations, args.algorithm)
    except ValueError as e:
        print(e)
        sys.exit(1)

    print(f"Cost of path: {cost}")
    graph.print_path(path)


if __name__ == "__main__":
    main()
