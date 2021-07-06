import sys
import argparse
from path_finding.graph import GridGraph
from path_finding.path import path_find


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--start_symbol", default="R")
    parser.add_argument("--end_symbol", default="G")
    parser.add_argument("--algorithm", choices=("bfs", "dijkstra", "a-star"), default="a-star")
    parser.add_argument("--no-diagonal", action="store_true")
    args = parser.parse_args()

    graph = GridGraph.from_file(args.data, no_diagonal=args.no_diagonal)
    start = next(graph.find(args.start_symbol))
    end = next(graph.find(args.end_symbol))

    try:
        path, cost = path_find(graph, start, end, args.algorithm)
    except ValueError as e:
        print(e)
        sys.exit(1)

    print(f"Cost of path: {cost}")
    graph.print_path(path)


if __name__ == "__main__":
    main()
