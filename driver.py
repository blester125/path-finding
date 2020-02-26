import sys
import argparse
from graph import Graph
from path import path_find


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--start_symbol", default="R")
    parser.add_argument("--end_symbol", default="G")
    parser.add_argument("--algorithm", choices=("bfs", "dijkstra", "a-star"), default="a-start")
    args = parser.parse_args()

    graph = Graph.from_file(args.data)
    start = graph.find(args.start_symbol)
    end = graph.find(args.end_symbol)

    try:
        path = path_find(graph, start, end, args.algorithm)
    except ValueError as e:
        print(e)
        sys.exit(1)

    graph.print_path(path)


if __name__ == "__main__":
    main()
