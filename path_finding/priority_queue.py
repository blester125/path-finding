import heapq
from typing import Optional, Sequence, Callable, Any


class PriorityQueue:
    def __init__(self,
                 data: Optional[Sequence[Any]] = None,
                 key: Optional[Callable[[Any], float]]= None
    ):
        """Priority queue as a class.

        Args:
          data: Initial data in the queue.
          key: A function that calculates the priority value for items in the
            queue. If not provided the priority of each items will just depend
            on the comparison or items.
        """
        self.heap = list(data) if data is not None else []
        if key:
            self.heap = [(key(item), item) for item in self.heap]
        heapq.heapify(self.heap)
        self.key = key

    def put(self, item: Any):
        """Add item to the queue."""
        if self.key:
            item = (self.key(item), item)
        heapq.heappush(self.heap, item)

    def get(self) -> Any:
        """Get the first item from the queue."""
        item = heapq.heappop(self.heap)
        if self.key:
            return item[1]
        return item

    def __len__(self):
        return len(self.heap)

    def empty(self):
        # `len(self)` will proxy to `__len__` which will check the length
        # of the `self.heap` list.
        return len(self) == 0


def test_heap():
    import random
    data = list(range(10))
    random.shuffle(data)
    max_heap = PriorityQueue(data, key=lambda x: -x)
    min_heap = PriorityQueue(data)
    assert min_heap.get() == 0
    assert max_heap.get() == 9
    max_heap.put(100)
    assert max_heap.get() == 100
    min_heap.put(-100)
    assert min_heap.get() == -100


def test_heap_property():
    """Test my heap by using it for heap sort."""
    import random
    data = [random.randint(1, 1000) for _ in range(random.randint(50, 100))]
    gold = sorted(data)
    h = PriorityQueue(data)
    res = []
    while len(h):
        res.append(h.get())
    assert res == gold

def test_keyed_data():
    index = list(range(10))
    data = list("ABCDEFGHI")
    pairs = [(i, d) for i, d in zip(index, data[::-1])]
    h = PriorityQueue(pairs)
    assert h.get() == (0, "I")

def test_empty():
    h = PriorityQueue()
    assert h.empty()
    h.put(None)
    assert not h.empty()


if __name__ == "__main__":
    test_heap()
    test_heap_property()
    test_keyed_data()
    test_empty()
    print("Smoke Tests Pass!")
