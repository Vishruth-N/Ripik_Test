"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import random
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import List, Any, Tuple, Callable, Optional, Union
from .constants import BatchSizeLinking


class RandomizedSet(object):
    def __init__(self) -> None:
        self._map = {}
        self._list = []

    def add(self, val: Any) -> None:
        if val not in self._map:
            self._list.append(val)
            self._map[val] = len(self._list) - 1

    def discard(self, val: Any) -> None:
        if val in self._map:
            n = len(self._list)
            i = self._map[val]
            if i != n - 1:
                tmp = self._list[n - 1]
                self._list[n - 1] = val
                self._list[i] = tmp
                self._map[tmp] = i
            del self._map[val]
            self._list.pop()

    def get_random(self) -> Any:
        return random.choice(self._list)

    def get_length(self) -> int:
        return len(self._list)

    def __iter__(self):
        return self._list.__iter__()

    def __contains__(self, val: Any):
        return val in self._map


class DSU:
    def __init__(self) -> None:
        self.parents = {}
        self.ranks = {}

    def exists(self, x):
        return x in self.parents

    def add(self, x):
        self.parents[x] = x
        self.ranks[x] = 1

    def find(self, x):
        if not self.exists(x):
            return x

        if self.parents[x] == x:
            return x
        self.parents[x] = self.find(self.parents[x])
        return self.parents[x]

    def union(self, x, y):
        if not self.exists(x):
            self.add(x)
        if not self.exists(y):
            self.add(y)

        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.ranks[x] < self.ranks[y]:
            x, y = y, x
        self.parents[y] = x
        if self.ranks[x] == self.ranks[y]:
            self.ranks[x] += 1


# Optional[Callable[[Any, float]]]
def merge_intervals(
    arr: List[Any], keyL=lambda x: x[0], keyR=lambda x: x[1]
) -> List[Tuple[float, float]]:
    """Merge intervals algorithm: O(n)"""
    if len(arr) == 0:
        return arr

    arr.sort(key=keyL)

    merged_intervals = []
    curr_L = keyL(arr[0])
    curr_R = keyR(arr[0])
    for interval in arr[1:]:
        start_time, end_time = keyL(interval), keyR(interval)

        if start_time <= curr_R:
            curr_R = max(curr_R, end_time)
        else:
            merged_intervals.append([curr_L, curr_R])
            curr_L = start_time
            curr_R = end_time

    merged_intervals.append([curr_L, curr_R])
    return merged_intervals


def close_subtract(a: float, b: float) -> float:
    """Subtract b from a"""
    return 0 if np.isclose(a, b) else a - b


def get_num_hours(start: Union[str, datetime], end: Union[str, datetime]):
    """Calculate num hours between end and start"""
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")

    assert end >= start
    secs = (end - start).total_seconds()
    return np.round(secs / 3600, 1)


def multidict(k: int, t: Callable[[], Any]):
    assert k > 0

    def _multidict(k: int):
        if k == 1:
            return defaultdict(t)
        return defaultdict(lambda: _multidict(k - 1))

    return _multidict(k)


def compare_linking_op(available: float, needed: float, mode: int) -> bool:
    diff = close_subtract(available, needed)

    if mode == BatchSizeLinking.GT.value and diff > 0:
        return True
    elif mode == BatchSizeLinking.GE.value and diff >= 0:
        return True
    elif mode == BatchSizeLinking.EQ.value and diff == 0:
        return True
    elif mode == BatchSizeLinking.LE.value and diff <= 0:
        return True
    elif mode == BatchSizeLinking.LT.value and diff < 0:
        return True
    elif mode == BatchSizeLinking.NA.value:
        return True

    return False
