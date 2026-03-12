from __future__ import annotations
from collections import OrderedDict
import math, random
from typing import Any, List

def normalize_logweights(logw: List[float]) -> List[float]:
    if not logw:
        return []
    m = max(logw)
    xs = [math.exp(w - m) for w in logw]
    s = sum(xs)
    return [math.log(x / s) for x in xs]

def ess_from_logweights(logw: List[float]) -> float:
    ws = [math.exp(x) for x in normalize_logweights(logw)]
    d = sum(w*w for w in ws)
    return 1.0 / d if d > 0 else 0.0

def systematic_resample(logw: List[float], rng: random.Random) -> List[int]:
    ws = [math.exp(x) for x in normalize_logweights(logw)]
    n = len(ws)
    positions = [(rng.random() + i) / n for i in range(n)]
    indexes, cumulative, j = [], 0.0, 0
    for i, w in enumerate(ws):
        cumulative += w
        while j < n and positions[j] <= cumulative:
            indexes.append(i)
            j += 1
    while len(indexes) < n:
        indexes.append(n - 1)
    return indexes
class LRUCache:
    def __init__(self, maxsize: int = 256):
        self.maxsize = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()
    def get(self, key: str):
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]
    def set(self, key: str, value: Any):
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)
