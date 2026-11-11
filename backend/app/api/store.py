"""Thread-safe in-memory stores with a capacity limit (oldest items are evicted).
Everything is lost on restart. Swap this file for a database/Redis later without touching the routers."""
import threading
import uuid
from collections import OrderedDict
from typing import Any, List, Tuple

from fastapi import HTTPException


class Store:
    def __init__(self, kind: str, capacity: int):
        self.kind = kind
        self.capacity = capacity
        self._items: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def add(self, item: Any) -> str:
        item_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._items[item_id] = item
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return item_id

    def get(self, item_id: str) -> Any:
        with self._lock:
            if item_id not in self._items:
                raise HTTPException(status_code=404, detail=f"{self.kind} '{item_id}' not found")
            return self._items[item_id]

    def delete(self, item_id: str) -> None:
        with self._lock:
            if item_id not in self._items:
                raise HTTPException(status_code=404, detail=f"{self.kind} '{item_id}' not found")
            del self._items[item_id]

    def items(self) -> List[Tuple[str, Any]]:
        with self._lock:
            return list(self._items.items())

    def __len__(self) -> int:
        return len(self._items)


datasets = Store("dataset", capacity=50)
models = Store("model", capacity=100)
jobs = Store("job", capacity=200)
