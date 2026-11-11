"""Plugin registry: modules self-register by name; the API/UI discover them here."""
from typing import Any, Dict, List


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, type] = {}

    def register(self, name: str):
        def decorator(cls):
            cls.name = name
            self._items[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> type:
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {list(self._items)}")
        return self._items[name]

    def create(self, name: str, **params: Any):
        return self.get(name)(**params)

    def describe(self) -> List[dict]:
        """Metadata the frontend will use to build parameter forms."""
        return [
            {"name": n, "params": getattr(c, "default_params", {})}
            for n, c in self._items.items()
        ]


FEATURE_EXTRACTORS = Registry("feature extractor")
CLASSIFIERS = Registry("classifier")
CLUSTERERS = Registry("clusterer")
