"""Vector store factory.

One store per backend name, built once per process. Both implementations hold expensive
state -- an HNSW graph in RAM, or an HTTP connection pool and a warmed collection handle --
so constructing one per request would put index load or a TCP handshake inside the 200ms
budget. The benchmark asks for several names on purpose (it compares `qdrant` against
`hnsw`), which is why the cache is keyed by name rather than being a single global.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from voxrag.config import settings
from voxrag.contracts import VectorStore

if TYPE_CHECKING:  # imported lazily at runtime so an unused backend costs nothing
    from voxrag.store.hnsw_store import HnswStore
    from voxrag.store.qdrant_store import QdrantStore

_LOG = logging.getLogger(__name__)

_CACHE: dict[str, VectorStore] = {}
_LOCK = threading.Lock()

#: Accepted aliases -> canonical backend. "qdrant_local" forces the embedded backend even
#: when `qdrant_url` is set, which is how the benchmark contrasts brute-force against ANN.
_ALIASES = {
    "qdrant": "qdrant",
    "qdrant_server": "qdrant",
    "qdrant_local": "qdrant_local",
    "hnsw": "hnsw",
    "hnswlib": "hnsw",
}


def get_vector_store(name: str | None = None) -> VectorStore:
    """Return the process-wide store for `name`, defaulting to `settings.vector_store`."""
    key = _ALIASES.get((name or settings.vector_store).strip().lower())
    if key is None:
        raise ValueError(
            f"unknown vector store {name or settings.vector_store!r}; "
            f"expected one of {sorted(_ALIASES)}"
        )
    store = _CACHE.get(key)
    if store is not None:
        return store
    # Double-checked under the lock: two concurrent first requests must not each build an
    # index, and the loser's HnswStore would hold a second copy of the graph in memory.
    with _LOCK:
        store = _CACHE.get(key)
        if store is None:
            store = _build(key)
            _CACHE[key] = store
    return store


def _build(key: str) -> VectorStore:
    if key == "hnsw":
        from voxrag.store.hnsw_store import HnswStore as _HnswStore

        return _HnswStore()

    try:
        from voxrag.store.qdrant_store import QdrantStore as _QdrantStore
    except ImportError as exc:
        # A missing qdrant-client should degrade to the in-process index rather than take
        # the whole service down; retrieval still works, only the "real vector DB" arm is off.
        _LOG.error("qdrant-client unavailable (%s); falling back to the in-process hnsw store", exc)
        from voxrag.store.hnsw_store import HnswStore as _HnswStore

        return _HnswStore()

    return _QdrantStore(local=key == "qdrant_local")


def __getattr__(name: str) -> Any:
    """Lazy re-export, so `from voxrag.store import HnswStore` works without importing
    qdrant-client (or vice versa) for callers that only need one backend."""
    if name == "HnswStore":
        from voxrag.store.hnsw_store import HnswStore as value

        return value
    if name == "QdrantStore":
        from voxrag.store.qdrant_store import QdrantStore as value

        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["HnswStore", "QdrantStore", "get_vector_store"]
