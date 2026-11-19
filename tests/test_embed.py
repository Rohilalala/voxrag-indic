r"""The embedding layer: the e5 prefixes, and the cache key that must include `is_query`.

Every failure mode should have exactly one owning test, and two of them belong here —
"wrong e5 prefix" and "cache returning another query's vector". Neither had an owner: there was no embedding test file at all, and a skeptic audit proved it by mutating
`onnx_encoder.py` two ways and watching the whole suite stay green:

    LRU ignores is_query (a query vector served for a passage)   602 passed   SURVIVED
    prefix always "query: " (passages embedded as queries)       602 passed   SURVIVED

Both are silent in production. e5 is trained with asymmetric prefixes, so embedding a passage
as a query moves it to a different region of the space and costs recall with nothing in the
logs, about 4 points of nDCG. `conftest.FakeEmbedder` cannot catch either, by its
own docstring: it records `is_query` and deliberately does not let it change the vector.

So these tests use the REAL `OnnxEncoder` code paths with the model swapped out. `_forward` is
the single seam where text becomes floats, so stubbing it leaves prefixing, batching, dedupe
and cache keying all under test while loading no ONNX graph — a unit test that downloads a
118M-parameter model is an integration test wearing a disguise.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from voxrag.config import settings
from voxrag.embed.onnx_encoder import OnnxEncoder, _LruCache


class TestLruCacheKeying:
    """The cache key is `(text, is_query)`. Dropping the flag is a silent correctness bug."""

    def test_the_same_text_caches_separately_as_query_and_as_passage(self) -> None:
        # THE mutation this file exists for. e5 embeds "query: X" and "passage: X" to different
        # points, so a cache keyed on text alone serves whichever arrived first to both callers.
        cache = _LruCache(capacity=8)
        as_query = np.array([1.0, 0.0], dtype=np.float32)
        as_passage = np.array([0.0, 1.0], dtype=np.float32)
        cache.put(("mumbai", True), as_query)
        cache.put(("mumbai", False), as_passage)

        assert np.array_equal(cache.get(("mumbai", True)), as_query)
        assert np.array_equal(cache.get(("mumbai", False)), as_passage)
        assert cache.stats()["size"] == 2, "one entry means the flag was dropped from the key"

    def test_a_miss_is_none_rather_than_a_neighbouring_vector(self) -> None:
        cache = _LruCache(capacity=4)
        cache.put(("indexed", True), np.array([1.0], dtype=np.float32))
        assert cache.get(("never seen", True)) is None
        assert cache.get(("indexed", False)) is None


class TestLruCacheMechanics:
    """Capacity, recency and the counters the analytics endpoint reports."""

    def test_evicts_least_recently_used_and_counts_it(self) -> None:
        cache = _LruCache(capacity=2)
        for i, text in enumerate(("a", "b")):
            cache.put((text, True), np.array([float(i)], dtype=np.float32))
        cache.get(("a", True))                    # a is now the most recent
        cache.put(("c", True), np.array([2.0], dtype=np.float32))

        assert cache.get(("b", True)) is None, "b was least recently used and should be gone"
        assert cache.get(("a", True)) is not None
        assert cache.stats()["evictions"] == 1

    def test_hits_and_misses_are_counted_and_uncounted_probes_are_not(self) -> None:
        cache = _LruCache(capacity=4)
        cache.put(("x", True), np.array([1.0], dtype=np.float32))
        cache.get(("x", True))
        cache.get(("absent", True))
        before = cache.stats()
        cache.get(("x", True), count=False)        # the async fast path's probe
        after = cache.stats()

        assert (before["hits"], before["misses"]) == (1, 1)
        assert (after["hits"], after["misses"]) == (1, 1), "an uncounted probe must not book a hit"

    def test_capacity_zero_disables_the_cache_without_raising(self) -> None:
        cache = _LruCache(capacity=0)
        cache.put(("x", True), np.array([1.0], dtype=np.float32))
        assert cache.get(("x", True)) is None
        assert cache.stats()["size"] == 0


class _StubEncoder(OnnxEncoder):
    """The real encoder with the ONNX graph removed.

    Built with `__new__` so no model is resolved or downloaded. `_forward` records exactly what
    it was handed — which is the point: these tests are about the strings that reach the model,
    not the floats that come back.
    """

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.seen: list[list[str]] = []
        self._n_query_encodes = 0
        self._n_passage_encodes = 0

    def _forward(self, prefixed: list[str]) -> np.ndarray:
        self.seen.append(list(prefixed))
        # Deterministic unit vectors keyed on the PREFIXED text, so a wrong prefix produces a
        # different vector and any assertion downstream of it moves.
        out = np.zeros((len(prefixed), self.dim), dtype=np.float32)
        for i, text in enumerate(prefixed):
            # blake2b, not hash(): PYTHONHASHSEED is randomised per process, so builtin hash
            # makes "query: X" and "passage: X" collide on some runs and not others -- a test
            # that fails once in a while and passes on re-run, which is the worst kind. This
            # file was written because a flaky-looking suite turned out to be real; it must not
            # add a genuine flake of its own.
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
            out[i] = np.frombuffer(digest, dtype=np.uint8)[: self.dim].astype(np.float32) + 1.0
        return out


def _encoder() -> _StubEncoder:
    return _StubEncoder.__new__(_StubEncoder)  # type: ignore[return-value]


@pytest.fixture
def encoder() -> _StubEncoder:
    enc = _encoder()
    _StubEncoder.__init__(enc)
    return enc


class TestE5Prefixes:
    """e5 is asymmetric: the prefix decides which half of the space the text lands in."""

    def test_queries_get_the_query_prefix_and_passages_the_passage_prefix(
        self, encoder: _StubEncoder
    ) -> None:
        encoder._encode_uncached(["what is a corporation"], is_query=True)
        encoder._encode_uncached(["A corporation is a legal entity."], is_query=False)

        assert encoder.seen[0] == [f"{settings.embed_query_prefix}what is a corporation"]
        assert encoder.seen[1] == [f"{settings.embed_passage_prefix}A corporation is a legal entity."]

    def test_the_two_prefixes_are_actually_different(self) -> None:
        # Guards the config as well as the code: setting both to the same string would make
        # every assertion above pass while silently discarding e5's asymmetry.
        assert settings.embed_query_prefix != settings.embed_passage_prefix

    def test_an_already_prefixed_text_is_not_prefixed_twice(self, encoder: _StubEncoder) -> None:
        # "query: query: X" is a different, worse point in embedding space.
        already = f"{settings.embed_query_prefix}what is a corporation"
        encoder._encode_uncached([already], is_query=True)
        assert encoder.seen[0] == [already]

    def test_query_and_passage_encodes_are_counted_separately(self, encoder: _StubEncoder) -> None:
        encoder._encode_uncached(["a", "b"], is_query=True)
        encoder._encode_uncached(["c"], is_query=False)
        assert (encoder._n_query_encodes, encoder._n_passage_encodes) == (2, 1)


class TestEncodeWithCache:
    """The full path: dedupe, cache, and the flag that must survive both."""

    def test_the_same_text_is_encoded_once_per_role_not_once_per_text(
        self, encoder: _StubEncoder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "embed_cache_size", 32)
        encoder._cache = _LruCache(capacity=32)

        first = encoder.encode(["mumbai"], is_query=True)
        second = encoder.encode(["mumbai"], is_query=True)      # cache hit, no new forward
        as_passage = encoder.encode(["mumbai"], is_query=False)  # different role, must encode

        assert np.array_equal(first, second)
        assert len(encoder.seen) == 2, "the passage role must not be served from the query entry"
        assert encoder.seen[0] == [f"{settings.embed_query_prefix}mumbai"]
        assert encoder.seen[1] == [f"{settings.embed_passage_prefix}mumbai"]
        assert not np.array_equal(first, as_passage)

    def test_duplicates_inside_one_call_are_encoded_once(
        self, encoder: _StubEncoder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "embed_cache_size", 32)
        encoder._cache = _LruCache(capacity=32)
        out = encoder.encode(["same", "same", "other"], is_query=True)

        assert out.shape == (3, encoder.dim)
        assert sorted(encoder.seen[0]) == sorted(
            [f"{settings.embed_query_prefix}same", f"{settings.embed_query_prefix}other"]
        )
        assert np.array_equal(out[0], out[1]), "identical inputs must produce identical vectors"

    def test_empty_input_returns_an_empty_matrix_of_the_right_width(
        self, encoder: _StubEncoder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "embed_cache_size", 8)
        encoder._cache = _LruCache(capacity=8)
        out = encoder.encode([], is_query=True)
        assert out.shape == (0, encoder.dim)
        assert encoder.seen == []
