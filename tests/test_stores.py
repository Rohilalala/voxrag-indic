"""Vector stores: HnswStore in process, plus a Qdrant round trip when a server is running.

Vectors here are built inside a single plane so the expected ranking is arithmetic rather
than guesswork: chunk `i` is `cos(theta_i)*e0 + sin(theta_i)*e1` with theta increasing, and
the query is `e0`. Cosine similarity is then exactly `cos(theta_i)`, strictly decreasing in
`i`, so "nearest first" has one correct answer that the test can name.

Everything is hermetic. HnswStore takes an explicit path, so nothing touches
`settings.hnsw_path` or the built 120k index; the Qdrant tests skip themselves unless
127.0.0.1:6333 is already listening, and use a throwaway collection when it is.
"""

from __future__ import annotations

import asyncio
import math
import socket
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import voxrag.store as store_pkg
import voxrag.store.hnsw_store as hnsw_mod
from voxrag.config import settings
from voxrag.contracts import Chunk, ChunkStrategyName, Lang, ScoredChunk, VectorStore
from voxrag.store import get_vector_store
from voxrag.store.hnsw_store import HnswStore

DIM = settings.embed_dim


@pytest.fixture(autouse=True)
def _small_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """hnswlib pre-allocates its graph: the production 65k x 384 float32 arena is 100MB, and
    every store ever built stays alive for the process because __init__ registers an atexit
    flush. A test suite that builds a dozen would carry >1GB. Shrinking the arena also means
    the resize path gets exercised rather than never reached."""
    monkeypatch.setattr(hnsw_mod, "_INITIAL_CAPACITY", 64)


def _plane_vector(theta: float) -> np.ndarray:
    """Unit vector at angle `theta` in the e0/e1 plane. Cosine to e0 is exactly cos(theta)."""
    vec = np.zeros(DIM, dtype=np.float32)
    vec[0] = math.cos(theta)
    vec[1] = math.sin(theta)
    return vec


#: Probe pointing at e0, so similarity to `_plane_vector(theta)` is cos(theta).
QUERY = _plane_vector(0.0)


def _chunk(
    chunk_id: str,
    *,
    lang: Lang = Lang.HIN,
    strategy: ChunkStrategyName = ChunkStrategyName.PASSAGE_NATIVE,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=f"passage text for {chunk_id}",
        lang=lang,
        strategy=strategy,
    )


def _fan(
    n: int,
    *,
    langs: list[Lang] | None = None,
    strategies: list[ChunkStrategyName] | None = None,
    spread: float = math.pi / 3,
) -> tuple[list[Chunk], np.ndarray]:
    """`n` chunks fanned out from the query by increasing angle, so chunk `i` is the i-th
    nearest neighbour of QUERY by construction."""
    chunks = [
        _chunk(
            f"c{i:03d}",
            lang=(langs[i] if langs else Lang.HIN),
            strategy=(strategies[i] if strategies else ChunkStrategyName.PASSAGE_NATIVE),
        )
        for i in range(n)
    ]
    vectors = np.vstack([_plane_vector(spread * i / n) for i in range(n)])
    return chunks, vectors


@pytest.fixture
def store(tmp_path: Path) -> HnswStore:
    return HnswStore(tmp_path / "hnsw")


# ======================================================================================
# HnswStore: search ordering and scores
# ======================================================================================


def test_hnsw_store_satisfies_the_protocol(store: HnswStore) -> None:
    assert isinstance(store, VectorStore)


async def test_upsert_then_search_returns_the_nearest_vectors_in_score_order(
    store: HnswStore,
) -> None:
    chunks, vectors = _fan(12)
    await store.upsert(chunks, vectors)

    hits = await store.search(QUERY, top_k=5)

    assert [hit.chunk.chunk_id for hit in hits] == [c.chunk_id for c in chunks[:5]]
    assert [hit.score for hit in hits] == sorted((h.score for h in hits), reverse=True)
    assert await store.count() == 12


async def test_scores_are_cosine_similarity_not_distance(store: HnswStore) -> None:
    """hnswlib reports 1 - cosine; the store must invert it or every ranking downstream that
    treats `score` as "higher is better" silently flips."""
    chunks, vectors = _fan(4)
    await store.upsert(chunks, vectors)

    hits = await store.search(QUERY, top_k=4)

    expected = [float(np.dot(QUERY, vectors[i])) for i in range(4)]
    assert [hit.score for hit in hits] == pytest.approx(expected, abs=1e-5)
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)  # the query is an indexed vector


async def test_hits_carry_one_based_dense_rank_and_mirror_the_score(store: HnswStore) -> None:
    """RRF reads `dense_rank`; a 0-based rank there changes every fused score."""
    chunks, vectors = _fan(6)
    await store.upsert(chunks, vectors)

    hits = await store.search(QUERY, top_k=4)

    assert [hit.dense_rank for hit in hits] == [1, 2, 3, 4]
    assert all(hit.dense_score == hit.score for hit in hits)
    assert all(hit.lexical_score is None and hit.lexical_rank is None for hit in hits)


async def test_chunks_come_back_fully_hydrated(store: HnswStore) -> None:
    """The sidecar holds the whole pydantic Chunk so a hit needs no second lookup."""
    original = _chunk("rich", lang=Lang.BEN, strategy=ChunkStrategyName.PARENT_CHILD)
    original = original.model_copy(
        update={
            "ordinal": 3,
            "char_start": 40,
            "char_end": 220,
            "n_tokens": 37,
            "title": "শিরোনাম",
            "context_text": "wider parent context",
            "parent_id": "parent-1",
            "meta": {"is_selected": True},
        }
    )
    await store.upsert([original], _plane_vector(0.0).reshape(1, -1))

    hit = (await store.search(QUERY, top_k=1))[0]
    assert hit.chunk == original
    assert hit.chunk.generation_text == "wider parent context"


async def test_search_on_an_empty_index_returns_empty_rather_than_raising(
    store: HnswStore,
) -> None:
    assert await store.search(QUERY, top_k=5) == []
    assert await store.count() == 0
    assert await store.health() is True


async def test_nonpositive_top_k_returns_empty(store: HnswStore) -> None:
    chunks, vectors = _fan(3)
    await store.upsert(chunks, vectors)
    assert await store.search(QUERY, top_k=0) == []
    assert await store.search(QUERY, top_k=-1) == []


async def test_top_k_larger_than_the_corpus_returns_the_whole_corpus(store: HnswStore) -> None:
    chunks, vectors = _fan(3)
    await store.upsert(chunks, vectors)
    hits = await store.search(QUERY, top_k=50)
    assert len(hits) == 3


async def test_upsert_of_nothing_is_a_no_op(store: HnswStore) -> None:
    await store.upsert([], np.empty((0, DIM), dtype=np.float32))
    assert await store.count() == 0


# ======================================================================================
# HnswStore: filtering
# ======================================================================================


async def test_lang_filter_never_returns_another_language(store: HnswStore) -> None:
    langs = [Lang.HIN, Lang.TAM, Lang.BEN] * 8
    chunks, vectors = _fan(24, langs=langs)
    await store.upsert(chunks, vectors)

    for lang in (Lang.HIN, Lang.TAM, Lang.BEN):
        hits = await store.search(QUERY, top_k=5, lang=lang)
        assert hits, f"{lang} returned nothing"
        assert {hit.chunk.lang for hit in hits} == {lang}
        assert [hit.dense_rank for hit in hits] == list(range(1, len(hits) + 1))


async def test_strategy_filter_never_returns_another_strategy(store: HnswStore) -> None:
    strategies = [ChunkStrategyName.PASSAGE_NATIVE, ChunkStrategyName.PARENT_CHILD] * 10
    chunks, vectors = _fan(20, strategies=strategies)
    await store.upsert(chunks, vectors)

    hits = await store.search(QUERY, top_k=6, strategy=ChunkStrategyName.PARENT_CHILD)
    assert hits
    assert {hit.chunk.strategy for hit in hits} == {ChunkStrategyName.PARENT_CHILD}


async def test_lang_and_strategy_filters_compose(store: HnswStore) -> None:
    langs = [Lang.HIN, Lang.TAM] * 12
    strategies = [ChunkStrategyName.PASSAGE_NATIVE, ChunkStrategyName.PARENT_CHILD] * 12
    chunks, vectors = _fan(24, langs=langs, strategies=strategies)
    await store.upsert(chunks, vectors)

    hits = await store.search(
        QUERY, top_k=4, lang=Lang.TAM, strategy=ChunkStrategyName.PARENT_CHILD
    )

    assert hits
    for hit in hits:
        assert hit.chunk.lang is Lang.TAM
        assert hit.chunk.strategy is ChunkStrategyName.PARENT_CHILD


async def test_a_filter_that_matches_nothing_returns_empty(store: HnswStore) -> None:
    chunks, vectors = _fan(10)  # all Hindi
    await store.upsert(chunks, vectors)
    assert await store.search(QUERY, top_k=5, lang=Lang.MAL) == []


async def test_a_selective_filter_still_returns_top_k(store: HnswStore) -> None:
    """The over-fetch path. hnswlib applies the predicate *during* traversal, so a selective
    filter makes the graph walk drift away from the region that actually matches: 294 Hindi
    chunks clustered around the query and 6 Tamil ones in a separate, far-away cluster the
    unfiltered walk never visits. The store must still come back with a full top_k, ranked."""
    rng = np.random.default_rng(7)

    def unit(vec: np.ndarray) -> np.ndarray:
        return (vec / np.linalg.norm(vec)).astype(np.float32)

    query = unit(rng.standard_normal(DIM))
    far_centroid = unit(rng.standard_normal(DIM))
    near = [unit(query * 0.9 + rng.standard_normal(DIM) * 0.25) for _ in range(294)]
    far = [unit(far_centroid + rng.standard_normal(DIM) * 0.05) for _ in range(6)]
    vectors = np.vstack(near + far)
    chunks = [
        _chunk(f"c{i:03d}", lang=Lang.HIN if i < len(near) else Lang.TAM)
        for i in range(len(near) + len(far))
    ]
    await store.upsert(chunks, vectors)

    # Unfiltered, no Tamil chunk is anywhere near the top -- the filter really is selective.
    unfiltered = await store.search(query, top_k=20)
    assert all(hit.chunk.lang is Lang.HIN for hit in unfiltered)

    hits = await store.search(query, top_k=5, lang=Lang.TAM)

    assert len(hits) == 5, "a selective filter came back short of top_k"
    assert {hit.chunk.lang for hit in hits} == {Lang.TAM}
    # Ranked correctly too, not merely non-empty: the true 5 best of the 6 matching chunks.
    similarities = {
        c.chunk_id: float(np.dot(query, v)) for c, v in zip(chunks, vectors, strict=True)
    }
    expected = sorted(
        (c.chunk_id for c in chunks if c.lang is Lang.TAM),
        key=lambda cid: -similarities[cid],
    )[:5]
    assert [hit.chunk.chunk_id for hit in hits] == expected


# ======================================================================================
# HnswStore: replacement, validation, persistence, concurrency
# ======================================================================================


async def test_reupserting_a_chunk_id_replaces_it_and_updates_its_filter_keys(
    store: HnswStore,
) -> None:
    """The filter keys are parallel arrays plus a counter; a replacement that forgot to
    decrement the old key would leave the chunk reachable through a filter it no longer
    matches, or make a matching one unreachable."""
    chunks, vectors = _fan(4)
    await store.upsert(chunks, vectors)
    assert await store.count() == 4

    moved = chunks[0].model_copy(update={"lang": Lang.TAM, "text": "rewritten"})
    await store.upsert([moved], vectors[0].reshape(1, -1))

    assert await store.count() == 4  # replaced, not appended
    assert await store.search(QUERY, top_k=5, lang=Lang.TAM) != []
    hindi = await store.search(QUERY, top_k=5, lang=Lang.HIN)
    assert moved.chunk_id not in {hit.chunk.chunk_id for hit in hindi}
    assert len(hindi) == 3

    tamil = await store.search(QUERY, top_k=5, lang=Lang.TAM)
    assert [hit.chunk.chunk_id for hit in tamil] == [moved.chunk_id]
    assert tamil[0].chunk.text == "rewritten"


async def test_reupserting_moves_the_vector_and_not_just_the_payload(store: HnswStore) -> None:
    """Re-indexing has to update the point's *coordinates*, not only its stored Chunk.

    A store that replaced the payload but left the old vector in the graph would keep
    answering the old neighbourhood with the new text -- retrieval quality collapses and
    nothing raises, because every hit still hydrates a valid, current-looking chunk. hnswlib
    updates an element in place when `add_items` is given a label it already holds; this
    pins that, because losing it is invisible from the payload side.
    """
    near, far = _plane_vector(0.0), _plane_vector(1.0)
    chunks = [_chunk("mover"), _chunk("anchor")]
    await store.upsert(chunks, np.vstack([near, far]))
    assert [h.chunk.chunk_id for h in await store.search(far, top_k=2)] == ["anchor", "mover"]

    # Same chunk_id, new coordinates: it now sits on top of `anchor`.
    await store.upsert([chunks[0].model_copy(update={"text": "moved"})], far.reshape(1, -1))

    assert await store.count() == 2  # moved, not duplicated
    moved = next(h for h in await store.search(far, top_k=2) if h.chunk.chunk_id == "mover")
    assert moved.score == pytest.approx(1.0, abs=1e-5), "the vector did not move with the chunk"
    assert moved.chunk.text == "moved"
    # And it has genuinely left the origin rather than being indexed in both places: its
    # similarity to the old probe is now cos(1.0), not 1.0. Asserted on the score, not on the
    # ordering -- `mover` and `anchor` are the same point now, so their relative order is a tie.
    at_origin = await store.search(near, top_k=2)
    assert {h.chunk.chunk_id for h in at_origin} == {"mover", "anchor"}
    stale = next(h for h in at_origin if h.chunk.chunk_id == "mover")
    assert stale.score == pytest.approx(math.cos(1.0), abs=1e-5), "old vector left behind"


async def test_a_filter_matching_fewer_than_top_k_returns_all_of_them(store: HnswStore) -> None:
    """The over-fetch clamp is `min(top_k * 4, matching)`; asking hnswlib for more neighbours
    than exist raises rather than returning a short list, so this path must not come back
    empty-handed just because the filter is narrower than the request."""
    langs = [Lang.HIN, Lang.HIN, Lang.HIN, Lang.TAM, Lang.HIN, Lang.TAM]
    chunks, vectors = _fan(6, langs=langs)
    await store.upsert(chunks, vectors)

    hits = await store.search(QUERY, top_k=5, lang=Lang.TAM)

    assert [hit.chunk.chunk_id for hit in hits] == ["c003", "c005"]  # only two match at all
    assert [hit.dense_rank for hit in hits] == [1, 2]


async def test_shape_mismatches_are_rejected(store: HnswStore) -> None:
    chunks, vectors = _fan(3)
    with pytest.raises(ValueError, match="does not match"):
        await store.upsert(chunks, vectors[:2])
    with pytest.raises(ValueError, match="dim"):
        await store.upsert(chunks, np.zeros((3, DIM + 1), dtype=np.float32))
    with pytest.raises(ValueError, match="dim"):
        await store.search(np.zeros(DIM - 1, dtype=np.float32), top_k=1)
    assert await store.count() == 0  # a rejected upsert leaves nothing behind


async def test_persistence_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "hnsw"
    original = HnswStore(path)
    langs = [Lang.HIN, Lang.TAM] * 5
    chunks, vectors = _fan(10, langs=langs)
    await original.upsert(chunks, vectors)
    before = [(h.chunk.chunk_id, h.score) for h in await original.search(QUERY, top_k=5)]
    await original.save()

    reloaded = HnswStore(path)

    assert await reloaded.count() == 10
    after = [(h.chunk.chunk_id, h.score) for h in await reloaded.search(QUERY, top_k=5)]
    assert [cid for cid, _ in after] == [cid for cid, _ in before]
    assert [s for _, s in after] == pytest.approx([s for _, s in before], abs=1e-6)
    # Filter keys are rebuilt from the sidecar, not from the graph, so they must survive too.
    tamil = await reloaded.search(QUERY, top_k=5, lang=Lang.TAM)
    assert tamil and {hit.chunk.lang for hit in tamil} == {Lang.TAM}
    # And the reloaded store keeps accepting writes at the right labels.
    extra, extra_vecs = _fan(1)
    await reloaded.upsert(
        [extra[0].model_copy(update={"chunk_id": "later"})], extra_vecs
    )
    assert await reloaded.count() == 11


async def test_a_stale_index_of_the_wrong_dimension_is_rebuilt_not_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_index keeps the constructed dim, so a mismatched file loads and then segfaults on
    query. The meta sidecar exists purely to catch that before the C++ side is touched."""
    path = tmp_path / "hnsw"
    original = HnswStore(path)
    chunks, vectors = _fan(4)
    await original.upsert(chunks, vectors)
    await original.save()

    monkeypatch.setattr(settings, "embed_dim", DIM // 2)
    try:
        rebuilt = HnswStore(path)
        assert await rebuilt.count() == 0
        assert await rebuilt.search(np.zeros(DIM // 2, dtype=np.float32), top_k=3) == []
    finally:
        monkeypatch.undo()


async def test_concurrent_searches_during_upsert_do_not_corrupt_state(
    store: HnswStore,
) -> None:
    """hnswlib tolerates concurrent knn_query but not a query racing add_items, and
    resize_index reallocates the graph out from under an in-flight reader."""
    chunks, vectors = _fan(80)
    seed, seed_vectors = chunks[:8], vectors[:8]
    await store.upsert(seed, seed_vectors)

    async def writer(start: int) -> None:
        await store.upsert(chunks[start : start + 8], vectors[start : start + 8])

    async def reader() -> list[ScoredChunk]:
        return await store.search(_plane_vector(0.01), top_k=5)

    tasks: list[asyncio.Future[object]] = []
    for start in range(8, 80, 8):
        tasks.append(asyncio.ensure_future(writer(start)))
        tasks.extend(asyncio.ensure_future(reader()) for _ in range(4))
    results = await asyncio.gather(*tasks)

    assert await store.count() == 80

    known = {c.chunk_id: c for c in chunks}
    for result in results:
        if result is None:
            continue
        assert isinstance(result, list)
        for hit in result:
            # A torn label->chunk mapping shows up here as a chunk that never existed, or as
            # one whose payload does not belong to its id.
            assert hit.chunk.chunk_id in known
            assert hit.chunk == known[hit.chunk.chunk_id]
            assert -1.0001 <= hit.score <= 1.0001
        assert [h.score for h in result] == sorted((h.score for h in result), reverse=True)

    # The index is still coherent afterwards: every chunk is its own nearest neighbour.
    for i in (0, 17, 40, 79):
        top = await store.search(vectors[i], top_k=1)
        assert top[0].chunk.chunk_id == chunks[i].chunk_id


async def test_warmup_is_safe_on_an_empty_and_on_a_populated_store(store: HnswStore) -> None:
    await store.warmup()  # empty: must not probe an index with no elements
    chunks, vectors = _fan(5)
    await store.upsert(chunks, vectors)
    await store.warmup()
    assert store.info()["vectors"] == 5
    assert store.info()["dim"] == DIM


# ======================================================================================
# Factory
# ======================================================================================


def test_unknown_store_name_raises_with_the_valid_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown vector store"):
        get_vector_store("faiss")


def test_aliases_resolve_to_one_cached_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building a store loads an index off disk, so two names for one backend must not build
    it twice -- the loser's graph would be a second full copy in RAM."""
    monkeypatch.setattr(settings, "hnsw_path", tmp_path / "hnsw")
    monkeypatch.setattr(store_pkg, "_CACHE", {})

    first = get_vector_store("hnsw")
    assert get_vector_store("hnswlib") is first
    assert get_vector_store("  HNSW  ") is first
    assert isinstance(first, HnswStore)


# ======================================================================================
# Qdrant -- only when a server is already listening
# ======================================================================================
#
# These are the one exception to the suite's autouse socket block, and they are opt-in twice
# over: `skipif` decides at collection time whether a server exists at all, and
# `allow_network` (registered by conftest) lifts the block only for the tests that survive
# that. On a laptop with the wifi off, or on a CI box with no services, both marks resolve to
# "skipped with a reason" and nothing here can fail. The probe below runs at import time,
# before the autouse fixture is installed, so it sees the real socket layer.


def _qdrant_listening(host: str = "127.0.0.1", port: int = 6333) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        try:
            return sock.connect_ex((host, port)) == 0
        except OSError:
            return False


#: Paired with `@pytest.mark.allow_network` on every Qdrant test; see the note above.
qdrant_server = pytest.mark.skipif(
    not _qdrant_listening(),
    reason="no Qdrant server on 127.0.0.1:6333 (run `docker compose up -d` to include these)",
)


@pytest.fixture
def qdrant_store() -> Iterator[object]:
    """A throwaway collection on the running server, dropped afterwards either way."""
    from voxrag.store.qdrant_store import QdrantStore

    store = QdrantStore(
        url="http://127.0.0.1:6333", collection=f"voxrag_test_{uuid.uuid4().hex[:12]}"
    )
    try:
        yield store
    finally:
        asyncio.run(_drop(store))


async def _drop(store: object) -> None:
    client = store._client  # type: ignore[attr-defined]
    try:
        await client.delete_collection(store.collection)  # type: ignore[attr-defined]
    finally:
        await client.close()


async def _await_count(store: object, expected: int, timeout_s: float = 10.0) -> int:
    """Upserts are issued with wait=False, so the collection converges shortly after."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    count = await store.count()  # type: ignore[attr-defined]
    while count < expected and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
        count = await store.count()  # type: ignore[attr-defined]
    return count


@qdrant_server
@pytest.mark.allow_network
async def test_qdrant_round_trips_and_filters(qdrant_store: object) -> None:
    langs = [Lang.HIN, Lang.TAM] * 5
    chunks, vectors = _fan(10, langs=langs)
    await qdrant_store.upsert(chunks, vectors)  # type: ignore[attr-defined]
    assert await _await_count(qdrant_store, 10) == 10

    hits = await qdrant_store.search(QUERY, top_k=5)  # type: ignore[attr-defined]
    assert [hit.chunk.chunk_id for hit in hits] == [c.chunk_id for c in chunks[:5]]
    assert [hit.dense_rank for hit in hits] == [1, 2, 3, 4, 5]
    assert all(hit.dense_score == hit.score for hit in hits)
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[0].chunk == chunks[0]  # the whole Chunk round-trips through the payload

    tamil = await qdrant_store.search(QUERY, top_k=5, lang=Lang.TAM)  # type: ignore[attr-defined]
    assert tamil and {hit.chunk.lang for hit in tamil} == {Lang.TAM}

    none = await qdrant_store.search(QUERY, top_k=5, lang=Lang.MAL)  # type: ignore[attr-defined]
    assert none == []
    assert await qdrant_store.search(QUERY, top_k=0) == []  # type: ignore[attr-defined]
    assert await qdrant_store.health() is True  # type: ignore[attr-defined]


@qdrant_server
@pytest.mark.allow_network
async def test_qdrant_reupsert_replaces_rather_than_duplicates(qdrant_store: object) -> None:
    """Point ids are uuid5 of the chunk_id, so re-indexing must overwrite in place."""
    chunks, vectors = _fan(3)
    await qdrant_store.upsert(chunks, vectors)  # type: ignore[attr-defined]
    assert await _await_count(qdrant_store, 3) == 3

    edited = chunks[0].model_copy(update={"text": "rewritten passage"})
    await qdrant_store.upsert([edited], vectors[0].reshape(1, -1))  # type: ignore[attr-defined]
    await asyncio.sleep(0.2)

    assert await qdrant_store.count() == 3  # type: ignore[attr-defined]
    hits = await qdrant_store.search(QUERY, top_k=1)  # type: ignore[attr-defined]
    assert hits[0].chunk.text == "rewritten passage"


@qdrant_server
@pytest.mark.allow_network
async def test_qdrant_rejects_the_wrong_vector_dimension(qdrant_store: object) -> None:
    chunks, _ = _fan(2)
    with pytest.raises(ValueError, match="dim"):
        await qdrant_store.upsert(chunks, np.zeros((2, DIM + 3), dtype=np.float32))  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="dim"):
        await qdrant_store.search(np.zeros(DIM - 3, dtype=np.float32))  # type: ignore[attr-defined]
