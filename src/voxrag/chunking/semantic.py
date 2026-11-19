"""Semantic chunking: split where the topic actually changes.

Every sentence is embedded, the cosine distance between consecutive sentences is measured,
and a boundary is placed wherever that distance spikes above the
`semantic_breakpoint_percentile` percentile *of this document's own distances*. A per
document threshold matters: an absolute cutoff that works for a dense technical passage
shreds a narrative one, and the corpus mixes both across 14 languages.

The embedder is imported inside the method, never at module import. Importing it at the
top would make `import voxrag.chunking` pull in onnxruntime and load a 118M-parameter
model -- a cost the API process pays at startup for a strategy it may never run.

When the embedder is unavailable (not built yet, ONNX missing, model download failed) this
falls back to an IDF-weighted token-overlap distance over the same sentences. It is a
weaker signal but it is a real signal, and a dead optional dependency degrades the
strategy instead of failing the index build.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

from voxrag.chunking.base import BaseChunker, Span, split_sentences, token_index
from voxrag.config import settings
from voxrag.contracts import Chunk, ChunkStrategyName, Document

#: Attribute names tried, in order, when locating the project's embedder. The embed module
#: is owned by another component; this list covers the plausible accessors and any object
#: exposing `.encode` is accepted, so a rename there degrades to the lexical path rather
#: than raising.
_EMBEDDER_ACCESSORS = (
    "get_embedder",
    "get_default_embedder",
    "load_embedder",
    "default_embedder",
    "embedder",
    "EMBEDDER",
    "E5Embedder",
    "OnnxEmbedder",
    "Embedder",
)

_embedder_cache: Any | None = None
_embedder_resolved = False

#: Sentences pooled on each side of a candidate boundary. 2 gives enough context to drown
#: out single-sentence noise while keeping the comparison local; larger values start folding
#: a neighbouring topic into the pool and flatten the peak that identifies the seam.
_WINDOW = 2


def _resolve_embedder() -> Any | None:
    """Locate and memoise the project embedder. Resolution is attempted exactly once."""
    global _embedder_cache, _embedder_resolved
    if _embedder_resolved:
        return _embedder_cache
    _embedder_resolved = True
    try:
        import voxrag.embed as embed_mod
    except Exception:
        return None
    for attr in _EMBEDDER_ACCESSORS:
        obj = getattr(embed_mod, attr, None)
        if obj is None:
            continue
        try:
            candidate = obj() if callable(obj) else obj
        except Exception:
            # e.g. `Embedder` re-exported as the Protocol -- not instantiable.
            continue
        if hasattr(candidate, "encode"):
            _embedder_cache = candidate
            return candidate
    return None


def _dense_distances(texts: list[str]) -> list[float] | None:
    """Per-gap distance between the context BEFORE and the context AFTER each position.

    Comparing bare adjacent sentences is too noisy to locate a topic seam. Measured on a
    six-sentence Hindi document (three sentences about Delhi, then three about mangoes, so
    the true seam is gap 2) the adjacent-sentence distances were
        [0.1069, 0.1357, 0.1559, 0.1394, 0.1821]
    -- the peak (gap 4) sits INSIDE the mango topic and the real seam ranks only third.

    Smoothing with OVERLAPPING windows makes it worse, not better: a window centred on
    either side of the seam contains sentences from both topics, so the two look alike and
    the boundary is blurred out entirely (that variant peaked at gap 1).

    What works is a non-overlapping comparison: at each gap, pool the `_WINDOW` sentences
    before it and the `_WINDOW` after it, and measure across. Pooling is done on the
    sentence vectors rather than by re-embedding concatenated text, so this still costs
    exactly one batched encode of N sentences.
    """
    embedder = _resolve_embedder()
    if embedder is None:
        return None
    try:
        import numpy as np

        # One batched call for the whole document: per-sentence calls would pay the ONNX
        # session overhead N times over.
        vecs = np.asarray(embedder.encode(texts), dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] != len(texts):
            return None
        # The Embedder contract promises L2-normalised rows; re-normalising is cheap
        # insurance against a non-conforming implementation making every distance ~0.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-12)

        n = len(texts)
        out: list[float] = []
        for i in range(n - 1):
            left = vecs[max(0, i - _WINDOW + 1) : i + 1].mean(axis=0)
            right = vecs[i + 1 : min(n, i + 1 + _WINDOW)].mean(axis=0)
            # Renormalise: the mean of unit vectors is not a unit vector, and skipping this
            # would make every distance depend on how much the pooled sentences disagree
            # rather than on how far apart the two sides are.
            left = left / max(float(np.linalg.norm(left)), 1e-12)
            right = right / max(float(np.linalg.norm(right)), 1e-12)
            out.append(float(1.0 - float(np.dot(left, right))))
        return out
    except Exception:
        return None


def _breakpoint_threshold(distances: list[float], p: float) -> float:
    """Linearly-interpolated percentile of the distance distribution.

    Deliberately NOT `telemetry.percentile`, which is nearest-rank. On a typical MS MARCO
    passage there are 4-8 sentences, so 3-7 distances; nearest rank at p92 lands exactly
    on the maximum, `d > threshold` is then never true, and SEMANTIC silently degenerates
    into PASSAGE_NATIVE for the entire corpus. Interpolating puts the threshold just below
    the largest gap, so the strongest topic shift in a short passage does become a split.
    """
    if not distances:
        return 0.0
    ordered = sorted(distances)
    if len(ordered) == 1:
        # A single gap is no evidence of a topic shift; returning it suppresses the split.
        return ordered[0]
    k = max(0.0, min(100.0, p)) / 100.0 * (len(ordered) - 1)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _lexical_distances(doc_text: str, sentences: list[Span]) -> list[float]:
    """IDF-weighted cosine distance over encoder tokens -- the no-embedder fallback.

    Encoder tokens rather than words: Tamil, Malayalam and Telugu agglutinate heavily, so
    whole-word overlap between two related sentences is often zero while subword overlap
    is informative.
    """
    idx = token_index(doc_text)
    bags: list[dict[str, int]] = []
    for sp in sentences:
        lo, hi = idx.range(sp.start, sp.end)
        bag: dict[str, int] = {}
        for s, e in idx.spans[lo:hi]:
            tok = doc_text[s:e].strip().lower()
            if tok:
                bag[tok] = bag.get(tok, 0) + 1
        bags.append(bag)

    n_docs = max(1, len(bags))
    df: dict[str, int] = {}
    for bag in bags:
        for tok in bag:
            df[tok] = df.get(tok, 0) + 1
    idf = {tok: math.log(1.0 + n_docs / c) for tok, c in df.items()}

    weighted = [{t: c * idf[t] for t, c in bag.items()} for bag in bags]
    out: list[float] = []
    for a, b in pairwise(weighted):
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0.0 or nb == 0.0:
            out.append(1.0)
            continue
        smaller, larger = (a, b) if len(a) <= len(b) else (b, a)
        dot = sum(v * larger.get(t, 0.0) for t, v in smaller.items())
        out.append(float(max(0.0, 1.0 - dot / (na * nb))))
    return out


class SemanticChunker(BaseChunker):
    name = ChunkStrategyName.SEMANTIC

    def chunk(self, doc: Document) -> list[Chunk]:
        max_tokens = max(settings.semantic_max_tokens, settings.semantic_min_tokens)
        sentences = split_sentences(doc.text, doc.lang, max_tokens=max_tokens)
        if not sentences:
            return []
        if len(sentences) == 1:
            return self._from_spans(doc, sentences)

        distances = _dense_distances([sp.text for sp in sentences])
        backend = "dense"
        if distances is None or len(distances) != len(sentences) - 1:
            distances = _lexical_distances(doc.text, sentences)
            backend = "lexical_fallback"

        threshold = _breakpoint_threshold(
            list(distances), settings.semantic_breakpoint_percentile
        )
        breaks = {i for i, d in enumerate(distances) if d > threshold}

        segments = self._segments(len(sentences), breaks)
        segments = self._enforce_max(doc.text, sentences, segments, distances, max_tokens)

        # Cap the minimum-size floor at half the document. A fixed floor silently cancels
        # every boundary on a short passage: a 49-token document against
        # semantic_min_tokens=48 has no legal two-way split, so the merge step folds a
        # correctly-detected topic seam straight back into one chunk and SEMANTIC degenerates
        # into PASSAGE_NATIVE. Halving keeps the floor meaningful for long passages (where it
        # is what stops single-sentence fragments) while letting short ones split at all.
        doc_tokens = self._tokens(doc.text, sentences, (0, len(sentences)))
        effective_min = min(settings.semantic_min_tokens, max(1, doc_tokens // 2))

        segments = self._enforce_min(
            doc.text, sentences, segments, distances, effective_min, max_tokens
        )

        # A detected boundary must never be erased entirely by the size floor. `_enforce_min`
        # merges undersized neighbours, and on a short passage that can absorb every break --
        # leaving one chunk and turning SEMANTIC into an expensive PASSAGE_NATIVE without any
        # error to notice. Measured case: a 48-token document splits correctly at the topic
        # seam into 22 and 26 tokens, then both are merged for being under the floor. If the
        # detector found seams and the floor consumed all of them, keep the strongest one.
        if len(segments) == 1 and breaks and len(sentences) > 1:
            strongest = max(breaks, key=lambda i: distances[i])
            segments = [(0, strongest + 1), (strongest + 1, len(sentences))]

        chunks: list[Chunk] = []
        for ordinal, (a, b) in enumerate(segments):
            start, end = sentences[a].start, sentences[b - 1].end
            chunks.append(
                self._build(
                    doc,
                    ordinal,
                    start,
                    end,
                    meta={
                        "semantic_backend": backend,
                        "breakpoint_threshold": round(threshold, 6),
                        "n_sentences": b - a,
                    },
                )
            )
        return chunks

    @staticmethod
    def _segments(n: int, breaks: set[int]) -> list[tuple[int, int]]:
        """Half-open sentence-index ranges; a break at i cuts between sentence i and i+1."""
        segments: list[tuple[int, int]] = []
        start = 0
        for i in range(n - 1):
            if i in breaks:
                segments.append((start, i + 1))
                start = i + 1
        segments.append((start, n))
        return segments

    @staticmethod
    def _tokens(text: str, sentences: list[Span], seg: tuple[int, int]) -> int:
        a, b = seg
        return token_index(text).count(sentences[a].start, sentences[b - 1].end)

    def _enforce_max(
        self,
        text: str,
        sentences: list[Span],
        segments: list[tuple[int, int]],
        distances: list[float],
        max_tokens: int,
    ) -> list[tuple[int, int]]:
        """Recursively cut over-long segments at their weakest interior seam.

        Cutting at the largest interior distance rather than at an arbitrary token count is
        what keeps this strategy semantic once the length constraint bites.
        """
        out: list[tuple[int, int]] = []
        stack = list(reversed(segments))
        while stack:
            a, b = stack.pop()
            if b - a <= 1 or self._tokens(text, sentences, (a, b)) <= max_tokens:
                out.append((a, b))
                continue
            cut = max(range(a, b - 1), key=lambda i: distances[i])
            stack.append((cut + 1, b))
            stack.append((a, cut + 1))
        out.sort()
        return out

    def _enforce_min(
        self,
        text: str,
        sentences: list[Span],
        segments: list[tuple[int, int]],
        distances: list[float],
        min_tokens: int,
        max_tokens: int,
    ) -> list[tuple[int, int]]:
        """Absorb undersized segments into whichever neighbour they are closer to.

        A 12-token orphan chunk is nearly always retrieval noise. It is merged towards the
        smaller boundary distance -- the more semantically continuous side -- and only when
        the merge stays inside `max_tokens`.
        """
        segs = list(segments)
        changed = True
        while changed and len(segs) > 1:
            changed = False
            for i, seg in enumerate(segs):
                if self._tokens(text, sentences, seg) >= min_tokens:
                    continue
                prev_d = distances[segs[i - 1][1] - 1] if i > 0 else math.inf
                next_d = distances[seg[1] - 1] if i + 1 < len(segs) else math.inf
                order = [i - 1, i + 1] if prev_d <= next_d else [i + 1, i - 1]
                for j in order:
                    if not 0 <= j < len(segs):
                        continue
                    lo, hi = (min(i, j), max(i, j))
                    merged = (segs[lo][0], segs[hi][1])
                    if self._tokens(text, sentences, merged) <= max_tokens:
                        segs[lo : hi + 1] = [merged]
                        changed = True
                        break
                if changed:
                    break
        return segs
