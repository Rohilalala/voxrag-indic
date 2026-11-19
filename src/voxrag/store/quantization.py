"""Qdrant vector quantization: compression as a lever on the dense-search tail.

Dense search is the dominant cost in this system (measured: p50 68.3ms of a 108.3ms CORE) and
it runs over 2,268,974 x 384 float32 vectors -- roughly 3.5GB that has to stay RAM-resident or
the tail collapses. Quantization attacks both halves of that at once: the stored vectors get
smaller, so more of the working set stays resident, and integer distance computation is
cheaper than float32.

What it costs is RECALL, and how much is a property of THIS index and THIS embedder. The usual
claim -- "int8 with rescoring loses almost nothing" -- is somebody else's measurement on
somebody else's corpus. So nothing here is on by default: `settings.quantization` is "none",
and `scripts/measure_quantization.py` applies each kind to a throwaway COPY of the collection
and prints the latency/recall trade so the decision is made from a table.

The three kinds, and what each one actually does to a vector:

* ``none``   -- 384 x float32 = 1536 bytes per vector. The measured baseline.
* ``scalar`` -- 384 x int8 = 384 bytes, 4x smaller. Each dimension is linearly mapped into one
  byte using the `quantile` bounds of the observed value distribution. With ``rescore`` on,
  the quantized vectors only have to produce the right CANDIDATE SET; the final ordering is
  recomputed from the original float32 vectors, which Qdrant keeps.
* ``binary`` -- 384 x 1 bit = 48 bytes, 32x smaller. Only the SIGN of each dimension survives.
  Qdrant's own guidance is that this needs high-dimensional vectors (1024+) and models trained
  tolerant of it; multilingual-e5-small is 384-dim and is neither. Expect recall to fall hard.
  It is included so the measurement can demonstrate that instead of leaving it as folklore --
  not because it is a candidate configuration for this index.

``always_ram`` is set on both compressed kinds and matters more than it looks: it pins the
QUANTIZED copy in memory even when the originals are allowed to sit on disk. Without it the
compressed vectors can be paged out too, and the entire point of compressing them is gone.
"""

from __future__ import annotations

import logging
from typing import Final

from qdrant_client import AsyncQdrantClient, models

_LOG = logging.getLogger(__name__)

#: Every kind this module knows how to build. Qdrant also offers product and turbo
#: quantization; neither is wired up, because neither has been measured here.
KINDS: Final[tuple[str, ...]] = ("none", "scalar", "binary")

#: Clipping quantile for int8 scalar quantization.
#:
#: The int8 range has to be mapped onto some [min, max] window of the observed component
#: values. Taking the true min/max lets a handful of outlier components stretch the window and
#: spend precision on values almost nothing uses; 0.99 discards the extreme 1% and clips it
#: instead, so the other 99% is encoded more finely. This is Qdrant's own recommended value
#: and is not tuned here -- tuning it without re-running the retrieval eval would be exactly
#: the unmeasured change this module exists to avoid.
_SCALAR_QUANTILE: Final[float] = 0.99


def normalise_kind(kind: str | None) -> str:
    """Canonicalise a configured kind, raising on anything unrecognised.

    Deliberately strict. A typo in `VOXRAG_QUANTIZATION` that silently fell back to "none"
    would produce a benchmark labelled "scalar" that measured float32, which is worse than a
    startup failure.
    """
    resolved = (kind or "none").strip().lower()
    if resolved not in KINDS:
        raise ValueError(f"unknown quantization kind {kind!r}; expected one of {list(KINDS)}")
    return resolved


def quantization_config(kind: str) -> models.QuantizationConfig | None:
    """The qdrant_client model for `kind`, or None for "none" (no quantization block).

    None rather than `Disabled`: at CREATE time the field is simply omitted, which is what an
    unquantized collection looks like. `Disabled` is only meaningful as an UPDATE, where it
    means "remove the quantization that is currently there" -- see `apply_to_collection`.
    """
    resolved = normalise_kind(kind)
    if resolved == "none":
        return None
    if resolved == "scalar":
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=_SCALAR_QUANTILE,
                always_ram=True,
            )
        )
    return models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True))


def kind_of(config: object | None) -> str:
    """Map a quantization config read back off a live collection to a kind name.

    Used to decide whether a collection already holds the configuration being asked for, and
    to report the truth in `/api/health` rather than the intent held in settings.
    """
    if config is None or config is models.Disabled.DISABLED:
        return "none"
    if isinstance(config, models.ScalarQuantization):
        return "scalar"
    if isinstance(config, models.BinaryQuantization):
        return "binary"
    if isinstance(config, models.ProductQuantization):
        return "product"
    # Something this module did not configure -- name it rather than pretending it is "none",
    # which would make an unexpected configuration invisible in the health output.
    return "other"


async def apply_to_collection(
    client: AsyncQdrantClient, collection: str, kind: str
) -> bool:
    """Patch quantization onto an EXISTING collection. Returns True if anything changed.

    `update_collection` re-encodes the vectors already stored, so a 2.27M-point index can be
    moved between configurations without re-embedding anything -- which is the only reason
    trying this is cheap enough to be worth doing at all.

    Three things to know before measuring the result:

    * The rebuild is ASYNCHRONOUS. The call returns as soon as the optimiser accepts the new
      config; the collection then sits at status=yellow until every segment is re-encoded.
      Searching before it goes green measures a mixture of both configurations.
    * Quantized vectors are only built for INDEXED segments. A collection below the
      collection's `indexing_threshold` (20000 on the serving index) keeps its segments plain
      and answers by exact search over the original float32 vectors: this call succeeds,
      `get_collection` reports the new kind back, and nothing about search changes. Measured
      on a 3000-point throwaway collection -- unindexed, binary quantization returned the
      identical top-10 as unquantized; at `indexing_threshold=100`, 6 of those 10. Check
      `indexed_vectors_count` before believing a result.
    * It is a write against a live collection. Never point this at the serving one to satisfy
      curiosity: an in-progress re-encode competes for exactly the CPU the latency numbers are
      about, and if the recall turns out to be bad you have already shipped it.

    No-ops when the collection is already in the requested state, so it is safe to call on
    every start-up.
    """
    wanted = normalise_kind(kind)
    info = await client.get_collection(collection)
    current = kind_of(info.config.quantization_config)
    if current == wanted:
        return False

    config = quantization_config(wanted)
    await client.update_collection(
        collection,
        # Passing None here would mean "leave it alone", not "turn it off". Going back to
        # unquantized requires the explicit Disabled sentinel.
        quantization_config=config if config is not None else models.Disabled.DISABLED,
    )
    _LOG.info(
        "collection %s quantization %s -> %s (re-encode runs in the background; "
        "wait for status=green before measuring)",
        collection,
        current,
        wanted,
    )
    return True


def search_params(
    kind: str,
    *,
    rescore: bool = True,
    oversampling: float = 2.0,
    hnsw_ef: int | None = None,
) -> models.SearchParams:
    """SearchParams carrying the quantization block for `kind`.

    Both knobs buy recall back with latency, and where the right point sits depends on the
    index -- it cannot be guessed, which is why neither is hardcoded:

    * ``rescore`` re-ranks the quantized shortlist using the original full-precision vectors.
      It costs one extra pass over the shortlist (not the index), and it is what makes int8
      viable: the compressed vectors then only need to CONTAIN the right answers, not order
      them. Turning it off is the cheap-and-lossy setting.
    * ``oversampling`` multiplies how many candidates the quantized pass fetches before
      rescoring. 2.0 means a top-10 request pulls 20 quantized candidates and re-ranks them to
      10. It repairs the recall the compression cost, at the price of a larger graph walk.

    For kind "none" the block is omitted entirely rather than sent with ``ignore=True``: the
    request must be byte-for-byte the one the unquantized baseline was measured with, so that
    a comparison between kinds is a comparison of the collection and not of the query.
    """
    resolved = normalise_kind(kind)
    if resolved == "none":
        return models.SearchParams(hnsw_ef=hnsw_ef)
    return models.SearchParams(
        hnsw_ef=hnsw_ef,
        quantization=models.QuantizationSearchParams(
            rescore=rescore,
            # Below 1.0 would mean fetching FEWER candidates than the caller asked for, which
            # Qdrant rejects with a 400. Clamping keeps a config typo from failing every
            # search; the effective value is what gets reported, so nothing is hidden.
            oversampling=max(1.0, float(oversampling)),
        ),
    )


__all__ = [
    "KINDS",
    "apply_to_collection",
    "kind_of",
    "normalise_kind",
    "quantization_config",
    "search_params",
]
