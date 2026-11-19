"""Interface contracts for the whole system.

Every module in voxrag implements a Protocol declared here and exchanges only the
pydantic models declared here. Nothing else crosses a module boundary. This file is
the single source of truth for those shapes -- change it deliberately, not casually.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------


class Lang(str, Enum):
    """The 14 Indic languages in ai4bharat/MSMARCO-XI, plus English pivot."""

    ASM = "as"
    BEN = "bn"
    GUJ = "gu"
    HIN = "hi"
    KAN = "kn"
    MAL = "ml"
    MAR = "mr"
    NEP = "ne"
    ORI = "or"
    PAN = "pa"
    SAN = "sa"
    TAM = "ta"
    TEL = "te"
    URD = "ur"
    ENG = "en"


#: HF file stem per language, e.g. "hi" -> "hin" giving train/hintrain.parquet.
LANG_FILE_STEM: dict[Lang, str] = {
    Lang.ASM: "asm",
    Lang.BEN: "ben",
    Lang.GUJ: "guj",
    Lang.HIN: "hin",
    Lang.KAN: "kan",
    Lang.MAL: "mal",
    Lang.MAR: "mar",
    Lang.NEP: "nep",
    Lang.ORI: "ori",
    Lang.PAN: "pan",
    Lang.SAN: "san",
    Lang.TAM: "tam",
    Lang.TEL: "tel",
    Lang.URD: "urd",
}

#: Unicode script block per language, used by the script-based language detector.
LANG_SCRIPT: dict[Lang, str] = {
    Lang.ASM: "Bengali",
    Lang.BEN: "Bengali",
    Lang.GUJ: "Gujarati",
    Lang.HIN: "Devanagari",
    Lang.KAN: "Kannada",
    Lang.MAL: "Malayalam",
    Lang.MAR: "Devanagari",
    Lang.NEP: "Devanagari",
    Lang.ORI: "Oriya",
    Lang.PAN: "Gurmukhi",
    Lang.SAN: "Devanagari",
    Lang.TAM: "Tamil",
    Lang.TEL: "Telugu",
    Lang.URD: "Arabic",
    Lang.ENG: "Latin",
}


class ChunkStrategyName(str, Enum):
    """Every chunking strategy the system can index with.

    The task explicitly asks for a *vast* chunking approach, so these are all real and
    all benchmarked against each other in `voxrag.chunking.evaluate`.
    """

    #: Naive token-count windows. Baseline only -- exists to be beaten.
    FIXED = "fixed"
    #: Fixed windows with configurable token overlap so answers spanning a boundary survive.
    FIXED_OVERLAP = "fixed_overlap"
    #: Sentence-aligned windows, expanded with N neighbouring sentences at retrieval time.
    SENTENCE_WINDOW = "sentence_window"
    #: Split where consecutive-sentence embedding cosine distance spikes (topic shift).
    SEMANTIC = "semantic"
    #: Recursive descent through paragraph -> sentence -> clause separators.
    RECURSIVE = "recursive"
    #: One chunk per MS MARCO passage; the corpus is already human-segmented.
    PASSAGE_NATIVE = "passage_native"
    #: Small child chunks indexed for precision, parent chunk returned for context.
    PARENT_CHILD = "parent_child"
    #: Prepends title/lang/url metadata into the embedded text so filters are semantic.
    METADATA_AWARE = "metadata_aware"


class StageName(str, Enum):
    """Pipeline stages, in canonical execution order. Every stage is timed."""

    STT = "stt"
    INPUT_GUARDRAIL = "input_guardrail"
    QUERY_EMBED = "query_embed"
    RETRIEVE_DENSE = "retrieve_dense"
    RETRIEVE_LEXICAL = "retrieve_lexical"
    FUSE = "fuse"
    RERANK = "rerank"
    GENERATE = "generate"
    OUTPUT_GUARDRAIL = "output_guardrail"
    TOTAL = "total"
    #: `TOTAL` minus `STT` -- the number the <200ms budget is measured against.
    CORE = "core"


class GenerationMode(str, Enum):
    #: Extractive + template. Deterministic, grounded by construction, <200ms budget.
    FAST = "fast"
    #: Abstractive via gemma4:e4b. Better prose, reported with its own latency profile.
    QUALITY = "quality"


class RefusalReason(str, Enum):
    OFF_TOPIC = "off_topic"
    UNSAFE = "unsafe"
    PROMPT_INJECTION = "prompt_injection"
    NO_RELEVANT_CONTEXT = "no_relevant_context"
    UNGROUNDED_ANSWER = "ungrounded_answer"
    LOW_CONFIDENCE = "low_confidence"
    EMPTY_TRANSCRIPT = "empty_transcript"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    #: Query and evidence are in different languages, and nothing could confirm the evidence
    #: answers the question. Distinct from OFF_TOPIC on purpose: OFF_TOPIC is a judgement that
    #: the passages are about something else, this one is the absence of a judgement.
    UNVERIFIED_CROSS_LANGUAGE = "unverified_cross_language"


# --------------------------------------------------------------------------------------
# Core data models
# --------------------------------------------------------------------------------------


class Document(BaseModel):
    """A source record from MSMARCO-XI, before chunking."""

    doc_id: str
    text: str
    lang: Lang
    title: str | None = None
    url: str | None = None
    #: The dataset's own `query` field -- retained as a pseudo-title signal.
    source_query: str | None = None
    #: The dataset's own `Answer` field -- gold answer, used for eval, never for serving.
    gold_answer: str | None = None
    #: `is_selected` in MS MARCO: whether this passage answers `source_query`.
    is_selected: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """An indexable unit. `text` is what gets embedded; `context_text` is what the
    generator sees (identical except for window/parent strategies)."""

    chunk_id: str
    doc_id: str
    text: str
    lang: Lang
    strategy: ChunkStrategyName
    #: Position of this chunk within its parent document, 0-based.
    ordinal: int = 0
    #: Character offsets into `Document.text`, for citation highlighting.
    char_start: int = 0
    char_end: int = 0
    n_tokens: int = 0
    title: str | None = None
    url: str | None = None
    #: Wider text handed to the generator. Falls back to `text` when unset.
    context_text: str | None = None
    #: For PARENT_CHILD: the chunk_id of the parent whose text should be returned.
    parent_id: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def generation_text(self) -> str:
        return self.context_text or self.text


class ScoredChunk(BaseModel):
    """A chunk with retrieval provenance attached."""

    chunk: Chunk
    #: Fused final score, higher is better. Comparable only within one result set.
    score: float
    #: Raw dense cosine similarity, if the chunk was found densely.
    dense_score: float | None = None
    #: Raw BM25 score, if the chunk was found lexically.
    lexical_score: float | None = None
    #: Cross-encoder score, if reranking ran.
    rerank_score: float | None = None
    #: 1-based rank in each retriever, for RRF diagnostics.
    dense_rank: int | None = None
    lexical_rank: int | None = None


class Citation(BaseModel):
    """A pointer from a span of the answer back to the chunk that supports it."""

    chunk_id: str
    doc_id: str
    #: Offsets into the *answer* text that this citation supports.
    answer_start: int
    answer_end: int
    #: Offsets into the *chunk* text that provide the support.
    source_start: int
    source_end: int
    quote: str


class StageTiming(BaseModel):
    """Wall-clock for one stage. Milliseconds, monotonic clock."""

    stage: StageName
    ms: float
    #: Set when the stage was skipped (e.g. rerank disabled, STT bypassed for text input).
    skipped: bool = False
    #: Free-form stage detail: token counts, candidate counts, cache hits.
    detail: dict[str, Any] = Field(default_factory=dict)


class Transcript(BaseModel):
    """Output of the STT stage."""

    text: str
    lang: Lang | None = None
    #: Provider's own confidence in [0,1], when it reports one.
    confidence: float | None = None
    provider: str = "unknown"
    duration_s: float | None = None
    #: True when a fallback provider served this after the primary failed.
    from_fallback: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)


class GuardrailVerdict(BaseModel):
    """Result of one guardrail check. `allowed=False` short-circuits the pipeline."""

    name: str
    allowed: bool
    reason: RefusalReason | None = None
    #: Confidence that the block is correct, in [0,1].
    score: float = 0.0
    #: Human-readable explanation, surfaced in the UI and the trace.
    detail: str = ""
    #: A guardrail may rewrite the text it inspected (e.g. PII redaction).
    rewritten_text: str | None = None
    #: The rail returned a verdict but its evidence was missing -- a dependency it needed was
    #: down, too slow, or unreadable. `run_rails` turns this into a `trace.degrade`, which is
    #: what puts it in the artifact. Without it a fail-CLOSED rail refusing because nothing
    #: could check the pair is indistinguishable, in every stored run, from one that checked
    #: and decided: measured, three cold cross-language queries refused at ~1.53s each with
    #: `degradations: null` while ollama never loaded the model at all.
    degraded: bool = False


class Answer(BaseModel):
    """The generated answer plus everything needed to audit it."""

    text: str
    mode: GenerationMode
    citations: list[Citation] = Field(default_factory=list)
    #: Fraction of answer tokens covered by at least one citation, in [0,1].
    groundedness: float = 0.0
    #: Set when the system declined to answer. `text` then holds the refusal message.
    refused: bool = False
    refusal_reason: RefusalReason | None = None
    model: str = ""
    lang: Lang | None = None


class PipelineResult(BaseModel):
    """The complete, auditable record of one voice query. This is the API response."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    transcript: Transcript | None = None
    query_text: str
    detected_lang: Lang | None = None
    answer: Answer
    retrieved: list[ScoredChunk] = Field(default_factory=list)
    guardrails: list[GuardrailVerdict] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    #: Populated when a stage failed but the harness recovered.
    degradations: list[str] = Field(default_factory=list)
    #: Number of retries the harness performed across all stages.
    retries: int = 0

    def ms(self, stage: StageName) -> float:
        for t in self.timings:
            if t.stage is stage:
                return t.ms
        return 0.0

    @property
    def core_ms(self) -> float:
        """Latency of everything except speech-to-text. The <200ms budget."""
        return self.ms(StageName.CORE)


# --------------------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------------------


@runtime_checkable
class STTProvider(Protocol):
    """Speech-to-text. Implementations: sarvam (primary), elevenlabs, local whisper."""

    name: str

    async def transcribe(
        self, audio: bytes, *, mime_type: str = "audio/wav", lang_hint: Lang | None = None
    ) -> Transcript: ...

    async def health(self) -> bool: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a Document into Chunks. Must be deterministic and side-effect free."""

    name: ChunkStrategyName

    def chunk(self, doc: Document) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    """Dense text encoder. `dim` must match the vector store's configured dimension."""

    name: str
    dim: int

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        """Returns float32 array of shape (len(texts), dim), L2-normalised."""
        ...

    async def encode_async(self, texts: list[str], *, is_query: bool = False) -> np.ndarray: ...


@runtime_checkable
class VectorStore(Protocol):
    """Dense ANN index. Implementations: qdrant (default), hnswlib (in-process)."""

    name: str

    async def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    async def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 20,
        lang: Lang | None = None,
        strategy: ChunkStrategyName | None = None,
    ) -> list[ScoredChunk]: ...

    async def count(self) -> int: ...

    async def health(self) -> bool: ...


@runtime_checkable
class LexicalIndex(Protocol):
    """Sparse/keyword retrieval, script-aware for Indic text."""

    name: str

    def index(self, chunks: list[Chunk]) -> None: ...

    def search(
        self, query: str, *, top_k: int = 20, lang: Lang | None = None
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    """Optional precision stage between fusion and generation."""

    name: str

    def rerank(self, query: str, candidates: list[ScoredChunk], *, top_k: int) -> list[ScoredChunk]: ...


@runtime_checkable
class Generator(Protocol):
    """Turns query + retrieved context into a cited Answer."""

    name: str
    mode: GenerationMode

    async def generate(
        self, query: str, context: list[ScoredChunk], *, lang: Lang | None = None
    ) -> Answer: ...

    async def health(self) -> bool: ...


@runtime_checkable
class Guardrail(Protocol):
    """A single check. Input rails see the query; output rails see query+answer+context."""

    name: str
    #: "input" or "output".
    phase: str

    async def check(
        self,
        *,
        query: str,
        answer: Answer | None = None,
        context: list[ScoredChunk] | None = None,
        lang: Lang | None = None,
    ) -> GuardrailVerdict: ...


__all__ = [
    "LANG_FILE_STEM",
    "LANG_SCRIPT",
    "Answer",
    "Chunk",
    "ChunkStrategyName",
    "Chunker",
    "Citation",
    "Document",
    "Embedder",
    "GenerationMode",
    "Generator",
    "Guardrail",
    "GuardrailVerdict",
    "Lang",
    "LexicalIndex",
    "PipelineResult",
    "RefusalReason",
    "Reranker",
    "STTProvider",
    "ScoredChunk",
    "StageName",
    "StageTiming",
    "Transcript",
    "VectorStore",
]
