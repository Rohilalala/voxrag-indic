"""Abstractive answer generation over Ollama's /api/chat -- the QUALITY path.

Same wire format, two possible targets: a local Ollama daemon holding `settings.ollama_model`,
or `settings.hosted_generator_url` when one is configured, because the deployed box cannot
hold a 9.6GB model. The switch happens once, in `__init__`; nothing downstream knows or cares.

This module trusts the model for prose and for nothing else:

* **Citations** are parsed out of the model's own ``[n]`` markers and re-resolved against the
  context list by position, then pointed at the specific sentence of the chunk that supports
  the claim. A marker naming a context that was not supplied is dropped, not invented.
* **Groundedness** is measured here, by lexical coverage of the answer against the context
  that was actually sent. A self-reported confidence from a 4B model is not evidence.
* **Health** confirms the model is *pulled*, not merely that a daemon is listening. A
  generator that reports healthy and then 404s on an unpulled model is worse than one that
  reports unhealthy, because the harness would route real traffic into it.

Everything that can fail here does so as `GenerationError`, which is what
`generate.generate_with_fallback` catches to degrade to the extractive path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import regex

from voxrag.chunking.base import split_sentences
from voxrag.config import settings
from voxrag.contracts import (
    Answer,
    Citation,
    GenerationMode,
    Lang,
    RefusalReason,
    ScoredChunk,
)
from voxrag.generate.extractive import best_supporting_span, content_terms
from voxrag.generate.prompts import build_messages

# --------------------------------------------------------------------------------------
# Module-level constants (not in config.py -- they describe this file's own mechanics)
# --------------------------------------------------------------------------------------

#: Connect timeout, kept far below `settings.ollama_timeout_s`. The read timeout has to
#: cover a 4B model's decode; a *connect* that takes more than this means nothing is
#: listening, and the fallback to the extractive path must not wait 30s to find that out.
_CONNECT_TIMEOUT_S = 1.5

#: /api/tags is a directory listing; if it has not answered in this long the daemon is gone.
_HEALTH_TIMEOUT_S = 2.0

#: Health is cached for this long. `generate_with_fallback` may consult it on every request
#: and an uncached probe would add a round trip to each one; 10s is short enough that a
#: daemon dying mid-demo is noticed within one question.
_HEALTH_TTL_S = 10.0

_LOG = logging.getLogger(__name__)

#: Matches [1], [2,3] and [1; 2]. Used both to resolve citations and to strip the markers
#: before measuring groundedness -- otherwise the digits count as ungrounded answer content.
_CITATION_RE = regex.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")
_CITATION_SPLIT_RE = regex.compile(r"[,;]")

#: Token budget for `entails`. The reply is one word; anything beyond that is the model
#: ignoring the instruction, and a wider budget would only buy a longer thing to reject.
#: `settings.ollama_think` is False, so no part of this is spent on hidden reasoning.
_ENTAILMENT_MAX_TOKENS = 4

#: Per-passage character cap in the entailment prompt. Untuned, and stated as such: nothing
#: measured 600. The question asked is "does this evidence answer that question", which the
#: opening of a passage settles, and the cap bounds prompt size across `context_top_k`
#: passages so the verifier stays cheap enough to sit in front of an answer.
_ENTAILMENT_MAX_PASSAGE_CHARS = 600

#: A verdict is the FIRST word, not a word found anywhere: "the answer is yes" is a model that
#: did not follow a one-word instruction, and `CrossLanguageRail` refuses on an unparseable
#: reply. Being strict here therefore fails closed, which is the safe direction.
_VERDICT_RE = regex.compile(r"^\W*(yes|no)\b", regex.IGNORECASE)

#: Sentence-final punctuation stripped before comparing a reply against the refusal message.
#: The danda and the Urdu full stop are here for the same reason chunking/base.py knows about
#: them: a model that completes the refusal line in an Indic script terminates it with those,
#: not with ".", and the refusal would then not compare equal.
_TRAILING_PUNCT = " .।۔"  # noqa: RUF001 -- these ARE the Indic terminators, not typos


class GenerationError(RuntimeError):
    """The remote generator could not produce an answer. Callers degrade rather than fail."""


def _normalise_model(name: str) -> str:
    """Ollama reports `gemma4:e4b`; a request for `gemma4` means `gemma4:latest`."""
    name = name.strip()
    return name if ":" in name else f"{name}:latest"


def _extract_content(payload: dict[str, Any]) -> str:
    """Pull assistant text out of a non-streamed response.

    Three shapes are accepted: Ollama /api/chat (`message.content`), Ollama /api/generate
    (`response`), and the OpenAI-compatible shape some hosted proxies put in front of the
    same model. Cheap insurance -- `hosted_generator_url` points at someone else's box.
    """
    message = payload.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(payload.get("response"), str):
        return payload["response"]
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            for key in ("message", "delta"):
                part = first.get(key)
                if isinstance(part, dict) and isinstance(part.get("content"), str):
                    return part["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    return ""


def _extract_delta(payload: dict[str, Any]) -> str:
    """Token delta from one streamed NDJSON/SSE frame. Same shapes as `_extract_content`."""
    return _extract_content(payload)


def groundedness(answer_text: str, context: Sequence[ScoredChunk]) -> float:
    """Fraction of the answer's content tokens that occur somewhere in the given context.

    Occurrence-weighted, not set-weighted: an answer that repeats one grounded phrase and
    then invents a clause should be penalised for the clause, and a set-based score would
    hide it behind the repetition. Citation markers are stripped first so their digits are
    not counted as ungrounded content.
    """
    tokens = content_terms(_CITATION_RE.sub(" ", answer_text))
    if not tokens:
        return 0.0
    supported: set[str] = set()
    for sc in context:
        supported.update(content_terms(sc.chunk.generation_text))
    return sum(1 for t in tokens if t in supported) / len(tokens)


def parse_citations(
    answer_text: str, context: Sequence[ScoredChunk], lang: Lang | None = None
) -> list[Citation]:
    """Resolve the model's ``[n]`` markers onto chunk ids, one citation per claim sentence.

    The answer is segmented with the same Indic-aware splitter used for chunking, so a Hindi
    answer yields per-sentence citations rather than one span covering the whole paragraph.
    Each citation's source offsets point at the sentence *inside* the chunk that best
    supports that claim, found by term overlap -- pointing at the whole chunk would be
    technically true and useless for highlighting.
    """
    citations: list[Citation] = []
    seen: set[tuple[str, int, int]] = set()
    for sentence in split_sentences(answer_text, lang):
        claim = sentence.text
        indices: list[int] = []
        for match in _CITATION_RE.finditer(claim):
            for raw in _CITATION_SPLIT_RE.split(match.group(1)):
                raw = raw.strip()
                if raw.isdigit():
                    n = int(raw)
                    # Markers beyond the supplied context are model noise; drop them rather
                    # than resolve them to whatever happens to sit at that position.
                    if 1 <= n <= len(context):
                        indices.append(n)
        terms = content_terms(_CITATION_RE.sub(" ", claim))
        for n in dict.fromkeys(indices):
            chunk = context[n - 1].chunk
            source = chunk.generation_text
            span = best_supporting_span(terms, source, chunk.lang)
            if span is None:
                continue
            key = (chunk.chunk_id, sentence.start, span.start)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    answer_start=sentence.start,
                    answer_end=sentence.end,
                    source_start=span.start,
                    source_end=span.end,
                    quote=span.text,
                )
            )
    return citations


def _entailment_messages(query: str, context: Sequence[ScoredChunk]) -> list[dict[str, str]]:
    """Prompt for a one-word answerability verdict.

    Deliberately not `prompts.build_messages`: that prompt asks for a cited answer *in the
    query's language* and instructs the model to emit `settings.refusal_message` when it
    cannot -- so a model that judged the evidence insufficient would reply with a refusal
    sentence, not with NO, and the caller would read that as unparseable. This asks one
    question and constrains the reply to one token.
    """
    passages = "\n\n".join(
        f"[{n}] {hit.chunk.generation_text[:_ENTAILMENT_MAX_PASSAGE_CHARS]}"
        for n, hit in enumerate(context, 1)
    )
    return [
        {
            "role": "system",
            "content": (
                "You verify whether evidence answers a question. The evidence is often in a "
                "different language from the question; judge the meaning, not the language. "
                "Reply with exactly one word: YES or NO. Never explain."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\nEvidence:\n{passages}\n\n"
                "Does the evidence contain the information needed to answer the question? "
                "Answer YES or NO."
            ),
        },
    ]


class OllamaGenerator:
    """`contracts.Generator` in QUALITY mode, plus a token stream for the websocket route."""

    mode = GenerationMode.QUALITY

    def __init__(self) -> None:
        hosted = (settings.hosted_generator_url or "").strip()
        self._hosted = bool(hosted)
        self._base = (hosted or settings.ollama_base_url).rstrip("/")
        self._model = settings.hosted_generator_model if self._hosted else settings.ollama_model
        self.name = f"{'hosted' if self._hosted else 'ollama'}:{self._model}"
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._health: tuple[float, bool] | None = None
        #: Detached model load started by `_schedule_preload`. Held so it is not garbage
        #: collected mid-flight, and so a second timeout does not start a second one.
        self._preload: asyncio.Task[None] | None = None

    # -- transport ----------------------------------------------------------------------

    def _url(self, path: str) -> str:
        # A hosted URL may already be the full endpoint rather than a base; appending blindly
        # would produce .../api/chat/api/chat and a 404 on every request.
        return self._base if self._base.endswith(path) else self._base + path

    async def _get_client(self) -> httpx.AsyncClient:
        """One keep-alive client for the process.

        Built lazily and inside the running loop, so importing this module never opens a
        socket and never binds a client to a loop that is about to be replaced. Keep-alive
        matters: a fresh TCP handshake per token stream is pure added first-token latency.
        """
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                if self._client is None or self._client.is_closed:
                    headers = {"Content-Type": "application/json"}
                    if self._hosted and settings.hosted_generator_key:
                        headers["Authorization"] = f"Bearer {settings.hosted_generator_key}"
                    self._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(
                            settings.ollama_timeout_s, connect=_CONNECT_TIMEOUT_S
                        ),
                        headers=headers,
                        limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
                    )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _payload(
        self, query: str, context: Sequence[ScoredChunk], lang: Lang | None, *, stream: bool
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": build_messages(query, context, lang),
            "stream": stream,
            # Reasoning models route their chain-of-thought into a separate `thinking` field
            # and will happily spend the entire num_predict budget there, returning an EMPTY
            # `content`. Ollama ignores this key for models that do not support it, so it is
            # safe to send unconditionally. See settings.ollama_think for the measurements.
            "think": settings.ollama_think,
            "options": {
                "temperature": settings.generate_temperature,
                "num_predict": settings.generate_max_tokens,
            },
        }

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        """4xx means our request is wrong (bad model, bad auth); retrying cannot fix it."""
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            return code >= 500 or code in (408, 429)
        return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))

    async def _backoff(self, attempt: int) -> None:
        # Exponential, from settings.retry_backoff_ms. Bounded by settings.max_retries, so
        # the worst case is a fixed, budgetable delay rather than an open-ended retry storm.
        await asyncio.sleep(settings.retry_backoff_ms * (2**attempt) / 1000.0)

    async def _request_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(settings.max_retries + 1):
            try:
                client = await self._get_client()
                response = await client.post(self._url(path), json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:  # ValueError covers a non-JSON body
                last = exc
                if not self._retryable(exc) or attempt == settings.max_retries:
                    break
                await self._backoff(attempt)
        raise GenerationError(f"{self.name} {path} failed: {last}") from last

    # -- Generator protocol -------------------------------------------------------------

    async def generate(
        self, query: str, context: list[ScoredChunk], *, lang: Lang | None = None
    ) -> Answer:
        ctx = list(context)[: settings.context_top_k]
        if not ctx:
            # No context means the only honest answer is the refusal, and calling a 4B model
            # to have it say so costs seconds.
            return self._refuse(lang)

        data = await self._request_json("/api/chat", self._payload(query, ctx, lang, stream=False))
        text = _extract_content(data).strip()
        if not text:
            # Name the actual cause instead of "empty message". A reasoning model that used
            # its whole budget on chain-of-thought looks identical to a dead daemon from the
            # caller's side, and that ambiguity cost real debugging time here.
            thinking = (data.get("message") or {}).get("thinking") or ""
            if thinking:
                raise GenerationError(
                    f"{self.name} produced {len(thinking)} chars of reasoning and no answer "
                    f"(done_reason={data.get('done_reason')!r}, "
                    f"tokens={data.get('eval_count')}). Set VOXRAG_OLLAMA_THINK=false or raise "
                    "VOXRAG_GENERATE_MAX_TOKENS."
                )
            raise GenerationError(
                f"{self.name} returned an empty message (done_reason={data.get('done_reason')!r})"
            )
        return self.finalise(text, ctx, lang)

    async def stream(
        self, query: str, context: list[ScoredChunk], lang: Lang | None = None
    ) -> AsyncIterator[str]:
        """Yield token deltas as they arrive. Consumed by the websocket route."""
        ctx = list(context)[: settings.context_top_k]
        if not ctx:
            yield settings.refusal_message
            return

        payload = self._payload(query, ctx, lang, stream=True)
        emitted = False
        last: Exception | None = None
        for attempt in range(settings.max_retries + 1):
            try:
                client = await self._get_client()
                async with client.stream("POST", self._url("/api/chat"), json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        frame = line.strip()
                        if not frame:
                            continue
                        if frame.startswith("data:"):  # SSE-wrapping proxies
                            frame = frame[5:].strip()
                        if frame == "[DONE]":
                            return
                        try:
                            obj = json.loads(frame)
                        except json.JSONDecodeError:
                            continue  # keep-alive comment or a partial frame
                        if not isinstance(obj, dict):
                            continue
                        piece = _extract_delta(obj)
                        if piece:
                            emitted = True
                            yield piece
                        if obj.get("done") is True:
                            return
                return
            except (httpx.HTTPError, GenerationError) as exc:
                last = exc
                # Once bytes are on the wire a retry would duplicate the answer's prefix, so
                # a mid-stream failure is terminal; only a failure before the first token
                # can be safely retried.
                if emitted or not self._retryable(exc) or attempt == settings.max_retries:
                    break
                await self._backoff(attempt)
        raise GenerationError(f"{self.name} stream failed: {last}") from last

    async def health(self) -> bool:
        now = time.monotonic()
        cached = self._health
        if cached is not None and now - cached[0] < _HEALTH_TTL_S:
            return cached[1]
        ok = await self._probe()
        self._health = (now, ok)
        return ok

    # -- entailment, for guardrails.cross_language ----------------------------------------

    async def entails(self, query: str, context: Sequence[ScoredChunk]) -> bool | None:
        """Does this evidence answer this question? `None` means the model could not say.

        `CrossLanguageRail` turns `None` into a REFUSAL, so every unclear outcome has to
        arrive here as `None` rather than as a guess: no daemon, an HTTP failure, an empty
        reply, or a reply that is not a bare YES/NO. That is why this does not raise
        `GenerationError` the way `generate` does -- the caller has nothing to degrade to, and
        an exception and a `None` would mean the same refusal anyway.

        Cheap by construction: temperature 0 (a verdict is not creative writing),
        `_ENTAILMENT_MAX_TOKENS` of budget, truncated passages, and the same keep-alive
        client, retry policy and cached health probe the answer path already pays for.
        """
        ctx = list(context)[: settings.context_top_k]
        if not ctx or not query.strip():
            return None
        # Cached for _HEALTH_TTL_S, so once the daemon is known to be down this costs nothing
        # and converts a connect timeout per request into an immediate "cannot decide".
        if not await self.health():
            return None

        payload = {
            "model": self._model,
            "messages": _entailment_messages(query, ctx),
            "stream": False,
            # Same landmine as the answer path: with thinking on, a reasoning model spends the
            # whole num_predict budget on hidden tokens and returns empty content -- which
            # here would refuse every cross-language pair. See settings.ollama_think.
            "think": settings.ollama_think,
            "options": {"temperature": 0.0, "num_predict": _ENTAILMENT_MAX_TOKENS},
        }
        try:
            data = await self._request_json("/api/chat", payload)
        except GenerationError:
            return None
        except asyncio.CancelledError:
            # The caller's ceiling fired. Cancelling the HTTP request aborts the model LOAD
            # with it, which is why a cold verifier could never become warm: measured against
            # the live daemon, three consecutive cross-language queries each refused after
            # ~1.53s with "the verifier did not answer within 1500ms" and `ollama ps` was
            # still empty afterwards. The ceiling is deliberately below a cold load (6089.7ms
            # measured here, vs 267-385ms warm), so the fix is not a bigger ceiling -- it is
            # finishing the load in the background so the NEXT query is the warm one. That is
            # what config.cross_language_timeout_ms already claims this design costs.
            self._schedule_preload()
            raise
        match = _VERDICT_RE.match(_extract_content(data).strip())
        if match is None:
            return None
        return match.group(1).casefold() == "yes"

    def _schedule_preload(self) -> None:
        """Finish the aborted load in the background, detached from the cancelled request.

        At most one in flight: ollama serialises loads of the same model anyway, so a second
        task would only queue behind the first while holding another connection.
        """
        if self._preload is not None and not self._preload.done():
            return
        self._preload = asyncio.create_task(self.warm())

    async def warm(self) -> bool:
        """Load the model, waiting as long as it takes. Returns whether it is now resident.

        An empty-prompt /api/generate is ollama's own way to load a model and produce
        nothing. Two callers, one need: `Harness.warmup` at startup, because nothing there
        loaded gemma4 (it resolves settings.default_mode, which is FAST -- the extractive
        generator -- and `generate.warmup` only asks /api/tags whether the model exists); and
        `_schedule_preload` after a verifier timeout, so an eviction costs one refusal rather
        than every refusal. Unbounded on purpose in both: neither caller is holding a user's
        answer, and a bounded load is exactly the thing that could never finish.
        """
        try:
            await self._request_json("/api/generate", {"model": self._model, "stream": False})
        except Exception as exc:  # a warm-up must never surface as a request error
            _LOG.warning("%s could not be preloaded: %r", self.name, exc)
            return False
        return True

    # -- internals ----------------------------------------------------------------------

    async def _probe(self) -> bool:
        """Healthy means: the daemon answers AND it has actually pulled the model."""
        try:
            client = await self._get_client()
            response = await client.get(self._url("/api/tags"), timeout=_HEALTH_TIMEOUT_S)
        except (httpx.HTTPError, OSError):
            return False
        if response.status_code >= 500:
            return False
        if response.status_code >= 400:
            # A hosted proxy need not implement /api/tags at all, and it answered, so the box
            # is up. We cannot verify the pull there; refusing to use a reachable hosted model
            # over a missing introspection endpoint would be the worse error.
            return self._hosted
        try:
            models = response.json().get("models")
        except ValueError:
            return self._hosted
        if not isinstance(models, list) or not models:
            return self._hosted
        available = {
            _normalise_model(str(m.get("model") or m.get("name") or ""))
            for m in models
            if isinstance(m, dict)
        }
        return _normalise_model(self._model) in available

    def finalise(self, text: str, context: Sequence[ScoredChunk], lang: Lang | None) -> Answer:
        """Wrap raw model text into an audited Answer. Shared with the streaming caller."""
        refusal = settings.refusal_message.strip()
        # The prompt asks for the refusal line verbatim; models still add a trailing period
        # or a stray marker, so compare on the stripped prefix rather than for equality.
        if text.startswith(refusal) or text.rstrip(_TRAILING_PUNCT) == refusal.rstrip(
            _TRAILING_PUNCT
        ):
            return self._refuse(lang)
        return Answer(
            text=text,
            mode=GenerationMode.QUALITY,
            citations=parse_citations(text, context, lang),
            groundedness=groundedness(text, context),
            model=self._model,
            lang=lang or (context[0].chunk.lang if context else None),
        )

    def _refuse(self, lang: Lang | None) -> Answer:
        return Answer(
            text=settings.refusal_message,
            mode=GenerationMode.QUALITY,
            groundedness=0.0,
            refused=True,
            refusal_reason=RefusalReason.NO_RELEVANT_CONTEXT,
            model=self._model,
            lang=lang,
        )


__all__ = ["GenerationError", "OllamaGenerator", "groundedness", "parse_citations"]
