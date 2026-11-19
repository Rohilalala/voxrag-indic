"""The system prompt and message assembly for the abstractive (QUALITY) generator.

Three properties this file exists to guarantee, in priority order:

1. **Answer only from the numbered context.** Everything else in the pipeline -- retrieval,
   the groundedness score, the output guardrail -- is wasted if the model is free to answer
   from its own weights.
2. **Answer in the language of the question.** The corpus is 14 Indic languages. A 4B model
   asked a Hindi question over Hindi context will happily reply in English unless told not
   to, and an English answer to a Hindi question is a failed answer, not a partial one. The
   instruction names the target language in *its own script*, which is a far stronger signal
   to a multilingual model than the English exonym alone.
3. **Refuse verbatim.** The refusal must be byte-identical to ``settings.refusal_message`` so
   the caller can detect it by comparison instead of by classifying free prose.

Every token here is prefill latency on the QUALITY path, so: no persona, no few-shot
examples, no politeness. The language instruction is deliberately the *last* line of the
system prompt -- the 15 per-language variants then share one identical prefix, which is what
an inference server's KV prefix cache can actually reuse across requests.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from voxrag.chunking.base import truncate_to_tokens
from voxrag.config import settings
from voxrag.contracts import Lang, ScoredChunk

# --------------------------------------------------------------------------------------
# Module-level constants (not in config.py -- they describe this file's own mechanics)
# --------------------------------------------------------------------------------------

#: Per-chunk context ceiling, in encoder tokens. The largest chunk any strategy produces is
#: PARENT_CHILD's parent (settings.parent_chunk_tokens = 384), so this truncates nothing in
#: the default configuration and bounds prefill for a strategy that emits something larger.
MAX_CONTEXT_TOKENS_PER_CHUNK = 384

#: MS MARCO titles are occasionally a whole sentence. A title is a retrieval hint, not
#: context, so it gets a hard character cap rather than a token budget.
_MAX_TITLE_CHARS = 80

#: Language name in English plus the endonym in its own script. The script is the part that
#: actually steers a multilingual decoder; the English name disambiguates for the tokeniser.
LANGUAGE_NAMES: dict[Lang, str] = {
    Lang.ASM: "Assamese (অসমীয়া)",
    Lang.BEN: "Bengali (বাংলা)",
    Lang.GUJ: "Gujarati (ગુજરાતી)",
    Lang.HIN: "Hindi (हिन्दी)",
    Lang.KAN: "Kannada (ಕನ್ನಡ)",
    Lang.MAL: "Malayalam (മലയാളം)",
    Lang.MAR: "Marathi (मराठी)",
    Lang.NEP: "Nepali (नेपाली)",
    Lang.ORI: "Odia (ଓଡ଼ିଆ)",
    Lang.PAN: "Punjabi (ਪੰਜਾਬੀ)",
    Lang.SAN: "Sanskrit (संस्कृतम्)",
    Lang.TAM: "Tamil (தமிழ்)",
    Lang.TEL: "Telugu (తెలుగు)",
    Lang.URD: "Urdu (اردو)",
    Lang.ENG: "English",
}

#: Constant across every request and every language -- see the prefix-cache note above.
_RULES = """You answer questions from a retrieved passage set. Obey every rule.

1. Use only facts stated in the numbered CONTEXT. Never use outside knowledge.
2. End every sentence with the bracketed number of the context it came from, like [1] or [2].
3. Answer in at most three sentences. No preamble, no restating the question.
4. If the CONTEXT does not answer the question, reply with exactly this line and nothing else:
{refusal}"""


@lru_cache(maxsize=len(Lang) + 1)
def system_prompt(lang: Lang | None = None) -> str:
    """The system message for one target language. Cached: it is the same string forever."""
    named = LANGUAGE_NAMES.get(lang) if lang is not None else None
    tail = (
        f"5. Write your answer in {named}. Do not answer in any other language."
        if named
        else "5. Write your answer in the same language and script as the question. "
        "Do not translate it."
    )
    return f"{_RULES.format(refusal=settings.refusal_message)}\n{tail}"


def format_context(
    context: Sequence[ScoredChunk], *, max_tokens_per_chunk: int = MAX_CONTEXT_TOKENS_PER_CHUNK
) -> str:
    """Render retrieved chunks as the numbered block the citation rule refers to.

    Numbering is 1-based and positional: ``[n]`` in the model's output maps back to
    ``context[n - 1]``, which is how `ollama_gemma._parse_citations` resolves chunk ids.
    """
    blocks: list[str] = []
    for i, sc in enumerate(context, start=1):
        body = truncate_to_tokens(sc.chunk.generation_text.strip(), max_tokens_per_chunk)
        title = (sc.chunk.title or "").strip()[:_MAX_TITLE_CHARS]
        header = f"[{i}] {title}" if title else f"[{i}]"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def build_messages(
    query: str, context: Sequence[ScoredChunk], lang: Lang | None = None
) -> list[dict[str, str]]:
    """The /api/chat message list: stable system turn, per-request user turn.

    The context lives in the *user* turn on purpose. Servers that cache a KV prefix key it on
    the leading messages, so a per-request system turn would defeat the cache entirely.
    """
    return [
        {"role": "system", "content": system_prompt(lang)},
        {"role": "user", "content": f"CONTEXT:\n{format_context(context)}\n\nQUESTION: {query}"},
    ]


__all__ = [
    "LANGUAGE_NAMES",
    "MAX_CONTEXT_TOKENS_PER_CHUNK",
    "build_messages",
    "format_context",
    "system_prompt",
]
