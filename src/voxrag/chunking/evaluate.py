# ruff: noqa: RUF001 -- the demo passages below are real Urdu, Hindi, Tamil and Bengali;
# the "ambiguous" characters flagged in them are that text's own punctuation.
"""Benchmark every chunking strategy on the same document sample.

    python -m voxrag.chunking.evaluate --docs data/docs.jsonl --sample 500

The table this prints is the evidence behind `settings.active_strategy`. Six things are
measured, because "which chunker is best" is not one question:

* **size distribution** (mean / p50 / p95 tokens) -- whether chunks actually land near the
  budget, and `>enc` counts the ones the encoder would truncate at `embed_max_tokens`;
* **fragmentation** -- share of chunks that end mid-sentence. A chunk cut mid-clause both
  embeds badly and reads badly when quoted back as a citation;
* **redundancy** -- share of duplicated token 5-grams within a document. This is the cost
  of overlap: index weight spent storing the same words twice;
* **coverage** -- share of the document's non-whitespace characters reachable from some
  chunk. Anything below 100% means the strategy is silently dropping corpus text;
* **index cost** -- vectors, their bytes at `embed_dim` float32, and the tokens that must
  be pushed through the encoder to build the index;
* **chunking latency** -- ms per document, which is the "chunking" half of the pipeline's
  latency budget.

No metric decides alone. Fragmentation is minimised by never splitting (PASSAGE_NATIVE),
redundancy by never overlapping, index cost by emitting one huge chunk -- the point of
printing them together is that the tradeoffs are visible instead of asserted.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from voxrag.chunking import REGISTRY, get_chunker
from voxrag.chunking.base import (
    embed_budget_tokens,
    split_sentences,
    token_index,
    tokenizer_available,
)
from voxrag.config import settings
from voxrag.contracts import Chunk, ChunkStrategyName, Document, Lang
from voxrag.telemetry import percentile

#: Width of the duplicate-detection n-gram. 5 encoder tokens is roughly a clause: short
#: enough to catch real overlap, long enough that ordinary function-word repetition
#: ("of the", "में से") does not register as redundancy.
NGRAM = 5

#: Bytes of index per vector, at float32. Payload bytes are measured, not estimated.
_BYTES_PER_FLOAT = 4

#: Columns the comparison table needs before rich starts truncating headings.
_MIN_TABLE_WIDTH = 116

_TEXT_KEYS = ("text", "passage", "passage_text", "body", "content", "chunk")
_ID_KEYS = ("doc_id", "id", "_id", "pid", "passage_id", "docid")
_LANG_KEYS = ("lang", "language", "lang_code", "locale")

#: Synthetic smoke-test corpus for `--demo`. Six passages in four scripts, sized above
#: `chunk_target_tokens` so every strategy actually splits, and covering the segmentation
#: edge cases: danda and double danda, the Arabic full stop and question mark, Latin
#: abbreviations and decimals, and one passage with no terminal punctuation at all.
#: NOT the MSMARCO-XI corpus -- numbers from it are a check on the code, never a result.
_DEMO_DOCS: list[dict[str, Any]] = [
    {
        "doc_id": "demo-hi-1",
        "lang": "hi",
        "title": "माउंट एवरेस्ट",
        "source_query": "माउंट एवरेस्ट की ऊंचाई कितनी है",
        "text": (
            "माउंट एवरेस्ट पृथ्वी का सबसे ऊँचा पर्वत है। इसकी ऊँचाई समुद्र तल से 8,848.86 मीटर मापी गई है। "
            "यह नेपाल और तिब्बत की सीमा पर स्थित है और महालंगूर हिमाल श्रृंखला का हिस्सा है। "
            "नेपाली में इसे सगरमाथा और तिब्बती में चोमोलुंगमा कहा जाता है। "
            "1953 में एडमंड हिलेरी और तेनजिंग नोर्गे इसकी चोटी पर पहुँचने वाले पहले व्यक्ति बने। "
            "हर साल सैकड़ों पर्वतारोही इसे चढ़ने का प्रयास करते हैं, लेकिन मौसम और ऑक्सीजन की कमी सबसे बड़ी चुनौती बनी रहती है। "
            "आठ हजार मीटर से ऊपर के क्षेत्र को मृत्यु क्षेत्र कहा जाता है, जहाँ शरीर धीरे-धीरे क्षय होने लगता है। "
            "चढ़ाई का मुख्य मौसम अप्रैल और मई में होता है, जब हवाएँ अपेक्षाकृत शांत रहती हैं। "
            "नेपाल सरकार हर पर्वतारोही से परमिट शुल्क लेती है, जो देश की आय का एक बड़ा स्रोत है। "
            "हाल के वर्षों में चोटी पर जमा कचरा एक गंभीर पर्यावरणीय समस्या बन गया है॥"
        ),
    },
    {
        "doc_id": "demo-ta-1",
        "lang": "ta",
        "title": "நீரிழிவு நோய்",
        "source_query": "நீரிழிவு நோய் அறிகுறிகள் என்ன",
        "text": (
            "நீரிழிவு நோய் என்பது இரத்தத்தில் சர்க்கரையின் அளவு அதிகரிக்கும் ஒரு நிலை. "
            "அடிக்கடி சிறுநீர் கழித்தல், அதிக தாகம், விவரிக்க முடியாத எடை இழப்பு ஆகியவை பொதுவான அறிகுறிகள். "
            "மங்கலான பார்வை மற்றும் மெதுவாக ஆறும் காயங்களும் எச்சரிக்கை அறிகுறிகளாகும். "
            "வகை 1 நீரிழிவில் கணையம் இன்சுலினை உற்பத்தி செய்வதில்லை, வகை 2 இல் உடல் இன்சுலினை சரியாகப் பயன்படுத்துவதில்லை. "
            "வகை 2 நீரிழிவு உலகளவில் அனைத்து வழக்குகளிலும் தோராயமாக தொண்ணூறு சதவீதம் ஆகும். "
            "உணவுக் கட்டுப்பாடு, உடற்பயிற்சி மற்றும் மருந்துகள் மூலம் இதைக் கட்டுப்படுத்த முடியும். "
            "சிகிச்சை அளிக்கப்படாத நீரிழிவு சிறுநீரகம், கண் மற்றும் நரம்புகளுக்கு நிரந்தர சேதத்தை ஏற்படுத்தும். "
            "ஆண்டுதோறும் இரத்தப் பரிசோதனை செய்துகொள்வது ஆரம்ப கட்டத்திலேயே நோயைக் கண்டறிய உதவும்."
        ),
    },
    {
        "doc_id": "demo-bn-1",
        "lang": "bn",
        "title": "সুন্দরবন",
        "source_query": "সুন্দরবন কোথায় অবস্থিত",
        "text": (
            "সুন্দরবন বিশ্বের বৃহত্তম ম্যানগ্রোভ বন। এটি বাংলাদেশ ও ভারতের পশ্চিমবঙ্গ রাজ্য জুড়ে বিস্তৃত। "
            "বনের প্রায় ষাট শতাংশ বাংলাদেশে এবং বাকি অংশ ভারতে পড়েছে। "
            "এই বন রয়েল বেঙ্গল টাইগারের প্রধান আবাসস্থল হিসেবে পরিচিত। "
            "১৯৯৭ সালে ইউনেস্কো একে বিশ্ব ঐতিহ্যবাহী স্থান হিসেবে ঘোষণা করে। "
            "নদীর জোয়ার-ভাটা এবং লবণাক্ত জল এখানকার বাস্তুতন্ত্রকে অনন্য করে তুলেছে। "
            "সুন্দরী গাছ থেকেই বনের নাম এসেছে বলে অনেকে মনে করেন। "
            "ঘূর্ণিঝড়ের সময় এই বন উপকূলীয় জনপদের জন্য প্রাকৃতিক প্রাচীর হিসেবে কাজ করে। "
            "মধু সংগ্রহ এবং মাছ ধরা স্থানীয় বহু পরিবারের প্রধান জীবিকা। "
            "জলবায়ু পরিবর্তনের ফলে সমুদ্রপৃষ্ঠের উচ্চতা বৃদ্ধি এই বনের জন্য বড় হুমকি॥"
        ),
    },
    {
        "doc_id": "demo-ur-1",
        "lang": "ur",
        "title": "چائے",
        "source_query": "چائے کہاں سے آئی؟",
        "text": (
            "چائے دنیا میں پانی کے بعد سب سے زیادہ پیا جانے والا مشروب ہے۔ "
            "اس کی ابتدا قدیم چین سے ہوئی، جہاں اسے دوا کے طور پر استعمال کیا جاتا تھا۔ "
            "کیا آپ جانتے ہیں کہ سبز اور کالی چائے ایک ہی پودے سے حاصل ہوتی ہیں؟ "
            "فرق صرف پتوں کی تیاری اور آکسیڈیشن کے عمل کا ہے۔ "
            "سبز چائے کے پتوں کو آکسیڈیشن سے پہلے ہی بھاپ دی جاتی ہے۔ "
            "برطانوی دور میں آسام اور دارجیلنگ میں چائے کے بڑے باغات لگائے گئے۔ "
            "آج بھارت دنیا کے سب سے بڑے پیدا کنندگان اور صارفین میں شمار ہوتا ہے۔ "
            "برصغیر میں دودھ، الائچی اور ادرک کے ساتھ بنی مسالہ چائے سب سے زیادہ مقبول ہے۔ "
            "ایک پیالی چائے میں کافی کے مقابلے میں تقریباً آدھی کیفین ہوتی ہے۔"
        ),
    },
    {
        "doc_id": "demo-en-1",
        "lang": "en",
        "title": "Photosynthesis",
        "source_query": "how does photosynthesis work",
        "text": (
            "Photosynthesis is the process by which green plants convert light energy into "
            "chemical energy. Chlorophyll in the chloroplasts absorbs light, mostly in the "
            "blue and red parts of the spectrum. Water taken up by the roots is split, "
            "releasing O2 as a by-product. Dr. Melvin Calvin mapped the carbon-fixation "
            "cycle that now bears his name, work for which he received the 1961 Nobel Prize. "
            "Roughly 3.5 kg of carbon dioxide is fixed per kg of glucose produced. The "
            "light-dependent reactions occur in the thylakoid membranes, while the Calvin "
            "cycle runs in the stroma. C4 plants such as maize and sugarcane concentrate "
            "carbon dioxide before fixing it, which makes them far more efficient in hot, "
            "dry conditions. Roughly half of all photosynthesis on Earth is carried out not "
            "by land plants at all, but by marine phytoplankton. Prof. J. K. Haldane and "
            "others argued that the oxygen this produced reshaped the early atmosphere."
        ),
    },
    {
        "doc_id": "demo-hi-2",
        "lang": "hi",
        "title": None,
        "source_query": "भारतीय रेल के बारे में",
        "text": (
            "भारतीय रेल दुनिया के सबसे बड़े रेल नेटवर्क में से एक है और यह प्रतिदिन लगभग दो करोड़ यात्रियों "
            "को उनके गंतव्य तक पहुँचाती है इसका संचालन रेल मंत्रालय के अधीन होता है और इसमें लाखों कर्मचारी "
            "काम करते हैं मालगाड़ियाँ कोयला खाद्यान्न सीमेंट और उर्वरक जैसी वस्तुओं की ढुलाई करती हैं जिससे "
            "देश की अर्थव्यवस्था को गति मिलती है पहली यात्री रेलगाड़ी 1853 में बोरीबंदर से ठाणे के बीच चली थी "
            "और उसने चौंतीस किलोमीटर की दूरी तय की थी आज नेटवर्क की कुल लंबाई अड़सठ हजार किलोमीटर से अधिक है "
            "और इसमें सात हजार से ज्यादा स्टेशन शामिल हैं विद्युतीकरण का काम तेजी से चल रहा है जिससे डीजल पर "
            "निर्भरता और प्रदूषण दोनों घट रहे हैं"
        ),
    },
]


# --------------------------------------------------------------------------------------
# Document loading
# --------------------------------------------------------------------------------------


def _coerce_lang(value: Any) -> Lang | None:
    """Accept 'hi', 'hin', 'Hindi', 'hi-IN' -- ingest formats vary and this is read-only."""
    if isinstance(value, Lang):
        return value
    if not isinstance(value, str) or not value:
        return None
    raw = value.strip()
    for candidate in (raw, raw.split("-")[0].split("_")[0]):
        try:
            return Lang(candidate.lower())
        except ValueError:
            pass
    from voxrag.contracts import LANG_FILE_STEM

    lowered = raw.lower()
    for lang, stem in LANG_FILE_STEM.items():
        if lowered == stem:
            return lang
    return None


def _to_document(obj: dict[str, Any], fallback_id: str) -> Document | None:
    """Best-effort mapping of one JSONL record onto Document. Unusable rows are skipped."""
    text = next((obj[k] for k in _TEXT_KEYS if isinstance(obj.get(k), str) and obj[k].strip()), None)
    if text is None:
        return None
    doc_id = next((str(obj[k]) for k in _ID_KEYS if obj.get(k) is not None), fallback_id)
    lang = next((_coerce_lang(obj.get(k)) for k in _LANG_KEYS if obj.get(k)), None) or Lang.ENG
    return Document(
        doc_id=doc_id,
        text=text,
        lang=lang,
        title=obj.get("title") or None,
        url=obj.get("url") or None,
        source_query=obj.get("source_query") or obj.get("query") or None,
        gold_answer=obj.get("gold_answer") or obj.get("Answer") or None,
        is_selected=bool(obj.get("is_selected", False)),
    )


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def load_documents(
    path: Path, *, sample: int, seed: int, langs: set[Lang] | None = None
) -> list[Document]:
    """Reservoir-sample `sample` documents from a JSONL file in one pass.

    Reservoir rather than head-N so the sample is not biased toward whichever language
    ingest happened to write first, and seeded so the README table is reproducible.
    """
    reservoir: list[Document] = []
    seen = 0
    rng = random.Random(seed)
    for i, obj in enumerate(_iter_jsonl(path)):
        doc = _to_document(obj, fallback_id=f"row-{i}")
        if doc is None or (langs and doc.lang not in langs):
            continue
        seen += 1
        if sample <= 0 or len(reservoir) < sample:
            reservoir.append(doc)
        else:
            j = rng.randrange(seen)
            if j < sample:
                reservoir[j] = doc
    return reservoir


def demo_documents() -> list[Document]:
    return [d for d in (_to_document(o, o["doc_id"]) for o in _DEMO_DOCS) if d is not None]


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


@dataclass
class StrategyStats:
    strategy: ChunkStrategyName
    n_docs: int = 0
    n_chunks: int = 0
    failures: int = 0
    tokens: list[float] = field(default_factory=list)
    ms_per_doc: list[float] = field(default_factory=list)
    over_window: int = 0
    frag_chunks: int = 0
    ngrams_total: int = 0
    ngrams_unique: int = 0
    covered_chars: int = 0
    content_chars: int = 0
    payload_bytes: int = 0
    embed_tokens: int = 0

    def as_dict(self) -> dict[str, Any]:
        vec_bytes = self.n_chunks * settings.embed_dim * _BYTES_PER_FLOAT
        return {
            "strategy": self.strategy.value,
            "n_docs": self.n_docs,
            "n_chunks": self.n_chunks,
            "chunks_per_doc": self.n_chunks / self.n_docs if self.n_docs else 0.0,
            "tokens_mean": sum(self.tokens) / len(self.tokens) if self.tokens else 0.0,
            "tokens_p50": percentile(self.tokens, 50.0),
            "tokens_p95": percentile(self.tokens, 95.0),
            "over_encoder_window_pct": _pct(self.over_window, self.n_chunks),
            "fragmentation_pct": _pct(self.frag_chunks, self.n_chunks),
            "redundancy_pct": _pct(self.ngrams_total - self.ngrams_unique, self.ngrams_total),
            "coverage_pct": _pct(self.covered_chars, self.content_chars),
            "index_mb": (vec_bytes + self.payload_bytes) / 1e6,
            "vector_mb": vec_bytes / 1e6,
            "embed_tokens": self.embed_tokens,
            "ms_per_doc_mean": (
                sum(self.ms_per_doc) / len(self.ms_per_doc) if self.ms_per_doc else 0.0
            ),
            "ms_per_doc_p95": percentile(self.ms_per_doc, 95.0),
            "failures": self.failures,
        }


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def _ngram_keys(text: str) -> list[tuple[str, ...]]:
    """Token 5-grams of `text`, as tuples of the tokens' surface forms."""
    idx = token_index(text)
    toks = [text[s:e].strip() for s, e in idx.spans]
    toks = [t for t in toks if t]
    if len(toks) < NGRAM:
        return []
    return [tuple(toks[i : i + NGRAM]) for i in range(len(toks) - NGRAM + 1)]


def _content_chars(text: str) -> int:
    return sum(1 for c in text if not c.isspace())


def _covered_chars(text: str, chunks: list[Chunk]) -> int:
    """Non-whitespace characters inside the union of the chunks' character ranges."""
    if not chunks:
        return 0
    ranges = sorted((c.char_start, c.char_end) for c in chunks)
    merged: list[list[int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(_content_chars(text[a:b]) for a, b in merged)


def evaluate_documents(
    docs: list[Document], strategies: Iterable[ChunkStrategyName]
) -> dict[ChunkStrategyName, StrategyStats]:
    """Run every strategy over every document and collect the comparison metrics."""
    budget = embed_budget_tokens()
    # Warm the tokenisation cache first so no single strategy is charged for it; in
    # production the index build tokenises each document once as well.
    sentence_ends: dict[str, set[int]] = {}
    for doc in docs:
        token_index(doc.text)
        sentence_ends[doc.doc_id] = {sp.end for sp in split_sentences(doc.text, doc.lang)}

    results: dict[ChunkStrategyName, StrategyStats] = {}
    for name in strategies:
        chunker = get_chunker(name)
        stats = StrategyStats(strategy=name)
        for doc in docs:
            t0 = time.perf_counter()
            try:
                chunks = chunker.chunk(doc)
            except Exception:
                # One pathological document must not void the whole comparison; it is
                # counted and the run continues.
                stats.failures += 1
                continue
            stats.ms_per_doc.append((time.perf_counter() - t0) * 1000.0)
            stats.n_docs += 1
            stats.n_chunks += len(chunks)

            ends = sentence_ends[doc.doc_id]
            grams: set[tuple[str, ...]] = set()
            n_grams = 0
            for ch in chunks:
                stats.tokens.append(float(ch.n_tokens))
                stats.embed_tokens += min(ch.n_tokens, budget)
                stats.payload_bytes += len(ch.text.encode("utf-8"))
                if ch.context_text:
                    stats.payload_bytes += len(ch.context_text.encode("utf-8"))
                if ch.n_tokens > budget:
                    stats.over_window += 1
                if ch.char_end not in ends:
                    stats.frag_chunks += 1
                keys = _ngram_keys(ch.text)
                n_grams += len(keys)
                grams.update(keys)
            stats.ngrams_total += n_grams
            stats.ngrams_unique += len(grams)
            stats.content_chars += _content_chars(doc.text)
            stats.covered_chars += _covered_chars(doc.text, chunks)
        results[name] = stats
    return results


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def _render(console: Console, rows: list[dict[str, Any]], *, source: str) -> None:
    table = Table(
        title=f"Chunking strategy comparison  ({source})",
        caption=(
            f"target={settings.chunk_target_tokens}t  overlap={settings.chunk_overlap_tokens}t  "
            f"parent={settings.parent_chunk_tokens}t  child={settings.child_chunk_tokens}t  "
            f"encoder window={settings.embed_max_tokens}t (usable {embed_budget_tokens()}t)  "
            f"tokenizer={'e5' if tokenizer_available() else 'FALLBACK ratio'}"
        ),
        header_style="bold",
        show_lines=False,
    )
    table.add_column("strategy", style="bold cyan", no_wrap=True)
    table.add_column("chunks", justify="right")
    table.add_column("/doc", justify="right")
    table.add_column("tok\nmean", justify="right")
    table.add_column("tok\np50", justify="right")
    table.add_column("tok\np95", justify="right")
    table.add_column(">enc\n%", justify="right")
    table.add_column("frag\n%", justify="right")
    table.add_column("redun\n%", justify="right")
    table.add_column("cover\n%", justify="right")
    table.add_column("index\nMB", justify="right")
    table.add_column("embed\nktok", justify="right")
    table.add_column("ms\n/doc", justify="right")

    best_frag = min((r["fragmentation_pct"] for r in rows), default=0.0)
    cheapest = min((r["index_mb"] for r in rows), default=0.0)
    for r in rows:
        marker = " *" if r["strategy"] == settings.active_strategy.value else ""
        table.add_row(
            r["strategy"] + marker,
            f"{r['n_chunks']:,}",
            f"{r['chunks_per_doc']:.2f}",
            f"{r['tokens_mean']:.0f}",
            f"{r['tokens_p50']:.0f}",
            f"{r['tokens_p95']:.0f}",
            _warn(r["over_encoder_window_pct"], 0.5),
            _good(r["fragmentation_pct"], best_frag),
            _warn(r["redundancy_pct"], 15.0),
            _warn_low(r["coverage_pct"], 99.9),
            _good(r["index_mb"], cheapest, fmt="{:.2f}"),
            f"{r['embed_tokens'] / 1000.0:.1f}",
            f"{r['ms_per_doc_mean']:.2f}",
        )
    console.print(table)
    console.print("  * = settings.active_strategy, the strategy that builds the served index")


def _good(value: float, best: float, *, fmt: str = "{:.1f}") -> str:
    text = fmt.format(value)
    return f"[green]{text}[/green]" if value <= best + 1e-9 else text


def _warn(value: float, limit: float, *, fmt: str = "{:.1f}") -> str:
    text = fmt.format(value)
    return f"[yellow]{text}[/yellow]" if value > limit else text


def _warn_low(value: float, limit: float, *, fmt: str = "{:.1f}") -> str:
    text = fmt.format(value)
    return f"[red]{text}[/red]" if value < limit else text


def _reading(console: Console, rows: list[dict[str, Any]]) -> None:
    """Derived commentary. Everything here is read off the numbers just printed."""
    if not rows:
        return
    by = {r["strategy"]: r for r in rows}
    cheapest = min(rows, key=lambda r: r["index_mb"])
    most_redundant = max(rows, key=lambda r: r["redundancy_pct"])
    finest = min(rows, key=lambda r: r["tokens_p50"] or 1e9)
    best_frag = min(r["fragmentation_pct"] for r in rows)
    cleanest = [r["strategy"] for r in rows if r["fragmentation_pct"] <= best_frag + 1e-9]

    console.print("\n[bold]Reading the table[/bold]")
    console.print(
        f"  - cheapest index: [cyan]{cheapest['strategy']}[/cyan] at "
        f"{cheapest['index_mb']:.2f} MB / {cheapest['n_chunks']:,} vectors."
    )
    console.print(
        f"  - cleanest boundaries ({best_frag:.1f}% of chunks ending mid-sentence): "
        f"[cyan]{', '.join(cleanest)}[/cyan]."
    )
    console.print(
        f"  - most duplicated text: [cyan]{most_redundant['strategy']}[/cyan] at "
        f"{most_redundant['redundancy_pct']:.1f}% repeated {NGRAM}-grams -- that is what "
        "overlap and repeated metadata headers cost in index weight."
    )
    console.print(
        f"  - finest retrieval unit: [cyan]{finest['strategy']}[/cyan] at a p50 of "
        f"{finest['tokens_p50']:.0f} tokens."
    )

    native = by.get(ChunkStrategyName.PASSAGE_NATIVE.value)
    if native and native["n_chunks"]:
        matched = [
            r["strategy"]
            for r in rows
            if r["strategy"] != native["strategy"]
            and r["fragmentation_pct"] <= native["fragmentation_pct"] + 1e-9
        ]
        console.print(
            "\n  [bold]On PASSAGE_NATIVE.[/bold] An MS MARCO passage is already a "
            "human-curated retrieval unit -- one topic, a few sentences, cut to answer a "
            "query. That makes it the baseline every other strategy has to beat on "
            "*retrieval quality*, because on cost and boundary cleanliness it is close to "
            f"unbeatable here: {native['index_mb']:.2f} MB, "
            f"{native['fragmentation_pct']:.1f}% fragmentation, "
            f"{native['over_encoder_window_pct']:.1f}% of chunks longer than the encoder "
            "window."
        )
        console.print(
            "  Matching its boundary cleanliness: "
            + (", ".join(matched) if matched else "none")
            + ". Index cost relative to it: "
            + ", ".join(
                f"{r['strategy']} {r['index_mb'] / native['index_mb']:.1f}x"
                for r in rows
                if r["strategy"] != native["strategy"] and native["index_mb"]
            )
            + "."
        )
        console.print(
            "  Everything above 1.0x is buying precision (smaller units) or recall "
            "(overlap, parent context) with index size. That is the trade PARENT_CHILD "
            "makes, and why it is the served default: child-sized vectors for matching, "
            "parent-sized text for generating."
        )

    semantic = by.get(ChunkStrategyName.SEMANTIC.value)
    if semantic and semantic["chunks_per_doc"] < 1.5:
        console.print(
            f"\n  [bold]On SEMANTIC.[/bold] It emits {semantic['chunks_per_doc']:.2f} chunks "
            "per document here, i.e. it mostly declines to split. That is the correct "
            "answer, not a bug: a passage written to answer one query rarely contains a "
            "topic shift large enough to clear the "
            f"p{settings.semantic_breakpoint_percentile:g} sentence-distance threshold, and "
            f"semantic_min_tokens={settings.semantic_min_tokens} forbids the leftover "
            "fragment. Semantic chunking earns its cost on long multi-topic documents, "
            "not on MS MARCO passages."
        )
    over = [r["strategy"] for r in rows if r["over_encoder_window_pct"] > 0.5]
    if over:
        console.print(
            f"\n  [yellow]Encoder-window warning[/yellow]: {', '.join(over)} emit chunks longer "
            f"than the usable {embed_budget_tokens()}-token window; the embedder sees only "
            "their prefix. SEMANTIC does this by design (semantic_max_tokens="
            f"{settings.semantic_max_tokens}) -- its segments follow topic, not the encoder."
        )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m voxrag.chunking.evaluate",
        description="Compare every chunking strategy on the same document sample.",
    )
    p.add_argument("--docs", type=Path, default=Path("data/docs.jsonl"), help="JSONL of documents")
    p.add_argument("--sample", type=int, default=500, help="documents to sample; 0 = all")
    p.add_argument("--seed", type=int, default=13, help="sampling seed, for reproducibility")
    p.add_argument("--lang", action="append", default=None, help="restrict to a language (repeatable)")
    p.add_argument(
        "--strategies",
        default=None,
        help="comma-separated subset of strategies (default: all eight)",
    )
    p.add_argument("--demo", action="store_true", help="use the built-in synthetic sample")
    p.add_argument("--json", dest="as_json", action="store_true", help="emit JSON, no table")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # The table has 13 numeric columns and is meant to be pasted into the README; at
    # rich's default 80-column fallback (which is what a pipe gets) every heading is
    # ellipsised into uselessness.
    console = Console(width=max(_MIN_TABLE_WIDTH, Console().size.width))

    if args.strategies:
        try:
            wanted = [ChunkStrategyName(s.strip()) for s in args.strategies.split(",") if s.strip()]
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return 2
    else:
        wanted = list(REGISTRY)

    if args.demo:
        docs = demo_documents()
        source = f"{len(docs)} SYNTHETIC demo docs -- smoke test, not corpus results"
    elif not args.docs.exists():
        console.print(
            f"[red]no such file: {args.docs}[/red]\n"
            "Build it first (the ingest module writes data/docs.jsonl), or run with --demo "
            "to smoke-test the strategies on the built-in synthetic sample."
        )
        return 2
    else:
        langs = {lg for lg in (_coerce_lang(x) for x in (args.lang or [])) if lg} or None
        docs = load_documents(args.docs, sample=args.sample, seed=args.seed, langs=langs)
        source = f"{len(docs)} docs from {args.docs}"

    if not docs:
        console.print("[red]no usable documents in the sample[/red]")
        return 2

    results = evaluate_documents(docs, wanted)
    rows = [results[name].as_dict() for name in wanted]

    if args.as_json:
        json.dump(
            {"source": source, "settings": _settings_snapshot(), "strategies": rows},
            sys.stdout,
            ensure_ascii=False,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0

    _render(console, rows, source=source)
    _reading(console, rows)
    failed = [r["strategy"] for r in rows if r["failures"]]
    if failed:
        console.print(f"\n[yellow]strategies with per-document failures: {', '.join(failed)}[/yellow]")
    return 0


def _settings_snapshot() -> dict[str, Any]:
    return {
        "chunk_target_tokens": settings.chunk_target_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "sentence_window_size": settings.sentence_window_size,
        "semantic_breakpoint_percentile": settings.semantic_breakpoint_percentile,
        "semantic_min_tokens": settings.semantic_min_tokens,
        "semantic_max_tokens": settings.semantic_max_tokens,
        "parent_chunk_tokens": settings.parent_chunk_tokens,
        "child_chunk_tokens": settings.child_chunk_tokens,
        "embed_max_tokens": settings.embed_max_tokens,
        "embed_dim": settings.embed_dim,
        "active_strategy": settings.active_strategy.value,
        "tokenizer": settings.embed_model_repo if tokenizer_available() else "fallback-ratio",
        "ngram": NGRAM,
    }


if __name__ == "__main__":
    raise SystemExit(main())
