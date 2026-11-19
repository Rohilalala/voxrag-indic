"""Which language is this, when the script alone cannot say.

`input_rails._SCRIPT_RANGES` answers "which alphabet", and for ten of this corpus's fourteen
languages that IS the answer. For the other four it is not: Devanagari carries Hindi, Marathi,
Nepali and Sanskrit, and the Bengali script carries Bengali and Assamese. A default per script
cannot resolve those -- whichever language is not the default loses every time, which is
exactly what happened: Marathi, Nepali, Sanskrit and Assamese queries were all routed to the
Hindi or Bengali partition of the index and retrieved passages about unrelated subjects.

So a family with more than one member gets a real classifier: multinomial Naive Bayes over
character 3-grams, trained on the corpus's own queries by `scripts/train_lang_id.py`. Held-out
accuracy is 0.977 on the Devanagari family and 0.992 on the Bengali one, at ~6us per query --
three orders of magnitude below the 200ms budget and small enough that the model ships in the
wheel. The measured confusion matrix is stored inside the model file under `evaluation`.

The classifier never overrides an explicit hint and is never consulted for an unambiguous
script: those paths cost nothing and gain nothing from it.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import threading
from pathlib import Path

from voxrag.contracts import Lang

_LOG = logging.getLogger(__name__)

#: Model artefact, written by `scripts/train_lang_id.py`. Committed: `data/` is gitignored and
#: a detector that silently degrades when a file is missing is how the original bug survived.
MODEL_PATH = Path(__file__).with_name("lang_id_model.json.gz")

SCHEMA = "voxrag.lang_id/1"
NGRAM_N = 3

#: Below this many characters a query carries too few n-grams to distinguish sister languages,
#: and the caller's script default is the better answer. Measured: two-word Devanagari queries
#: are common in the corpus, one-word ones are not queries.
MIN_CHARS = 6


def ngrams(text: str, n: int = NGRAM_N) -> list[str]:
    """Character n-grams with boundary padding, so word-initial and word-final sequences --
    which is where Indic language morphology differs most -- are features in their own right."""
    padded = f" {text.strip()} "
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


class _Model:
    """Loaded lazily and once: reading and parsing the model is ~40ms, which belongs in
    warmup, not in the first user request."""

    def __init__(self) -> None:
        self.families: dict[str, dict[str, tuple[dict[str, float], float]]] = {}
        self.evaluation: dict[str, object] = {}

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> _Model:
        model = cls()
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"{path} has schema {payload.get('schema')!r}, expected {SCHEMA!r}")
        for script, langs in payload["families"].items():
            model.families[script] = {
                code: (params["logp"], float(params["unseen"])) for code, params in langs.items()
            }
        model.evaluation = payload.get("evaluation", {})
        return model


_INSTANCE: _Model | None = None
_LOAD_LOCK = threading.Lock()
_FAILED = False


def _model() -> _Model | None:
    """The loaded model, or None if it could not be read.

    A missing or corrupt model is a degradation, not an outage: detection falls back to the
    script default, which is what the system did before this module existed. It is logged at
    ERROR because that fallback silently mis-routes four languages.
    """
    global _INSTANCE, _FAILED
    if _INSTANCE is not None or _FAILED:
        return _INSTANCE
    with _LOAD_LOCK:
        if _INSTANCE is None and not _FAILED:
            try:
                _INSTANCE = _Model.load()
            except Exception as exc:
                _FAILED = True
                _LOG.error(
                    "language-ID model unavailable (%r); shared-script languages will fall "
                    "back to the script default and retrieve from the wrong partition",
                    exc,
                )
    return _INSTANCE


def prewarm() -> bool:
    """Load the model now. Called from harness warmup so no request pays the parse."""
    return _model() is not None


def classify(text: str, candidates: tuple[Lang, ...], script: str) -> tuple[Lang | None, float]:
    """Pick the most likely language of `text` among `candidates`.

    Returns the language and the per-character log-probability margin over the runner-up: a
    scale-free confidence, since a longer query accumulates more evidence for whichever
    language it is in. Returns (None, 0.0) when the model cannot decide -- no model, no
    trained family, unknown candidates, or a query too short to carry evidence.
    """
    if len(text.strip()) < MIN_CHARS or len(candidates) < 2:
        return None, 0.0
    model = _model()
    if model is None:
        return None, 0.0
    family = model.families.get(script)
    if not family:
        return None, 0.0

    grams = ngrams(text)
    if not grams:
        return None, 0.0

    best: Lang | None = None
    best_score = -math.inf
    runner_up = -math.inf
    for lang in candidates:
        params = family.get(lang.value)
        if params is None:
            continue
        logp, unseen = params
        score = 0.0
        for gram in grams:
            score += logp.get(gram, unseen)
        if score > best_score:
            best, best_score, runner_up = lang, score, best_score
        elif score > runner_up:
            runner_up = score

    if best is None or runner_up == -math.inf:
        return None, 0.0
    return best, (best_score - runner_up) / len(grams)


def evaluation() -> dict[str, object]:
    """The held-out accuracy and confusion matrix the shipped model was validated at."""
    model = _model()
    return dict(model.evaluation) if model else {}


__all__ = ["MIN_CHARS", "MODEL_PATH", "NGRAM_N", "SCHEMA", "classify", "evaluation", "ngrams", "prewarm"]
