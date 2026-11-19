"""Runtime configuration. Every tunable lives here; nothing reads os.environ directly."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from voxrag.contracts import ChunkStrategyName, GenerationMode, Lang

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="VOXRAG_", extra="ignore", case_sensitive=False
    )

    # -- speech to text -----------------------------------------------------------------
    #: "sarvam" | "elevenlabs" | "local". Sarvam is primary: the corpus is Indic.
    stt_provider: str = "sarvam"
    #: Used when the primary provider errors or times out. "local" needs no key.
    stt_fallback_provider: str | None = "local"
    sarvam_api_key: str | None = None
    sarvam_base_url: str = "https://api.sarvam.ai"
    #: Verified against docs.sarvam.ai on 2026-08-15: POST /speech-to-text, auth header
    #: `api-subscription-key`, models `saaras:v3` (default) and `saaras:v4`.
    sarvam_model: str = "saaras:v3"
    #: `transcribe` keeps the source language. `translate` would silently return English,
    #: which would break retrieval against an Indic index -- do not change this casually.
    sarvam_mode: str = "transcribe"
    #: Sarvam wants BCP-47-ish codes (hi-IN, ta-IN) and "unknown" for auto-detect. Note the
    #: Odia code is `od-IN`, not the ISO `or` used by contracts.Lang -- see stt/sarvam.py.
    sarvam_auto_language: str = "unknown"
    elevenlabs_api_key: str | None = None
    elevenlabs_base_url: str = "https://api.elevenlabs.io"
    elevenlabs_model: str = "scribe_v2"
    stt_timeout_s: float = 12.0

    # -- embedding ----------------------------------------------------------------------
    #: Multilingual, covers all 14 Indic scripts, 384-dim, ~118M params, ONNX-quantised.
    embed_model_repo: str = "intfloat/multilingual-e5-small"
    embed_onnx_file: str = "onnx/model_quantized.onnx"
    embed_dim: int = 384
    embed_max_tokens: int = 192
    embed_batch_size: int = 64
    #: e5 models require these prefixes; wrong prefix costs ~4 points of nDCG.
    embed_query_prefix: str = "query: "
    embed_passage_prefix: str = "passage: "
    #: intra-op threads for onnxruntime.
    #:
    #: Measured on M5 Pro (600 passages, int8 graph), threads -> bulk / single-query:
    #:   2 -> 57 chunks/s, 1.72ms      6 -> 105 chunks/s, 1.85ms      10 -> 114 chunks/s, 1.69ms
    #: Single-query latency is flat because a ~20-token query is too small to parallelise, so
    #: raising this is free for serving and roughly doubles indexing throughput. Kept modest
    #: by default so that concurrent requests do not oversubscribe the cores; the index
    #: builder raises it via VOXRAG_EMBED_THREADS.
    #:
    #: Note: the only quantised graph intfloat publishes is `model_qint8_avx512_vnni.onnx`,
    #: which targets x86 AVX-512 VNNI. On ARM those kernels do not map to native dotprod, so
    #: bulk throughput is well below what this model should do. It is kept anyway because
    #: query latency is ~1.4ms either way -- irrelevant against a 200ms budget -- and the
    #: alternative is a 470MB fp32 download for a one-off indexing speedup.
    embed_threads: int = 4
    #: LRU size for the query-embedding cache. Repeat queries skip the encoder entirely.
    embed_cache_size: int = 4096

    # -- chunking -----------------------------------------------------------------------
    #: Strategy used to build the served index. Others are built for the benchmark.
    active_strategy: ChunkStrategyName = ChunkStrategyName.PARENT_CHILD
    chunk_target_tokens: int = 160
    chunk_overlap_tokens: int = 32
    #: SENTENCE_WINDOW: sentences of context added on each side at retrieval time.
    sentence_window_size: int = 2
    #: SEMANTIC: percentile of cosine-distance spikes that becomes a split point.
    semantic_breakpoint_percentile: float = 92.0
    semantic_min_tokens: int = 48
    semantic_max_tokens: int = 320
    #: PARENT_CHILD: child is embedded for precision, parent is shown to the generator.
    parent_chunk_tokens: int = 384
    child_chunk_tokens: int = 96
    #: METADATA_AWARE: fold the passage's own `source_query` into the embedded header.
    #:
    #: OFF by default because it makes this corpus's benchmark meaningless. MSMARCO-XI's
    #: `source_query` IS the graded evaluation query, so embedding it puts the answer key
    #: inside the document. Measured on the eight-strategy comparison: Recall@10 = 0.996 and
    #: Recall@20 = 1.000, against 0.764 for the next best strategy. That is leakage, not
    #: retrieval quality.
    #:
    #: Worth turning ON for a production index -- indexing likely queries next to a passage
    #: is doc2query-style expansion and genuinely helps on queries the index has not seen --
    #: but any evaluation run with it enabled is measuring memorisation.
    metadata_include_source_query: bool = False

    # -- vector store -------------------------------------------------------------------
    #: "qdrant" | "hnsw". qdrant satisfies the "vector DB" requirement; hnsw is the
    #: in-process ANN path and the latency floor we measure against.
    vector_store: str = "qdrant"
    #: MUST point at a real Qdrant server for the benchmarked configuration. Embedded/local
    #: Qdrant (qdrant_path below) silently does BRUTE-FORCE exact search and ignores both
    #: hnsw_ef_search and payload indexes -- verified from its own runtime warnings. That is
    #: fine for a 5k-chunk dev loop and wrong for a latency claim, so `docker compose up -d`
    #: (see docker-compose.yml) before running the benchmark.
    qdrant_url: str | None = "http://127.0.0.1:6333"
    qdrant_api_key: str | None = None
    #: Zero-setup embedded fallback used only when `qdrant_url` is unset. Brute-force.
    qdrant_path: Path = DATA_DIR / "qdrant"
    #: The 14-language index, which is the corpus this system claims to serve. It is NOT the
    #: default for cosmetic reasons: `msmarco_xi` is the earlier three-language collection and
    #: is still on the server, and a `lang` filter for a language a collection does not hold
    #: matches nothing and returns an EMPTY result -- no error, no degradation, no log line.
    #: With the old default, eleven of fourteen languages were served by the lexical arm alone
    #: and dense retrieval scored nDCG@10 0.043 against lexical's 0.275.
    qdrant_collection: str = "msmarco_xi14"
    #: HNSW build/search params. ef_search dominates the latency/recall tradeoff.
    hnsw_m: int = 32
    hnsw_ef_construct: int = 200
    #: 96 rather than the 192 the sweep preferred. 192 is worth +0.0059 nDCG@10 held out
    #: (significant, but a sixth of the noise floor) and costs the tail the budget is graded
    #: on: over 500 balanced queries, CORE p70 90.4ms vs 70.1ms and 96.4% vs 99.6% of queries
    #: inside 200ms. A graded latency requirement outranks a 1.5% relative quality gain.
    hnsw_ef_search: int = 96
    hnsw_path: Path = DATA_DIR / "hnsw"
    #: Qdrant vector quantization: "none" | "scalar" | "binary". See store/quantization.py.
    #:
    #: THE DEFAULT IS OFF BECAUSE IT HAS NOT BEEN MEASURED ON THIS INDEX. Quantization is a
    #: recall change wearing a memory change's clothes: int8 discards precision in the stored
    #: vectors and binary discards all of it but the sign. Rescoring usually recovers most of
    #: the loss for int8 and does not for binary at 384 dimensions -- but "usually" is a claim
    #: about other people's corpora. Turning this on WITHOUT re-running
    #: `python -m voxrag.eval.retrieval` ships an unmeasured recall change, and the reported
    #: retrieval numbers stop describing what the system does.
    #:
    #: `scripts/measure_quantization.py` runs that comparison on a throwaway copy of the
    #: collection and prints latency and Recall@10 per kind side by side.
    #: MEASURED on this index and enabled: scripts/measure_quantization.py on a copy gave
    #:   none    0.45GB searched, p50 5.1ms  p90 11.9ms  p100 18.1ms  nDCG@10 0.2789
    #:   scalar  0.11GB searched, p50 4.0ms  p90  4.9ms  p100  8.7ms  nDCG@10 0.2789
    #: Identical R@1, R@10, MRR and nDCG -- rescoring against the retained float32 vectors
    #: recovered the compression loss exactly, with zero spread across repeats. The gain is
    #: concentrated in the TAIL (p90 2.4x, p100 2.1x), which is where this system's latency
    #: problem lives: a 14-language working set thrashing cache.
    quantization: str = "scalar"
    #: Re-rank the quantized shortlist with the original full-precision vectors.
    #: OFF since the corpus went to 9,028,804 chunks (~40% of the validation split).
    #:
    #: Rescoring re-ranks the int8 shortlist against the ORIGINAL float32 vectors, which at
    #: this size are ~13.8GB and cannot stay resident -- so every query became a disk read
    #: storm. Measured on the live index, same queries, only this flag moving:
    #:     rescore=on    p50 72.2ms   p90 159.7ms   p100 575.2ms
    #:     rescore=off   p50 10.4ms   p90  14.4ms   p100  24.7ms
    #: n=140 each, with_payload=True (the production shape), and each configuration WARMED ON
    #: ITS OWN PATH before timing. An earlier version of this comment claimed 123.4 vs 2.9ms;
    #: those runs shared one warm-up, so the first config measured paid the cold cost and the
    #: gap was part cache state. The direction held; the magnitude did not.
    #: End to end that was CORE p70 259.3ms with 41% of queries inside the 200ms budget --
    #: a FAILING graded requirement -- against 7.8ms p50 on the old 2.27M index.
    #:
    #: It is off because it was measured to buy NOTHING, not merely to buy less than it cost.
    #: Three configurations over the same 350 graded queries on the 9M index:
    #:     ef=96 rescore=on    dense nDCG@10 0.268   hybrid 0.280
    #:     ef=96 rescore=off   dense nDCG@10 0.266   hybrid 0.288
    #:     ef=48 rescore=on    dense nDCG@10 0.253   hybrid 0.256
    #: on-vs-off differs by 0.002 dense and -0.008 hybrid, both an order of magnitude inside
    #: the 0.0328 noise floor. Lowering ef instead IS a real loss and was rejected.
    #:
    #: The older note claiming scalar+rescore scored identically to no quantization was
    #: measured with rescore ON and is not evidence that rescoring earns its cost; on this
    #: index it never did, it was just cheap enough at 2.27M for nobody to notice.
    quantization_rescore: bool = False
    #: Candidates the quantized pass fetches per requested hit before rescoring. Higher trades
    #: latency for the recall the compression cost. Ignored when quantization is "none".
    quantization_oversampling: float = 2.0

    # -- retrieval ----------------------------------------------------------------------
    #: TUNED: 10 scored as well as 20 (nDCG@10 0.3822 vs 0.3813) at HALF the dense latency
    #: (30.8ms vs 40.0ms per query in the sweep), and 40 was worse and twice as slow again.
    retrieve_top_k: int = 10
    lexical_top_k: int = 20
    #: Candidates surviving fusion and handed to the reranker.
    fuse_top_k: int = 12
    #: Chunks handed to the generator.
    context_top_k: int = 4
    #: Reciprocal Rank Fusion constant. 60 is the value from the original RRF paper; 10 is
    #: what this corpus measured. A smaller k weights the very top ranks far more, which is
    #: the right shape when both arms are strong and their top-3 is where the answer is.
    rrf_k: int = 10
    #: Weight on the dense arm during fusion; lexical gets (1 - this).
    #:
    #: TUNED, and the tuning only became possible after the fusion knobs stopped being frozen
    #: into function defaults at import (see store/hybrid.py). With them live, this and rrf_k
    #: are the two knobs that move the number at all.
    #:
    #: {dense_weight 0.5, rrf_k 10, hnsw_ef_search 192, retrieve_top_k 10} vs the previous
    #: {0.65, 60, 96, 20}, validated on 560 HELD-OUT queries (a different balanced_sample
    #: seed from the one the sweep tuned on), 3 interleaved repeats each:
    #:   nDCG@10  0.3564 -> 0.3855   (+0.0291, 95% CI [+0.0129, +0.0464])
    #:   paired bootstrap p=0.0006, randomization p=0.0011 -- both required to agree
    #: The three repeats were bit-identical, so the run-to-run spread that the 0.0328 noise
    #: floor describes is not what produced this: retrieval here is deterministic.
    dense_weight: float = 0.5
    enable_lexical: bool = True
    enable_rerank: bool = False
    rerank_model_repo: str = "BAAI/bge-reranker-v2-m3"
    bm25_path: Path = DATA_DIR / "bm25"

    # -- generation ---------------------------------------------------------------------
    default_mode: GenerationMode = GenerationMode.FAST
    extractive_semantic_rerank: bool = True
    #: Sentences re-embedded per request. Measured ~2ms/sentence, so 8 costs ~16ms against a
    #: 200ms budget. Shortlisted by the cheap score first, so this stays bounded regardless
    #: of how many sentences the retrieved chunks contain.
    extractive_semantic_candidates: int = 8
    #: Weight on semantic similarity when blending with the lexical score. Below 0.5 so exact
    #: term matches still win when they exist -- names, numbers and codes must not be diluted.
    extractive_semantic_weight: float = 0.45
    ollama_base_url: str = "http://127.0.0.1:11434"
    #: 4.5B effective params, 128K context, multimodal. The quality path.
    ollama_model: str = "gemma4:e4b"
    ollama_timeout_s: float = 30.0
    #: gemma4 is a REASONING model: it emits a separate `thinking` field and, left on, spends
    #: its whole token budget there. Measured with num_predict=160 the answer came back
    #: literally empty (done_reason="length", 160 tokens, all of them reasoning) and QUALITY
    #: mode failed every request. With thinking off: 874ms and 22 tokens for the same correct,
    #: cited Hindi answer -- versus 4265ms with it on. A grounded extract from four retrieved
    #: passages does not need chain-of-thought, so this stays off.
    ollama_think: bool = False
    generate_max_tokens: int = 160
    generate_temperature: float = 0.2
    #: Hosted generator for the deployed box, where a 9.6GB local model is not viable.
    hosted_generator_url: str | None = None
    hosted_generator_key: str | None = None
    hosted_generator_model: str = "gemma4:cloud"

    # -- guardrails ---------------------------------------------------------------------
    enable_guardrails: bool = True
    #: Below this fused retrieval score the system refuses rather than guesses.
    min_retrieval_score: float = 0.28
    #: ABSOLUTE floor on the best retrieved chunk's raw dense cosine.
    #:
    #: This catches only the degenerate case where retrieval matched essentially nothing. It
    #: is deliberately LOW, and the reason is a failed experiment worth recording so nobody
    #: repeats it.
    #:
    #: The intent was an off-topic gate at 0.78, calibrated on measured pairs:
    #:     relevant    0.8151 .. 0.8790
    #:     irrelevant  0.7253 .. 0.7592
    #: Those irrelevant pairs were ARTIFICIAL -- bank and cooking queries scored against
    #: corporate text. Re-measured against the real 2.27M-chunk index, the separation
    #: vanishes:
    #:     "what is the price of bitcoin right now"  0.8803   (off-topic)
    #:     "what is a checking account"              0.8748   (legitimate)
    #:     "what is the capital of India"            0.8484   (off-topic)
    #:     "कॉर्पोरेशन क्या है?"                        0.8793   (legitimate)
    #: An off-topic query outscores a legitimate one. No threshold separates them, and 0.78
    #: additionally produced a false refusal on "how do banks store passwords" (0.8500).
    #:
    #: The reason is structural, not a tuning failure: MS MARCO covers nearly every topic, so
    #: every query retrieves something topically close. "Passage about a city being a state
    #: capital" and "passage naming India's capital" are neighbours in embedding space. What
    #: is actually needed is ANSWERABILITY detection -- does this passage answer THIS question
    #: about THESE entities -- which similarity cannot express and which is an open problem in
    #: retrieval, not a constant to tune.
    min_dense_similarity: float = 0.60
    #: Fraction of the query's DISTINCTIVE terms that must appear in the retrieved passages,
    #: enforced by `guardrails.grounding.EntityGroundingRail`. This is the answer to the
    #: similarity failure documented above: "capital of India" is really a question about
    #: *India*, and the Stuttgart passage retrieved for it never says India. Entity presence
    #: separates the two cases that cosine could not.
    #:
    #: CALIBRATED on 200 real MSMARCO queries balanced across the 10 correctly-detected
    #: languages, plus 10 off-topic queries. The two populations separate almost perfectly,
    #: but not where intuition suggests:
    #:     off-topic            p50 0.00  p90 0.00   (they match NOTHING, not merely little)
    #:     answerable           p10 0.06  p50 0.59
    #: Swept:
    #:     thresh   false refusals   off-topic caught
    #:      0.05             10%              100%
    #:      0.25             19%              100%
    #:      0.50             40%              100%
    #: Every threshold catches all the off-topic queries, so the only thing raising it buys is
    #: false refusals. 0.05 is therefore the whole usable signal: refuse when NONE of the
    #: query's distinctive terms appear, and never on "few". An earlier 0.50 shipped for one
    #: commit and refused 53% of real queries.
    #:
    #: RE-MEASURED after retrieval was fixed, on 474 positive and 474 negative pairs built
    #: mechanically rather than written by hand (scripts/eval_grounding.py: a positive is a
    #: query whose own source passage was retrieved, a negative is that query against another
    #: query's passages). The "catches all off-topic queries at any threshold" claim above came
    #: from ten hand-authored queries and does NOT survive an independent set:
    #:     thresh   false refusals   off-topic caught   balanced accuracy
    #:      0.02             3.6%             82.1%              89.2%
    #:      0.05             4.4%             82.3%              88.9%   <- configured
    #:      0.20            10.1%             86.9%              88.4%
    #:      0.34            15.6%             95.8%              90.1%   <- best balanced
    #:      0.67            33.5%            100.0%              83.2%
    #: 0.34 maximises balanced accuracy on that table and is NOT configured, which is a
    #: judgement, but it is a measured one. 0.34 was run end to end over the same 500
    #: language-balanced queries (`data/bench_ground034.json` against `data/bench_final.json`):
    #:     threshold   false refusals   adversarial refusal accuracy
    #:      0.05                 4.0%                          65.4%
    #:      0.34                23.2%                          76.9%
    #: So it buys 11.5 points of adversarial accuracy for 19.2 points of false refusals --
    #: refusing about one real question in four. 0.05 keeps the answer path usable; the
    #: failure mode on the other side is one this project has already shipped once, when an
    #: earlier 0.50 refused 53% of real queries.
    #:
    #: The residual 4.4% are genuine vocabulary mismatch -- MSMARCO relevant passages often
    #: paraphrase the query (कॉर्पोरेशन vs निगम). That is the cost of a lexical rail, paid
    #: knowingly. Prefix matching was tried as a fix for it and measured worse: it cut false
    #: refusals to 1.9% and dropped off-topic catching to 77.8% (scripts/eval_grounding.py).
    min_entity_grounding: float = 0.05
    #: The rail above is lexical, so it SKIPS pairs whose query language is absent from the
    #: retrieved passages -- see guardrails/cross_language.py, which owns exactly that case.
    #: With this on, QUALITY mode asks gemma4 whether the evidence answers the question; with
    #: it off, every cross-language pair is refused without a model call. Both settings refuse
    #: on doubt: the switch decides only whether doubt is resolvable at all. Turning it off is
    #: the documented fallback if the entailment rail misses its ship gate -- safe and
    #: zero-latency, and it costs English as an input language.
    #:
    #: The gate was run: `scripts/eval_crosslang_rail.py`, artifact `data/eval_crosslang.json`,
    #: 150 on-topic + 100 off-topic English queries, none of them authored -- the on-topic set
    #: is `Eng_Query` for indexed rows whose own passage came back from live retrieval, the
    #: off-topic set is `Eng_Query` for rows outside the indexed slice (only 10,136 of ~98k
    #: parquet query_ids were ingested), retrieved live so each negative's evidence is the
    #: index's best match out of 2,268,974 vectors.
    #: CURRENT RESULT, `data/eval_crosslang_fixed.json`, n=450 on-topic / 300 off-topic:
    #:     false refusals (must answer)   10.7%  95% CI [8.1%, 13.9%]   gate <= 15%  PASS
    #:     off-topic caught               87.3%  95% CI [83.1%, 90.6%]  gate >= 80%  PASS
    #:     entailment call                p50 877ms  p90 1073ms  max 1750ms, 0 timeouts
    #:
    #: Getting to that number took two corrections, both worth keeping because both were
    #: measurement faults rather than rail faults, and an independent audit found the first:
    #:   30.9%  ceiling at 1500ms, which sat on the p90 of the call it timed, so one pair in
    #:          ten was refused for being SLOW. Raising it to 3000ms removed 14.7 points and
    #:          took rail timeouts to zero.
    #:    16.2%  the eval kept a query as a positive when its gold passage appeared anywhere
    #:          in the 12 fused chunks, while `entails` judges `context_top_k` = 4. It was
    #:          asking the verifier about four passages and scoring it against a fifth it had
    #:          never seen: all 73 residual refusals read "the verifier says these passages do
    #:          not answer it" over evidence spanning eight or nine languages. Correct answers
    #:          to a badly asked question. Fixing the slice removed a further 5.5 points.
    #:    10.7%  both fixed. The catch rate did not move (87.3% either way), which is the
    #:          check that the correction touched the positive side only, as it should.
    #: Earlier comments here cited 14.0% and 12.9%; neither is in any artefact in `data/` --
    #: one quoted a run that had been overwritten, the other pooled two runs of different
    #: sizes and reported the result as a single measurement.
    #: For scale, the lexical `min_entity_grounding` rail scores 4.4% / 82.3% on its own
    #: mechanical set, so this buys the English path a comparable catch rate for ~3x the false
    #: refusals and ~700ms. Both gate numbers are LOWER bounds on quality: MS MARCO is
    #: topically dense, so some unindexed queries really are answerable from an indexed
    #: passage, and every one of those is a mislabelled negative.
    #:
    #: Where it is weak, stated because a demo will hit it: on 10 hand-written off-domain
    #: questions (authored, so reported apart and never folded into the 86%) it caught 6/10.
    #: The four it allowed are time-sensitive or entity-current questions the corpus has a
    #: topical neighbour for -- bitcoin price, last night's football, "what is my account
    #: balance", the current US president. gemma4 judges whether the passages are ABOUT the
    #: question, not whether they are current, so "is there a passage on this topic" is the
    #: real signal. The agentive ones (book a table, send an email, write a poem) are caught.
    cross_language_verification: bool = True
    #: Ceiling on that entailment call, enforced by the rail itself so it returns its own
    #: fail-CLOSED refusal rather than being killed by `run_rails`' fail-OPEN rail timeout.
    #: Measured against the live daemon (gemma4:e4b, one Bengali passage, think off, one
    #: English query), four consecutive calls:
    #:     6243.4ms   <- first call, model not yet resident
    #:      259.0ms
    #:      259.0ms
    #:      257.1ms
    #: So ~6x the warm call. It is deliberately NOT sized to cover a cold load: a 6.2s wait in
    #: front of an answer is worse than a refusal, and refusing the first cross-language query
    #: after an eviction is the safe direction. This is a ceiling on a QUALITY-mode call and
    #: has nothing to do with the 200ms FAST budget, which never reaches a verifier.
    #:
    #: "The first query after an eviction" was FALSE as originally shipped, and it took two
    #: changes elsewhere to make it true. Cancelling the request at this ceiling also aborts
    #: the model LOAD, so a cold verifier could never become warm: measured on a production
    #: -shaped context (context_top_k=4, 1938-char user message), a cold `entails` costs
    #: 6089.7ms and warm ones 267.7 / 271.7 / 322.6 / 384.7 / 289.3 / 271.5ms, and three
    #: consecutive live QUALITY cross-language queries each refused at ~1.53s with `ollama ps`
    #: still empty afterwards -- every English query refused, forever, with `degradations`
    #: empty. Nothing in startup loaded the model either: `Harness.warmup` resolves
    #: `default_mode` (FAST -> extractive) and `generate.warmup` only probes /api/tags. So:
    #: `generate.warm_verifier` loads it at startup, and `OllamaGenerator.entails` finishes an
    #: aborted load in the background so the next query is the warm one. The refusals this
    #: ceiling does produce now carry `GuardrailVerdict.degraded`, which `run_rails` turns
    #: into a trace degradation -- a run in which the verifier never executed must not look
    #: like one in which it judged every pair.
    #:
    #: RAISED 1500 -> 3000 after an independent audit measured what 1500 actually does under
    #: load. The warm figures above (260-385ms) were taken with the box otherwise idle. Timed
    #: inside a 760-call evaluation instead, the same call runs p50 1016ms, p90 1502ms, max
    #: 1524ms -- so the ceiling sat exactly on the p90 of the thing it was timing, and roughly
    #: one cross-language pair in ten was refused for being SLOW rather than for being wrong.
    #: Measured consequence: false refusals on the on-topic set read 30.9% against a 15% gate,
    #: and dropped to 16.2% on the same population once this ceiling moved. A metric that is a
    #: function of machine load is not a property of the rail.
    #:
    #: 3000ms clears the loaded max with headroom and still refuses rather than waiting out a
    #: cold load (6.2s), which is the trade the paragraph above argues for and this preserves.
    #: The cost of waiting is QUALITY-mode latency, which is not budgeted; the cost of timing
    #: out is a false refusal on a question the system could have answered.
    cross_language_timeout_ms: float = 3000.0
    #: Below this lexical-overlap-with-context ratio an answer is judged ungrounded.
    min_groundedness: float = 0.55
    #: Queries shorter than this many characters are treated as empty transcripts.
    min_query_chars: int = 3
    max_query_chars: int = 512
    refusal_message: str = (
        "I could not find that in the indexed passages, so I will not guess."
    )

    # -- harness ------------------------------------------------------------------------
    #: Per-stage wall-clock ceiling. Exceeding it degrades rather than fails the request.
    #: Measured on the 183k-chunk index (120 real queries, steady state):
    #:   dense search p50=3.2ms p90=9.0ms p99=38.5ms max=41.5ms
    #: So these ceilings are ~4x p99, not targets. They exist to bound a pathological
    #: request, and the earlier tighter values (80ms dense / 60ms lexical) were firing on
    #: COLD requests -- the first query in a language pays for loading that language's
    #: segments and BM25 index -- which silently degraded a correct answer to lexical-only.
    #: A slightly slower correct answer beats a fast worse one, and the trace records the
    #: real time either way, so nothing is hidden by being generous here.
    stage_timeout_ms: dict[str, float] = Field(
        default_factory=lambda: {
            "query_embed": 60.0,
            # 150ms was too tight and it silently destroyed hybrid retrieval. Under
            # concurrency (4 in flight against a 4-CPU Qdrant) dense search runs 68ms at p50
            # and 152ms at p100 -- right on the old ceiling -- so the arm timed out on most
            # requests and RRF fused a single arm. Measured consequence: hybrid scored
            # IDENTICALLY to lexical (R@10 0.435, nDCG 0.275) while dense alone scored 0.509
            # / 0.349. The system was quietly running as lexical-only and reporting itself
            # as hybrid.
            #
            # 400ms is a bound on pathological requests, not a target: the budget is enforced
            # by measuring CORE, and a stage ceiling that fires in normal operation trades a
            # correct answer for a fast worse one.
            "retrieve_dense": 400.0,
            "retrieve_lexical": 200.0,
            "rerank": 120.0,
            "generate": 25_000.0,
            "input_guardrail": 25.0,
            "output_guardrail": 40.0,
        }
    )
    max_retries: int = 2
    retry_backoff_ms: float = 40.0
    #: Consecutive failures before a provider is tripped out of the rotation.
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_s: float = 30.0

    # -- budget / analytics -------------------------------------------------------------
    #: The task's contract: chunking + retrieval + everything to final output.
    latency_budget_ms: float = 200.0
    #: Percentiles reported by the benchmark. The task asks for P50/P70/P100.
    report_percentiles: tuple[float, ...] = (50.0, 70.0, 90.0, 95.0, 99.0, 100.0)
    trace_ring_size: int = 5000

    # -- ingest -------------------------------------------------------------------------
    hf_dataset: str = "ai4bharat/MSMARCO-XI"
    #: Languages actually indexed. Kept small by default so a laptop can rebuild fast.
    #: Languages the system will serve. This is NOT only an ingest knob: `LanguageRail` reads
    #: it to decide which scripts the corpus can answer in, so a language missing here is
    #: REFUSED at the door with "that language is not in the indexed corpus" even when its
    #: vectors are sitting in Qdrant. Leaving it at a three-language default after indexing
    #: all fourteen silently disabled eleven of them.
    #:
    #: Keep it in step with what is actually indexed. Narrow it only to serve a subset
    #: deliberately, and know that doing so turns those languages off for users too.
    ingest_langs: list[Lang] = Field(
        default_factory=lambda: [
            Lang.ASM,
            Lang.BEN,
            Lang.GUJ,
            Lang.HIN,
            Lang.KAN,
            Lang.MAL,
            Lang.MAR,
            Lang.NEP,
            Lang.ORI,
            Lang.PAN,
            Lang.SAN,
            Lang.TAM,
            Lang.TEL,
            Lang.URD,
        ]
    )
    #: Rows per language pulled from the validation split. None means "all of it".
    ingest_limit_per_lang: int | None = 20_000
    ingest_split: str = "validation"
    hf_token: str | None = None

    # -- server -------------------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
