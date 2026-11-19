"""Dense embedding. One ONNX encoder, one process-wide instance.

    from voxrag.embed import get_embedder

    emb = get_embedder()                       # loads + warms once per process
    q = await emb.encode_async(["नमस्ते"], is_query=True)   # (1, 384) float32, unit norm
    p = emb.encode(passages)                   # is_query=False -> "passage: " prefix

`get_embedder()` raises `EmbedderUnavailableError` if no ONNX graph can be resolved;
callers should treat that as "dense retrieval is unavailable" and degrade to lexical
rather than substituting fake vectors.
"""

from __future__ import annotations

from voxrag.embed.onnx_encoder import EmbedderUnavailableError, OnnxEncoder, get_embedder

__all__ = ["EmbedderUnavailableError", "OnnxEncoder", "get_embedder"]
