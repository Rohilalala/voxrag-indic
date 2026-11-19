"""The `voxrag` command.

    voxrag ingest      [...]   download MSMARCO-XI and write data/docs.jsonl
    voxrag index       [...]   chunk, embed and index that corpus
    voxrag strategies  [...]   the 8-way chunking comparison table
    voxrag bench       [...]   latency benchmark: P50 / P70 / P100 and the budget verdict
    voxrag serve       [...]   run the API + demo UI

Each subcommand forwards its arguments untouched to the module that owns the work, so
`voxrag ingest --help` prints the ingest module's own options rather than a copy of them
that would drift. Only `serve` is parsed here, because uvicorn is the thing being
configured and no other module owns it.

Delegation targets are imported lazily. `voxrag serve` must not pay for pyarrow and
huggingface_hub on the way to opening a socket, and `voxrag ingest` must not load
onnxruntime.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from voxrag.config import settings

#: subcommand -> (module path, one-line help). Resolved to `<module>.main(argv)` at call time.
_DELEGATES: dict[str, tuple[str, str]] = {
    "ingest": ("voxrag.ingest.msmarco", "download MSMARCO-XI parquet -> data/docs.jsonl"),
    "index": ("voxrag.index", "chunk + embed + index a corpus into the vector store"),
    "strategies": ("voxrag.chunking.evaluate", "compare every chunking strategy"),
    "bench": ("voxrag.bench.latency", "latency benchmark with P50/P70/P100 analytics"),
}

_USAGE = f"""usage: voxrag <command> [options]

commands:
  ingest       {_DELEGATES["ingest"][1]}
  index        {_DELEGATES["index"][1]}
  strategies   {_DELEGATES["strategies"][1]}
  bench        {_DELEGATES["bench"][1]}
  serve        run the API and the demo UI (default {settings.host}:{settings.port})

`voxrag <command> --help` shows that command's own options.
"""


def _delegate(command: str, argv: Sequence[str]) -> int:
    module_path, _ = _DELEGATES[command]
    try:
        module = __import__(module_path, fromlist=["main"])
    except ImportError as exc:
        # Distinguishes "that module has not been written/installed" from "that module is
        # broken", because the fix is different and only the first is expected during a build.
        print(
            f"voxrag {command}: cannot load {module_path} ({exc}).\n"
            f"If a dependency is missing, install the project: uv pip install -e '.[dev]'",
            file=sys.stderr,
        )
        return 2
    main: Callable[[Sequence[str] | None], int] | None = getattr(module, "main", None)
    if main is None:
        print(f"voxrag {command}: {module_path} has no main()", file=sys.stderr)
        return 2
    return int(main(list(argv)) or 0)


def _serve(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="voxrag serve", description="Run the API and demo UI.")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--log-level", default=settings.log_level.lower())
    parser.add_argument(
        "--reload", action="store_true", help="restart on source changes (development only)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        # Each worker loads its own ONNX session, HNSW graph and BM25 matrices -- several
        # hundred MB apiece -- and each keeps its own trace ring, so /api/analytics would
        # report whichever worker answered rather than the whole service.
        help="uvicorn workers; >1 multiplies model memory and splits the trace ring",
    )
    args = parser.parse_args(list(argv))

    import uvicorn

    uvicorn.run(
        "voxrag.api.app:create_app",
        factory=True,  # a fresh app per worker/reload, with its own warmed state
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=args.reload,
        workers=None if (args.reload or args.workers <= 1) else args.workers,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(_USAGE, end="")
        return 0 if args else 2
    if args[0] in ("-V", "--version"):
        print("voxrag 0.1.0")
        return 0

    command, rest = args[0], args[1:]
    if command == "serve":
        return _serve(rest)
    if command in _DELEGATES:
        return _delegate(command, rest)

    print(f"voxrag: unknown command {command!r}\n\n{_USAGE}", end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
