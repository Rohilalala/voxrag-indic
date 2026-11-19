"""Launch a command in its own session so it outlives the shell that started it.

macOS ships no `setsid`, and `nohup ... &` is not enough on its own: the job stays in the
launching process group, so when the parent tool tears that group down the job dies with it.
A long index build was killed at 6% exactly this way. `start_new_session=True` calls
`setsid()` in the child, which is the portable equivalent.

Usage: python scripts/detach.py <logfile> <command> [args...]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__.strip())
        return 2

    log_path, command = Path(argv[1]), argv[2:]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    print(f"detached pid={proc.pid} log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
