"""Launch a long job in its own session, so it survives the tool that started it.

macOS has no `setsid(1)` -- the usage line in `index_all_langs.sh` assumes GNU coreutils and
silently fails with "command not found" on this machine, which leaves the job attached to the
caller's process group and killed with it. Double-fork + os.setsid() is the portable form.

    .venv/bin/python scripts/detach_run.py data/index40.log KEY=VALUE ... -- cmd args
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    log_path, *rest = sys.argv[1:]
    env_pairs, cmd = [], []
    seen_sep = False
    for item in rest:
        if item == "--":
            seen_sep = True
        elif seen_sep:
            cmd.append(item)
        else:
            env_pairs.append(item)
    if not cmd:
        print("usage: detach_run.py <log> [ENV=VAL ...] -- <cmd> [args]", file=sys.stderr)
        return 2

    env = dict(os.environ)
    for pair in env_pairs:
        key, _, value = pair.partition("=")
        env[key] = value

    if os.fork() != 0:
        return 0                      # parent returns immediately
    os.setsid()                       # child becomes its own session leader
    if os.fork() != 0:
        os._exit(0)                   # first child exits so the job cannot reacquire a tty
    with open(log_path, "ab", buffering=0) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    os.close(0)
    os._exit(subprocess.run(cmd, env=env).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
