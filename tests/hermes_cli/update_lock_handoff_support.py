"""Run marker assertions in a real child of the test's orchestrating updater."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent

from hermes_cli.update_lock import HANDOFF_PID_ENV

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_handoff_child(marker, body, *, via=(), handoff=True):
    """Keep the claimant below the owner, regardless of the test runner's ancestors.

    A sleeper child named in HANDOFF_PID_ENV is not a parent: the runtime-host
    fence correctly keeps walking above pytest when it cannot reach that PID.
    Do not bypass that fence to test marker delegation or legacy handoffs.
    ``via`` inserts a controlled stage (or runtime-host stand-in) between them.
    """
    preamble = (
        f"import os, sys, time\nfrom pathlib import Path\nsys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from hermes_cli.update_lock import UpdateLock, read_live_update\n"
        f"marker = Path({str(marker)!r})\n"
    )
    command = [sys.executable, "-B", "-c", preamble + dedent(body)]
    env = {**os.environ}
    env.pop(HANDOFF_PID_ENV, None)
    if handoff:
        env[HANDOFF_PID_ENV] = str(os.getpid())
    result = subprocess.run(
        [*via, *command], env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
