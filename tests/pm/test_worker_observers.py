"""Progress observers cannot stall other range writers in the isolated worker."""
from __future__ import annotations

import json
import threading
import time

import pytest

from pm import paths
from pm.lock import Facts
from tests.pm._fixtures import client as client, isolated_python as isolated_python
from tests.pm._range_server import RangeHandler, dl_server as dl_server
from tests.pm.test_worker import _node_archive


@pytest.mark.platforms("posix")
def test_slow_worker_observer_does_not_hold_range_writers(client, dl_server, monkeypatch):
    # Every range contains several chunks: one write per range could otherwise
    # appear to make progress even while all writers wait on the bitmap lock.
    _node_archive(dl_server, b"#!/bin/sh\nexit 0\n#" + b"x" * (64 << 20))
    snapshots = []
    advanced = []
    release = threading.Event()
    # Each response sends exactly one downloader chunk before rendezvousing
    # with the callback. No range can finish before the observer starts.
    arrivals = threading.Barrier(9)

    def before_chunk(handler, start, served):
        if served == 1 << 20:
            arrivals.wait(timeout=15)
            assert release.wait(timeout=15), "observer never released response bytes"

    monkeypatch.setattr(RangeHandler, "before_range_chunk", before_chunk)

    def observe(done, total, ranges):
        snapshots.append((done, total, ranges))
        if not done or advanced:
            return
        part, = paths.partials_root().glob("*.part")
        arrivals.wait(timeout=15)
        # Use the actual requests, not their arrival order. A writer writes
        # before acquiring the bitmap lock, so its SECOND chunk could land on
        # the old implementation too. Only a third chunk proves it got past
        # that lock while this callback remained blocked across worker IPC.
        positions = [start + (2 << 20) for _, start, _ in RangeHandler.ranges_seen]
        assert len(positions) == 8
        with part.open("rb") as stream:
            for position in positions:
                stream.seek(position)
                assert stream.read(1) == b"\0", "response escaped the gate"
        release.set()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with part.open("rb") as stream:
                written = 0
                for position in positions:
                    stream.seek(position)
                    written += stream.read(1) == b"x"
            if written >= 2:
                advanced.append(True)
                break
            time.sleep(0.01)
        else:
            advanced.append(False)

    try:
        client.ensure("node", explicit=True, download_progress=observe)
    finally:
        release.set()
        arrivals.abort()
    assert advanced == [True], "range writers stalled until the progress callback returned"
    assert snapshots[-1][0] == snapshots[-1][1]
    assert [done for done, _, _ in snapshots] == sorted(done for done, _, _ in snapshots)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("action", ["pause", "error"])
def test_worker_observer_failure_preserves_resumable_bytes(client, dl_server, action):
    from pm.downloader import DownloadPaused

    _node_archive(dl_server, b"#!/bin/sh\nexit 0\n#" + b"x" * (32 << 20))
    pause = threading.Event()
    failure = ValueError("observer failed")
    seen = []

    def observe(done, total, ranges):
        seen.append((done, total, ranges))
        if done:
            if action == "pause":
                pause.set()
            else:
                raise failure

    with pytest.raises(DownloadPaused if action == "pause" else ValueError) as caught:
        client.ensure("node", explicit=True, pause_event=pause, download_progress=observe)
    if action == "error":
        assert caught.value is failure
    assert Facts(paths.facts_path()).get("node") is None
    sidecar, = paths.partials_root().glob("*.ranges")
    durable = sum(end - start for start, end in json.loads(sidecar.read_text())["ranges"])
    assert durable >= seen[-1][0] and durable > 0
    pause.clear()
    resumed = []
    client.ensure("node", explicit=True, pause_event=pause,
                  download_progress=lambda *args: resumed.append(args))
    assert resumed[0][0] == durable
    assert resumed[-1][0] == resumed[-1][1]
    assert not list(paths.partials_root().glob("*.part"))
    assert Facts(paths.facts_path()).get("node") is not None
