"""The explorer launcher: one server, found again rather than started twice."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import launch_explorer as le  # noqa: E402


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(le, "_state_dir", lambda: tmp_path)
    # Ports nothing listens on, so no real server is ever touched.
    monkeypatch.setattr(le, "PORTS", {"region": 1, "simple": 2})
    return tmp_path


def test_nothing_listening_is_not_running(state):
    assert not le.is_running("region", timeout=0.2)


def test_a_workbook_travels_in_the_url(state):
    url = le.url_of("region", "/data/my report/report.xlsx")
    assert url == "http://localhost:1/?workbook=%2Fdata%2Fmy+report%2Freport.xlsx"
    assert le.url_of("simple") == "http://localhost:2/"


def test_launches_at_the_same_moment_take_turns(state):
    inside, overlaps = [0], []

    def launch():
        with le._Lock(state / "explorer_region.lock"):
            inside[0] += 1
            overlaps.append(inside[0])
            time.sleep(0.05)
            inside[0] -= 1

    threads = [threading.Thread(target=launch) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert overlaps == [1, 1, 1, 1], "never two holders at once"
    assert not (state / "explorer_region.lock").exists()


def test_a_lock_left_by_a_dead_launch_is_taken_over(state):
    lock = state / "explorer_region.lock"
    lock.write_text("12345")
    old = time.time() - le.STALE_LOCK_S - 5
    os.utime(lock, (old, old))
    with le._Lock(lock):
        assert lock.read_text() == str(os.getpid())


def test_a_second_launch_finds_the_running_server(state, monkeypatch):
    started = []
    monkeypatch.setattr(le, "start_server", lambda app: started.append(app) or None)
    monkeypatch.setattr(le, "wait_until_running", lambda app, process=None: True)
    running = {"now": False}
    monkeypatch.setattr(le, "is_running", lambda app, timeout=1.0: running["now"])

    assert le.ensure_running("region") == (True, True)
    running["now"] = True
    assert le.ensure_running("region") == (True, False)
    assert started == ["region"], "only the first launch started a server"


def test_stop_with_nothing_running_is_harmless(state):
    assert le.stop("region") is False
