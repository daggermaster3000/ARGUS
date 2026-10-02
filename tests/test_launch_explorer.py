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
    monkeypatch.setattr(le, "is_current", lambda app: True)

    assert le.ensure_running("region") == (True, True)
    running["now"] = True
    assert le.ensure_running("region") == (True, False)
    assert started == ["region"], "only the first launch started a server"


def test_stop_with_nothing_running_is_harmless(state):
    assert le.stop("region") is False


def _server_on_the_port(monkeypatch, current: bool, ours: bool):
    """A server answering on the port; *current*: today's code; *ours*: we can stop it."""
    calls = {"started": [], "stopped": []}
    answering = {"now": True}
    monkeypatch.setattr(le, "is_running", lambda app, timeout=1.0: answering["now"])
    monkeypatch.setattr(le, "is_current", lambda app: current)

    def stop(app):
        calls["stopped"].append(app)
        if ours:
            answering["now"] = False
        return ours

    monkeypatch.setattr(le, "stop", stop)
    monkeypatch.setattr(le, "start_server", lambda app: calls["started"].append(app) or None)
    monkeypatch.setattr(le, "wait_until_running", lambda app, process=None: True)
    monkeypatch.setattr(le, "_wait_until_stopped", lambda app, timeout=15.0: not answering["now"])
    return calls


def test_a_server_from_before_an_update_is_restarted(state, monkeypatch):
    calls = _server_on_the_port(monkeypatch, current=False, ours=True)

    assert le.ensure_running("region") == (True, True)
    assert calls == {"started": ["region"], "stopped": ["region"]}


def test_a_server_that_is_not_ours_is_never_opened(state, monkeypatch):
    calls = _server_on_the_port(monkeypatch, current=False, ours=False)
    opened = []
    monkeypatch.setattr(le.webbrowser, "open", lambda url, new=0: opened.append(url))

    with pytest.raises(le.ForeignServer):
        le.ensure_running("region")
    assert le.main([]) == 1
    assert calls["started"] == [] and opened == []


def test_the_current_server_is_reused_as_is(state, monkeypatch):
    calls = _server_on_the_port(monkeypatch, current=True, ours=True)

    assert le.ensure_running("region") == (True, False)
    assert calls == {"started": [], "stopped": []}


def test_current_means_this_app_this_code_and_alive(state, monkeypatch):
    monkeypatch.setattr(le, "_runs", lambda pid, script: pid == 42)
    record = state / "explorer_region.json"

    def write(**changes):
        entry = {"pid": 42, "port": 1, "script": str(le.APPS["region"]),
                 "version": le.code_version(), **changes}
        record.write_text(le.json.dumps(entry), encoding="utf-8")

    write()
    assert le.is_current("region")
    write(version="0")  # started before the code changed
    assert not le.is_current("region")
    write(script=str(le.APPS["simple"]))  # the other explorer
    assert not le.is_current("region")
    write(pid=7)  # that server is gone
    assert not le.is_current("region")
    record.write_text('{"pid": 42, "port": 1}', encoding="utf-8")  # a launcher from before
    assert not le.is_current("region")


def test_editing_the_package_changes_the_code_version(tmp_path, monkeypatch):
    (tmp_path / "apps").mkdir()
    (tmp_path / "microscopy_viewer" / "widgets").mkdir(parents=True)
    module = tmp_path / "microscopy_viewer" / "widgets" / "x.py"
    module.write_text("")
    (tmp_path / "apps" / "app.py").write_text("")
    monkeypatch.setattr(le, "PROJECT_ROOT", tmp_path)
    before = le.code_version()

    later = time.time() + 10
    os.utime(module, (later, later))

    assert le.code_version() != before


def test_stop_leaves_a_reused_pid_alone(state, monkeypatch):
    killed = []
    monkeypatch.setattr(le, "_runs", lambda pid, script: False)
    monkeypatch.setattr(le.os, "kill", lambda pid, sig: killed.append(pid))
    (state / "explorer_region.json").write_text('{"pid": 4242}', encoding="utf-8")

    assert le.stop("region") is False
    assert killed == []
    assert not (state / "explorer_region.json").exists()
