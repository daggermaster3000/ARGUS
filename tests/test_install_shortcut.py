"""The macOS app written by install_shortcut.py."""

from __future__ import annotations

import os
import plistlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import install_shortcut as shortcut  # noqa: E402

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the macOS app is a POSIX shell script")


@pytest.fixture
def no_lsregister(monkeypatch):
    """Keep Launch Services out of it: the test app lives in a temp folder."""
    real_run = shortcut.subprocess.run

    def run(command, *args, **kwargs):
        if "lsregister" in str(command[0]):
            return None
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(shortcut.subprocess, "run", run)


def test_app_bundle_runs_the_launcher(tmp_path, no_lsregister):
    bundle = shortcut.create_mac_shortcut(tmp_path, "Microscopy Viewer", console=False)

    assert bundle == tmp_path / "Microscopy Viewer.app"
    with open(bundle / "Contents" / "Info.plist", "rb") as handle:
        info = plistlib.load(handle)
    assert info["CFBundleIdentifier"] == shortcut.BUNDLE_ID
    executable = bundle / "Contents" / "MacOS" / info["CFBundleExecutable"]
    assert os.access(executable, os.X_OK)
    script = executable.read_text()
    assert str(shortcut.LAUNCHER) in script
    assert script.splitlines()[-1].startswith("exec ")
    extensions = info["CFBundleDocumentTypes"][0]["CFBundleTypeExtensions"]
    assert {"ims", "tif", "zarr"} <= set(extensions)


def test_reinstalling_replaces_our_app_but_not_someone_elses(tmp_path, no_lsregister):
    shortcut.create_mac_shortcut(tmp_path, "Microscopy Viewer", console=False)
    shortcut.create_mac_shortcut(tmp_path, "Microscopy Viewer", console=False)

    other = tmp_path / "Other.app" / "Contents"
    other.mkdir(parents=True)
    with open(other / "Info.plist", "wb") as handle:
        plistlib.dump({"CFBundleIdentifier": "com.example.other"}, handle)
    with pytest.raises(shortcut.InstallError):
        shortcut.create_mac_shortcut(tmp_path, "Other", console=False)
    assert shortcut.remove_mac_shortcut(tmp_path, "Other") == []
    assert other.exists()


def test_uninstall_removes_app_and_command(tmp_path, no_lsregister):
    bundle = shortcut.create_mac_shortcut(tmp_path, "Microscopy Viewer", console=False)
    command = shortcut.create_mac_shortcut(tmp_path, "Microscopy Viewer", console=True)
    assert command.suffix == ".command" and os.access(command, os.X_OK)

    removed = shortcut.remove_mac_shortcut(tmp_path, "Microscopy Viewer")

    assert set(removed) == {bundle, command}
    assert not bundle.exists() and not command.exists()
