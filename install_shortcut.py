"""Create the "Microscopy Viewer" desktop shortcut.

Run once, from the Python environment that has napari installed::

    python install_shortcut.py

The shortcut points at ``pythonw.exe`` from that same environment with
``launch_viewer.py`` as its argument, so double-clicking it starts the viewer
with no console window and no manual environment activation. Because the target
is an executable, Windows appends dropped file paths to the argument list, which
is what makes drag-and-drop onto the shortcut work.

Other useful forms::

    python install_shortcut.py --start-menu     # also add a Start Menu entry
    python install_shortcut.py --console        # keep a console for debugging
    python install_shortcut.py --uninstall      # remove the shortcuts again
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SHORTCUT_NAME = "Microscopy Viewer"
PROJECT_ROOT = Path(__file__).resolve().parent
LAUNCHER = PROJECT_ROOT / "launch_viewer.py"

#: The dashboard shortcut. Always a console shortcut, never a windowless one: the
#: terminal is the point. A plate's first neighbour search takes the better part
#: of a minute, and a browser tab that is merely sitting there looks exactly like
#: one that has hung — the console is where it says which step it is on.
DASHBOARD_NAME = "Microscopy Viewer Dashboard"
ASSETS = PROJECT_ROOT / "assets"
DESCRIPTION = "Open microscopy images (.ims, TIFF, OME-TIFF, OME-Zarr) in napari"


class InstallError(RuntimeError):
    """Raised when the shortcut cannot be created."""


# ---------------------------------------------------------------------------
# Locating things
# ---------------------------------------------------------------------------


def find_interpreter(console: bool) -> Path:
    """The interpreter the shortcut should launch.

    ``pythonw.exe`` is preferred because it runs a GUI without a console window;
    ``--console`` selects ``python.exe`` so tracebacks stay visible.
    """
    current = Path(sys.executable).resolve()
    if console:
        return current
    candidate = current.with_name("pythonw.exe")
    if candidate.exists():
        return candidate
    # Virtual environments keep pythonw.exe in Scripts/ alongside python.exe.
    alternative = current.parent / "Scripts" / "pythonw.exe"
    if alternative.exists():
        return alternative
    print(f"note: pythonw.exe not found next to {current}; using it directly (a console will appear)")
    return current


def ensure_icon() -> Path | None:
    """A stable ``.ico`` path for the shortcut, copied into ``assets/``.

    napari ships an icon; copying it means the shortcut keeps working if napari
    is later reinstalled or upgraded.
    """
    target = ASSETS / "microscopy_viewer.ico"
    if target.exists():
        return target
    try:
        import napari

        source = Path(napari.__file__).parent / "resources" / "icon.ico"
    except Exception:
        return None
    if not source.exists():
        return None
    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def special_folder(name: str) -> Path:
    """Resolve ``Desktop`` or ``Programs`` via the shell, honouring OneDrive redirection."""
    try:
        import win32com.client

        shell = win32com.client.Dispatch("WScript.Shell")
        return Path(str(shell.SpecialFolders(name)))
    except Exception:
        pass
    if name == "Desktop":
        for candidate in (
            Path(os.environ.get("USERPROFILE", Path.home())) / "OneDrive" / "Desktop",
            Path.home() / "OneDrive" / "Desktop",
            Path(os.environ.get("USERPROFILE", Path.home())) / "Desktop",
            Path.home() / "Desktop",
        ):
            if candidate.is_dir():
                return candidate
        raise InstallError("could not locate the Desktop folder")
    programs = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    if programs.is_dir():
        return programs
    raise InstallError("could not locate the Start Menu Programs folder")


# ---------------------------------------------------------------------------
# Writing the .lnk
# ---------------------------------------------------------------------------


def _write_with_pywin32(
    link: Path, target: Path, arguments: str, workdir: Path, icon: Path | None
) -> bool:
    try:
        import win32com.client
    except ImportError:
        return False
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(str(link))
    shortcut.TargetPath = str(target)
    shortcut.Arguments = arguments
    shortcut.WorkingDirectory = str(workdir)
    shortcut.Description = DESCRIPTION
    if icon is not None:
        shortcut.IconLocation = f"{icon},0"
    shortcut.save()
    return True


def _write_with_vbscript(
    link: Path, target: Path, arguments: str, workdir: Path, icon: Path | None
) -> bool:
    """Fallback that drives the same COM object through ``cscript``.

    Used when pywin32 is unavailable, so no extra package is needed to install.
    """
    icon_line = f'link.IconLocation = "{icon},0"' if icon is not None else ""
    script = f"""
Set shell = WScript.CreateObject("WScript.Shell")
Set link = shell.CreateShortcut("{link}")
link.TargetPath = "{target}"
link.Arguments = "{arguments.replace('"', '""')}"
link.WorkingDirectory = "{workdir}"
link.Description = "{DESCRIPTION}"
{icon_line}
link.Save
"""
    handle = tempfile.NamedTemporaryFile("w", suffix=".vbs", delete=False, encoding="mbcs")
    try:
        handle.write(script)
        handle.close()
        result = subprocess.run(
            ["cscript", "//nologo", handle.name], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise InstallError(f"cscript failed: {result.stderr.strip() or result.stdout.strip()}")
        return True
    finally:
        try:
            os.unlink(handle.name)
        except OSError:
            pass


def create_shortcut(
    directory: Path, name: str, console: bool, arguments: str | None = None
) -> Path:
    """Write ``<directory>/<name>.lnk``.

    *arguments* defaults to the viewer's launcher script; the dashboard passes
    ``-m microscopy_viewer.dashboard`` instead, which is the same environment and
    a different program.
    """
    if arguments is None and not LAUNCHER.exists():
        raise InstallError(f"launcher not found: {LAUNCHER}")

    interpreter = find_interpreter(console)
    if not interpreter.exists():
        raise InstallError(f"interpreter not found: {interpreter}")

    link = directory / f"{name}.lnk"
    if arguments is None:
        arguments = f'"{LAUNCHER}"'
    icon = ensure_icon()

    if not _write_with_pywin32(link, interpreter, arguments, PROJECT_ROOT, icon):
        _write_with_vbscript(link, interpreter, arguments, PROJECT_ROOT, icon)

    if not link.exists():
        raise InstallError(f"shortcut was not created at {link}")
    return link


def remove_shortcut(directory: Path, name: str) -> bool:
    link = directory / f"{name}.lnk"
    if link.exists():
        link.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _check_environment() -> list[str]:
    """Warn about anything missing before the user double-clicks the shortcut."""
    problems: list[str] = []
    for module, note in (
        ("napari", "required — the viewer itself"),
        ("qtpy", "required — Qt bindings wrapper"),
        ("h5py", "required for .ims files"),
        ("tifffile", "required for TIFF/OME-TIFF"),
        ("pandas", "required for Excel export"),
        ("openpyxl", "required for Excel export"),
        ("zarr", "optional — OME-Zarr support"),
        ("imageio_ffmpeg", "optional — .mov / .mp4 movie export"),
    ):
        try:
            __import__(module)
        except Exception:
            problems.append(f"{module} is not installed ({note})")
    return problems


def _check_dashboard() -> list[str]:
    """Warn about anything the dashboard needs and does not have."""
    problems: list[str] = []
    for module, note in (
        ("streamlit", "required — the dashboard is a Streamlit page"),
        ("anndata", "required — it reads .h5ad"),
        ("scanpy", "required — scaling, PCA, clustering"),
        ("squidpy", "required — the spatial statistics"),
        ("matplotlib", "required — every plot on the page"),
        ("leidenalg", "strongly recommended — without it the clustering falls back to k-means"),
        ("sklearn", "recommended — the exact neighbour search, four times faster than the default"),
    ):
        try:
            __import__(module)
        except Exception:
            problems.append(f"{module} is not installed ({note})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=None, help=f"shortcut name (default: {SHORTCUT_NAME})")
    parser.add_argument("--start-menu", action="store_true", help="also create a Start Menu entry")
    parser.add_argument("--no-desktop", action="store_true", help="skip the desktop shortcut")
    parser.add_argument("--console", action="store_true", help="launch via python.exe so a console stays open")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="make the spatial dashboard's shortcut instead of the viewer's; it always "
        "keeps its console, which is where the timings appear",
    )
    parser.add_argument("--uninstall", action="store_true", help="delete the shortcuts instead of creating them")
    args = parser.parse_args(argv)

    # The dashboard is a console program by definition, whatever --console says.
    console = bool(args.console or args.dashboard)
    arguments = "-m microscopy_viewer.dashboard" if args.dashboard else None
    name = args.name or (DASHBOARD_NAME if args.dashboard else SHORTCUT_NAME)

    targets: list[tuple[str, Path]] = []
    if not args.no_desktop:
        targets.append(("Desktop", special_folder("Desktop")))
    if args.start_menu:
        targets.append(("Start Menu", special_folder("Programs")))
    if not targets:
        print("Nothing to do: --no-desktop was given without --start-menu.")
        return 1

    if args.uninstall:
        for label, directory in targets:
            if remove_shortcut(directory, name):
                print(f"Removed: {directory / (name + '.lnk')}")
            else:
                print(f"Not present on the {label}: {name}.lnk")
        return 0

    problems = _check_dashboard() if args.dashboard else _check_environment()
    for problem in problems:
        print(f"warning: {problem}")

    for label, directory in targets:
        link = create_shortcut(directory, name, console, arguments)
        print(f"{label} shortcut created: {link}")

    print()
    print(f"Interpreter : {find_interpreter(console)}")
    if args.dashboard:
        print("Runs        : -m microscopy_viewer.dashboard")
        print(
            "Double-click the shortcut to start the dashboard. A terminal opens with it "
            "and stays open: that is where the timings appear, and closing it stops the "
            "server."
        )
    else:
        print(f"Launcher    : {LAUNCHER}")
        print("Double-click the shortcut to start, or drop image files onto it to open them.")
    if problems:
        print("\nInstall the missing packages listed above before using the shortcut.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
