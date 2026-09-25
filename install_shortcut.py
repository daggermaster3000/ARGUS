"""Create the "Microscopy Viewer" desktop shortcut, on Windows or macOS.

Run once, from the Python environment that has napari installed::

    python install_shortcut.py

On Windows the shortcut points at ``pythonw.exe`` from that same environment with
``launch_viewer.py`` as its argument, so double-clicking it starts the viewer
with no console window and no manual environment activation. Because the target
is an executable, Windows appends dropped file paths to the argument list, which
is what makes drag-and-drop onto the shortcut work.

On macOS it is a small ``Microscopy Viewer.app`` whose only job is to run that
same interpreter on ``launch_viewer.py``. Files dropped on it (or opened with
*Open With*) reach the viewer as ``FileOpen`` events, which it listens for.

Other useful forms::

    python install_shortcut.py --start-menu     # also add a Start Menu entry (macOS: ~/Applications)
    python install_shortcut.py --console        # keep a console for debugging
    python install_shortcut.py --uninstall      # remove the shortcuts again
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SHORTCUT_NAME = "Microscopy Viewer"
PROJECT_ROOT = Path(__file__).resolve().parent
LAUNCHER = PROJECT_ROOT / "launch_viewer.py"
ASSETS = PROJECT_ROOT / "assets"
DESCRIPTION = "Open microscopy images (.ims, TIFF, OME-TIFF, OME-Zarr) in napari"
IS_MAC = sys.platform == "darwin"
#: Written into the macOS app, and checked before ``--uninstall`` deletes one, so
#: an unrelated app that happens to share the name is never removed.
BUNDLE_ID = "local.microscopy-viewer.launcher"


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


def ensure_icon(suffix: str = ".ico") -> Path | None:
    """A stable icon path for the shortcut, copied into ``assets/``.

    napari ships an icon (``.ico`` for Windows, ``.icns`` for macOS); copying it
    means the shortcut keeps working if napari is later reinstalled or upgraded.
    """
    target = ASSETS / f"microscopy_viewer{suffix}"
    if target.exists():
        return target
    try:
        import napari

        source = Path(napari.__file__).parent / "resources" / f"icon{suffix}"
    except Exception:
        return None
    if not source.exists():
        return None
    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def special_folder(name: str) -> Path:
    """Resolve ``Desktop`` or ``Programs`` via the shell, honouring OneDrive redirection.

    On macOS ``Programs`` is the user's own ``~/Applications``, which Launchpad
    and Spotlight index, and which needs no administrator rights.
    """
    if IS_MAC:
        folder = Path.home() / ("Desktop" if name == "Desktop" else "Applications")
        if name != "Desktop":
            folder.mkdir(exist_ok=True)
        if not folder.is_dir():
            raise InstallError(f"could not locate {folder}")
        return folder
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


def create_shortcut(directory: Path, name: str, console: bool) -> Path:
    """Write ``<directory>/<name>.lnk`` pointing at the launcher."""
    if not LAUNCHER.exists():
        raise InstallError(f"launcher not found: {LAUNCHER}")

    interpreter = find_interpreter(console)
    if not interpreter.exists():
        raise InstallError(f"interpreter not found: {interpreter}")

    link = directory / f"{name}.lnk"
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
# macOS: a minimal .app bundle
# ---------------------------------------------------------------------------


def _document_extensions() -> list[str]:
    """File extensions Finder should let the user drop on the app."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from microscopy_viewer.loaders import SUPPORTED_SUFFIXES
    except Exception:
        SUPPORTED_SUFFIXES = (".ims", ".imaris", ".tif", ".tiff", ".btf", ".tf8", ".lsm", ".stk", ".zarr", ".ngff")
    return [suffix.lstrip(".") for suffix in SUPPORTED_SUFFIXES]


def _mac_launcher_script(interpreter: Path) -> str:
    """The shell script the app (or ``.command`` file) runs.

    Finder starts apps with a bare environment, so the interpreter is named by
    its full path and its ``bin`` folder is put on ``PATH`` for anything the
    viewer shells out to (ffmpeg, for movies). ``exec`` keeps the process the one
    macOS launched, which is what lets dropped files be delivered to it.
    """
    q = shlex.quote
    return f"""#!/bin/sh
# Written by install_shortcut.py. Run it again after moving the project or
# changing Python environment.
PATH={q(str(interpreter.parent))}:"$PATH"
export PATH
cd {q(str(PROJECT_ROOT))} || exit 1
# Older macOS adds a -psn_... process serial number argument; it is not a file.
for arg do
    shift
    case "$arg" in
        -psn_*) ;;
        *) set -- "$@" "$arg" ;;
    esac
done
exec {q(str(interpreter))} {q(str(LAUNCHER))} "$@"
"""


def _mac_info_plist(name: str, icon: Path | None) -> dict:
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from microscopy_viewer import __version__ as version
    except Exception:
        version = "1.0"
    info = {
        "CFBundleName": name,
        "CFBundleDisplayName": name,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": "microscopy-viewer",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "NSHighResolutionCapable": True,
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "Microscopy image",
                "CFBundleTypeRole": "Viewer",
                "CFBundleTypeExtensions": _document_extensions(),
                # Offered under Open With, never made the default for .tif.
                "LSHandlerRank": "Alternate",
            },
            {
                # OME-Zarr stores are folders.
                "CFBundleTypeName": "Folder",
                "CFBundleTypeRole": "Viewer",
                "LSItemContentTypes": ["public.folder"],
                "LSHandlerRank": "None",
            },
        ],
    }
    if icon is not None:
        info["CFBundleIconFile"] = icon.name
    return info


def _is_our_app(bundle: Path) -> bool:
    try:
        with open(bundle / "Contents" / "Info.plist", "rb") as handle:
            return plistlib.load(handle).get("CFBundleIdentifier") == BUNDLE_ID
    except (OSError, plistlib.InvalidFileException):
        return False


def _mac_interpreter() -> Path:
    # Not resolved: a virtual environment's python is a symlink, and following
    # it would start the base interpreter without the environment's packages.
    return Path(sys.executable)


def create_mac_shortcut(directory: Path, name: str, console: bool) -> Path:
    """Write ``<directory>/<name>.app``, or ``<name>.command`` with *console*.

    A ``.command`` file opens in Terminal, so the log stays in view.
    """
    if not LAUNCHER.exists():
        raise InstallError(f"launcher not found: {LAUNCHER}")
    interpreter = _mac_interpreter()
    if not interpreter.exists():
        raise InstallError(f"interpreter not found: {interpreter}")
    script = _mac_launcher_script(interpreter)

    if console:
        command = directory / f"{name}.command"
        command.write_text(script, encoding="utf-8")
        command.chmod(0o755)
        return command

    bundle = directory / f"{name}.app"
    if bundle.exists():
        if not _is_our_app(bundle):
            raise InstallError(f"{bundle} exists and was not made by this script; not replacing it")
        shutil.rmtree(bundle)
    macos = bundle / "Contents" / "MacOS"
    resources = bundle / "Contents" / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir()

    executable = macos / "microscopy-viewer"
    executable.write_text(script, encoding="utf-8")
    executable.chmod(0o755)

    icon = ensure_icon(".icns")
    if icon is not None:
        shutil.copyfile(icon, resources / icon.name)
    with open(bundle / "Contents" / "Info.plist", "wb") as handle:
        plistlib.dump(_mac_info_plist(name, icon), handle)

    # Tell Launch Services about it now, so Finder shows the icon and offers it
    # under Open With without waiting to notice the new app by itself.
    lsregister = Path(
        "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
        "LaunchServices.framework/Support/lsregister"
    )
    if lsregister.exists():
        subprocess.run([str(lsregister), "-f", str(bundle)], capture_output=True)
    return bundle


def remove_mac_shortcut(directory: Path, name: str) -> list[Path]:
    removed = []
    bundle = directory / f"{name}.app"
    if bundle.exists() and _is_our_app(bundle):
        shutil.rmtree(bundle)
        removed.append(bundle)
    command = directory / f"{name}.command"
    if command.exists() and str(LAUNCHER) in command.read_text(encoding="utf-8", errors="replace"):
        command.unlink()
        removed.append(command)
    return removed


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=SHORTCUT_NAME, help=f"shortcut name (default: {SHORTCUT_NAME})")
    parser.add_argument(
        "--start-menu", "--applications", action="store_true",
        help="also create a Start Menu entry (macOS: an app in ~/Applications)",
    )
    parser.add_argument("--no-desktop", action="store_true", help="skip the desktop shortcut")
    parser.add_argument(
        "--console", action="store_true",
        help="keep a console open: python.exe on Windows, a Terminal .command file on macOS",
    )
    parser.add_argument("--uninstall", action="store_true", help="delete the shortcuts instead of creating them")
    args = parser.parse_args(argv)

    if not IS_MAC and sys.platform != "win32":
        print("Shortcuts are made on Windows and macOS only; run `microscopy-viewer` instead.")
        return 1

    targets: list[tuple[str, Path]] = []
    if not args.no_desktop:
        targets.append(("Desktop", special_folder("Desktop")))
    if args.start_menu:
        targets.append(("Applications" if IS_MAC else "Start Menu", special_folder("Programs")))
    if not targets:
        print("Nothing to do: --no-desktop was given without --start-menu.")
        return 1

    if args.uninstall:
        for label, directory in targets:
            if IS_MAC:
                removed = remove_mac_shortcut(directory, args.name)
                for path in removed:
                    print(f"Removed: {path}")
                if not removed:
                    print(f"Not present in {label}: {args.name}.app")
            elif remove_shortcut(directory, args.name):
                print(f"Removed: {directory / (args.name + '.lnk')}")
            else:
                print(f"Not present on the {label}: {args.name}.lnk")
        return 0

    problems = _check_environment()
    for problem in problems:
        print(f"warning: {problem}")

    for label, directory in targets:
        if IS_MAC:
            link = create_mac_shortcut(directory, args.name, args.console)
        else:
            link = create_shortcut(directory, args.name, args.console)
        print(f"{label} shortcut created: {link}")

    print()
    print(f"Interpreter : {_mac_interpreter() if IS_MAC else find_interpreter(args.console)}")
    print(f"Launcher    : {LAUNCHER}")
    print("Double-click the shortcut to start, or drop image files onto it to open them.")
    if IS_MAC:
        print("Drag it into the Dock to keep it there.")
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
