"""The guided tour: what it says, where it points, and whether it has been seen.

The tour walks through one experiment the way it is actually done — find the
samples, outline the regions, segment, analyse — pointing at the control for
each step. The overlay that draws it is :mod:`microscopy_viewer.widgets.tour`;
this module is the script, kept free of Qt so it can be read and checked on its
own.

Each step names its target by panel and widget attribute rather than holding a
widget, because the panels are built after this module is imported and any of
them can fail to build (a missing optional dependency). A step whose target is
missing is still shown, centred, so the tour never breaks halfway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .utils import get_logger

logger = get_logger("onboarding")

#: Bump when the tour changes enough that people who saw the old one should see
#: the new one once.
TOUR_VERSION = 1


@dataclass(frozen=True)
class TourStep:
    """One stop of the tour."""

    title: str
    body: str
    #: Panel identifier in ``app.docks`` to bring forward, or "" for none.
    panel: str = ""
    #: Attribute path from the app to the widget to highlight, e.g.
    #: ``"experiment_widget._choose_button"``. Empty centres the bubble.
    target: str = ""
    #: Tab of a ``QTabWidget`` (``<panel widget>._tabs``) to switch to first.
    tab: str = ""


TOUR: tuple[TourStep, ...] = (
    TourStep(
        "Welcome to ARGUS",
        "A quick tour of how to process one experiment, from a folder of .ims files. "
        "It takes about a minute.\n\n"
        "Press Next to continue, Back to go back, and End tour or Esc to stop at any "
        "time. The Tour button in the toolbar starts it again.",
    ),
    TourStep(
        "Open a single file",
        "For one dataset, Open Images (or drag files onto the window) adds it to the "
        "viewer. For a whole experiment, use the folder panel on the left instead.",
        target="toolbar._buttons.open",
    ),
    TourStep(
        "1 · Choose the experiment folder",
        "Pick the folder the acquisition was saved into. Every readable file gets a "
        "thumbnail; nothing is opened yet.",
        panel="experiment",
        target="experiment_widget._choose_button",
    ),
    TourStep(
        "2 · Browse the samples",
        "Double-click a sample to show it. Opening another replaces it, and each "
        "sample's stored brain regions and label maps come up with it.\n\n"
        "Select several (Shift / Ctrl-click) to write to or process them together.",
        panel="experiment",
        target="experiment_widget._grid",
    ),
    TourStep(
        "3 · Outline the brain regions",
        "Add region puts the canvas into drawing mode: click around a region, "
        "double-click to close it, then type its name in the table.",
        panel="regions",
        target="regions_widget._add_button",
    ),
    TourStep(
        "4 · Save the outlines into the sample",
        "The outlines are written inside the .ims itself, so they travel with the "
        "file and come back whenever the sample is opened.",
        panel="regions",
        target="regions_widget._save_button",
    ),
    TourStep(
        "…or into many samples at once",
        "When the samples are mounted alike, one set of outlines can be written into "
        "every selected sample from here.",
        panel="experiment",
        target="experiment_widget._write_rois_button",
    ),
    TourStep(
        "5 · Set up segmentation",
        "Choose the Cellpose model, 2D or 3D mode (we will do 2D + Stitch), the expected nucleus diameter and "
        "the device. Segment tries it on the channel on screen.",
        panel="segmentation",
        tab="Setup",
        target="segmentation_widget._run_button",
    ),
    TourStep(
        "6 · Segment the whole folder",
        "Run on selected segments every sample selected in the folder panel with "
        "those settings, and writes each label map back into its own file.",
        panel="segmentation",
        tab="Batch",
        target="segmentation_widget.batch._run_button",
    ),
    TourStep(
        "7 · Analyse",
        "Analyse selected reads the stored labels and outlines back out of the "
        "files and measures every region and every cell. Nothing has to be open.",
        panel="analysis",
        target="analysis_widget._run_button",
    ),
    TourStep(
        "8 · The report",
        "Each run leaves a dated folder in the experiment folder: the workbook, "
        "violin plots, a cell PCA, and cell_outlines.npz for the region explorer "
        "app (apps/region_explorer.py). Save report to… writes another copy.",
        panel="analysis",
        target="analysis_widget._export_button",
    ),
    TourStep(
        "That's it",
        "Hover over any button for details. The Tour button in the toolbar runs this "
        "again whenever you want it.",
        target="toolbar._buttons.tour",
    ),
)


def resolve(root, path: str):
    """Follow a dotted attribute path from *root*; dict keys work too. ``None`` if broken."""
    current = root
    for part in path.split(".") if path else ():
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


# ---------------------------------------------------------------------------
# Whether it has been seen
# ---------------------------------------------------------------------------


def state_file() -> Path:
    from .runtime import app_data_dir

    return app_data_dir() / "onboarding.json"


def has_seen(path: Path | None = None) -> bool:
    target = path or state_file()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return int(data.get("tour_version", 0)) >= TOUR_VERSION


def mark_seen(path: Path | None = None, finished: bool = True) -> None:
    """Remember the tour was shown — finished or stopped, it is not shown again."""
    target = path or state_file()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"tour_version": TOUR_VERSION, "finished": bool(finished)}),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("could not record that the tour was seen", exc_info=True)
