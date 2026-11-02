"""Phenoscapes: the 4i processing pipeline of the brain organoid morphodynamics paper.

The pipeline itself is not in this repository. It lives in
`quadbio/morphodynamics_human_brain_organoid`_, under
``4i_analysis/4i_processing/phenoscapes``, and this module is what the
**Phenoscape** panel needs to drive it: the steps in order, the default
configuration, the folders each step reads and writes, and the command that runs
them.

.. _quadbio/morphodynamics_human_brain_organoid:
   https://github.com/quadbio/morphodynamics_human_brain_organoid

**It runs in a Python of its own.** Phenoscapes pins numpy 1.26, cellpose 3 and
itk-elastix, uses matplotlib and scipy calls that later releases removed, starts
a multiprocessing pool, and its own entry point points ``sys.stdout`` at a file
for good. None of that can share a process with napari. So the panel starts
:mod:`microscopy_viewer.phenoscape_runner` as a child process, in whichever
interpreter has phenoscapes installed, and reads what it prints.

**It is file-based, like the original.** Every step reads one folder and writes
the next, under one output directory per run, laid out exactly as
``phenoscapes.cli.process_sample`` lays it out. A folder written by the panel can
be picked up by the original command line and the other way round.

The input is the original's too: ``<input>/stitched/<sample>/*.tif``, one file
per cycle and channel, named so that :data:`DEFAULT_REGEX` (or the config's
``regex``) finds the cycle and channel in it; and a stain table, a CSV with at
least ``cycle_id``, ``channel_id`` and ``stain``, where a cycle whose stain is
empty is a background (elution) cycle.

No Qt here, and nothing from phenoscapes: this module has to import in napari's
Python, where phenoscapes does not.
"""

from __future__ import annotations

import copy
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .utils import get_logger

logger = get_logger("phenoscape")

#: Where the pipeline's source lives, for :func:`download_source`.
SOURCE_URL = (
    "https://raw.githubusercontent.com/quadbio/morphodynamics_human_brain_organoid/"
    "main/4i_analysis/4i_processing/phenoscapes/"
)
#: The package's files, as the repository has them.
SOURCE_FILES = (
    "__init__.py",
    "bg_subtraction.py",
    "cli.py",
    "cropping_denoising.py",
    "feature_extraction.py",
    "masking.py",
    "montage.py",
    "morphometrics.py",
    "registration.py",
    "requirements.txt",
    "sc.py",
    "segmentation.py",
    "speckle_removal.py",
    "utils.py",
)

#: The file name pattern phenoscapes assumes when the config gives none: the
#: names the Yokogawa stitching writes.
DEFAULT_REGEX = (
    r"_c(?P<cycle_id>\d{3})_(?P<well_id>[A-Z]\d{2})_r(?P<region_id>\d{3})_t(?P<timeline_id>\d{3})"
    r"_z\d{3}_c\d{1}_stitch_ch(?P<channel_id>\d{2})\.tif$"
)

#: The modules the runner imports for every run, which is what "phenoscapes
#: works in this Python" means. Checked one by one so a missing elastix is named
#: as such.
REQUIRED_MODULES = (
    "phenoscapes",
    "yaml",
    "pandas",
    "skimage",
    "itk",
    "smo",
    "cellpose",
    "scanpy",
    "cv2",
)

#: Modules one step needs, mapped to that step and to how to get them.
#:
#: Kept apart from :data:`REQUIRED_MODULES` because a missing one is not a
#: broken environment. ``louvain`` has no wheel past Python 3.11, so pip builds
#: igraph's C core from source and stops at "You need to install CMake before
#: compiling igraph" — and the panel used to answer that by advising the very
#: install that had just failed. Everything except the Overview plots runs
#: without it.
OPTIONAL_MODULES = {
    "louvain": (
        "the Overview plots step, for scanpy's Louvain clustering",
        "it has no wheel past Python 3.11, so pip builds igraph's C core from "
        "source: install CMake first (python -m pip install cmake), then "
        "python -m pip install louvain",
    ),
}

#: Everything the check probes: what every run needs, then what one step needs.
CHECKED_MODULES = REQUIRED_MODULES + tuple(OPTIONAL_MODULES)


@dataclass(frozen=True)
class Step:
    """One stage of the pipeline."""

    #: Name the runner knows it by.
    key: str
    #: What the panel calls it.
    title: str
    #: Where its on/off switch is in the config: ``("cellpose", "run")``.
    switch: tuple[str, str]
    #: The folder under the output directory it writes into.
    output: str
    #: One sentence for the tooltip.
    about: str


#: Every step, in the order ``process_sample`` runs them.
STEPS: tuple[Step, ...] = (
    Step("simple_masking", "Initial mask", ("simple_masking", "run"), "masks",
         "A rough organoid mask from the reference cycle: every channel maxed, "
         "blurred, Otsu-thresholded, the largest object kept."),
    Step("alignment", "Align cycles", ("alignment", "run"), "aligned",
         "Elastix, rigid then affine, every cycle onto the reference cycle's nuclei; "
         "the reference moves forward at each of the alignment reference cycles."),
    Step("alignment_check", "Alignment check", ("alignment_check", "run"), "check_aligned",
         "Overlays of the reference cycle's nuclei with two cycles at a time, as PNGs, "
         "to look at before trusting the alignment."),
    Step("refined_masking", "Refined mask", ("refined_masking", "run"), "masks_refined",
         "The mask again, from the aligned stains rather than one cycle."),
    Step("speckle_removal", "Speckle removal", ("speckle_removal", "run"), "speckle_masks",
         "Masks of antibody speckles per channel, applied during background subtraction."),
    Step("cropping_denoising", "Crop and denoise", ("cropping_denoising", "run"), "denoised",
         "Crop to the mask, optional SMO background correction, non-local-means denoising."),
    Step("bg_subtraction", "Background subtraction", ("bg_subtraction", "run"), "bg_subtracted",
         "Each stain minus the elution cycles either side of it, weighted by distance "
         "and scaled by non-negative least squares inside the mask."),
    Step("cellpose", "Segment (Cellpose)", ("cellpose", "run"), "segmented",
         "Cellpose on the reference cycle's nuclei (or the configured channel), on CPU."),
    Step("feature_extraction", "Feature tables", ("feature_extraction", "run"), "feature_tables",
         "Per-cell area, centroid and mean / median / max / min of every stain, as a CSV."),
    Step("feature_morphology_extraction", "Morphometrics",
         ("feature_morphology_extraction", "run"), "feature_tables_morphometrics",
         "Shape features of the organoid contour and of each cell relative to it."),
    Step("to_h5ad", "AnnData", ("analysis_pipeline", "to_h5ad"), "anndata",
         "The feature table as .h5ad: the chosen statistic of each stain, log1p, in X."),
    Step("label_overlay", "Label overlay", ("analysis_pipeline", "label_overlay"), "label_overlay",
         "The segmentation coloured over the nuclei, as a PNG."),
    Step("overview_montage", "Overview montage", ("analysis_pipeline", "overview_montage"),
         "montages", "Every stain side by side, annotated, as one picture."),
    Step("overview_plots", "Overview plots", ("analysis_pipeline", "overview_plots"),
         "plots_overview", "UMAP and spatial plots of Louvain clusters and every stain."),
)

STEP_KEYS = tuple(step.key for step in STEPS)


def get_step(key: str) -> Step:
    for step in STEPS:
        if step.key == key:
            return step
    raise KeyError(key)


def default_config() -> dict:
    """``phenoscapes.cli.load_default_config``, copied so it needs no phenoscapes.

    Kept key for key identical to the original, so a config written here reads
    there and a config written there loads here.
    """
    return {
        "regex": None,
        "metadata": None,
        "reference_cycle": 1,
        "stain_nuclei": "DAPI",
        "channel_nuclei": 0,
        "channel_stains": [1, 2, 3],
        "stitching": {"run": True, "channel": "01", "cycle": "cycle1", "mask": True},
        "simple_masking": {"run": True, "sigma": 100, "n_binary_operations": 50,
                           "channels": [0, 1, 2, 3]},
        "alignment": {"run": True, "reference_cycles": [1, 6, 11, 16], "mask": False,
                      "param_maps": {"rigid": None, "affine": None},
                      "mask_bright_spots": False},
        "alignment_check": {"run": True},
        "refined_masking": {"run": True, "threshold": 0.1, "sigma": 100,
                            "use_only_dapi": False, "n_binary_operations": 50, "cycle_dapi": 1},
        "speckle_removal": {"run": False, "percentile": 2},
        "cropping_denoising": {"run": True, "average_nuclei": False, "cropping": True,
                               "mask_zeros_smo": False, "denoising": True, "smo": True,
                               "multiprocessing": True, "n_processes": 8},
        "bg_subtraction": {"run": True, "bg_cycles": None, "cycle_dapi": 1, "cycles": None},
        "cellpose": {"run": True, "model": "cyto2", "segment_dual_channel": False,
                     "diameter": 20, "flow_threshold": -3, "cellprob_threshold": 1,
                     "pretrained_model": False, "segment_channel": False,
                     "segment_cycle": False},
        "feature_extraction": {"run": True, "expand_labels": None},
        "feature_morphology_extraction": {"run": False},
        "analysis_pipeline": {"run": True, "label_overlay": True, "overview_montage": True,
                              "to_h5ad": True, "overview_plots": True, "slice_step": 5000,
                              "stat": "median"},
        "cleanup": {"run": False},
    }


def complete_config(config: dict | None) -> dict:
    """*config* with every key the runner reads, the defaults filling the gaps.

    A config file written for an older phenoscapes lacks the newer keys
    (``speckle_removal``, ``mask_bright_spots``), and the original reads them
    with ``config[...]`` and dies on the first one missing. Filling them from
    the defaults is what the original would have done had it been asked.
    """
    merged = default_config()
    for key, value in (config or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def is_enabled(config: dict, step: Step) -> bool:
    section, flag = step.switch
    return bool((config.get(section) or {}).get(flag, False))


def set_enabled(config: dict, step: Step, on: bool) -> None:
    section, flag = step.switch
    config.setdefault(section, {})[flag] = bool(on)
    # The four analysis steps are gated twice in the original: by their own flag
    # and by analysis_pipeline.run over them. One switch per step on the panel
    # means the outer one follows the inner four.
    if section == "analysis_pipeline":
        block = config["analysis_pipeline"]
        block["run"] = any(
            block.get(get_step(key).switch[1], False)
            for key in ("to_h5ad", "label_overlay", "overview_montage", "overview_plots")
        )


def enabled_steps(config: dict) -> list[str]:
    """The steps the config turns on, in pipeline order."""
    return [step.key for step in STEPS if is_enabled(config, step)]


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------


def stitched_dir(dir_input: str | Path) -> Path:
    return Path(dir_input) / "stitched"


def list_samples(dir_input: str | Path) -> list[str]:
    """The samples under ``<input>/stitched``: one folder each, holding TIFFs."""
    root = stitched_dir(dir_input)
    if not root.is_dir():
        return []
    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and any(p.suffix.lower() in (".tif", ".tiff") for p in child.iterdir())
    )


def count_images(dir_input: str | Path, sample: str, regex: str | None = None) -> tuple[int, int]:
    """How many TIFFs a sample has, and how many of them the regex can read."""
    folder = stitched_dir(dir_input) / sample
    if not folder.is_dir():
        return 0, 0
    pattern = re.compile(regex or DEFAULT_REGEX, flags=re.IGNORECASE)
    files = [p.name for p in folder.iterdir() if ".tif" in p.name]
    return len(files), sum(1 for name in files if pattern.search(name))


def output_dirs(dir_output: str | Path, config: dict) -> dict[str, Path]:
    """Every folder ``process_sample`` uses, by its variable name there."""
    out = Path(dir_output)
    refined = out / "masks_refined"
    return {
        "masks": out / "masks",
        "aligned": out / "aligned",
        "check_aligned": out / "check_aligned",
        # The original's own rule: without a refined mask, the simple one stands in.
        "masks_refined": refined if is_enabled(config, get_step("refined_masking")) else out / "masks",
        "speckle_masks": out / "speckle_masks",
        "denoised": out / "denoised",
        "bg_subtracted": out / "bg_subtracted",
        "segmented": out / "segmented",
        "feature_tables": out / "feature_tables",
        "feature_tables_morphometrics": out / "feature_tables_morphometrics",
        "label_overlay": out / "label_overlay",
        "montages": out / "montages",
        "anndata": out / "anndata",
        "plots_overview": out / "plots_overview",
        "avg_nuclei": out / "avg_nuclei",
        "logs": out / "logs",
    }


@dataclass(frozen=True)
class Output:
    """Something a step wrote for one sample, that the viewer can open."""

    step: str
    path: Path
    #: ``image``, ``labels``, ``picture`` (a PNG), ``table`` or ``anndata``.
    kind: str

    @property
    def name(self) -> str:
        return f"{get_step(self.step).title}: {self.path.name}"


def sample_outputs(dir_output: str | Path, sample: str, config: dict) -> list[Output]:
    """What is on disk for *sample*, step by step. Only files that exist."""
    dirs = output_dirs(dir_output, config)
    found: list[Output] = []

    def add(step: str, paths, kind: str) -> None:
        found.extend(Output(step, p, kind) for p in sorted(paths) if p.is_file())

    add("simple_masking", [dirs["masks"] / f"{sample}.tif"], "labels")
    add("alignment", (dirs["aligned"] / sample).glob("*.tif"), "image")
    add("alignment_check", (dirs["check_aligned"] / sample).glob("*.png"), "picture")
    add("refined_masking", [Path(dir_output) / "masks_refined" / f"{sample}.tif"], "labels")
    add("speckle_removal", (dirs["speckle_masks"] / sample).glob("*.tif"), "labels")
    add("cropping_denoising", (dirs["denoised"] / sample).glob("*.tif"), "image")
    add("bg_subtraction", (dirs["bg_subtracted"] / sample).glob("*.tif"), "image")
    add("cellpose", [dirs["segmented"] / f"{sample}.tif"], "labels")
    add("feature_extraction", [dirs["feature_tables"] / f"{sample}.csv"], "table")
    add("feature_morphology_extraction",
        [dirs["feature_tables_morphometrics"] / f"{sample}.csv"], "table")
    add("to_h5ad", [dirs["anndata"] / f"{sample}.h5ad"], "anndata")
    add("label_overlay", [dirs["label_overlay"] / f"{sample}.png"], "picture")
    add("overview_montage", (dirs["montages"] / sample).glob("*.png"), "picture")
    add("overview_plots", dirs["plots_overview"].glob(f"{sample}_*.png"), "picture")
    return found


def channel_of(path: str | Path, regex: str | None = None) -> tuple[int, int] | None:
    """(cycle, channel) from a file name, or ``None`` when the regex cannot read it."""
    match = re.search(regex or DEFAULT_REGEX, Path(path).name, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group("cycle_id")), int(match.group("channel_id"))
    except (IndexError, ValueError):
        return None


def stain_names(metadata_file: str | Path | None) -> dict[tuple[int, int], str]:
    """(cycle, channel) → stain, from the stain table. Empty when there is none."""
    if not metadata_file or not Path(metadata_file).is_file():
        return {}
    try:
        import pandas as pd

        table = pd.read_csv(metadata_file)
    except Exception:  # noqa: BLE001 - a bad table is reported by the run, not here
        logger.debug("could not read the stain table", exc_info=True)
        return {}
    if not {"cycle_id", "channel_id", "stain"} <= set(table.columns):
        return {}
    return {
        (int(row.cycle_id), int(row.channel_id)): str(row.stain)
        for row in table.itertuples()
        if isinstance(row.stain, str) and row.stain
    }


def check_metadata(metadata_file: str | Path) -> str | None:
    """What is wrong with a stain table, or ``None`` when nothing is."""
    path = Path(metadata_file)
    if not path.is_file():
        return f"No file at {path}."
    try:
        import pandas as pd

        table = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - the message is the point
        return f"Could not read it as CSV: {exc}"
    missing = {"cycle_id", "channel_id", "stain"} - set(table.columns)
    if missing:
        return "Missing column(s): " + ", ".join(sorted(missing))
    return None


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------

#: Lines the runner prints to say where it is. Anything else is the pipeline's
#: own output and is shown as it comes.
MARK = "@@phenoscape"


def runner_path() -> Path:
    """The runner script, started by path so the other Python needs no ARGUS."""
    return Path(__file__).with_name("phenoscape_runner.py")


def child_environment(source_dir: str | Path | None) -> dict[str, str]:
    """The environment for the child: phenoscapes importable, plots drawn off-screen."""
    env = dict(os.environ)
    if source_dir:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(source_dir), existing) if p)
    # The original calls plt.show() nowhere it matters, but pyplot still opens a
    # GUI backend on import; a child process with a window is a window nobody asked for.
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def check_command(python: str) -> list[str]:
    """A command that prints which of :data:`CHECKED_MODULES` import in *python*."""
    probe = (
        "import importlib, sys\n"
        f"for name in {list(CHECKED_MODULES)!r}:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "        print('ok', name)\n"
        "    except Exception as exc:\n"
        "        print('missing', name, '-', type(exc).__name__, str(exc).splitlines()[0] if str(exc) else '')\n"
        "print('python', sys.version.split()[0])\n"
    )
    return [python, "-c", probe]


def parse_check(output: str) -> tuple[list[str], list[str], str]:
    """(importable, missing with reasons, python version) from :func:`check_command`'s output."""
    ok, missing, version = [], [], ""
    for line in output.splitlines():
        if line.startswith("ok "):
            ok.append(line[3:].strip())
        elif line.startswith("missing "):
            missing.append(line[8:].strip())
        elif line.startswith("python "):
            version = line[7:].strip()
    return ok, missing, version


def module_of(entry: str) -> str:
    """The module name out of one ``missing`` line, which carries its reason too."""
    return str(entry).split(" ", 1)[0].strip()


def split_missing(missing: Sequence[str]) -> tuple[list[str], list[str]]:
    """``(blocking, optional)`` out of what :func:`parse_check` found missing.

    A run needs everything in :data:`REQUIRED_MODULES`; the rest costs one step.
    Telling the two apart is the difference between "this Python will not work"
    and "this Python will not draw the overview plots".
    """
    blocking, optional = [], []
    for entry in missing:
        (optional if module_of(entry) in OPTIONAL_MODULES else blocking).append(str(entry))
    return blocking, optional


def optional_note(missing: Sequence[str]) -> str:
    """One sentence per optional module that is missing: what it costs, and how to get it."""
    lines = []
    for entry in missing:
        name = module_of(entry)
        detail = OPTIONAL_MODULES.get(name)
        if detail is None:
            continue
        needed_by, remedy = detail
        lines.append(f"{name} is missing; only {needed_by} needs it. To install it, {remedy}.")
    return " ".join(lines)


@dataclass(frozen=True)
class Request:
    """Everything one run needs, written to a file the runner reads."""

    samples: tuple[str, ...]
    dir_input: str
    dir_output: str
    metadata_file: str
    config: dict
    steps: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "samples": list(self.samples),
            "dir_input": self.dir_input,
            "dir_output": self.dir_output,
            "metadata_file": self.metadata_file,
            "config": self.config,
            "steps": list(self.steps),
        }

    def problems(self) -> list[str]:
        """What stops this run from starting, in words. Empty when nothing does."""
        found = []
        if not self.samples:
            found.append("No sample chosen.")
        if not stitched_dir(self.dir_input).is_dir():
            found.append(f"No 'stitched' folder in {self.dir_input or 'the input folder'}.")
        if not self.dir_output:
            found.append("No output folder.")
        problem = check_metadata(self.metadata_file) if self.metadata_file else "No stain table."
        if problem:
            found.append(f"Stain table: {problem}")
        unknown = [key for key in self.steps if key not in STEP_KEYS]
        if unknown:
            found.append("Unknown step(s): " + ", ".join(unknown))
        # A step the config switches off is a step the later ones do not expect:
        # with the refined mask off, the simple mask is what everything reads.
        off = [key for key in self.steps
               if key in STEP_KEYS and not is_enabled(self.config, get_step(key))]
        if off:
            found.append("Switched off in the config: " + ", ".join(off))
        if not self.steps:
            found.append("No step to run.")
        return found


def write_request(request: Request, folder: str | Path) -> Path:
    """Write *request* as JSON into *folder* and return the file."""
    import json

    Path(folder).mkdir(parents=True, exist_ok=True)
    path = Path(folder) / "phenoscape_request.json"
    path.write_text(json.dumps(request.as_dict(), indent=2), encoding="utf-8")
    return path


def run_command(python: str, request_file: str | Path) -> list[str]:
    return [python, "-u", str(runner_path()), str(request_file)]


def parse_mark(line: str) -> tuple[str, list[str]] | None:
    """``("step-start", [sample, key])`` and the like, or ``None`` for an ordinary line."""
    if not line.startswith(MARK):
        return None
    parts = line[len(MARK):].strip().split("\t")
    return (parts[0], parts[1:]) if parts and parts[0] else None


def save_config(config: dict, path: str | Path) -> None:
    import yaml

    with open(path, "w", encoding="utf-8") as handle:
        yaml.dump(config, handle, sort_keys=False)


def load_config(path: str | Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return complete_config(yaml.safe_load(handle) or {})


def config_text(config: dict) -> str:
    import yaml

    return yaml.dump(config, sort_keys=False)


def parse_config_text(text: str) -> dict:
    import yaml

    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError("The config has to be a mapping of keys to values.")
    return complete_config(loaded)


def download_source(folder: str | Path, timeout: float = 30.0) -> Path:
    """Fetch the phenoscapes package into ``<folder>/phenoscapes`` and return *folder*.

    *folder* is then what goes on ``PYTHONPATH``. Its dependencies are not
    installed by this: ``requirements.txt`` comes down beside the code for that.
    """
    target = Path(folder) / "phenoscapes"
    target.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        with urllib.request.urlopen(SOURCE_URL + name, timeout=timeout) as response:
            (target / name).write_bytes(response.read())
    logger.info("phenoscapes downloaded to %s", target)
    return Path(folder)


def default_python() -> str:
    return sys.executable
