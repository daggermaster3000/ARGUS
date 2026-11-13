"""The OASIS paper's data package: raw scans, segmentation, junctions, counts, methods, results and figures in one folder.

Builds ``D:\\Slidescanner extracted\\oasis-paper\\`` from the analysis folders, so the publication can be shared and
reproduced without the working folders around it:

    data/
      raw/                  the 614 analysed scans (.ome.tiff; NTFS hard links to the originals: no extra space here,
                            full files when the folder is copied elsewhere)
      design/               every scan's genotype / experiment / slide, the pre-screen decisions, the 614 analysed
      segmentation/         per section: every cell (Cellpose nuclei grown by 6 µm) as label maps and a per-cell table;
                            the run's settings and its review / QC table
      junction/             per section: the junction line and region (label map), the run's section record, and the
                            ARGUS store the line and region were drawn / reviewed in (.argus.h5)
      counting/             per section: the counting method's AF488 mask objects (median + 8 robust SD; also 3, 5),
                            thresholds, each cell's cover, the section outline used to drop objects outside the section
      annotations/          the two annotators' AF488+ clicks; the hand-drawn junctions and their evaluation
      derived/              section sizes and the slim cell cache the notebooks read
    methods/                METHODS.md, the code at the analysis commit (zip), the Python environments, the executed
                            notebooks (.ipynb and .html)
    results/                section_browser/ (open section_browser.html; the figure builder beside it), tables/
    figures/                the paper's figures (PDF, PNG, SVG)
    README.md, SHA256SUMS

Derived files are copied (frozen; later reruns of the analysis do not change them). Re-running skips files already
there with the same size.

    python notebooks/official_analysis/paper_package.py              # build or complete the package
    python notebooks/official_analysis/paper_package.py --checksums  # also write SHA256SUMS (~1 h: the raw scans)

No Qt.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# ---------------------------------------------------------------- Settings
DATA = Path(r"D:\Slidescanner extracted")
PAPER = DATA / "oasis-paper"
RUN = DATA / "cells_analysis" / "2026-10-28_090440 re-scored"     # the cells-analysis run (614 sections)
OUT = DATA / "run_analysis_20261028"
BROWSER = OUT / "section_browser"
FIGURES = OUT / "figures_official"
DESIGN_CSV = DATA / "Image_File_List.csv"
PRESCREEN = Path(r"C:\Users\qfavey\Downloads\oasis_prescreen_211026.csv")   # the pre-screen decisions (OASIS)
JUNCTION_EVAL = DATA / "junction_eval"
PICKS = DATA / "picks_20261020_0742.csv"
#: the annotators, as the notebooks name them (folder → name in the package)
ANNOTATORS = {"james_counts": "annotator_1", "quillan_counts": "annotator_2"}
NOTEBOOKS = [ROOT / "notebooks" / "official_analysis" / "01_validation.ipynb",
             ROOT / "notebooks" / "official_analysis" / "02_biology.ipynb"]
CONDA_PYTHON = Path(r"C:/Users/qfavey/.conda/envs/napari-env-q/python.exe")   # executes the notebooks
# --------------------------------------------------------------------------


def say(text: str) -> None:
    print(time.strftime("%H:%M:%S"), text, flush=True)


def copy(src: Path, dst: Path) -> bool:
    """Copy *src* to *dst* unless a file of the same size is there; True when copied."""
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def link(src: Path, dst: Path) -> bool:
    """A hard link *dst* to *src* (same volume), unless it is there already."""
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)
    return True


def robocopy(src: Path, dst: Path) -> None:
    """Mirror-free copy of a folder tree (new and changed files only), many files at a time."""
    code = subprocess.run(["robocopy", str(src), str(dst), "/E", "/MT:16", "/NFL", "/NDL", "/NJH", "/NJS", "/NP"],
                          capture_output=True, text=True).returncode
    if code >= 8:                                      # robocopy: < 8 is success (1 = files copied)
        raise RuntimeError(f"robocopy {src} -> {dst} failed ({code})")


def sections() -> pd.DataFrame:
    """The 614 analysed sections with their design, from the section browser's data and the design list."""
    data = json.loads((BROWSER / "data.json").read_text(encoding="utf-8"))
    rows = [{"name": s["name"], "id": s["id"], "genotype": s["genotype"], "experiment": s["experiment"],
             "slide": s["slide"], "QC status": s["review"], "region drawn by": s["region_by"],
             "junction length (µm)": s["length_um"], "home side sign": s["sign"],
             "tissue area (mm²)": s["area_mm2"], "nuclei": s["n_nuclei"],
             "human-counted": bool(s["humans"])} for s in data["sections"]]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the parts
def build_design(table: pd.DataFrame) -> None:
    d = PAPER / "data" / "design"
    copy(DESIGN_CSV, d / "Image_File_List.csv")
    if PRESCREEN.exists():
        copy(PRESCREEN, d / "oasis_prescreen.csv")
    table.to_csv(d / "analysed_sections.csv", index=False, encoding="utf-8")
    say(f"design: {len(table)} analysed sections")


def build_raw(table: pd.DataFrame) -> None:
    n = 0
    for name in table["name"]:
        n += link(DATA / f"{name}.ome.tiff", PAPER / "data" / "raw" / f"{name}.ome.tiff")
    say(f"raw: {n} new hard links ({len(table)} scans)")


def build_segmentation(table: pd.DataFrame) -> None:
    d = PAPER / "data" / "segmentation"
    copy(RUN / "README.txt", d / "run_README.txt")
    copy(RUN / "counts.xlsx", d / "run_counts_and_review.xlsx")
    n = 0
    for name in table["name"]:
        n += copy(RUN / "cells" / f"{name}.csv", d / "cells" / f"{name}.csv")
        for kind in ("nuclei", "grown"):
            n += copy(RUN / "labels" / f"{name} {kind}.tif", d / "labels" / f"{name} {kind}.tif")
    say(f"segmentation: {n} files copied")


def build_junction(table: pd.DataFrame) -> None:
    d = PAPER / "data" / "junction"
    rows, n = [], 0
    for name in table["name"]:
        info = json.loads((RUN / "cells" / f"{name}.json").read_text(encoding="utf-8"))
        n += copy(RUN / "cells" / f"{name}.json", d / "sections" / f"{name}.json")
        n += copy(RUN / "labels" / f"{name} region.tif", d / "region" / f"{name} region.tif")
        store = DATA / f"{name}.ome.tiff.argus.h5"
        if store.exists():
            n += copy(store, d / "argus_store" / store.name)
        line = info.get("line")
        line = json.loads(line) if isinstance(line, str) else line
        for i, (y, x) in enumerate(line or []):
            rows.append({"name": name, "point": i, "y (µm)": y, "x (µm)": x})
    pd.DataFrame(rows).to_csv(d / "junction_lines.csv", index=False, encoding="utf-8")
    say(f"junction: {n} files copied, {len(rows)} line points")


def build_counting(table: pd.DataFrame) -> None:
    n = 0
    for name in table["name"]:
        for f in ("objects.parquet", "cells.parquet", "summary.json", "outline.npz"):
            n += copy(OUT / "af488_mask" / name / f, PAPER / "data" / "counting" / "af488_mask" / name / f)
    say(f"counting: {n} files copied")


def build_annotations() -> None:
    d = PAPER / "data" / "annotations"
    n = 0
    for folder, label in ANNOTATORS.items():
        n += copy(OUT / folder / "clicks.parquet", d / label / "clicks.parquet")
        for f in sorted((OUT / folder / "cells").glob("*.parquet")):
            n += copy(f, d / label / "cells" / f.name)
    h = d / "junctions_hand_drawn"
    for f in ("annotated_all.csv", "scores.csv", "seeding.jsonl"):
        n += copy(JUNCTION_EVAL / f, h / f)
    if PICKS.exists():
        n += copy(PICKS, h / PICKS.name)
    # the hand-drawn lines and regions are stored in each section's ARGUS store
    for name in pd.read_csv(JUNCTION_EVAL / "annotated_all.csv")["name"]:
        store = DATA / f"{name}.ome.tiff.argus.h5"
        if store.exists():
            n += copy(store, h / "argus_store" / store.name)
    say(f"annotations: {n} files copied")


def build_derived() -> None:
    d = PAPER / "data" / "derived"
    for f in ("section_scales.csv", "section_tissue_area.csv"):
        copy(OUT / f, d / f)
    robocopy(OUT / "cache", d / "cell_cache")
    say("derived: section sizes and the cell cache")


def build_methods() -> None:
    d = PAPER / "methods"
    d.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    target = d / "code" / f"ARGUS_{commit[:7]}.zip"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "archive", "--format=zip", "-o", str(target), "HEAD"], cwd=ROOT, check=True)
    (d / "code" / "COMMIT.txt").write_text(f"{commit}\n", encoding="utf-8")
    env = d / "environment"
    env.mkdir(parents=True, exist_ok=True)
    for label, py in (("notebooks_conda_env", CONDA_PYTHON), ("scripts_python", Path(sys.executable))):
        out = subprocess.run([str(py), "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        version = subprocess.run([str(py), "--version"], capture_output=True, text=True).stdout.strip()
        (env / f"{label}.txt").write_text(f"# {version}\n{out}", encoding="utf-8")
    nb = d / "notebooks"
    nb.mkdir(parents=True, exist_ok=True)
    for path in NOTEBOOKS:
        copy(path, nb / path.name)
        subprocess.run([str(CONDA_PYTHON), "-m", "jupyter", "nbconvert", "--to", "html", "--output-dir", str(nb),
                        str(path)], capture_output=True, check=True)
    (d / "METHODS.md").write_text(methods_text(commit), encoding="utf-8")
    say(f"methods: code at {commit[:7]}, environments, notebooks, METHODS.md")


def methods_text(commit: str) -> str:
    """The Methods of the two notebooks, and how each part of the package was made."""
    def section(path: Path, heading: str) -> str:
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        for c in cells:
            s = "".join(c["source"])
            if c["cell_type"] == "markdown" and heading in s:
                return s[s.index(heading):].strip()
        raise ValueError(f"no {heading!r} in {path.name}")

    biology = section(NOTEBOOKS[1], "## Methods").replace("## Methods", "## Biology (02_biology)", 1)
    validation = section(NOTEBOOKS[0], "## Methods").replace("## Methods", "## Validation of the count (01_validation)", 1)
    return f"""# Methods

Code: ARGUS at commit `{commit}` (`code/ARGUS_{commit[:7]}.zip`; the repository's `notebooks/README.md` indexes
every script and notebook). Python environments: `environment/`.

{biology}

{validation}

## How each part of this package was made

The scripts read and write fixed paths under `D:\\Slidescanner extracted\\`; the table in the package's README maps
them to the folders here.

1. **Pre-screen** (OASIS, a separate QC app): every scan was screened for quality; the decisions are
   `data/design/oasis_prescreen.csv`. WT2 scans are in the design list but not in the study (by design).
2. **Junction** (ARGUS viewer, Junction panel): the junction between the two organoids was found automatically
   (minimum-cost path along the dark seam of a 5 µm overview), every line was reviewed by eye and redrawn by hand where
   wrong, and a junction region (counting band) was built along it. Line and region are stored in each scan's ARGUS
   store (`data/junction/argus_store/`).
3. **Segmentation** (ARGUS viewer, *Cells analysis* run `2026-10-28_090440 re-scored`, settings in
   `data/segmentation/run_README.txt`): Cellpose (cpsam) on the nuclear stain over the whole section, nuclei grown by
   6 µm; per cell its position, size, intensities, region membership, distance to the junction line and to the tissue
   edge (`data/segmentation/cells/`, label maps in `labels/`).
4. **Counting** (`notebooks/dataset/af488_masks.py`, with `section_outline.py`): the AF488 mask at median + 3, 5 and
   8 robust SD; 8 SD is the counting method (`data/counting/`).
5. **Section sizes**: `notebooks/official_analysis/tissue_area.py` (tissue area) and `notebooks/dataset/04` (assembloid
   length) → `data/derived/`.
6. **Analysis**: `notebooks/official_analysis/01_validation.ipynb` (agreement with the hand counts) and
   `02_biology.ipynb` (WT3 against GD1), executed in the conda environment
   (`jupyter nbconvert --execute --inplace`); executed copies in `notebooks/`.
7. **Results browser**: `notebooks/official_analysis/section_browser.py`, then `browser_tiles.py`,
   `browser_outlines.py` (run by it: `objects_data.py`, `browser_cellpose.py`) → `results/section_browser/`.
"""


def build_results() -> None:
    robocopy(BROWSER, PAPER / "results" / "section_browser")
    t = PAPER / "results" / "tables"
    for f in sorted(FIGURES.glob("*.csv")):
        copy(f, t / f.name)
    say("results: section browser and tables")


def build_figures() -> None:
    n = 0
    for f in sorted(FIGURES.iterdir()):
        if f.suffix.lower() in (".pdf", ".png", ".svg"):
            n += copy(f, PAPER / "figures" / f.name)
    say(f"figures: {n} files copied")


README = """# OASIS assembloids: AF488⁺ cells at and across the junction of WT3 and GD1 assembloids

Data, methods and results of the analysis comparing WT3 and GD1 assembloids: how many AF488-labelled (AF488⁺) cells lie
at the junction between the two fused organoids and how many crossed into the partner organoid. {n} sections
(WT3 {n_wt3}, GD1 {n_gd1}) on {n_slides} slides from three experiments, imaged on a slide scanner at 0.325 µm per pixel.

Package built {built} from the analysis at code commit `{commit}`.

## Start here

- **Results, interactive**: open `results/section_browser/section_browser.html` in a web browser (works straight from
  disk; Chrome, Edge or Firefox). Every section with its junction line, junction region, counted AF488⁺ objects
  (home white, crossed magenta), the human annotators' dots, a table and plots; zoom in for the full-resolution image
  and outlines. `figure_builder.html` beside it re-draws the paper's figures with any normalisation and writes the
  Methods text for them.
- **Paper figures**: `figures/` (PDF, PNG, SVG); their tables in `results/tables/`.
- **Methods**: `methods/METHODS.md`; the executed analysis notebooks in `methods/notebooks/` (`.html` to read,
  `.ipynb` to run).

## Contents

```
data/
  raw/                     {n} raw scans, <section>.ome.tiff (OME-TIFF; channels AF555, AF488, AF405 nuclear; z-stack)
  design/
    Image_File_List.csv    every scan: genotype, experiment, slide (WT2 is listed but not part of this study)
    oasis_prescreen.csv    the quality pre-screen of every scan (OASIS): pass, quality, ripped, folded, shape
    analysed_sections.csv  the {n} analysed sections: design, QC status of the junction review, who drew the region,
                           junction length, home side, tissue area, number of nuclei, whether counted by hand
  segmentation/
    cells/<section>.csv    every segmented cell of the section (Cellpose cpsam on the nuclear stain, grown by 6 µm):
                           position, size, intensities per channel, region membership, distance to the junction line
                           and to the tissue edge
    labels/<section> nuclei.tif, <section> grown.tif
                           label maps of the whole section (uint, 0 = background; label = the 'Label' column)
    run_README.txt         the cells-analysis run's settings
    run_counts_and_review.xlsx  the run's per-section counts and the junction review (QC status, decisions)
  junction/
    junction_lines.csv     every section's junction line: points (y, x) in µm
    region/<section> region.tif  the junction region (counting band, 300 µm) as a label map
    sections/<section>.json      the run's record of the section: line, region source, flags, segmentation settings
    argus_store/<section>.ome.tiff.argus.h5
                           the ARGUS store of the section (HDF5): ARGUS/Labels/<name> label maps of the junction
                           regions (automatic "fusion region"; "manual annotated junction region" where drawn by hand)
                           with the line in their attribute line_yx_um; ARGUS/ROIs/ the band outlines
  counting/af488_mask/<section>/
    objects.parquet        every object of the AF488 mask at median + 3, 5, 8 robust SD (column mask: kmad3/5/8;
                           kmad8 is the counting method): area (µm²), centroid (µm), distance to the junction line,
                           in the junction region
    summary.json           the section's thresholds, noise SD, objects dropped as outside the section
    cells.parquet          each segmented cell's cover by each mask
    outline.npz            the section outline objects must lie in (packed boolean, 5 µm grid)
  annotations/
    annotator_1/, annotator_2/  clicks.parquet: one row per AF488⁺ cell clicked by hand (y, x in µm; the segmented
                           cell it fell on); cells/: the run's cells of each hand-counted section. Annotator 1 counted
                           whole sections (14), annotator 2 from 50 µm before the junction line on (13).
    junctions_hand_drawn/  the 138 sections with a hand-drawn junction (annotated_all.csv: tuning / held-out split,
                           section classes), the first 31 picks, the finder's scores against them, and the ARGUS
                           stores holding the hand-drawn lines ("manual annotated junction region")
  derived/                 section sizes (tissue area, assembloid length) and the slim cell cache the notebooks read
methods/
  METHODS.md               Methods text, and how each part of this package was made
  code/ARGUS_<commit>.zip  the analysis code at the commit used (ARGUS; notebooks/README.md indexes the analysis)
  environment/             pip freeze of the two Python environments used
  notebooks/               01_validation (automated against hand counts), 02_biology (WT3 against GD1), executed
results/
  section_browser/         the interactive browser (above) and its data
  tables/                  the paper's tables (CSV): Table 1, GLMM, supplementary tables, every section's outcomes
figures/                   the paper's figures
SHA256SUMS                 checksums of every file (sha256sum -c SHA256SUMS)
```

**Coordinates.** Positions are in µm from the top-left corner of the scan: µm = pixel index × 0.325 (y down, x right),
as the label maps' pixel grid. Section names are the scan file names without `.ome.tiff`.

**Counting method.** An AF488⁺ cell is one connected object of the AF488 mask (maximum over z, smoothed 0.5 µm, less a
20 µm Gaussian background, thresholded at the median + 8 robust SD of the section's own tissue; objects < 20 µm²
dropped, holes < 10 µm² filled; objects outside the section not counted). *Crossed*: centroid on the far side of the
junction line extended past its ends. Full detail in `methods/METHODS.md`.

## Reproducing the analysis

Unzip `methods/code/`, install the environment (`methods/environment/`), and run the steps of `methods/METHODS.md`
in order. The scripts read fixed paths; put the data back in this layout or edit the settings block at the top of
each script / notebook:

| Path in the code (`D:\\Slidescanner extracted\\…`) | In this package |
|---|---|
| `<section>.ome.tiff`, `<section>.ome.tiff.argus.h5` | `data/raw/`, `data/junction/argus_store/` |
| `Image_File_List.csv` | `data/design/Image_File_List.csv` |
| `cells_analysis\\2026-10-28_090440 re-scored\\cells\\` | `data/segmentation/cells/` (`.csv`), `data/junction/sections/` (`.json`) |
| `cells_analysis\\2026-10-28_090440 re-scored\\labels\\` | `data/segmentation/labels/`, `data/junction/region/` |
| `cells_analysis\\2026-10-28_090440 re-scored\\counts.xlsx` | `data/segmentation/run_counts_and_review.xlsx` |
| `run_analysis_20261028\\af488_mask\\` | `data/counting/af488_mask/` |
| `run_analysis_20261028\\james_counts\\`, `quillan_counts\\` | `data/annotations/annotator_1/`, `annotator_2/` |
| `run_analysis_20261028\\cache\\`, `section_*.csv` | `data/derived/` |
| `run_analysis_20261028\\section_browser\\` | `results/section_browser/` |
| `run_analysis_20261028\\figures_official\\` | `figures/`, `results/tables/` |
| `junction_eval\\`, `picks_20261020_0742.csv` | `data/annotations/junctions_hand_drawn/` |
| `C:\\Users\\…\\Downloads\\oasis_prescreen_211026.csv` | `data/design/oasis_prescreen.csv` |
"""


def build_readme(table: pd.DataFrame) -> None:
    commit = (PAPER / "methods" / "code" / "COMMIT.txt").read_text(encoding="utf-8").strip()
    text = README.format(n=len(table), n_wt3=int((table["genotype"] == "WT3").sum()),
                         n_gd1=int((table["genotype"] == "GD1").sum()), n_slides=table["slide"].nunique(),
                         built=time.strftime("%d %b %Y"), commit=commit)
    (PAPER / "README.md").write_text(text, encoding="utf-8")
    say("README.md")


def checksums() -> None:
    """SHA256SUMS over every file of the package (relative paths), the raw scans included."""
    lines = []
    files = sorted(p for p in PAPER.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    for i, p in enumerate(files, 1):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        lines.append(f"{h.hexdigest()}  {p.relative_to(PAPER).as_posix()}")
        if i % 5000 == 0:
            say(f"checksums: {i}/{len(files)}")
    (PAPER / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"checksums: {len(files)} files")


def main(argv: list[str]) -> int:
    PAPER.mkdir(exist_ok=True)
    table = sections()
    build_design(table)
    build_raw(table)
    build_segmentation(table)
    build_junction(table)
    build_counting(table)
    build_annotations()
    build_derived()
    build_methods()
    build_results()
    build_figures()
    build_readme(table)
    if "--checksums" in argv:
        checksums()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
