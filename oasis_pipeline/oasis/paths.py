"""Every file the pipeline reads and writes, relative to the package folder (``oasis-paper/``).

The package folder is the one holding ``pipeline/`` (this code) and ``data/``; set ``OASIS_PAPER`` to use another.
Outputs go to ``<package>/rerun/`` (``OASIS_OUT`` to change it), never over the shipped ``results/`` and
``figures/``, so the two can be compared (``python check.py``).

Segmentation: the shipped one (``data/segmentation/``) unless the pipeline re-segmented (``rerun/segmentation/``,
``python run.py --segment``), in which case every later step uses the new one.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("OASIS_PAPER") or Path(__file__).resolve().parents[2])
DATA = ROOT / "data"
OUT = Path(os.environ.get("OASIS_OUT") or ROOT / "rerun")

# ---------------------------------------------------------------- inputs
RAW = DATA / "raw"                                            #: <section>.ome.tiff
DESIGN_CSV = DATA / "design" / "Image_File_List.csv"          #: genotype, experiment, slide of every scan
SECTIONS_CSV = DATA / "design" / "analysed_sections.csv"      #: the analysed sections
PRESCREEN_CSV = DATA / "design" / "oasis_prescreen.csv"
JUNCTION = DATA / "junction"                                  #: sections/<s>.json (line, …), region/<s> region.tif
SHIPPED_SEGMENTATION = DATA / "segmentation"                  #: cells/<s>.csv, labels/<s> nuclei|grown.tif
ANNOTATIONS = DATA / "annotations"
ANNOTATORS = {"human annotator 1": ANNOTATIONS / "annotator_1", "human annotator 2": ANNOTATIONS / "annotator_2"}
JUNCTIONS_HAND_DRAWN = ANNOTATIONS / "junctions_hand_drawn"

# ---------------------------------------------------------------- outputs
RERUN_SEGMENTATION = OUT / "segmentation"
MASKS = OUT / "counting" / "af488_mask"                       #: <s>/objects.parquet, cells.parquet, summary.json, outline.npz
DERIVED = OUT / "derived"
CACHE = DERIVED / "cell_cache"                                #: <s>.parquet: the cells near the line, slim
SCALES_CSV = DERIVED / "section_scales.csv"                   #: assembloid length per section
TISSUE_AREA_CSV = DERIVED / "section_tissue_area.csv"
FIGURES = OUT / "figures"                                     #: the paper's figures and tables
BROWSER = OUT / "results" / "section_browser"
NOTEBOOKS = OUT / "notebooks"                                 #: the executed notebooks


def segmentation() -> Path:
    """The segmentation every step reads: the re-run one when it is complete, else the shipped one."""
    done = RERUN_SEGMENTATION / "COMPLETE"
    return RERUN_SEGMENTATION if done.exists() else SHIPPED_SEGMENTATION


def scan_path(name: str) -> Path:
    return RAW / f"{name}.ome.tiff"


def section_json(name: str) -> Path:
    """The section's record: junction line (µm, y x), region source, flags, voxel size, shape, …"""
    return JUNCTION / "sections" / f"{name}.json"


def region_tif(name: str) -> Path:
    """The junction region (counting band) as a label map."""
    return JUNCTION / "region" / f"{name} region.tif"


def cells_csv(name: str) -> Path:
    """Every segmented cell of the section."""
    return segmentation() / "cells" / f"{name}.csv"


def nuclei_tif(name: str) -> Path:
    return segmentation() / "labels" / f"{name} nuclei.tif"


def grown_tif(name: str) -> Path:
    return segmentation() / "labels" / f"{name} grown.tif"


def review_xlsx() -> Path:
    """The junction review: QC status, decision, who drew the region (sheet Counts)."""
    return SHIPPED_SEGMENTATION / "run_counts_and_review.xlsx"


def section_names() -> list[str]:
    """The analysed sections, in name order."""
    import pandas as pd

    return sorted(pd.read_csv(SECTIONS_CSV)["name"].tolist())


def section_info(name: str) -> dict:
    """The section's record, its file paths pointed at this package."""
    import json

    info = json.loads(section_json(name).read_text(encoding="utf-8"))
    info["file"] = str(scan_path(name))
    redone = RERUN_SEGMENTATION / "cells" / f"{name}.json"
    if segmentation() == RERUN_SEGMENTATION and redone.exists():      # the number of cells of the new segmentation
        info.update(json.loads(redone.read_text(encoding="utf-8")))
    return info
