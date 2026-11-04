"""Hand counts (one 5 µm dot per AF488+ cell) placed like the run's cells.

Two annotators clicked AF488+ cells, each into a label map stored with the sample
(:data:`ANNOTATORS`): James ("James counts", the whole section) and Quillan
("Manual counts Quillan", the junction and the cells that crossed). For every
section of the run with the annotator's label map, writes into his folder
(``<OUT>/james_counts/`` or ``<OUT>/quillan_counts/``):

* ``clicks.parquet`` — one row per dot: its position, whether it is in the run's
  junction region, its signed distance to the run's junction line (the run's own
  sign, so it compares with the cells' column), beside the line or not, its
  distance to the tissue edge (that of the nearest segmented cell), and the
  segmented cell it is matched to: one to one, nearest pairs first, within
  :data:`MATCH_UM` (as ``validation/count_common.load_dots``), −1 for none;
* ``cells/<sample>.parquet`` — every segmented cell of the section (not only those
  near the line, as the 03/04 cache), with the columns the positivity rules need,
  each AF488 mask's cover (from ``af488_masks.py``) and ``clicked``.

Run it once per annotator (about a minute each); ``05_james_counts.ipynb`` reads
James's, ``publication/01_pipeline_validation.ipynb`` both::

    python notebooks/dataset/james_clicks.py            # James (the default)
    python notebooks/dataset/james_clicks.py quillan    # Quillan

No Qt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from microscopy_viewer import fusion as fu  # noqa: E402
from microscopy_viewer import ims_store  # noqa: E402
from microscopy_viewer import junction_analysis as ja  # noqa: E402

RUN = Path(r"D:\Slidescanner extracted\cells_analysis\2026-10-28_090440 re-scored")
OUT = Path(r"D:\Slidescanner extracted\run_analysis_20261028")
MASKS = OUT / "af488_mask"
#: Per annotator: the label map of his dots, and the folder his placed dots go to.
ANNOTATORS = {
    "james": ("James counts", OUT / "james_counts"),
    "quillan": ("Manual counts Quillan", OUT / "quillan_counts"),
}
#: James's, which the other ``james_*.py`` scripts read.
COUNTS, TARGET = ANNOTATORS["james"]
#: A dot and a cell further apart than this are not matched (µm).
MATCH_UM = 7.0

COLUMNS = {
    "Label": "label", "Centroid Y (µm)": "y", "Centroid X (µm)": "x", "In region": "region",
    "Distance to the junction line (µm)": "d", "Beside the junction line": "beside",
    "Distance to the tissue edge (µm)": "edge", "AF488 mean": "AF488 mean",
    "AF488 mean (median filter)": "AF488 mean (median filter)", "AF488 positive": "AF488 positive",
}


def match(dots_um: np.ndarray, cells_um: np.ndarray) -> np.ndarray:
    """Per dot, the index of its cell (one to one, nearest pairs first, within MATCH_UM), −1 for none."""
    pairs = []
    for k, near in enumerate(cKDTree(cells_um).query_ball_point(dots_um, MATCH_UM)):
        pairs += [(float(np.hypot(*(cells_um[j] - dots_um[k]))), k, j) for j in near]
    used_dot, used_cell, matched = set(), set(), np.full(len(dots_um), -1)
    for _dist, k, j in sorted(pairs):
        if k not in used_dot and j not in used_cell:
            used_dot.add(k)
            used_cell.add(j)
            matched[k] = j
    return matched


def section_info(name: str) -> dict:
    """The run's record of one section (file, pixel size, line, region file …)."""
    return json.loads((RUN / "cells" / f"{name}.json").read_text(encoding="utf-8"))


def run_cells(name: str) -> pd.DataFrame:
    """Every segmented cell of the section, with the columns in :data:`COLUMNS` (renamed)."""
    cells = pacsv.read_csv(RUN / "cells" / f"{name}.csv",
                           convert_options=pacsv.ConvertOptions(include_columns=list(COLUMNS))).to_pandas()
    return cells.rename(columns=COLUMNS)


def place(info: dict, cells: pd.DataFrame, yx_um: np.ndarray) -> pd.DataFrame:
    """Points (µm) placed like the run's cells: ``region``, ``d`` (the run's sign), ``beside``, ``edge``.

    The edge distance is that of the nearest segmented cell.
    """
    yx_um = np.asarray(yx_um, float).reshape(-1, 2)
    voxel = np.asarray(info["voxel_yx"], float)
    cells_um = cells[["y", "x"]].to_numpy(float)
    # The flip that turns the line's raw signed distance into the cells' stored one.
    line = np.asarray(info["line"], float)
    raw, beside, _length = fu.signed_distance(yx_um, line)
    some = np.random.default_rng(0).choice(len(cells), min(5000, len(cells)), replace=False)
    raw_cells, _b, _l = fu.signed_distance(cells_um[some], line)
    stored = cells["d"].to_numpy(float)[some]
    ok = np.isfinite(stored) & (np.abs(raw_cells) > 1)
    flip = float(np.sign(np.median(np.sign(stored[ok]) * np.sign(raw_cells[ok])))) if ok.any() else 1.0
    band = ja.read_labels(RUN / info["region_file"]) > 0
    rr = np.clip(np.round(yx_um[:, 0] / voxel[0]).astype(int), 0, band.shape[0] - 1)
    cc = np.clip(np.round(yx_um[:, 1] / voxel[1]).astype(int), 0, band.shape[1] - 1)
    _dist, nearest = cKDTree(cells_um).query(yx_um) if len(yx_um) else (None, np.zeros(0, int))
    return pd.DataFrame({"region": band[rr, cc], "d": raw * (flip or 1.0), "beside": beside,
                         "edge": cells["edge"].to_numpy(float)[nearest]})


def one_section(info: dict, counts: str = COUNTS, target: Path = TARGET) -> pd.DataFrame:
    """One section's dots of the label map *counts* (placed and matched); its cells go to *target*."""
    name, path = info["sample"], Path(info["file"])
    voxel = np.asarray(info["voxel_yx"], float)
    cells = run_cells(name)
    cover = MASKS / name / "cells.parquet"
    if cover.exists():
        cover = pd.read_parquet(cover).set_index("Label")
        for column in cover.columns:
            cells[column] = cells["label"].map(cover[column]).to_numpy()

    lab, _attrs = ims_store.load_labels(path, counts)
    lab = np.asarray(lab).reshape(np.shape(lab)[-2:])
    ids = np.unique(lab[lab > 0])
    centres = np.asarray(ndimage.center_of_mass(lab > 0, lab, ids)).reshape(-1, 2)
    dots_um = centres * voxel
    cells_um = cells[["y", "x"]].to_numpy(float)
    matched = match(dots_um, cells_um)
    cells["clicked"] = False
    cells.loc[matched[matched >= 0], "clicked"] = True
    placed = place(info, cells, dots_um)

    target.joinpath("cells").mkdir(parents=True, exist_ok=True)
    cells.to_parquet(target / "cells" / f"{name}.parquet", index=False)
    return pd.DataFrame({
        "name": name, "dot": ids, "y": dots_um[:, 0], "x": dots_um[:, 1], **{c: placed[c] for c in placed},
        "cell": matched, "matched": matched >= 0,
        "label": np.where(matched >= 0, cells["label"].to_numpy()[np.maximum(matched, 0)], -1),
    })


def main(argv: list[str]) -> int:
    annotator = argv[0].lower() if argv else "james"
    if annotator not in ANNOTATORS:
        print(f"unknown annotator {annotator!r}: one of {', '.join(ANNOTATORS)}")
        return 2
    counts, target = ANNOTATORS[annotator]
    infos = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RUN / "cells").glob("*.json"))]
    infos = [i for i in infos if counts in ims_store.list_labels(Path(i["file"]))]
    clicks = []
    for info in infos:
        clicks.append(one_section(info, counts, target))
        c = clicks[-1]
        print(f"{info['sample']}: {len(c)} dots, {c['matched'].mean():.0%} on a cell, {c['region'].mean():.0%} in the region")
    pd.concat(clicks, ignore_index=True).to_parquet(target / "clicks.parquet", index=False)
    print(f"{len(infos)} sections of {counts!r} → {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
