# Re-running the analysis

This folder holds the complete analysis code of the paper (the `oasis` Python package, two notebooks and a runner).
With the data beside it (`../data/`), one command re-makes every count, statistic, figure, table and the
interactive section browser from the raw scans and the stored junctions, into `../rerun/`; a second command compares
the re-run with the shipped results.

## 1. Install

Python 3.10 and the packages in `requirements.txt` (the versions the re-run was checked with):

```
python -m venv .venv
.venv\Scripts\activate          (Windows)      source .venv/bin/activate   (macOS / Linux)
pip install -r requirements.txt
```

## 2. Run

From this folder:

```
python run.py                    # everything except segmentation, ~3–4 h on a 4-core machine with 16 GB of memory
python check.py                  # compare rerun/ with the shipped data and results
```

| Step | What it does | Time | Output |
|---|---|---|---|
| `counting` | the AF488 mask of every section, the counted objects, each cell's cover (`oasis/counting.py`) | ~1 h | `rerun/counting/` |
| `sizes` | tissue area, assembloid length, the slim cell cache (`oasis/sizes.py`) | ~10 min | `rerun/derived/` |
| `notebooks` | `notebooks/01_validation.ipynb` (the count against the hand counts) and `02_biology.ipynb` (WT3 against GD1): every statistic, figure and table | ~15 min | `rerun/figures/`, `rerun/notebooks/` |
| `browser` | the section browser and figure builder, with full-resolution tiles and outlines | ~2–3 h | `rerun/results/section_browser/` |

`python run.py --steps counting,notebooks` runs some steps only; `--workers 8` processes more sections at a time
(each needs ~2 GB of memory); `--sections 5` tries the pipeline on the first five sections (the notebooks need all
of them). Every step skips what it has already done, so an interrupted run continues where it stopped.

**Segmentation** (optional, GPU): `pip install -r requirements-segmentation.txt` (after a CUDA build of torch), then
`python run.py --segment` segments every section again with Cellpose before the other steps, which then use the new
segmentation (~1 day for 614 sections on one GPU). Cellpose on a GPU is not bit-for-bit repeatable, so the cells
differ slightly from the shipped ones; the counting method (the AF488 mask) does not depend on them.

**What is not re-run**: the junction lines were found automatically and every one was reviewed by eye and redrawn
by hand where wrong (in an interactive image viewer); they are data here (`../data/junction/`), as are the
annotators' hand counts (`../data/annotations/`).

## 3. Check

`python check.py` compares, step by step:

- **counting**: each section's thresholds and number of objects per mask against `../data/counting/`;
- **tables**: every table of the paper (`rerun/figures/*.csv` against `../results/tables/`), cell by cell;
- **browser**: every section's counts against `../results/section_browser/`.

With the shipped segmentation the re-run is identical: checked on Windows 11 with Python 3.10 and
`requirements.txt` for every section's counts (614) and every table of the paper, and for the browser's counts on a
subset of sections (`python run.py --steps counting,sizes,browser --sections 3`).

## Code

```
run.py, check.py         the runner and the comparison
oasis/paths.py           every file read and written (inputs in ../data, outputs in ../rerun; OASIS_PAPER and
                         OASIS_OUT override them)
oasis/scan.py            reading the scans (OME-TIFF; channel planes, maximum over z)
oasis/cv.py              the AF488 mask: smoothing, local background, robust threshold, cleanup
oasis/section_outline.py the section's outline; objects outside the section are not counted
oasis/junction.py        the 5 µm overview, the section's outline, distance to the tissue edge
oasis/geometry.py        distance to the junction line (smoothed), beside the line or past its ends
oasis/sides.py           crossed or home: the side of the junction line extended past its ends
oasis/cells.py           label maps, local median, robust spread, placing cells and objects
oasis/counting.py        step "counting"
oasis/sizes.py           step "sizes"
oasis/segment.py         step "segment" (optional)
oasis/glmm.py            generalised linear mixed models (sensitivity analysis in 02_biology)
oasis/browser/           step "browser": section_browser.py, browser_tiles.py, browser_outlines.py,
                         objects_data.py, and the two pages (section_browser.html, figure_builder.html)
notebooks/               01_validation.ipynb, 02_biology.ipynb
```
