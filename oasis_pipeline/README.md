# OASIS assembloids: analysis code

The analysis code of the paper comparing WT3 and GD1 assembloids: how many AF488-labelled (AF488⁺) cells lie at the
junction between the two fused organoids, and how many crossed into the partner organoid. 614 sections (WT3 265,
GD1 349) on 36 slides from three experiments, imaged on a slide scanner at 0.325 µm per pixel.

From the raw scans, the stored junction interfaces and the Cellpose masks, the code re-makes every section's
counts, the primary outcome and the interactive section browser, and compares them with the shipped results.

**Reviewers using a coding agent:** point it at [`REVIEWER_AGENT.md`](REVIEWER_AGENT.md).

## The data

The code reads the data package (≈ 700 GB: raw scans, junction interfaces, Cellpose model and masks, the expected
results). It is not in this repository. **Data package: `<DOI / link to be added>`.**

Put this repository inside the package folder as `pipeline/`, or anywhere else with `OASIS_PAPER` set to the package:

```
oasis-paper/                 the data package
  data/                      inputs (raw scans, junctions, Cellpose model and masks, …)
  results/                   the shipped results the re-run is compared with
  pipeline/                  ← this repository (or set OASIS_PAPER=/path/to/oasis-paper)
  rerun/                     written by the code (OASIS_OUT to change it); nothing shipped is overwritten
```

## Install

Python 3.10 and the pinned packages (the versions the re-run was checked with; also checked with Python 3.13):

```
python -m venv .venv
.venv\Scripts\activate          (Windows)      source .venv/bin/activate   (macOS / Linux)
pip install -r requirements.txt
```

The GPU steps (optional) need a CUDA build of torch, then `pip install -r requirements-segmentation.txt`.

## Run

```
python run.py --steps cellpose,stats     # the paper's counts and primary outcome from the shipped masks (~2 min)
python run.py                            # everything except the GPU steps (~4–5 h with 4 workers)
python check.py                          # compare rerun/ with the shipped results
```

| Step | What it does | Time | Output (`rerun/…`) |
|---|---|---|---|
| `counting` | the AF488 intensity mask: the section's noise SD, the home side, the section outline (`oasis/counting.py`) | ~1 h | `counting/af488_mask/` |
| `sizes` | tissue area, assembloid length (`oasis/sizes.py`) | ~10 min | `derived/` |
| `cellpose` | **the paper's counts**: Cellpose masks at 3, 5 and 8 noise SD, each cell home, crossed or in the junction region (`oasis/cpself.py`) | ~1 min | `counting/cellpose_counts.csv` |
| `stats` | **the primary outcome**: WT3 ÷ GD1 of crossed cells per mm of junction interface, mixed model; sensitivity analyses (`oasis/stats.py`) | < 1 min | `statistics/primary_outcome.csv` |
| `browser` | the interactive section browser, with full-resolution tiles and outlines (`oasis/browser/`) | ~2.5–3.5 h | `results/section_browser/` |
| `infer` (`--infer`, GPU) | the Cellpose masks again with the shipped model | hours | `counting/cellpose/` |
| `segment` (`--segment`, GPU) | the nuclei segmentation again (used by the browser's per-nucleus methods only) | ~1 day | `segmentation/` |

`--steps` runs some steps only; `--workers 8` processes more sections at a time (each needs ~2 GB of memory);
`--sections 5` runs the first five sections only (the statistics need all of them). Each step skips what it has
already done, so an interrupted run continues where it stopped. Steps that need the `counting` step's results use the
shipped ones (`data/counting/af488_mask/`) for sections it has not re-made.

## Check

`python check.py` compares what was re-run, step by step:

- **counting**: each section's thresholds and number of objects per mask, against `data/counting/af488_mask/`;
- **cellpose**: each section's counts per method-threshold (all, home, crossed, junction region), against
  `results/counting/cellpose_counts.csv`;
- **stats**: the primary outcome and its sensitivity analyses, against `results/statistics/primary_outcome.csv`
  (when every section was counted);
- **browser**: each section's counts, against `results/section_browser/data.json`.

With the shipped masks the re-run is identical (checked on Windows 11 for all 614 sections' Cellpose counts and the
statistics, and for the counting and browser steps on a subset of sections). Cellpose on a GPU is not bit-for-bit
repeatable, so after `--infer` or `--segment` the counts are close, not identical.

## The method, in short

- **Cells** (`oasis/cpself.py`): Cellpose (`cpdino`, DINO backbone) fine-tuned on this dataset without hand labels
  (its training labels were the AF488 intensity mask's surest cells). It sees each section's AF488 and nuclear stain
  (maximum over z, each scaled to its 1st–99.9th percentile within the tissue); a mask is an AF488⁺ cell when its
  mean background-subtracted AF488 reaches **5 noise SD** of the section and its area is more than **20 µm²**.
- **Compartments** (`oasis/sides.py`, `oasis/cpself.py`): home or crossed by the cell's side of the junction interface
  (extended past its ends); the home side is the side with more AF488 mask area; junction region = in the stored band
  (2–302 µm into the partner organoid).
- **Primary outcome** (`oasis/stats.py`): log(1 + crossed cells per mm of junction interface), linear mixed model with
  genotype and experiment as fixed effects and slide as a random intercept (REML); the genotype effect as a fold.
- **What is data, not code**: the junction interfaces (found automatically, each reviewed by eye and redrawn by hand
  where wrong), the hand counts, and the trained Cellpose model.

## Code

```
run.py, check.py           the runner and the comparison
oasis/paths.py             every file read and written (inputs: OASIS_PAPER/data; outputs: OASIS_OUT, default ../rerun)
oasis/scan.py              reading the scans (OME-TIFF; channel planes, maximum over z)
oasis/cv.py                the AF488 intensity mask: smoothing, local background, robust threshold, cleanup
oasis/section_outline.py   the section's outline; cells outside the section are not counted
oasis/junction.py          the 5 µm overview, the section's outline, distance to the tissue edge
oasis/geometry.py          distance to the junction interface
oasis/sides.py             home or crossed: the side of the junction interface extended past its ends
oasis/cells.py             label maps, robust spread, placing cells
oasis/counting.py          step "counting"
oasis/sizes.py             step "sizes"
oasis/cpself.py            steps "infer" (GPU) and "cellpose": the paper's counting method
oasis/stats.py             step "stats": the primary outcome
oasis/segment.py           step "segment" (GPU, optional)
oasis/browser/             step "browser": section_browser.py (+ .html), browser_tiles.py, browser_outlines.py,
                           browser_small.py
```
