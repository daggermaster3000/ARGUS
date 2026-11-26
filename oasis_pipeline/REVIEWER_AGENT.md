# Guide for a reviewer's coding agent

You are helping a reviewer check the analysis of the OASIS assembloid paper. This file tells you what can be
verified, the cheapest way to verify it, how to read the results (including the interactive section browser), and
where each method parameter lives in the code. Read it fully before running anything; the full data are ≈ 700 GB
and the full re-run takes hours, but the central claims can be checked in minutes.

## 1. What the code reproduces

| Claim in the paper | How to verify | Cost |
|---|---|---|
| Every section's AF488⁺ cell counts (all, home, crossed, junction region) with the self-trained Cellpose method | `run.py --steps cellpose` + `check.py` | ~1 min, CPU, ~0.7 GB of data |
| Primary outcome: WT3 ÷ GD1 crossed cells per mm of junction interface = **1.83 (95 % CI 1.32–2.55), P = 0.0003**, 614 sections, 36 slides | `run.py --steps stats` + `check.py` | < 1 min |
| Sensitivity of the outcome to the signal threshold (3, 5, 8 noise SD) and size filter (20, 10, 5, 3 µm²) | rows of `rerun/statistics/primary_outcome.csv` | included above |
| The AF488 intensity mask (noise SD, home side, section outline) from the raw scans | `run.py --steps counting` + `check.py` | ~20 s per section, ~1 GB raw scan each |
| The Cellpose masks themselves, from the raw scans with the shipped model | `run.py --infer` | GPU, hours; not bit-for-bit repeatable |
| The section browser (outlines, counts per section, hand counts) | §4 | open the shipped one; rebuild a few sections in minutes |

**Not reproduced by this code** (inputs, not outputs): the junction interfaces (found automatically, reviewed by eye,
redrawn by hand where wrong), the manual cell counts, and the trained Cellpose model (training is described in the
Methods). The validation statistics against the manual counts and the paper's figures were made with separate
analysis notebooks, which are not part of this repository.

## 2. Setup

1. Locate the data package (a folder holding `data/` and `results/`). Set `OASIS_PAPER` to it unless this repository
   sits inside it as `pipeline/`. Set `OASIS_OUT` to a writable folder for outputs (default `<package>/rerun`).
2. Python 3.10 (3.13 also works) in a virtual environment: `pip install -r requirements.txt`. Do **not** install the
   GPU requirements unless you will run `--infer` or `--segment`.
3. Sanity check: `python -c "from oasis import paths; print(paths.ROOT, len(paths.section_names()))"` → the package
   path and `614`.

Data each path needs (download only these if the package is remote):

| Path | Needs | Size |
|---|---|---|
| Fast (§3 A) | `data/design/`, `data/junction/sections/`, `data/junction/region/`, `data/counting/af488_mask/*/objects.parquet`, `data/counting/cellpose/*.parquet`, `results/counting/`, `results/statistics/` | ~0.7 GB |
| Browser rebuild (§4) | the above + `data/raw/<section>.ome.tiff` of the chosen sections + `data/counting/cellpose/<id>.npz` + `data/segmentation/` of those sections | ~1–2 GB per section |
| Full | everything | ~700 GB |

## 3. Running efficiently

**A. Fast path — the counts and the primary outcome (minutes, CPU only).**

```
python run.py --steps cellpose,stats
python check.py
```

Expected:

```
cellpose counts: 614 sections from …/data/counting/cellpose -> …/counting/cellpose_counts.csv; cpself-5: 116493 cells, 10756 crossed
primary outcome (cpself-5, crossed AF488⁺ cells per mm of junction interface): WT3/GD1 = 1.83 (95 % CI 1.32–2.55), P = 0.00034, 614 sections on 36 slides
cellpose: 614 sections, 7368 method-thresholds compared, 0 differ in 0 sections
stats: primary outcome cpself-5 WT3/GD1 1.83 (1.32–2.55); identical to the shipped
```

**B. From the raw scans, a sample (≈ 20 s per section with 2 workers; 5 sections in under a minute).** Re-make the intensity mask and the counts for a few
sections and compare:

```
python run.py --steps counting,cellpose --sections 5 --workers 2
python check.py                       # counting: 5 sections compared, 0 differ; cellpose: 0 differ
```

`--sections N` always takes the first N sections by name. For particular sections, call the step from Python with
their names, e.g. `from oasis import counting, cpself; names = [...]; counting.main(names); cpself.count(names)`
(names as in `data/design/analysed_sections.csv`, column `name`).

**C. Full re-run (~4–5 h, 4 workers, ~2 GB of memory per worker):** `python run.py`, then `python check.py`.
Every step resumes where it stopped. Raise `--workers` only as memory allows.

**Avoid unless asked:** `--infer` and `--segment` (GPU, hours to a day, results close but not identical), and the
full `browser` step (2.5–3.5 h, ~8 GB of output).

## 4. The section browser

The browser shows every section with its junction interface, junction region, the counted cells' outlines (home and
crossed coloured differently), the manual counters' points and the blinded count squares, with a table and plots.

- **Shipped (complete)**: `results/section_browser/section_browser.html` in the data package. It works straight from
  disk in Chrome, Edge or Firefox. URL options: `?m=cpself-5` the counting method shown (the default), `#<section id>`
  opens that section (e.g. `#052-051_s15`), `?gallery` opens the thumbnails.
- **Rebuilt for a few sections**: `python run.py --steps counting,sizes,browser --sections 3` (≈ 5 min) →
  `rerun/results/section_browser/section_browser.html`; `check.py` compares its counts with the shipped browser.

Reading it without a screen (for an agent): `data.json` (also as `data.js`) holds everything the page shows.

```
data.json
  default            "cpself-5"
  methods[]          key, label, ks (thresholds), min_um2 (size filter), family, variant
  sections[]         one per section:
    id, name, genotype, experiment, slide, length_um (junction interface), area_mm2
    counts[<method>-<k>]   n (all), home, crossed, reg (junction region), area2, thr
    humans[]         manual counts: n (annotator), human (count), automated[<method>-<k>] (count in the same area),
                     compartments {home, region, crossed: {human, automated}}, dots {x, y, in} (picture pixels)
    squares[]        the blinded 120 µm count squares: code, compartment, human {annotator: count}, automated
    line, region     the junction interface and region (picture pixels; um_per_px gives the scale)
parts/masks/<id>__<method>-<k>.js   each counted cell: rings (outline, picture pixels), away (µm past the interface),
                                     side (1 crossed, 0 home, −1 no interface), reg (in the junction region)
img/<id>.jpg                         the section's AF488 overview (picture pixels)
```

To look at a section visually from code, open the page in a headless browser and take a screenshot of
`section_browser.html?m=cpself-5#<id>` (the page needs a few seconds to load its data).

## 5. Where each parameter lives

| Parameter (as in the Methods) | Value | Code |
|---|---|---|
| Input scaling for Cellpose | 1st–99.9th percentile of the section's tissue, clipped 0–1.5 | `oasis/cpself.py` `NORM_PCT`, `scaled()` |
| Cellpose flow / cell-probability thresholds | 0.4 / 0 | `oasis/cpself.py` `FLOW_THRESHOLD`, `CELLPROB_THRESHOLD` |
| Background subtraction for the signal | Gaussian 0.5 µm less Gaussian 20 µm | `oasis/cv.py` `SMOOTH_UM`, `BACKGROUND_UM`; `cpself.infer_one` |
| Noise SD | the section's robust SD (1.4826 × MAD of its tissue pixels) | `oasis/cv.py` (`classical_masks`), `counting.py` → `summary.json` `noise_sd` |
| Positivity threshold | signal ≥ 5 noise SD (3, 8 as sensitivity) | `oasis/cpself.py` `KS`, `DEFAULT` |
| Size filter | area > 20 µm² (10, 5, 3 as sensitivity) | `oasis/cpself.py` `SIZES` |
| In the section | centroid in the section outline (within 10 µm) and in the assembloid | `oasis/section_outline.py` `in_section` |
| Home / crossed | side of the junction interface extended past its ends; home = side with more 8 SD mask area | `oasis/sides.py`; `cpself.home_sign` |
| Junction region | the stored band, 2–302 µm into the partner organoid | `data/junction/region/`; `cpself.placed` |
| Primary model | log(1 + crossed per mm) ~ genotype + experiment, slide random intercept, REML | `oasis/stats.py` `fold()` |

## 6. Expected differences and pitfalls

- After `--infer` or `--segment` the counts differ slightly (Cellpose on a GPU is not deterministic); `check.py` then
  reports differences — this is expected, compare their size, not their presence.
- `stats` needs both genotypes; with `--sections N` it reports "too few for the model" and does nothing.
- Steps that need the intensity mask fall back to the shipped `data/counting/af488_mask/` for sections not re-run,
  so the fast path needs no raw scans.
- Paths: everything is read from `OASIS_PAPER/data` and written under `OASIS_OUT`; nothing shipped is modified.
- Windows consoles: set `PYTHONIOENCODING=utf-8` if µm or ÷ print as errors.
