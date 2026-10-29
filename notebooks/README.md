# Notebooks

Four kinds, in four folders. Within each, the notebooks are numbered in the
order of the pipeline, so `01_…` comes first.

## `explainers/` — how the method works

Step-by-step walk-throughs on two real sections drawn by hand, from one slide: one
where the finder worked (034-033_s13, 15 µm from the hand-drawn line) and one where
it failed (034-033_s17, 341 µm off, status ok). They change when the method changes,
not when new data arrives.

| Notebook | What it explains |
|---|---|
| `01_junction_region.ipynb` | The automatic junction region, step by step, on the section that worked and the one that failed. |
| `02_junction_pixels.ipynb` | The same, pixel by pixel, with an animation from the seeds to the line, and where the failed section goes wrong. |
| **`pipeline_explainer.html`** | **The whole pipeline as an interactive web page** — pre-screen, junction, cells, positivity, crossing, counting — with live figures on the two real sections (a *Sample* switch on each; `?sample=failed` in the address opens on the failed one), galleries of good and bad organoids and of failure modes. Each section has a collapsible **Minimal code** block (from `explainer_code.py`; `python notebooks/explainers/explainer_code_check.py` checks the junction steps give the package's result on the worked section). Open it in a browser; works offline. Rebuild with `python notebooks/explainers/build_explainer.py` (page in `explainer_template.html`, data computed with the package). |

## `validation/` — how well it works on the annotated data

Each ends with a dated **What it showed**. Data and caches live in
`D:\Slidescanner extracted\junction_eval\`.

Numbered in the order of the pipeline: the junction (01–02), which cells are
positive (03–05), counting them (06–09).

| Step | Notebook | Question |
|---|---|---|
| junction | `01_junction_evaluation.ipynb` | The junction finder against the hand-drawn lines: line distance, Dice, tuning, triage, two annotators. |
| junction | `02_junction_seeding.ipynb` | Seeds and other settings of the junction finder on all 138 hand-drawn sections (tuning and held-out): which change helps, on which kind of section; the label change in the price. |
| positivity | `03_positivity_metrics.ipynb` | Which rule should call a cell positive: thresholds from the band, a negative control, local contrast, two-population models; judged without and with the hand clicks. |
| positivity | `04_classical_vs_cellpose.ipynb` | AF488 segmented directly (background taken away, k × SD / triangle / Otsu) against Cellpose nuclei + median + 3 SD, median + 7 SD (median filter) or ring contrast, and against the clicks: same cells? more of their shape? Script `classical_vs_cellpose.py`, cache `junction_eval/classical/`. |
| positivity | `05_watershed_cells.ipynb` | Each cell grown over its own AF488 — a watershed from the nuclei on the distance transform of the median-filtered AF488 — and scored by ring contrast or median + 7 SD (median filter) on that, against the current rule (median + 7 SD (median filter) over nucleus + 3 µm), on the 39 scans and against the clicks. And merging the current rule's positive cells that lie in one labelled body (distance-transform watershed of the stain, h-maxima): the count halves, F1 against the clicks 0.34 → 0.43. Script `watershed_cells.py`, cache `junction_eval/watershed/`. |
| counting | `06_counts_in_band.ipynb` | Positive cells in the junction band, per sample and per genotype. |
| counting | `07_counts_crossed.ipynb` | Positive cells that passed the line into the other organoid: how many, how far, along the surface or not. |
| counting | `08_band_choice.ipynb` | How much the counts depend on which band is used: first annotator, second, or the finder's. |
| counting | `09_manual_vs_auto.ipynb` | Quillan's hand counts against the automatic counts, on the samples he counted. |

Scripts beside them, run from the repository root:

| Script | Does |
|---|---|
| `count_cells.py` | Segments each annotated sample once and measures every cell every way the notebooks compare (GPU, ~2 min a sample) → `junction_eval/cells/`. |
| `count_common.py` | What the count notebooks share: loading cells with their geometry (cached in `junction_eval/cell_tables/`), the bands, the clicks, the positivity rules, the statistics. |
| `junction_tuning.py` | Overview cache, scoring and Bayesian search for the junction finder; also the list of excluded samples and the second annotator's key. |
| `junction_walk.py` | The random-walker variants for `junction_evaluation`. |
| `junction_seeding.py` | Finds every section with a hand-drawn junction, splits them, runs every variant → `junction_eval/seeding.jsonl`, seed diagnostics; with variant names, reruns only those. |
| `junction_labels.py` | The change between the two labels as part of the cutting price, on the same sections → `junction_eval/labels.jsonl` (section 6 of `02_junction_seeding.ipynb`). |
| `junction_classes.py` | Every combination of five finder settings on every hand-drawn section, to choose settings separately for sections with and without a dark line → `junction_eval/classes.jsonl` (section 7 of `02_junction_seeding.ipynb`). |
| `junction_browser.py` + `.html` | An interactive page of all hand-drawn sections: error against any figure (including organoid polarity), each section's picture with the hand-drawn and found lines on hover → `junction_eval/browser/`. |
| `watershed_cells.py` | Each cell's territory from a nuclei-seeded watershed on the median-filtered AF488, its means and ring contrast, on the 39 scans (nuclei from their cells-analysis run) and the 12 clicked samples (segmented here, GPU) → `junction_eval/watershed/`, `watershed_clicked/`; with `bodies`, the labelled body each cell lies in → `bodies.parquet`. |
| `junction_downstream.py` | Junction analysis counts in each annotator's band and the finder's. |

## `junction_graph/` — the junction from the nuclei, not the pixels

Ideas for the junction finder that work on the segmented nuclei rather than the
5 µm overview. Scored as `validation/02_junction_seeding` (the 138 hand-drawn
sections, same split), so the numbers compare directly.

| Notebook | Question |
|---|---|
| `01_nuclei_graph_cut.ipynb` | The minimum cut made on a graph of the nuclei (Gabriel or Delaunay), each edge as dear to cut as its length, its nuclei's neighbours, the density round it or the change of the cells' own label make it, between the pixel finder's seeds: closer to the hand-drawn line than the pixel cut? Script `nuclei_graph.py`, cache `junction_eval/graph/`. |

## `dataset/` — the whole dataset

| File | What |
|---|---|
| `run_dataset.py` | The pipeline over every passing sample of the pre-screen table: junction (stored in the sample, QC picture), cells near the line, positivity under the validated rules, counts in the band and past the line, a picture per sample. Resumes where it stopped. Output: `D:\Slidescanner extracted\dataset_analysis_20261021\`. |
| `01_whole_dataset.ipynb` | The analysis of that run: QC, genotype comparison (samples, slides, mixed model), robustness, where the cells are, band choice. |
| `02_positivity_dataset.ipynb` | The positivity rules recomputed on that run's cells: median + k SD with the reference pooled per section, slide, experiment or dataset, ring contrast, negative control; how many cells each calls, how they agree, and the genotype result under each. |
| `03_cells_run.ipynb` | The viewer's *Cells analysis* run on the reviewed sections (`cells_analysis\2026-10-28_090440 re-scored`, 614 sections, whole sections segmented): ten positivity rules recomputed from the saved cells and the genotype result under each (conclusion matrix, forest plot); sample bias (scan → pre-screen → run funnel, sections per slide, covariates by genotype, covariate-adjusted model, leave one slide / experiment out); Prism-style slide figures, representative sections chosen by rule, and interactive plotly views to inspect outliers on hover. Settings (run folder, channel, distances) in one cell; slim cell cache and figures (SVG, PNG, HTML) in `D:\Slidescanner extracted\run_analysis_20261028\`. |

Notebooks are executed in the conda env that has Jupyter:

```powershell
C:/Users/qfavey/.conda/envs/napari-env-q/python.exe -m jupyter nbconvert --execute --inplace <notebook>
```
