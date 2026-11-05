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
| `03_cells_run.ipynb` | The viewer's *Cells analysis* run on the reviewed sections (`cells_analysis\2026-10-28_090440 re-scored`, 614 sections, whole sections segmented): ten positivity rules recomputed from the saved cells and the genotype result under each (conclusion matrix, forest plot); sample bias (scan → pre-screen → run funnel, sections per slide, covariates by genotype, covariate-adjusted model, leave one slide / experiment out); distance of the positive cells to the organoid's edge (edge rank among their zone's cells: along the surface or through the middle, home against crossed, by genotype); Prism-style slide figures, representative sections chosen by rule, and interactive plotly views to inspect outliers on hover. Settings (run folder, channel, distances) in one cell; slim cell cache and figures (SVG, PNG, HTML) in `D:\Slidescanner extracted\run_analysis_20261028\`. |
| `af488_masks.py` | The classical AF488 mask of `validation/04_classical_vs_cellpose` (background-subtracted AF488 at median + 3, 5, 8 robust SD) on every section of the cells-analysis run: each cell's cover by each mask, and the masks' own objects with their place → `D:\Slidescanner extracted\run_analysis_20261028\af488_mask\` (CPU, ~13 s a section, four at a time; resumes). |
| `04_genotype_by_rule.ipynb` | WT3 against GD1 only, under nine positivity rules (raw mean and median-filtered mean, each at median + 3, 5, 7 SD; the AF488 mask at 3, 5, 8 SD, from `af488_masks.py`), and the mask's own objects: AF488+ crossed cells and cells in the junction region (per section and per mm of junction), the crossed cells' mean distance to the organoid edge and past the line (µm, ÷ assembloid length and ÷ the AF488+ centroid's distance to the line), edge distance against distance past the line, and density histograms of where the AF488+ cells lie, normalised both ways. Section-level graphs (Prism style), mixed model per rule, one conclusion table; example cells of each kind with which rules call them, and bright-background regions the raw rules read as cells. Reads the cell cache of `03_cells_run`; figures in `D:\Slidescanner extracted\run_analysis_20261028\figures_genotype\`. |
| `james_clicks.py` | (`python james_clicks.py quillan` does the same for Quillan's 13 sections, into `quillan_counts\`.) James's hand counts ("James counts", one dot per AF488+ cell, whole section; 14 sections) placed like the run's cells — junction region, distance to the line and the edge — and matched one to one to the segmented cells within 7 µm. Writes `run_analysis_20261028\james_counts\`. |
| `james_masks.py` | The AF488 mask in 21 variants (background none / Gaussian 20 µm / local median 45 µm × 3, 5, 7, 8, 10 SD, Otsu, triangle) on James's 14 sections: each cell's cover, the mask's objects, and which object each dot falls in. Writes `james_counts\methods\`. |
| `james_cellpose488.py` | Cellpose (the run's model) on the AF488 channel itself, raw and less its local median, on James's 14 sections (GPU, ~1 min a section): objects, cell cover, dots hit, label maps. |
| `james_bodies.py` | On James's 14 sections: the run's positive cells merged into labelled bodies for five rules, and the nuclei-seeded watershed territories of validation/05 with their AF488 values. |
| `cellpose488_retrain.py` | Cellpose (cpsam) fine-tuned on the AF488 channel from the two annotators' clicks (masks grown from each dot over the AF488 label; unclicked cells in the counted area taught as background): `tiles`, `train` (2-fold cross-validation by section, and trained on one annotator to predict the other; 30 epochs), `predict`, `numbers`. Writes `cellpose_retrained\` and the `cellpose488_retrained` family under each annotator's `methods\`. Precise but finds ~half the cells: not used (publication §3e). |
| `05_james_counts.ipynb` | James's counts against 79 positivity methods in six families (Cellpose nuclei + rule; the AF488 mask with no background, a Gaussian 20 µm or a local median 45 µm taken off, at 3–10 SD, Otsu and triangle, as covered cells and as objects; Cellpose on AF488 itself; merged bodies; watershed territories), as validation/09 did for Quillan: where the dots are, per-section agreement (Spearman, CCC, R²) in the whole section, junction region and crossed zone, cell-by-cell precision / recall / F1, one scoreboard, whether each gives the same WT3-against-GD1 conclusions as his count (14 sections, and the whole run where it can be computed), and pictures of the cells where they disagree. Figures in `figures_james\`. |

## `official_analysis/` — the analysis as it goes into the paper

Publication figures (greyscale data, AF488 images with white / magenta overlays, figure-legend titles), human annotators anonymised. AF488+ cells are counted as objects of the AF488 mask (Gaussian 20 µm background, median + 5 SD; `dataset/af488_masks.py`), chosen in `dataset/05` and `publication/01`.

| File | What |
|---|---|
| `01_validation.ipynb` | Automated against human totals of AF488+ cells per section (27 sections, two annotators, each in the area they counted): Lin's CCC and ratio-scale Bland–Altman with section-bootstrap CIs; the per-nucleus rule and the fine-tuned Cellpose for comparison; whether the human and the automated count give the same WT3-against-GD1 conclusion; a gallery of 12 regions with human dots and automated outlines. |
| `02_biology.ipynb` | WT3 against GD1 on the 614 analysed sections with that count: cells in the junction region, crossed (primary), on the home side, the share crossed, the crossed cells' distance to the edge and past the line (mixed model, Holm); robustness (leave one experiment / slide out, slide level, covariates); representative images; conclusion and methods text. |
| `tissue_area.py` | Each analysed section's tissue (assembloid) area, from the nuclear stain with the tissue mask the AF488 count is thresholded in; writes `run_analysis_20261028/section_tissue_area.csv` (about 1 s a section) for `02_biology` Figure S2. |
| `section_browser.py` + `section_browser.html` | A browser of every analysed section (614): the whole section's AF488 (~1.3 µm per pixel) with the junction line, the junction region, the human dots and the outline of every AF488⁺ object or cell (home white, crossed magenta: past the junction line extended past its ends, `sides.py`) as toggled overlays; a choice of positivity method and threshold (AF488 mask with a Gaussian 20 µm or local median 45 µm background, or the per-nucleus rule on the median-filtered or raw mean; 3–8 robust SD; default the counting method, Gaussian 5 SD) that the table, the plots and the outlines follow; a sortable, filterable table of the per-section numbers; the automated-against-human scatter, the crossed-per-mm strip plot and a size-against-outcome plot (click a point to open its section); ←/→ through the table; a gallery of thumbnails. `python notebooks/official_analysis/section_browser.py` (resumes; ~20 s a section for the outlines, four at a time) writes `run_analysis_20261028\section_browser\`; open `section_browser.html` there (from disk, or `python -m http.server` in the folder and http://localhost:8000/section_browser.html). |

## `publication/` — the verdict for the paper

| File | What |
|---|---|
| `01_pipeline_validation.ipynb` | Is the pipeline fit for publication? One claim per step, each with its evidence (95 % CIs by section or slide bootstrap) and a verdict (PASS / PASS with caveat / FAIL): (1) the data — sample flow from scan to analysis and genotype-biased attrition; (2) junction detection — held-out accuracy, against two people's lines, triage, and whether the line error changes % positive; (3) nuclei and AF488 positivity against two independent annotators (Quillan 13 + James 14 sections: CCC, Bland–Altman, precision / recall / F1); (4) the WT3-against-GD1 result on 614 sections with its robustness (leave one slide / experiment out, slide level, every rule, body count, covariates, reweighting for attrition, "more cells or more migration?"); (5) the verdict table; (6) limitations, what to do before submission, and a draft of the methods and statistics text. Figures in `figures_publication\`. |

Notebooks are executed in the conda env that has Jupyter:

```powershell
C:/Users/qfavey/.conda/envs/napari-env-q/python.exe -m jupyter nbconvert --execute --inplace <notebook>
```
