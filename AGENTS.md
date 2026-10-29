# AGENTS.md

How to work in this repository. Written for coding agents (Claude Code, Codex, …)
and anyone pairing with them. `CLAUDE.md` imports this file.

ARGUS is a customised napari viewer for Imaris / TIFF / OME-Zarr microscopy, plus
analysis engines (segmentation, regions, assembloid junctions) and Streamlit
explorers in `apps/`. The user is a biologist-developer: the README is the user
manual, and results are judged on real slide-scanner data, not only on tests.

## Ground rules

- **Never commit or push unless asked.** Work on the current branch (`assembloids-study`
  at the time of writing); `main` is the PR target.
- Commit messages: Conventional Commits with a scope and a plain-English subject,
  e.g. `feat(fusion): add the fusion region panel, drop four unused panels`,
  `fix(experiment): skip macOS ._ companion files when listing a folder`.
- Look before deleting or overwriting. Folders outside git (e.g. `C:\Users\qfavey\oasis`)
  get a backup copy in the scratchpad *before* the first edit.
- Raw data lives on `D:\Slidescanner extracted\` and other drives. Read it freely;
  write only into an obvious output folder (e.g. `junction_eval\`), never beside the
  originals unless the feature is meant to (the in-file store, QC pictures).
- Scratch scripts, patch scripts and intermediate outputs go in the session
  scratchpad, never in the repo. Do not leave `__pycache__` or copies in `notebooks/`.

## Environments (Windows workstation, reached over Remote Desktop)

| Use | Interpreter |
|---|---|
| Repo tests, scripts, the desktop shortcut | Store Python 3.13: `python` |
| Executing notebooks | `C:/Users/qfavey/.conda/envs/napari-env-q/python.exe -m jupyter nbconvert --execute --inplace <nb>` |
| Fallback napari env | `C:\ProgramData\miniconda3\python.exe` (3.12, `pip install --user` only) |
| OASIS Streamlit app | `C:\Users\qfavey\oasis\.venv\Scripts\python.exe` (own venv) |

- Do **not** use `C:\ProgramData\miniconda3\envs\napari-env` (no Qt, Python 3.8).
- GPU: RTX 4090, torch `+cu130`. torch and torchvision must be the same build.
- Over RDP any napari window takes minutes to appear (software OpenGL). A slow
  window is not a hang; imports and file reading are seconds.
- Set `PYTHONIOENCODING=utf-8` if a script prints µm / arrows to a cp1252 console.
- Notebook setup cells carry an SSL guard (`ssl.create_default_context()` →
  stub `_load_windows_store_certs` on `SSLError`). Keep it in any new notebook.

## Architecture: engine + thin widget

Every feature is split in two, and this is the rule most worth keeping:

- **Engine** — `microscopy_viewer/<feature>.py`. Pure numpy / scipy / skimage /
  pandas. **No Qt, no napari imports.** Takes arrays and options dataclasses, returns
  arrays, tables and metrics. This is what tests, notebooks and batch runs call.
- **Widget** — `microscopy_viewer/widgets/<feature>_widget.py`. Qt controls, napari
  layers, and a background worker (`napari.qt.threading.thread_worker` / QThread)
  around the engine. No maths here.
- New panels are one `register_panel(PanelSpec(...))` call in
  `widgets/registry.py`; the factory imports the widget lazily.
- New file formats: a module with `can_read(path)` / `read(path) -> list[LayerSpec]`,
  registered in `_READERS` in `loaders/__init__.py`.
- Options live in a dataclass (e.g. `junction.JunctionOptions`) with module-level
  default constants (`BEND_UM`, `MEDIAN_UM`, …). The widget mirrors the defaults as
  its own constants and a test checks they agree.
- Experimental options that did not help stay in the engine, **off by default**, with
  the evaluation that decided it recorded in the notebook (tears, edge cost,
  random walk, `split_method="auto"`).

### Things that bit us

- **File handles and threads.** h5py / tifffile handles are owned per thread
  (`loaders._OWNER`). Background work releases only its own:
  `release(path, own_thread_only=True)`. A thumbnail thread closing the viewer's
  handle crashed the viewer once.
- **Don't wait on the folder scan.** Matching a table to samples uses the folder
  listing (`folder_paths`), not the explorer scan; selections made before the scan
  finishes are kept as pending and applied afterwards.
- Qt `clicked` passes `checked` as the first argument: connect with a lambda when
  the slot has optional parameters (`save_list(path=False)` bug).
- Stored regions: manual first (`fusion.MANUAL_KEY`, "manual annotated junction
  region"), automatic as fallback (`fusion.region_keys`). Every reader of a region
  goes through that, and records which one it used.
- **torch after Qt.** In a script, importing `qtpy.QtWidgets` before torch makes
  torch fail with `WinError 1114` (c10.dll), and segmentation then reports
  "needs the cellpose package". Do as the viewer does: `import microscopy_viewer`
  first, create `napari.Viewer()` before importing widget modules or torch.
- Guard geometry against degenerate input (a line collapsing to a point broke
  `fit_spline`).

## Tests

Tests are plain scripts, not pytest: each `tests/test_*.py` has a `check(cond, msg)`
helper, prints `ok` / `FAIL`, and exits non-zero on failure. They draw synthetic
samples (discs of "tissue", known junctions) so the expected answer is known.

```powershell
python tests/make_sample_data.py          # once; writes tests/sample_data/
python tests/test_junction.py             # run the file(s) you touched
python tests/smoke_gui.py                 # builds the real viewer; slow over RDP
```

- Add checks to the matching `tests/test_<feature>.py` for every engine change;
  a new engine module gets a new test file **and** a step in
  `.github/workflows/tests.yml` and a line in the README's *Checks* list.
- CI installs only the non-GUI dependencies (no napari, antspyx, cellpose), so
  engine tests must import without them.
- Streamlit (OASIS, `apps/`): headless `streamlit.testing.v1.AppTest`; run with the
  app's own venv.
- Run the tests before reporting work done, and say which ones ran.

## Evaluating algorithm changes (junction finder and similar)

Visual plausibility is not enough; measure against hand annotations.

- Ground truth: `D:\Slidescanner extracted\picks_20261020_0742.csv` (31 hand-drawn
  junctions). Cache and results: `D:\Slidescanner extracted\junction_eval\`
  (`scores.csv`, per-sample `*.npz`, `trials.jsonl`, `walk.jsonl`, `downstream.jsonl`).
- Excluded samples live in `junction_tuning.EXCLUDE` (with the reason); `split()` is
  drawn on all samples, then excluded ones dropped, so the held-out set never moves.
  The second annotator's region is `junction_tuning.SECOND_KEY`
  ("manual annotated junction region quillan").
- `notebooks/validation/junction_downstream.py` counts cells (Cellpose, GPU, ~1 min/sample) in
  the hand-drawn and the automatic band: the metric that matters is the change in
  % positive, not Dice.
- Harness: `notebooks/validation/junction_tuning.py` (`run_case`, cache, Bayesian `search`,
  candidate `RULES`) and `notebooks/validation/junction_walk.py` as the pattern for a new
  variant sweep: define `VARIANTS` as overrides of the defaults, run in a
  `ProcessPoolExecutor`, write JSONL, analyse in the notebook.
- Metrics: Dice / IoU on the band, Dice line-only, mean symmetric line distance,
  HD95, fraction within 50 / 100 µm, side agreement. Headline number is the
  **capped mean error** (300 µm cap; no junction = 300).
- Always report a **held-out** split (stratified by Ripped / Folded / Quality).
  Tuning on 31 samples overfits easily (BO: 73→64 µm on tuning, 158→163 held out).
  Change a default only if held-out improves.
- Since 26 Oct the evaluation set is **138 hand-drawn sections** (`junction_seeding.py` finds them;
  `junction_eval/annotated_all.csv`, 45 held out; the first 30 keep their old split). The new
  108 are much harder (many without a dark line): capped mean 175 µm against 67 on the first 30.
- Current baseline (bend 1000, tissue from every channel, no-seam fallback on since 26 Oct):
  capped mean 130 µm tuning / 159 µm held out on the 138; 66 µm on the first 30
  (`02_junction_seeding.ipynb`, `seeding.jsonl`).
- No seam: the cut's "Seam price" > 0.4 flags *no dark line at the junction* and the junction is
  found again from dominant seeds (`no_seam_price`, default `NO_SEAM_PRICE`). Seam price is the
  best single predictor of a bad line (Spearman 0.40). Dominant seeds everywhere help seamless
  sections but break seamed ones (first 30: 67 → 88 µm); seeding from every AF488 patch is worse
  (+25 µm). "default" in `seeding.jsonl` is the fallback off (`junction_seeding.BASE`).
- Label change in the price (`label_change`, 27 Oct, `junction_labels.py`, `labels.jsonl`): the
  steepness of AF488 − AF555 multiplies the price down to 0.3. Off: with tears priced it is best on
  tuning (130 → 118 µm) but held out only 159 → 154 µm (p 0.67, 7 fixed / 8 broken); the factor map
  is full of rings round every cell patch, not one valley at the junction.
- Per-class settings (27 Oct, `junction_classes.py`, `classes.jsonl`, 32 combinations): the second
  search after a no-line first cut takes its own settings (`no_seam_*`, defaults = the old fallback).
  Chosen per class on tuning: tuning 130 → 108 µm but held out 159 → 150 µm (p 0.29) and the first
  30 get worse (66 → 80 µm); both classes pick dominant seeds, so seam price is not the right split.
  Defaults unchanged.
  Triage: draw by hand when status is fail or Shape is Damaged (7/8 failures, no
  good line sent); the "check" status is not useful for triage. On kept samples
  % positive agrees within ~±2 pp, no bias.
- Write the result into `notebooks/validation/01_junction_evaluation.ipynb` as a new section
  (method → table/figures → dated "What it showed" with the decision), then execute
  the notebook in the conda env above and check it has 0 errors.

- Counting methods: `notebooks/validation/count_cells.py` segments each sample once (whole
  section, ~2 min, GPU) and stores per-cell values for every expansion / filter /
  statistic (and a 3–6 µm ring for local background) in `junction_eval/cells/`; the count notebooks do the bands,
  thresholds and crossing analysis from those without re-segmenting.

## Notebooks

`notebooks/README.md` is the index. Within each folder the notebooks are numbered
in pipeline order (`01_…`); a new one takes the number of its step and the ones
after it move up, with every reference to their file names updated.

- `explainers/` (`01_junction_region`, `02_junction_pixels`): how the method works,
  step by step. Update them when a step or a default changes.
- `validation/`: how well it works on the annotated data, each ending with a dated
  "What it showed". Junction (`01_junction_evaluation` — the finder vs hand-drawn
  lines; append, don't rewrite history — and `02_junction_seeding`), positivity
  (`03_positivity_metrics`, `04_classical_vs_cellpose`, `05_watershed_cells`) and
  counting (`06_counts_in_band`, `07_counts_crossed`, `08_band_choice`,
  `09_manual_vs_auto`). The count notebooks all load through `count_common.py` —
  put shared code there, not in a notebook. Their scripts (unnumbered) sit beside
  them; the kernel's cwd is the notebook's folder.
- `junction_graph/` (`01_nuclei_graph_cut`): the junction from the segmented nuclei
  instead of the overview's pixels; scored as `02_junction_seeding`.
- `dataset/` (`01_whole_dataset`, `02_positivity_dataset`): the whole dataset's run.
- Edit notebooks with a small script (load JSON → append/replace cells → dump with
  `ensure_ascii=False, indent=1`) or NotebookEdit; back up first, and look up cell
  indices by content, not by number. Execute each after editing and check it has
  0 error outputs.

## Documentation and style

- **Update the README with every user-visible change**: the panel's section under
  *Use*, the *Layout* tree for new modules, *Checks* for new tests.
- Write prose for a biologist: plain words, what a control does and why, units in
  µm. UI labels are short sentences ("Past the tissue edge", "Follow the tissue"),
  not jargon. Tooltips explain the trade-off.
- Code: match the surrounding style. `from __future__ import annotations`, type
  hints, module docstring saying what the module does and that it has no Qt,
  `#:` comments on constants and dataclass fields, short functions named for what
  they return (`band_along_tissue`, `cut_looks_wrong`).
- Keep each file's line endings (most are LF). Python `write_text` on Windows can
  turn them into CRLF; patch scripts should read bytes, detect `\r\n`, and write
  back the same.

## Tooling tips for agents

- Prefer the Edit tool for code changes. For multi-hunk patches, write a Python
  patch script to the scratchpad with the Write tool (bash heredocs mangle quotes
  and `\n` here), assert each target string occurs exactly once, then run it.
- Long sweeps (evaluation, BO) run in the background; keep the main thread working.
- Use subagents for broad, independent questions (e.g. "does sample class predict
  failure?"); keep edits to shared files in the main thread.
- Finish with: what changed, which tests ran and passed, what was not verified
  (e.g. "not looked at in the GUI"), and anything left uncommitted.

## Related project: OASIS

`C:\Users\qfavey\oasis` is the pre-screen / QC Streamlit app whose exports
(`oasis_prescreen*.csv`) feed the sample picker here. Not under git: back up before
editing, test with its `.venv` (Streamlit 1.65, `AppTest`), update its README, and
remind the user to restart `oasis app`.
