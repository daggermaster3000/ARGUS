# ARGUS — Microscopy Viewer

A customised [napari](https://napari.org) viewer for rapid microscopy image inspection.
It opens Imaris `.ims`, TIFF, OME-TIFF and OME-Zarr datasets, shows the acquisition
metadata alongside the image, and lets you measure and annotate in calibrated
micrometres — then export a snapshot for slides and the measurements as a spreadsheet.

It stays a plain Python project: no bundling, no standalone executable. Windows also
gets a script that drops a desktop shortcut pointing at the launcher, so the viewer
opens with a double-click.

Runs on Windows, macOS and Linux. Python 3.9 or newer.

---

## Install

```bash
git clone https://github.com/daggermaster3000/ARGUS.git
cd ARGUS
python -m pip install .
```

That pulls in napari and a Qt binding suited to your platform, then puts a
`microscopy-viewer` command on your PATH:

```bash
microscopy-viewer                       # empty viewer
microscopy-viewer cells.ims other.tif   # open files straight away
```

Working on the code instead of just using it? `python -m pip install -e .` installs
in place, so edits take effect without reinstalling.

<details>
<summary><b>Prefer a virtual environment (recommended if you have no napari yet)</b></summary>

```bash
python -m venv .venv
# Windows:            .venv\Scripts\activate
# macOS / Linux:      source .venv/bin/activate
python -m pip install .
```

Qt needs a graphical session. Over SSH use X forwarding or a desktop session; on a
headless Linux box the viewer will not start, though every non-GUI check still runs.
</details>

<details>
<summary><b>Already have a napari environment</b></summary>

Install into it without letting pip touch your napari:

```bash
python -m pip install --no-deps .
python -m pip install -r requirements.txt   # anything you are missing
```
</details>

### Windows: desktop shortcut

Optional, and Windows only:

```powershell
python -m pip install ".[shortcut]"
python install_shortcut.py
```

`install_shortcut.py` creates **Microscopy Viewer** on your desktop. It reports the
interpreter it wired the shortcut to, and warns about any missing package before you
double-click.

| Command | Effect |
|---|---|
| `python install_shortcut.py --start-menu` | also add a Start Menu entry |
| `python install_shortcut.py --console` | launch via `python.exe` so a console stays open (debugging) |
| `python install_shortcut.py --uninstall` | remove the shortcuts |

On macOS and Linux there is no shortcut installer — launch it from a terminal, or
make a `.desktop` entry pointing at the `microscopy-viewer` command.

## Use

- Run `microscopy-viewer` — with file paths after it to open them at startup.
- **Drop image files onto the open window** to add them to the current session.
- On Windows, **double-click** the shortcut, or **drop files onto the shortcut**.
- From a source checkout without installing: `python launch_viewer.py cells.ims`

Several datasets can be open at once; each channel becomes its own layer, blended
additively, so you can toggle and compare them from napari's layer list.

### Channel colours

Fluorescence channels use the display colour stored in the file, or the next entry
in a colour cycle when the file has none.

**Brightfield channels are always grayscale.** Transmitted light carries no emission
colour, but acquisition software still hands those channels a slot in its colour
cycle — a brightfield channel arriving tagged green is common. The reader matches on
the channel name and overrides the stored colour: `Brightfield`, `BF`, `TL`, `TL-BF`,
`DIC`, `Phase`, `PhC`, `Transmitted`, `Transmission`, `Trans`, `T-PMT` and `ESID` are
all recognised, case- and separator-insensitive. Short abbreviations are matched as
whole words, so `BFP` stays a fluorescence channel. To add a name your setup uses,
extend `_BRIGHTFIELD_TOKENS` in
[microscopy_viewer/loaders/layer_spec.py](microscopy_viewer/loaders/layer_spec.py).

### Toolbar

| Button | Shortcut | Does |
|---|---|---|
| Open Images | `Ctrl+O` | file picker, multi-select |
| Export Snapshot | `Ctrl+S` | save the current view as PNG / TIFF / JPEG |
| Export Measurements | `Ctrl+E` | write all ROI measurements to `.xlsx` |
| Export Slide | `Ctrl+P` | build a PowerPoint figure: a row per dataset, a column per channel, plus the merge. Also batch mode — works with nothing open |
| Auto Contrast | `Ctrl+Shift+A` | stretch each visible layer to the 0.5–99.5 percentile of what is on screen |
| Reset Contrast | `Ctrl+R` | back to the full data range |
| 2D / 3D (MIP) | `Ctrl+D` | switch the canvas between slice view and a 3D maximum-intensity projection (a *view* — to draw ROIs on a projection use **Create projection layers** in the ROI intensity comparison panel) |
| Toggle Scale Bar | `Ctrl+B` | show / hide the calibrated scale bar |
| Show/Hide Metadata | `Ctrl+M` | collapse the metadata dock |

### 3D and maximum-intensity projection

**2D / 3D (MIP)** switches the viewer into a volume rendering and puts every image
layer into `mip` mode.

napari renders a multiscale image at its *coarsest* pyramid level whenever the
viewer is in 3D — it picks `len(data) - 1` outright, regardless of zoom. Since
Imaris files open as pyramids, plain napari shows a small, blocky volume in 3D.
This viewer hands napari a single-level pyramid while in 3D so the level it picks
is the one we chose, and puts the full pyramid back on return to 2D, where
multiscale rendering works properly and is worth keeping.

The level chosen is the finest one the GPU can actually hold. The budget is read
from the card at runtime rather than assumed — free video memory and
`GL_MAX_3D_TEXTURE_SIZE`, since a volume is uploaded whole and can be refused
either for total size or for being too long on one axis. On an RTX 4090 with
22.5 GB free that works out at about **1.4 billion voxels**, against the 128M a
fixed default would have allowed, so full resolution is used in practice instead
of dropping to a coarser level. When a volume genuinely does not fit, the next
level down is used and the status bar says so; the voxel size is rescaled to
match, so ROIs, measurements and the scale bar stay correct either way.

Override the budget with `MICROSCOPY_VIEWER_3D_VOXEL_BUDGET` (in voxels). If the
GPU cannot be queried the old conservative 128M default applies.

### Local volume cache

Source data usually lives on a NAS, where re-reading a stack dominates the wait.
The first time a volume is shown in 3D it is copied to a local `.npy` under
`%LOCALAPPDATA%\MicroscopyViewer\volume_cache\` and memory-mapped, on a worker
thread so the viewer stays usable. Everything afterwards reads from local disk.

Measured on a 17 × 2040 × 2040 deconvolved stack over the network:

| | |
|---|---|
| first switch to 3D (cold, reads the NAS) | 28.8 s |
| re-entering 3D in the same session | 0.9 s |
| first switch to 3D in a later session | 9.8 s |
| cache written | 283 MB in 3.2 s |

Entries are keyed by the source file's path, size and modification time plus the
array's shape and dtype, so a changed file never resolves to a stale copy. The
cache is capped at 32 GB and pruned least-recently-used; a file still mapped by an
open layer is skipped rather than forced. Set `MICROSCOPY_VIEWER_CACHE_LIMIT` (in
bytes) to change the cap, or `MICROSCOPY_VIEWER_NO_CACHE=1` to switch it off — the
viewer works identically either way, just slower. Deleting the folder is safe at
any time.

### Metadata panel

Updates automatically as you select a different layer. It reports exposure time,
excitation and emission wavelengths, laser power, objective, numerical aperture,
pixel size, Z-step, time interval, channel names and acquisition date — whatever
the file actually contains. Everything the reader found but could not classify is
kept under **Other file metadata** rather than dropped, and **Copy to clipboard**
gives you the whole panel as tab-separated text.

If a file carries no voxel size, the panel says so explicitly and measurements
switch to pixels rather than silently reporting wrong micrometres.

### Measuring and annotating

The **Measurements** dock drives a standard napari Shapes layer:

1. **New ROI layer** — adds a Shapes layer matched to the active image's
   dimensionality and voxel size.
2. Pick a tool: line, polyline, polygon, rectangle or ellipse.
3. Draw on the image. The table fills in immediately.

| Shape | Reported |
|---|---|
| Line | length, ΔX, ΔY |
| Polyline | path length |
| Rectangle, polygon | area, perimeter |
| Ellipse | area, perimeter, major and minor axis |

Values are in µm and µm² whenever the file provided a voxel size. A line drawn
across Z reports a true 3D distance. ROI names are editable in the first column and
follow through to the spreadsheet and the on-canvas labels.

### ROI intensity comparison

A separate dock, tabbed behind **Measurements**, for asking whether a region is
brighter under one condition than another — and how separable the two pixel
populations actually are.

**Setup tab.** Assign open image layers to named conditions ("TEB + Biotin",
"TEB", "Biotin", …); **One per layer** fills the table from what is open, and
names are editable. Any number of conditions works. Draw rectangles or polygons
into a single `Comparison ROIs` layer, then assign them in the ROI table:

| Column | Meaning |
|---|---|
| **Applies to** | `All conditions` for a shared outline, or one condition for a ROI belonging to that sample only |
| **Role** | `Signal` or `Background` |
| **Compare as** | display label for the curve; give ROIs on different conditions the same label when they represent the same region |

A **shared** ROI is converted through world coordinates, so the same outline lands
on the same physical region even when pixel sizes differ between datasets. That
only works when the specimens are in the same place. When they are not — separate
acquisitions, a fish in a different corner of each field — give each condition its
own ROI by setting **Applies to** to that condition. Nothing else is required:
every measured ROI becomes its own curve. **One ROI set per condition** does the
assignment for you, dealing the drawn ROIs across the conditions in order.

Backgrounds work the same way. A background assigned to a specific condition wins
over a shared one, so each sample is corrected against its own nearby background —
which is the point when illumination or exposure differed between acquisitions.

**Normalise** rescales each condition's pixels against *its own* background before
plotting and comparing:

- *None* — raw, offset-corrected intensities.
- *Subtract background* — puts every condition on a common zero.
- *Divide by background* — fold-over-background, which is what makes separate
  acquisitions comparable when gain or illumination differed.

The choice drives the histograms, the AUC and a **Normalised mean** column; the
raw mean/median/std columns are always left untouched so the underlying numbers
stay visible. A background at or near zero is left alone rather than producing
infinities. Optionally set a camera offset and a saturation level.

For 3D/4D data choose whether to measure the current slice or a maximum or mean
projection over Z. Projections hold every other axis at its slider position, so a
projection of a time series stays on the displayed timepoint. Measurements always
read the full-resolution pyramid level, never a preview.

**Create projection layers** flattens each condition's stack into a real 2D layer
using the mode selected above — `… [MIP]` for a maximum projection, `… [Mean Z]`
for a mean. This is the button to use before drawing ROIs on a projection: the
toolbar's *2D / 3D (MIP)* switches the canvas into a volume rendering, which looks
right but cannot be drawn on, because napari's shape tools only work in 2D.

The new layers keep the source's pixel size, colormap and contrast, and record
where they came from in `mv_source_layer`. The condition table is repointed at
them automatically and the mode resets to *Current slice*, so what you draw on is
exactly what gets measured. Running it again updates the existing layers instead
of piling up duplicates. Layers that are already 2D are left alone. The projection
runs on a worker thread — a 17-plane 2040×2040 stack off a network drive takes a
few seconds and the viewer stays responsive throughout.

**Statistics tab.** Per comparison label and condition: pixel count, mean, median,
std, integrated intensity, min and max; then background mean and std, mean minus
background, the normalised mean, signal/background ratio, and SNR as
`(mean_signal − mean_background) / std_background`. Ratios that would divide by
zero are reported as `—` rather than infinity. Saturated pixels are counted and
called out in a warning, since clipping invalidates every ratio on that row.
Exports to CSV.

**Histogram tab.** Every measured ROI-on-condition pair is its own curve, listed
with a checkbox and all ticked by default, so whatever you drew is compared side
by side without any further setup — a shared ROI gives one curve per condition, a
per-condition ROI gives one curve per sample, and mixtures work too. Curves are
density-normalised step histograms on shared bins, with a log-y toggle and
**All** / **None** buttons. For any two curves it reports the **ROC AUC** with its
Mann-Whitney U p-value and a histogram **overlap coefficient**. Both are
rank- or density-based — no normality is assumed, which matters because ROI
intensities are typically skewed and multi-modal. AUC is the probability that a
random pixel from the second condition exceeds one from the first, so 0.5 means
indistinguishable and either extreme means fully separated. Saves the figure to
PNG.

ROIs larger than 200 000 pixels are subsampled with a fixed seed for the
histogram and the AUC, so the numbers are reproducible and the panel stays
responsive; the statistics themselves always use every pixel. Measurement runs on
a `thread_worker`, so the viewer keeps redrawing while a large ROI is measured.

### Exports

**Export Snapshot** saves what is on the canvas at 2× oversampling, so it stays
sharp in PowerPoint. The scale bar is included. A `.tif` target writes lossless
pixels and tags the file with its on-screen resolution.

**Export Measurements** writes an `.xlsx` with two sheets:

- `Measurements` — one row per reported quantity: ROI name, type, value, unit,
  image, channel, timestamp, shape type, whether it was calibrated, the slice the
  ROI sits on, vertex count and source ROI layer.
- `Acquisition` — the acquisition settings of every open dataset, so a shared
  spreadsheet still records which settings produced the numbers.

### PowerPoint figure slides

**Export Slide** builds the comparison figure that usually gets assembled by hand:
one table, a row per open dataset, a column per channel, and a merge column.
Z-stacks are flattened by maximum projection, so the slide shows the whole
specimen rather than one plane.

The dialog has two tables. The first picks which datasets appear and what each
row is called — the row label defaults to the file's stem, because Imaris stores
the acquiring machine's own path (`D:\Transfer\2026-07-27\TEB-BIOTIN_4.ims`) as
the image name, which is not a figure label. The second names the channels.

**The channel label is the point of the dialog.** It becomes the column heading,
printed in that channel's own display colour, and it starts as whatever the
microscope called the channel — `Confocal - Yellow`. Change it to the staining or
the antibody (`anti-biotin`) before exporting, or edit it in PowerPoint
afterwards: the headings are ordinary text runs, not part of the images.

#### Batch mode

**Add folder…** reads every supported file in a directory straight into slide
rows. Nothing is added to the viewer — opening thirty datasets as layers would
mean a hundred-odd entries in the layer list and every pyramid held open, for
images nobody is going to look at interactively. **Add files…** does the same for
a hand-picked selection, and both can be used with datasets already open: the
viewer's layers and the batch appear in one table, deduplicated by file path.

More samples than *Rows per slide* (default 4) go onto further slides rather than
being squeezed onto one. Every slide keeps the same columns and the same row
height, including a final slide holding one leftover sample, so the deck reads as
one figure and slide titles are numbered `(3 of 8)`.

Channel labels and merge ticks survive a further folder being added, so typing
`anti-CD31` is not undone by loading more files.

Because batch files were never on screen, **Contrast** defaults to *Auto per
image* — the same 0.5–99.5 percentile stretch as the Auto Contrast button. Switch
it to *As displayed* to use each layer's or file's own recorded range instead.
Exports of what is already open default the other way round.

A folder takes minutes, so the export reports which sample it is on and **Cancel
becomes Stop**, which halts after the current sample and writes nothing.

The real acquisition folder, 29 Imaris datasets on the NAS: 8 slides in 94 s
(3.3 s per sample), 30 MB.

Other behaviour worth knowing:

- **Merge** is an additive composite, the same blending napari uses on screen, so
  it matches what you were looking at. Brightfield channels are left out by
  default — added to a fluorescence merge they wash every colour out to grey —
  and the *In merge* tick puts one back.
- Each image gets a **scale bar** at a round length (1/2/5 × 10ⁿ µm, aiming for a
  fifth of the width). It is drawn white on dark panels and black on bright ones,
  so it stays visible on brightfield. The length is written in the row label.
- Only as much resolution as the slide needs is read. The images are embedded at
  900 px on the long edge, so a 2040² dataset is read from its 1020² pyramid
  level — a quarter of the bytes. On the NAS that took a two-sample, four-channel
  export from **31.6 s to 9.1 s**. Raise *Image size* to force finer levels.
- Samples need not share a channel set: the columns are the union across
  datasets, and a sample missing one gets a dash rather than a shifted row. A
  brightfield-only dataset gets a dash in the merge column too — an empty cell
  reads as a panel that failed to render.
- Pictures are separate shapes positioned over the table cells, because
  PowerPoint has no concept of an image *inside* a cell. Everything stays
  editable — restyle the table, move or resize any panel.

#### Overview slide

An acquisition folder often holds a low-magnification **overview** of the whole
slide, written by Imaris as a numbered series of fields — `overview_F0000.ims`
through `overview_F0024.ims`. Left alone those become twenty-five near-identical
rows in the figure, all of the same slide at 2×.

They are detected instead, stitched into one picture, and put on a slide of their
own at the front of the deck, with a numbered red box where each sample was
imaged and a legend giving its stage coordinates in millimetres. The numbers are
the deck's row order, so marker 3 is the third row of the figure slides.

None of this is registration. Every `.ims` records the absolute stage extents it
was acquired at (`ExtMin0`/`ExtMax0` and so on), overview fields and z-stacks
alike, so each field is placed where the stage says it was and each sample's
footprint is drawn at its own coordinates. Nothing is correlated or guessed.

- Fields overlap by about a tenth of their width and are **averaged across the
  overlap** with a weight that fades towards each field's edge, so there is no
  grid of seams. Intensities are otherwise untouched: the contrast stretch is
  applied to the whole mosaic at once, so no field is brightened relative to its
  neighbour, and the faint per-field shading of the raw brightfield data is left
  as acquired rather than flat-fielded away.
- A sample imaged **past the edge of the overview** widens the canvas rather than
  being clamped onto the border, and its legend entry says `outside the
  overview`. The part no field reached is left white.
- Boxes and numbers are PowerPoint **shapes**, not pixels burned into the
  picture, so a marker sitting on top of the specimen can be dragged off it.
- A file is only taken for an overview field when it carries the `_F####` suffix
  (three digits or more), records a stage position, and is a **single plane** — a
  multi-position z-stack experiment, which Imaris numbers the same way, stays in
  the figure. The aborted-acquisition stubs Imaris leaves behind as `_F0` are
  neither, and are skipped as unreadable along with anything else that fails.
- Untick **Overview** in the dialog to get the plain deck; the fields stay out of
  the sample table either way.

The real folder — 25 overview fields on the NAS — stitches to a 1600 px mosaic at
20 µm/px in **1.2 s** with the files in the OS cache; a cold run is dominated by
pulling the ~9 MB of each field across the share. Overview fields are written
without a resolution pyramid, so there is no coarser level to read; where a field
does have one, only the level that covers its share of the mosaic is read.

Needs `python-pptx`; the export says so plainly if it is missing.

## Supported formats

| Format | Notes |
|---|---|
| **Imaris `.ims`** | primary format. Reads the resolution-level pyramid as a napari multiscale image, crops the chunk padding, and derives voxel size from the dataset extents. Channel colours and display ranges come from the file. |
| **TIFF / OME-TIFF** | calibration from OME-XML, then the ImageJ description block, then the baseline resolution tags. Sub-resolution series load as a pyramid. |
| **OME-Zarr / NGFF** | optional. Reads `multiscales`, the `axes` list and the `omero` rendering block directly, so no `ome-zarr` package is needed. A Zarr store is a *folder*, so open one by dropping it on the window or passing it on the command line — the file picker only lists files. |

XY, XYZ, XYT, XYZT and multichannel layouts are all handled: the channel axis is
split into separate layers, and singleton T/Z axes are dropped so 2D images do not
get useless sliders. Large stacks load lazily through dask — only the slice you are
looking at is read.

## Layout

```
launch_viewer.py            what the shortcut points at; puts the project on sys.path
install_shortcut.py         creates/removes the desktop shortcut
requirements.txt
microscopy_viewer/
  __main__.py               CLI entry point; opens argv paths, reports startup failures
  app.py                    assembles the viewer: docks, toolbar, shortcuts, file opening
  loaders/
    __init__.py             dispatch by format; batch loading that survives bad files
    layer_spec.py           the reader -> GUI contract
    ims.py                  Imaris
    tiff.py                 TIFF / ImageJ / OME-TIFF
    ome_zarr.py             OME-Zarr / NGFF
  metadata.py               metadata model and the vendor-key synonym matching
  measurements.py           Shapes -> calibrated distances and areas
  intensity.py              ROI statistics, AUC / overlap; no Qt, runs off-thread
  exports.py                snapshots and the Excel workbook
  slides.py                 channel/merge rendering and the PowerPoint slide; no Qt
  contrast.py               auto / reset contrast
  rendering.py              full-resolution 3D / MIP for multiscale layers
  gpu.py                    GPU limits and the 3D voxel budget
  volume_cache.py           local disk cache for volumes read from slow storage
  dragdrop.py               application-wide drop handling
  widgets/
    registry.py             the panel manifest
    toolbar.py
    metadata_widget.py
    measurements_widget.py
    intensity_comparison.py
    slide_dialog.py         pick samples, name the stainings, write the .pptx
  utils.py                  logging, unit conversion, geometry helpers
tests/
  make_sample_data.py       synthetic .ims / TIFF / OME-Zarr samples
  test_readers.py           reader, measurement and export checks (no display needed)
```

To add a format, write a module with `can_read(path)` and `read(path) -> list[LayerSpec]`
and register it in `_READERS` in [microscopy_viewer/loaders/__init__.py](microscopy_viewer/loaders/__init__.py).
Nothing else needs to change: metadata display, measurement calibration and exports
all work off `LayerSpec`.

### Adding a panel

Dock widgets are declared in the manifest in
[microscopy_viewer/widgets/registry.py](microscopy_viewer/widgets/registry.py) —
`app._build_docks()` walks it, so a new panel is one `register_panel` call:

```python
register_panel(PanelSpec(
    identifier="my_panel",
    title="My panel",
    factory=lambda app: MyWidget(app.viewer),
    area="right",
    attribute="my_widget",      # optional: exposes app.my_widget
    tabify_with="measurements", # optional: share a tab stack
    order=40,
))
```

Factories are called during window construction and import their widget inside the
function, which keeps a panel's dependencies off the startup path. A panel that
fails to build is logged and skipped rather than taking the window down, and its
`attribute` is left as `None`. Built panels are available as `app.panels[identifier]`
with their docks in `app.docks[identifier]`.

## Checks

```powershell
python tests/make_sample_data.py    # writes tests/sample_data/
python tests/test_readers.py        # readers, measurement maths, Excel export — no Qt needed
python tests/test_intensity.py      # ROI statistics, AUC / overlap, CSV export — no Qt needed
python tests/test_rendering.py      # GPU voxel budget and the volume cache — no Qt needed
python tests/test_slides.py         # channel/merge rendering and the .pptx table — no Qt needed
python tests/test_overview.py       # overview detection, stitching, the locator slide — no Qt needed
python tests/smoke_gui.py           # builds the real viewer: docks, ROIs, snapshots, exports
```

`smoke_gui.py` wants a working OpenGL context because it takes real canvas
screenshots. Without one — under `QT_QPA_PLATFORM=offscreen`, or over a Remote
Desktop session that has dropped the context — it skips the screenshot checks and
runs everything else.

The sample data is also the quickest way to try the GUI without real acquisitions:

```powershell
python launch_viewer.py tests/sample_data/sample_4d_2ch.ims
```

## Troubleshooting

The shortcut runs `pythonw.exe`, which has no console, so nothing is printed if
startup fails. Two things to check:

- A dialog appears on a startup failure with the full traceback under **Show Details**.
- Every run logs to `%LOCALAPPDATA%\MicroscopyViewer\microscopy_viewer.log`.

To watch it work with a console instead, reinstall the shortcut with
`python install_shortcut.py --console`, or run `python launch_viewer.py --verbose`.

If the shortcut opens the wrong environment, re-run `install_shortcut.py` from the
interpreter you want — the shortcut always points at the interpreter that created it.

### First launch is slow, or hangs before the window appears

napari JIT-compiles part of its colormap handling with numba, and caches the result
next to the installed package. If napari lives in a read-only prefix — a system-wide
conda install under `C:\ProgramData`, for example — that cache cannot be written and
importing `napari.utils.colormaps` stalls for minutes on *every* launch.

`microscopy_viewer/__init__.py` heads this off by pointing `NUMBA_CACHE_DIR` at
`%LOCALAPPDATA%\MicroscopyViewer\numba_cache` before napari is imported, so the
compile happens once. If you already set `NUMBA_CACHE_DIR` yourself, that value is
left alone — make sure it points somewhere writable.

### napari's modal plugin warning

If any installed plugin still uses napari's deprecated plugin engine, napari opens a
modal *Installed Plugin Warning* while the window is being built and waits for a
click. Started from a shortcut that is fatal: the window appears but nothing loads
until someone finds and dismisses the box.

The viewer marks the plugins that are already installed as "already warned about" —
the same thing napari's own *Only warn me about newly installed plugins* checkbox
does — so a plugin installed later still raises the warning. Pass
`--warn-shimmed-plugins` to `launch_viewer.py` to leave napari's behaviour alone.

### Startup over Remote Desktop

Creating the window itself is the slow part in an RDP session, where OpenGL falls
back to software rendering: measured here at roughly three minutes from
double-click to a usable window, against about four seconds for everything else
(imports, viewer construction, reading the files). Locally it is a few seconds.
Image loading is unaffected — that is dask-backed and lazy either way.
