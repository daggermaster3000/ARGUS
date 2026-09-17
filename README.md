# ARGUS — Microscopy Viewer

[![tests](https://github.com/daggermaster3000/ARGUS/actions/workflows/tests.yml/badge.svg)](https://github.com/daggermaster3000/ARGUS/actions/workflows/tests.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

A customised [napari](https://napari.org) viewer for rapid microscopy image inspection.
It opens Imaris `.ims`, TIFF, OME-TIFF and OME-Zarr datasets, shows the acquisition
metadata alongside the image, and lets you measure and annotate in calibrated
micrometres — then export a snapshot for slides, a time lapse as a movie, and the
measurements as a spreadsheet.

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
| Export Movie | `Ctrl+Shift+M` | write the time series as a `.mov` for PowerPoint, or `.mp4` / `.gif` |
| Play / Pause | `Ctrl+Space` | play the time series at the rate set in the **Time series** panel |
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
thread so the viewer stays usable. Everything afterwards reads from local disk,
[time-series playback](#time-series) included — the two share one entry per
pyramid level rather than keeping a copy each.

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

### Time series

A time lapse gets its own dock under the canvas — **Time series** — with the
transport controls, the range being played, and a line saying where the frames
are being read from.

| Control | Does |
|---|---|
| Transport buttons | first timepoint, back one, play / pause, forward one, last timepoint |
| **fps** | playback rate, 0.5 to 60 |
| **Loop** | wrap at the end of the range instead of stopping |
| **Play from … to …** | the range that plays, and the range an export defaults to |
| **Cache locally** | start the copy described below without waiting for playback |
| **Export movie…** | the dialog described further down |

**The rate is kept, not the frame count.** Playing at 20 fps means twenty
timepoints a second whether or not every frame was ready in time: a tick that
arrives late advances by however many frame periods actually elapsed, so a slow
read drops a frame rather than stretching the whole clip out. napari's own
slider does the opposite, which is why a time lapse played from a network share
runs in slow motion there. Frames are never queued up behind a slow read either
— while a slice is still loading the clock keeps counting but nothing new is
requested, so the backlog cannot grow.

#### Why playback stutters, and what is done about it

The pixels are not slow to draw; they are slow to arrive. A timepoint is a
separate read, and the source is usually a NAS, so the frame rate is really the
round-trip rate of the share.

So every pyramid level of a time series that fits the budget is copied once to
a local `.npy` and memory-mapped, through the same cache the 3D view uses.
Unlike the 3D path the levels are replaced *in place*: same shapes, same voxel
size, same number of levels, so multiscale rendering carries on exactly as
before and nothing about how the image looks changes. Only where the bytes come
from does.

**The copy starts when you press play**, on a worker thread, with the status bar
saying what is being copied — not when the file is opened. Opening is when the
viewer is busiest, reading the first slice and building the first volume, and a
background thread pulling a whole time lapse over the same share at that moment
makes everything else slower, 3D included. Opening a file still *adopts* any
copy that already exists, which costs nothing. **Cache locally** in the panel
forces it at any time.

The entry is shared with the 3D view — same key, same file — so a dataset that
has been rotated in 3D is already local when you play it, and one that has been
played is already local when you switch to 3D. Nothing is copied twice.

A second thread walks ahead of the playhead, touching one byte per memory page
of the timepoints about to be shown, so they are resident in the operating
system's page cache before napari asks for them. It does one pass per frame and
then sleeps — it never reads on a timer of its own — and it does nothing at all
for arrays that are not already local, or while the viewer is in 3D, where
reading ahead would evict the pages the volume on screen was built from.

The budget is **8 GB per layer** (`MICROSCOPY_VIEWER_TIMELINE_BUDGET`, in
bytes). Levels are taken finest first, and one that does not fit is skipped
while the coarser ones are still cached — which is the useful outcome for a
series too big to hold: playback zoomed out, which is how a time lapse is
watched, comes off local disk, while a zoomed-in view still reads its small crop
from the source. Everything else about the cache — the 32 GB total cap, the LRU
pruning, the fingerprinting that stops a changed file resolving to a stale copy,
and `MICROSCOPY_VIEWER_NO_CACHE=1` to switch it all off — is as described under
[Local volume cache](#local-volume-cache).

Note the budget is *per layer*, and each channel is its own layer, so a
four-channel time lapse can ask for four times it. The total cache cap still
applies, and an entry a layer currently has open is skipped by the pruner rather
than pulled out from under it.

#### Movies for PowerPoint

**Export Movie** writes the range being played as a video. The frames are canvas
screenshots — the same thing **Export Snapshot** saves — so the file shows what
was on screen: the contrast you set, the layers you had visible, the ROIs, the
scale bar, and a 3D view if that is what you were looking at. Nothing is
re-rendered from the data with settings of its own.

| Setting | Notes |
|---|---|
| **File** | `.mov` (default), `.mp4` or `.gif` |
| **Timepoints … to …**, **Every nth** | inclusive range; a stride shortens a long series |
| **Frames/s** | playback rate of the file, independent of the panel's own rate |
| **Resolution** | canvas oversampling: 1×, 2× (sharp in PowerPoint) or 3× |
| **Quality** | encoder quality 0–10; 8 is enough for a slide |
| **Burn in the time** | draws the elapsed acquisition time into the corner, or the timepoint number when the file records no interval |

`.mov` and `.mp4` are both H.264 in `yuv420p`, which is what PowerPoint,
QuickTime and browsers can all decode. The two differ only in the container:
`.mov` is what was asked for here and plays inside a slide, `.mp4` is the safer
choice on an old PowerPoint. `.gif` needs no encoder at all and loops forever,
which suits a chat window better than a deck.

Frames are cropped to an even width and height, because H.264 cannot encode an
odd-sized 4:2:0 frame. Cropping loses a row of pixels; the alternative — letting
the encoder rescale the frame, which is what imageio does by default — would
resample the image and leave the burnt-in scale bar wrong.

Each frame is waited on before it is grabbed. napari loads slices on a worker
thread, so a screenshot taken the moment the slider moves shows the *previous*
timepoint, and a movie made that way is off by one frame throughout. **Cancel**
becomes **Stop** during a render, and a stopped or failed export deletes its
half-written file rather than leaving one to be dropped into a slide.

Writing `.mov` and `.mp4` needs `imageio-ffmpeg`, which ships its own ffmpeg
binary and installs with everything else. Without it playback and GIF export
still work and the dialog says what is missing:

```bash
python -m pip install imageio-ffmpeg
```

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

### Atlas registration

**Atlas registration** fits an open brain onto a reference atlas — Z-Brain, ZBB or
anything else that ships a reference volume — and reads the signal channel out per
anatomical region.

**One channel drives the fit, and it is the nuclear one.** DAPI is the only
channel with signal across the whole brain, which is what a global affine plus a
deformable pass needs. The others are assigned around it:

| Role | Channel | What happens to it |
|---|---|---|
| **Registration driver** | DAPI | Fitted against the atlas reference. The only channel the optimiser ever sees. |
| **Landmark / QC** | acetylated tubulin | Resampled and shown for eyeballing tracts. Contributes to the metric only if you tick it on. |
| **Carry-along** | anti-SV2 | Resampled through the driver's transform. Never fitted. |

That asymmetry is the point. A regional channel like anti-SV2 handed to the
optimiser produces a beautiful alignment of its own domains onto whatever the
atlas happens to have nearby — confident, and wrong. The landmark option is off
by default for a milder version of the same problem: a tract-rich channel pulls
the warp onto tracts and lets the space between them drift.

Roles are guessed from the channel names the reader already recorded, so a file
whose channels are called `dapi`, `Actub` and `sv2` arrives assigned correctly.
Every guess is a combo box. An unrecognised channel defaults to carry-along —
never to driving the fit.

**Atlas** — *Detect from a folder…* scans a download and picks the reference, the
region masks and the name list. A **nuclear reference is preferred**, since DAPI
against a nuclear channel is the same-modality case; falling back to tERK is
allowed, but it is reported on the panel and recorded in the result, because that
fit is the one to check before believing its output. All three paths can be set
by hand.

Region masks are read in all three forms atlases ship them: an integer label
volume, one binary mask per HDF5 dataset, and Z-Brain's `MaskDatabase.mat` — a
MATLAB v7.3 sparse logical matrix, one column per region. Those regions *overlap*
(a voxel is in a subdivision *and* in a neuropil inside it), so they are never
collapsed into a single volume. Nothing is held across regions: one Z-Brain mask
is 120 million voxels, and the sparse form is measured through its own indices
without building a mask at all.

#### Working with the Z-Brain download

The three files do not agree with each other, and two of them are named
misleadingly:

| File | What it is | Axis order |
|---|---|---|
| `AnatomyLabelDatabase.hdf5` | 29 averaged **anatomy stacks**, not labels. One of them, `Elavl3-H2BRFP_6dpf_MeanImageOf10Fish`, is the **nuclear reference you want for DAPI**. | `z, x, y` |
| `MaskDatabase.mat` | The actual 294 **region masks**, sparse. | `y, x, z` |
| `Ref20131120pt14pl2.nrrd` | The tERK reference — the cross-modality fallback. | `x, y, z` |

*Detect from a folder…* handles this: it looks **inside** multi-volume files, so
it finds the H2B nuclear stack rather than settling for tERK, and it prefers
`MaskDatabase.mat` over the anatomy database despite the latter's name. When the
reference file holds several volumes, a **Reference volume** dropdown appears for
picking a different one.

Axis order is reconciled on the way in — everything works in `(z, y, x)`. Where a
file gives no clue (a bare HDF5 of stacks), the reference is transposed onto the
grid the masks declare, and the panel says it did so. This matters more than it
sounds: a transposed reference does not raise anything, it just asks the
optimiser to find a 90° rotation, which it will not — it converges somewhere
confident and wrong.

Pointing **Region masks** at `AnatomyLabelDatabase.hdf5` is refused with a message
saying where the masks actually are: measured as masks, those intensity stacks
put every voxel inside every region.

Other behaviour worth knowing:

- Registration runs in **physical space**, using the calibrated voxel size the
  reader already put on each layer. An anisotropic stack and an atlas at a
  different resolution line up without resampling anything by hand.
- **Affine only** is a fast sanity pass. Run it first: a deformable registration
  cannot undo a mirrored or upside-down stack — it converges to a confident wrong
  answer — so check the overlay, and use **Flip axes** if handedness is wrong.
- The fit is done at or below **Fit at most** voxels; the transform is smooth, so
  it is found on a decimated volume and applied at full atlas resolution.
- The signal is warped **into atlas space** and scored against the atlas's own
  masks, so no region boundary is ever interpolated. The inverse transform is
  kept too, for bringing masks back onto the original stack.
- The run is on a `thread_worker`, so the window stays usable.
- The region table exports through the same writer as the measurements workbook.

Registration needs `antspyx`, which is a large, platform-fussy wheel and so is an
optional extra rather than a dependency:

```bash
python -m pip install ".[registration]"
```

Without it the panel still appears and says exactly that, the way the slide export
reports a missing `python-pptx`. `Backend` is a three-method interface, so
`itk-elastix` can be added beside ANTs — relevant on Apple Silicon, where antspyx
may need a source build.

Measured on a real 7 dpf larva (134 × 2040 × 2040, 558 Mvoxels, 1.29 × 0.61 ×
0.61 µm) registered onto a 54 × 510 × 510 nuclear reference: **43 s affine, 63 s
with the deformable pass**, both including resampling two channels at atlas
resolution.

One thing the software cannot do for you: an atlas is a *brain*, so a field
holding a whole larva has to be cropped to the head, or the fit will spend itself
matching yolk and trunk to a brain-shaped reference.

### Cellpose segmentation

**Segmentation** labels cells or nuclei with [Cellpose](https://cellpose.readthedocs.io)
and measures every object it finds, on the GPU when there is one.

Two things are handled that a bare `model.eval` call is not:

**The device is decided and shown before the run, not after.** Cellpose falls back
to the CPU without complaining, and on a 3D stack that is tens of seconds against
tens of minutes. The panel names the card it will use — `NVIDIA GeForce RTX 4090 —
24.6 GB, CUDA` — and when it says `CPU`, the tooltip says why that usually happens
(the default torch wheel is CPU-only). The batch size is sized from free video
memory rather than left at Cellpose's default of 8, which leaves a big card idle.

**Two stains at once, when the biology needs them.** A nuclear stain segments
reliably but only ever gives you nuclei. Pointing **Segment** at a membrane or
cytoplasmic channel and **Nuclei** at DAPI hands Cellpose both — cell channel
first, nuclear channel second, which is the pairing the two-channel models were
trained on — and what comes back is whole cells, with touching cells separated and
one mask per nucleus. On a kidney-organoid well here, a weak cytoplasmic channel
alone found 55 objects; the same channel with its DAPI found 626. Both stains are
decimated together, so a large volume stays paired.

**Noise can be filtered out before Cellpose sees it.** **Median filter** smooths
each channel with a square window of 2r+1 px before handing it over — shot noise
and hot pixels go, edges stay where they are, which is what a Gaussian would not
manage. Only the network's input is filtered: the masks come back on the original
grid and per-object intensities are still read from the raw channel. It is not
free, so it is off by default; the Batch segmentation panel estimates what it
will add to a plate run before the run starts.

**Round false positives can be thrown out by shape.** Cellpose labels debris,
beads and out-of-focus blobs along with the cells, and what those have in common
is that they come back as clean convex discs while a real nucleus packed against
its neighbours is dented by them. **Max solidity** measures every object with
scikit-image's `regionprops` and drops the ones above the cut — solidity being the
object's area over the area of its convex hull, 1.0 for anything convex. Set it to
`1.000` to measure without dropping anything: the **Solidity**, **Circularity**,
**Eccentricity** and **Extent** columns fill in on the Objects tab, and the cut can
be read off them rather than guessed. Objects that survive keep their original
label numbers, so a label in the table is still the label in the image.

Solidity measures *convexity*, which is not quite the same as roundness — a
convex ellipse scores as highly as a disc. On one kidney-organoid well here the
nuclei ran to a median solidity of 0.959 and a maximum of 0.991, and a cut at
0.985 removed 3 objects whose circularity was 1.00, while a cut at 0.98 removed 76
with a median eccentricity of 0.61 — elongated, not round. Start near the top of
the distribution, and use the circularity column to check what is being removed.

**Sizes are in µm, not pixels.** Cellpose's `diameter` is in XY pixels and its
`anisotropy` is the Z/XY voxel ratio; both are derived from the calibrated voxel
size the reader already put on the layer. A 5 µm nucleus stays 5 µm whether the
stack is 0.13 µm/px or 0.6 µm/px, and it stays right after a large volume has been
decimated — the conversion happens after the decimation, not before.

| Setting | What it does |
|---|---|
| **Segment** | The channel that is labelled. A nuclear stain with the `nuclei` model is the reliable case. |
| **Nuclei** | Optional second stain handed to Cellpose alongside the segmented one, which is how you get whole cells rather than nuclei: segment the membrane or cytoplasmic channel and point this at DAPI. The nuclei separate cells that touch and guarantee one mask per nucleus. Leave it at `— none —` for a single-channel run. |
| **Measure** | The channel intensities are read from. Segment on DAPI, measure on the reporter, and every row is signal per nucleus. |
| **Mode** | `2D + stitch` segments each plane and joins overlapping masks between planes — faster, and usually better on an anisotropic stack where a nucleus is four planes tall. `3D` computes flows in 3D. `2D per plane` leaves labels unconnected between planes. |
| **Diameter** | Expected object diameter in µm. The setting that matters most; automatic is worth overriding. |
| **Median filter** | Radius in pixels of a median filter applied to the channels before segmentation. `off` by default. Removes shot noise and hot pixels without moving edges. Costs time per image — see the estimate in the Batch panel. |
| **Max solidity** | Objects rounder than this are dropped after the run: `0.985` throws out the convex discs that debris and beads produce and leaves the cells. `off` skips the measurement entirely; `1.000` measures every object and drops none, which is how you choose the cut. Needs scikit-image, which arrives with cellpose. |
| **Segment at most** | Volumes above this are decimated **laterally** before segmentation — Z is left alone, since that is where objects are already only a few planes tall. The labels always come back on the original grid. |

**The model list is discovered, not hard-coded.** It holds three kinds of entry,
and the tooltip says which is which:

- what the installed cellpose ships — Cellpose 4.2 has `cpsam`, `cpsam_v2` and the
  two `cpdino` models; Cellpose 3 has the `nuclei` / `cyto3` zoo and a size model
  that estimates diameters for you;
- models you trained in the Cellpose GUI, listed by name and loaded by their path;
- loose weights in `~/.cellpose/models`.

The ↻ button rescans, so a model trained mid-session shows up without a restart,
and **Custom model** takes any file directly. Weights the installed version cannot
load are filtered out rather than offered — Cellpose 4 is one architecture and
cannot read a v3 zoo file, which otherwise fails minutes into a run with a tensor
shape mismatch. The filter matches whole zoo names, so a model of your own called
`nuclei_finetuned` still appears. The starting selection prefers a model already
downloaded, so the first Segment does not silently fetch a gigabyte of weights.

What comes back is a Labels layer at the source layer's own scale, and a table with
one row per object: voxel count, volume in µm³, equivalent diameter, centroid in µm,
the four shape descriptors when they were measured, and mean / median / std / max /
integrated intensity from the measure channel. It
exports through the same writer as the measurements workbook. Objects are measured
through the indices of the labelled voxels rather than a loop over labels, so the
memory it takes scales with the segmented fraction of the volume, not the volume.

The run is on a `thread_worker`, so the window stays usable while it goes.

Segmentation needs `cellpose`, which pulls in torch, so it is an optional extra
rather than a dependency:

```bash
# GPU: install a CUDA build of torch first, from https://pytorch.org
python -m pip install ".[segmentation]"
```

Without it the panel still appears and says exactly that, the way the atlas panel
reports a missing `antspyx`. `Backend` is a three-method interface here too, so
StarDist or micro-SAM can be added beside Cellpose.

### File explorer

A converted 4i plate is a few hundred images in one folder. Dropping it on the
window assembles a mosaic of every cycle — correct, and far more than anyone asked
for when the question is *what does well G/07 look like, and did the run pick up
its nuclei?* **File explorer** is how you ask that question.

Point it at the `.zarr` store — or press **Use the loaded plate** to take the path
off a layer already open — and press **Scan the plate**. Reading is metadata only,
so a store of a few hundred gigabytes lists in under a second.

```
Scan  →  46 wells × 7 acquisitions = 320 images
Well G/07, cycle 1 │ 4 channels │ 4 × 1 × 12000 × 12000 │ nuclei, DAPITEST, dapi-test-2
```

Tick wells and cycles, and every image they name appears in the table with **the
segmentations already sitting inside it**. That last column is the point: it is
how you see which wells a run reached without opening any of them.

**The miniature is free.** Each image carries its own pyramid, so the preview is
read from the bottom of it — a 12 000 × 12 000 channel previews from its 750 × 750
level, a few hundred kilobytes rather than 288 MB. A Z stack is projected at
maximum, which is what makes an organoid visible in one plane. Switch channels to
check the stain you care about before opening anything.

**The tables find their images.** A run names each object table after the image it
measured — `B_02_0.csv` is well `B/02`, image `0`, which is cycle 1 — so the
**Tables** column says which images have been *measured* as well as which have
been segmented, and the tables for the selected image are listed underneath ready
to open:

```
B/02 │ cycle 1 │ nuclei, dapi-test-2, analysis-1 │ …_analysis-1_objects
B/02 │ cycle 2 │ —                               │ —
B/02 │ cycle 3 │ nuclei                          │ —
7 image(s), 2 already segmented, 1 with an object table.
```

**Open in Measurement analysis** hands the selected table to that panel, which
colours this image's labels by any column in it. A table written for *another*
cycle of the same well is offered too, marked `(from B/02/0)` — a 4i plate images
the same cells every cycle, so it does describe these objects, but it was measured
somewhere else and the segmentation it refers to lives in the image it was run on.
It is never pre-selected for that reason.

The list is indexed once per scan, and refreshed when a batch run finishes.

**Open in the viewer** loads the selected rows, with **with segmentations** adding
every NGFF label set stored inside each image, on the same grid and already
aligned. Seven layers come back for one 4-channel well that has been segmented
three times. The layers are named `G/07 :: cycle 1 :: Ab1_DAPI` — the same names
opening the well folder gives them, so a table written by a batch run still finds
its layer in the Measurement analysis panel.

**This is the only place a plate is chosen.** Scanning here fills in the Batch
segmentation panel's well and channel lists, and lists the table folders beside the
store in the Measurement analysis panel. One store box for the window rather than
one per panel: two paths to keep in step is two lists that can disagree about
what is in the plate.

### Batch segmentation

**Batch segmentation** takes the settings you just got right on one image and runs
them over a whole plate. A Fractal-converted 4i plate here is 46 wells × 7
acquisitions = 320 images of 12 000 × 12 000 px; nobody is going to click through
that one at a time.

**The plate comes from the File explorer.** Scan it there and this panel fills
itself in — the well list, the acquisition list and the channels.

**Channels are matched, not indexed.** In a 4i plate the channel label changes
every cycle — `Ab1_DAPI`, `Ab2_DAPI`, … `Ab7_DAPI` — while `wavelength_id` does
not. The picker matches on the key that survives the plate, so choosing DAPI in
cycle 1 finds the right channel in cycle 7. A label match is a case-insensitive
substring, with an exact hit preferred, for plates that carry no wavelength ids.

**Masks are written back into the plate**, as NGFF `labels` groups:

```
AssayPlate_….zarr/B/02/0/
  0 … 4                 the channels, never opened for writing
  tables/               untouched
  labels/
    nuclei/
      0 … 4             the masks, uint32, on the image's own pyramid
```

That is the layout Fractal and `ome-zarr-py` expect, so the result is readable by
anything that reads the plate — including this viewer, which loads a label set
beside the channels it came from as a proper Labels layer at the same scale. The
label pyramid mirrors the image's own levels and is built by subsampling on a
stride, not averaging: a label map has no meaningful mean.

**The masks come back with the plate.** Opening the store assembles each label set
into a mosaic of its own on the same grid as the channels, so a segmented plate
opens with its segmentation on top of it — wells that have not been run yet are
blank rather than missing, which is what tells you where the run got to. The same
holds for a single well, and **Open the selected result** under the results table
does it for one image.

**An interrupted run is resumable.** An image that already has the named label set
is skipped rather than redone, and the well list says which those are, so a plate
that stopped overnight is restarted by pressing Run again. **Replace a label set
that is already there** is the override. One image failing — out of video memory,
a missing channel — is reported in the table as that image's outcome and the run
carries on; a plate of three hundred is not lost to one bad well.

By default only the *first* acquisition is selected. Every 4i cycle images the same
cells, so segmenting all seven gives you the same nuclei seven times and takes
seven times as long.

| Setting | What it does |
|---|---|
| **Wells** / **Acquisitions** | What to run. Everything, one row, one cycle — ctrl-click and shift-click. Wells that already carry the label set are marked. |
| **Segment** / **Nuclei** / **Measure** | As in the Segmentation panel, but resolved per image by wavelength or label. **Nuclei** gives whole cells instead of nuclei. |
| **every channel, in columns of its own** | Measure all the stains of each image rather than one, adding `Mean intensity (Green488-bCAT)` and its four companions per channel. One segmentation, four answers. |
| **Median filter** | The same setting as the Segmentation panel's, shown here because a plate run is where it costs real time; the two are kept in step. The line under it estimates what it adds to the selected images. |
| **Label set** | Name written under `labels/`. Give a second run a different name to keep both. |
| **Pyramid level** | Which level to segment. 0 is full resolution; each step up halves the image and quarters the time. The panel shows the resulting extent and µm/px. |
| **and an .h5ad beside each one** | Also write each table as AnnData for squidpy. See [Out to squidpy](#out-to-squidpy-scanpy-and-the-rest). |
| **as one file for the whole run** | One `.h5ad` for the run rather than one per image, with the well and cycle in `obs["image"]`. The CSVs are written either way. |
| **Tables** | A per-image object table and a plate-level summary CSV — one row per image with counts, median size, median solidity, how many the shape filter dropped, and what went wrong. |

**The intensity columns say which channel they came from.** `Mean intensity` is
the segmented channel unless **Measure** names another; every further channel gets
its own five columns named after it, and the run summary records both. On a 4i
plate, where the same stain is called `Ab1_DAPI` in cycle 1 and `Ab7_DAPI` in cycle
7, a column called nothing but "Mean intensity" is a column nobody can check
afterwards.

If **Measure** names a channel an image does not have, the intensities fall back to
the segmented channel — as they always did — but the image's note now says so
rather than putting one channel's numbers silently under another's heading. A
channel on a different grid from the labels stops the measurement instead of being
read through the wrong voxels.

Measuring every channel costs one extra read per channel: on a 12 000 × 12 000 well
with 18 000 objects, four channels measure in about 9 s.

> **Equivalent diameter changed in this version.** A plate mask is `(1, Y, X)` —
> three axes, one plane deep — and the diameter was being computed with the sphere
> formula, which spends a third of the volume on a Z extent the object does not
> have. A nucleus 20 µm across was reported as 8 µm. Objects are now measured as
> discs when the image is one plane deep and as spheres when it is a real volume;
> tables written before this understate diameter by about 2.4× on plate data. The
> voxel counts, volumes and intensities in those tables were always right.

Cellpose settings — model, diameter, mode, thresholds, the solidity filter, GPU —
are read from the **Segmentation** panel when the run starts, and shown here as one
line. There is no
second copy to keep in step: get one image right there, then run the plate here.

The run is on a `thread_worker` that yields one image at a time, so the window
stays usable, the table fills as it goes, and **Stop** takes effect at the next
image boundary with everything already finished safely on disk. On an RTX 4090 a
12 000 × 12 000 well takes about 35 s including the read and the write, so a
46-well cycle is roughly half an hour.

### Measurement analysis

A segmentation writes one row per object and then the numbers and the image go
their separate ways — the table into Excel, the masks into the viewer, with
nothing tying a row to the object it came from. **Measurement analysis** is the
way back.

Open any per-object table — the CSV a batch run writes, the workbook the
Segmentation panel exports, or a table from elsewhere — and the panel

- **shows it**, all of it. 18 000 rows is a normal well and opens instantly: the
  view reads the cells it is about to draw rather than building a widget per cell.
- **sorts by any column.** Click a header, click again to reverse it. The first
  row is then the largest, the roundest or the brightest object in the well — and
  one click from being on screen. Blanks sort last either way, so a descending
  sort really does start at the maximum rather than at the objects scikit-image
  could not fit a hull to.
- **colours the labels by any column.** Pick *Mean intensity* and the
  segmentation is redrawn as that measurement; pick *Solidity* and the round
  false positives stand out as one end of the scale. Objects with no row in the
  table are left transparent rather than painted the colour of zero, so a
  partially measured plate shows you exactly what was measured.
- **plots two columns against each other**, with the points carrying the same
  colours as the labels, so a cluster in the plot and a region in the image are
  recognisably the same thing.
- **finds the layer itself.** A batch table is named after the image it came
  from, so `G_07_0.csv` picks out `G/07 :: cycle 1 :: nuclei` in the layer list.
  Change it in the combo when the guess is wrong.
- **finds the tables themselves.** Scan a plate in the **File explorer** and the
  folders of tables sitting beside the store are listed under **Beside the
  plate**, while the explorer's own list offers the tables for one image — one per label set a run has written — with their contents in the combo
  next to it. Nothing is opened until you pick one; which of five segmentations
  you meant is not something to guess at.

```
AssayPlate_….zarr                       ← scanned in the File explorer
AssayPlate_…_nuclei_objects/            ← listed as "nuclei_objects"
AssayPlate_…_DAPITEST_objects/          ← listed as "DAPITEST_objects"
```

**Clicking a row goes to that object.** The label is selected in the viewer, the
camera centres on its centroid and zooms until it fills about a quarter of the
canvas — not the whole canvas, because an object with nothing around it is an
object you cannot place, and the neighbours are usually why you looked. On a
stack the Z slider steps to the plane the object is in, which the camera alone
would not do.

The centroids in the table are micrometres and a calibrated layer's world
coordinates are micrometres, so the centroid *is* the camera position: no
conversion, and it stays right when the viewer is showing a coarser pyramid
level. Untick **go to the object** to select the label without moving the camera.

Sorting and going are the pair that make the table usable: sort by solidity
descending, click the top row, and you are looking at the roundest object in the
well — the one most likely to be a bubble rather than a nucleus.

#### What the numbers are normalised against

**Nothing in an object table is normalised.** A run measures the pixels as they
were acquired and writes down raw grey levels. Cellpose does normalise its input
per channel, but that affects only where it draws the outlines — never a reported
number.

The one place scaling happens is the colour scale: where a value sits between a
low and a high. **What that low and high are computed over is a choice**, and it
changes the picture completely:

| Normalise over | What it is good for |
|---|---|
| **this image** | the structure *inside* one well. Every well uses its own range, so wells are not comparable — 40 % of the scale in one is not 40 % in another. |
| **this well, every cycle** | comparing the staining rounds of one well. |
| **this cycle, every well** | how a plate is normally read. One staining round shares a scale, so a well that is genuinely brighter looks brighter. |
| **the whole plate** | cycles pooled. Only when the cycles are the same stain — on a 4i plate they are not, and pooling puts a bright cycle's range on a dim cycle's objects. |

The scale line under the control says which was taken, how many images it covered,
and — when the scope is wider than one image — what this image alone would have
given, because the gap between those two numbers is the whole point:

```
343.4 … 3483 over this cycle, every well — 44 images, 557,915 objects
                          ·  this image alone would be 375.7 … 6176
```

That is well B/02 of the sample plate. Its own top is nearly double the plate's,
so painted against itself it looks like an ordinary well; painted against the
plate it is plainly one of the bright ones. The choice was per-image before this
and there was no way to ask the other question.

A wider scope reads one column from every table in the folder chosen under
**Beside the plate** — about a second for a 44-well plate, cached afterwards — and
pools the objects rather than averaging per-image percentiles, so a well with
sixteen objects does not count as much as one with forty-six thousand. The scope
is recorded on the layer alongside the range, so a figure made from it can say
what its colours mean.

**The dashboard scales separately**, and differently per tab: the spatial tabs
z-score within the one image they are showing, and the plate tab z-scores across
the whole sampled plate. That is the right thing for clustering — it is what stops
a PCA becoming a PCA of whichever column has the largest units — but it is not the
same question as the colour scale above.

**The colour scale is clipped to 1–99 % by default**, and the values at both ends
are shown. This matters more than it sounds: an object table always has a handful
of enormous outliers — two nuclei segmented as one — and stretching the scale to
the true maximum leaves every real object the same dark blue. Widen it to 0–100 %
to see the raw range.

**Selecting a region of the plot selects those objects in the image.** Set
**select** to `rectangle` and drag a band across one measurement, or to `lasso`
and draw round a cluster: the points you enclosed keep their colour, everything
else fades to 12 %, and the labels in the viewer fade with them. A group that is
only visible as a cluster in the numbers becomes a group you can see the position
of in the well — which nuclei they are, whether they are at an edge, whether they
are all the same organoid.

Faded rather than hidden, because the question being asked is *where* the cluster
is, and that needs the rest of the field faintly there to place it against. The
selected objects keep the colour their measurement gave them rather than turning
some highlight colour, so they stay readable as values. The status line reports
how many were caught and their distribution; **Clear** puts everything back.

Clicking a row selects that label in the viewer, which is how a suspicious number
gets looked at rather than argued about. **Reset** puts the ordinary random label
colours back.

| Setting | What it does |
|---|---|
| **File** | The table. CSV, TSV or Excel; a semicolon-separated CSV from a European Excel is detected rather than read as one column. **Reload** re-reads it after a run has rewritten it. |
| **Labels layer** | Which segmentation the table describes. Guessed from the file name. |
| **Colour by** | The column the colours come from. Numeric columns only; text columns are not offered. |
| **Colormap** | Perceptually uniform maps first — a measurement painted in a map with false edges is a measurement misread. |
| **Percentiles** | Where the colour scale starts and stops. |
| **select** | `rectangle` or `lasso` to pick objects out of the plot; `off` leaves the drag to pan the axes. |
| **x** / **y** | The scatter axes. Above 100 000 points the plot draws a random sample, and says so — random rather than the first N, because a table is written in label order and the first N would be one corner of the well. |

#### Out to squidpy, scanpy and the rest

**Export as AnnData…** writes the table as `.h5ad`, which is what the single-cell
stack reads:

| Where | What |
|---|---|
| `X` | the measurements, one row per object, one column per feature |
| `var_names` | the column names — `Solidity`, `Mean intensity (Red568-pSTAT)`, … |
| `obs` | the label id, the centroids, and any text column, indexed by label |
| `obsm["spatial"]` | the centroid as `(x, y)` in µm — or `(x, y, z)` for a real volume |
| `uns["microscopy_viewer"]` | which file and which image it came from, and the units |

The label is deliberately **not** a feature. It is an identifier, and clustering
on it would be clustering on the order Cellpose happened to number things in. The
centroids are not features either, for the same reason in reverse: leave them in
`X` and every clustering is partly a clustering on position, which is exactly what
the spatial analysis is supposed to discover rather than assume.

`obsm["spatial"]` is what `squidpy.gr.spatial_neighbors` builds its graph from, so
a well is three lines from a neighbourhood enrichment:

```python
import anndata as ad, squidpy as sq

adata = ad.read_h5ad("B_02_0.h5ad")      # 2971 objects x 27 features
sq.gr.spatial_neighbors(adata)            # coordinates already in micrometres
sq.gr.nhood_enrichment(adata, cluster_key="...")
```

**Or one file for the whole plate.** A plate is one experiment, and a folder of
forty-four files is forty-four files to concatenate before anything can be asked
about the plate as a whole. Two ways to get one:

- **…the whole folder as one** in this panel combines every table in the folder
  chosen under **Beside the plate**;
- **as one file for the whole run** in the Batch segmentation panel writes it as
  the run finishes, instead of one per image.

The well and cycle go in `obs["image"]` and are appended to the object names —
`100-G/09/0` — because label 100 exists in every well and concatenating without
that would give forty-four objects the same name. Tables are joined **outer**:
a 4i plate names its stains differently in every cycle, so two images can measure
different columns, and the inner join that is usual for single-cell data would
silently drop every column they did not share. A blank means "this image did not
measure that", which is the truth and is visible.

```python
a = ad.read_h5ad("…_analysis-1_objects.h5ad")    # 557 915 x 27, 44 images
well = a[a.obs["image"] == "G/09/0"]              # 46 394 objects
```

The Batch segmentation panel can write these as it goes — tick **and an .h5ad
beside each one** — built from the measured numbers rather than by reading the
CSV back, because a float that has been through a text file is not the float that
was measured.

`pip install "microscopy-viewer[analysis]"` for the writer; squidpy is left to
you, since what it pulls in depends on the analysis.

### Spatial dashboard

`.h5ad` in hand, **Spatial dashboard…** in the Measurement analysis panel opens a
browser tab with the squidpy statistics already wired up. It runs as its own
process: the spatial work is minutes of CPU that has no business blocking the
window the images are in, and Streamlit's event loop would fight Qt's. Closing
the viewer leaves it running.

Four tabs. The first asks a different kind of question from the other three:

| Tab | What it answers |
|---|---|
| **Across the plate** | which wells hold which phenotypes — every well at once, in feature space, as a UMAP |
| **Where they are** | the well as a scatter, coloured by phenotype or by any measurement, with how many objects fall in each group |
| **What they are** | which measurements separate the groups — a box per group, and any two features against each other |
| **Does it mean anything** | neighbourhood enrichment, Ripley's L, co-occurrence, Moran's I, centrality |

#### Across the plate

Feature space, not the well: *which wells differ from which* is not a spatial
question, and it is the one the other three tabs cannot ask, because their
coordinates are per image. Give it a combined `.h5ad` — the analysis panel's
**…the whole folder as one** — and it embeds every well together.

**Sampled per image, not overall.** This plate runs from 16 objects in one well
to 46 394 in another; a flat sample of the lot would be a picture of the big
wells with the small ones invisible in it. Each image contributes up to the same
quota, and a well smaller than the quota keeps everything it has:

```
557 915 objects, 44 images, from 16 to 46 394 each
  ->  29 644 objects, at most 750 per image
      the 16-object well kept all 16; a 46 394-object well capped at 750
```

Colour the map by phenotype, by **well**, by **row**, by **column**, or by any
measurement. Row and column are the ones to look at first: if the map separates
by plate row rather than by anything biological, what you are looking at is the
plate, not the sample.

Beside it, **what each well is made of** — the share of each well's objects in
each cluster — and the same thing **as a plate**, rows down and columns across,
because a plate is a physical object and the answer often is too: an edge effect,
a column of controls, a row that did not take. Grey cells are wells the plate does
not have, which is not the same as a well holding none of that cluster.

It sits behind an **Embed the plate** button because it is the slow one — about
45 s for 30 000 objects — and is remembered afterwards, so recolouring the map
costs nothing.

**Point at anything to see what it is.** The scatters are interactive: hovering a
point names its well, its cluster and its label, which is how you find out whether
a corner of the map is one well or many. The composition bars say the well, the
cluster and the share. Above 20 000 points the plot is subsampled and says so —
the page holds the data for hovering, and all of them would stop the tab
responding.

Two things to hold on to. Distances between clusters on a UMAP mean nothing; only
what is together and what is apart does. And a cluster that turns out to be 98 %
one well is usually that well looking different — staining, focus, density —
rather than a phenotype, so check a couple of its objects in the image before
believing it.

**Objects are grouped first**, because every neighbourhood statistic needs a label
per object. Either a Leiden clustering on the features — phenotypes, but ones you
then have to interpret — or **bins of one measurement**, which is cruder and far
easier to explain: quartiles of solidity is a perfectly good question to ask a
neighbourhood about, and one you can defend in a figure legend.

**The graph is a Delaunay triangulation by default**, which joins each object to
the ones it actually abuts. A fixed radius in micrometres misses the neighbours
in a sparse field and joins half the well in a dense one. kNN and radius are
there when you want them.

**These are not genes.** The preparation is a z-score and a PCA, not the
log1p-and-highly-variable-genes recipe a scanpy tutorial opens with — that belongs
to counts, and taking the log of a solidity is not a thing to do. Scaling is still
needed, because integrated intensity is six figures and solidity is below one.

**One image at a time.** `obsm["spatial"]` holds positions *within a well*, so two
wells overlap in that space; a graph across a plate would join objects that are
merely in the same corner of different wells. The sidebar picks the image, and a
run is subsampled above 20 000 objects — a permutation test on fifty thousand is
minutes, on eight thousand it is seconds and the answer is the same shape.

What it looks like on a real well:

```
one well  — G/07 cycle 1, 8000 of 18428 objects, leiden -> 10 groups, Delaunay
  neighbourhood enrichment   strongest self-association z = 36.6
  Moran's I  Median intensity (Green488-bCAT)  0.563   p_adj = 0
             Median intensity (FarRed641-CDH1) 0.540   p_adj = 0

the plate — 29 644 objects, 750 per image, 44 wells, leiden -> 17 groups
  scale + PCA  1.2 s   cluster  34 s   UMAP  11 s
  most well-to-well variation: cluster 14, 98 % of E/03 and under 1 % anywhere else
```

— the β-catenin signal comes in patches rather than cell by cell, and every
phenotype keeps its own company, which for an organoid is what you would hope.

#### Running it on its own

The dashboard is a program in its own right and does not need the viewer:

```powershell
python -m microscopy_viewer.dashboard G_07_0.h5ad     # or: microscopy-viewer-dashboard
python install_shortcut.py --dashboard                 # a desktop shortcut for it
```

**The terminal is the feature.** The shortcut opens a console and leaves it open,
and so does the viewer's button. A plate's first analysis spends ten to fifteen
seconds inside numba compiling scanpy's kernels, and a browser tab that is merely
thinking looks exactly like one that has hung. The console says which step it is
on and what each cost:

```
Microscopy Viewer — spatial dashboard
http://localhost:8501
Close this window to stop the server.

16:12:03  microscopy_viewer.dashboard  warm-up: compiled in 9.9 s; the first real run will not pay this
16:12:20  microscopy_viewer.dashboard  prepare: 29644 x 27 scaled, 15 component(s), 0.4 s
16:12:32  microscopy_viewer.dashboard  neighbours: exact (scikit-learn), k=15 over 29644 object(s) in 12.0 s
16:12:33  microscopy_viewer.dashboard  cluster: leiden found 14 group(s) in 0.8 s
16:12:33  microscopy_viewer.dashboard  neighbours: reusing the graph already built (k=15)
16:12:43  microscopy_viewer.dashboard  umap: 29644 object(s) laid out in 10.1 s
```

Closing the terminal stops the server; closing the viewer does not.

**It is served to this machine only.** Streamlit's own default binds every
interface, which on a university network would put an unauthenticated page
holding your object tables in front of anyone who can route to it. This binds
`127.0.0.1`.

#### Colour

One colour per group, in every plot on the tab: the UMAP, the well, the
composition bar, the box plot, Ripley's lines, the co-occurrence curves, and the
tick labels of the enrichment matrix. Reading these means carrying a colour from
one panel to the next, and a page where that does not hold cannot be read.

The palette is forty colours rather than matplotlib's ten, ordered greedily by
CIELAB distance so that the first twenty are at least 22 ΔE apart. Both halves
matter: with the default cycle a plate clustering into fourteen drew **clusters 0
and 10 in the same blue, 1 and 11 in the same orange, 2/12 and 3/13 likewise** —
four colliding pairs, in every plot, with nothing to say they were different.
Past forty groups the extras go grey rather than repeating a colour that already
means something else; hovering still names them.

Colours are shared *within* a tab, not across. The plate tab clusters every well
together and the spatial tabs cluster one well on its own, so cluster 3 in one is
not cluster 3 in the other — colouring them alike would claim a sameness that is
not there, and the page says so.

#### Why it used to be slow, and is less so

A plate embedding was 48 s and is now 26 s, none of it from doing less work:

| | Before | After |
|---|---|---|
| neighbour graph, 30 000 objects | 35 s — pynndescent, mostly numba compiling | **12 s** — scikit-learn, exact |
| the same graph for the UMAP | built a second time | **reused** |
| the compiler's fixed cost | paid on the first click | **paid at start-up**, in the background |

The neighbour search is nearly all of the cost, and nearly all of *that* was
compilation rather than arithmetic: pynndescent's parallel kernels cannot be
written to numba's on-disk cache, so the price is paid on every launch. Below
60 000 objects scikit-learn's **exact** search is handed to scanpy instead — no
compilation, no approximation, four times faster end to end. Above it the exact
search's quadratic term wins and pynndescent is the right tool again.

What remains is compiled while the browser is still opening, so a per-image
analysis that used to take 14 s now takes 0.1 s.

`pip install "microscopy-viewer[dashboard]"`, which includes `leidenalg` — without
it scanpy cannot run Leiden and the dashboard falls back to k-means, which needs
the number of groups decided in advance and says so on the page. A **separate
extra** from `[analysis]` on purpose: squidpy brings scanpy, scikit-learn and a graph library,
and because the dashboard is its own process it can perfectly well live in an
environment of its own rather than beside napari. The button says exactly what is
missing if it is not installed, and starts nothing.

### Third-party napari plugins

Most napari plugins declare their input as `napari.types.ImageData` and then treat
it as an array. A **multiscale** layer hands them `MultiScaleData` instead — a
sequence of pyramid levels, not an array — which they do not convert and cannot
use. Every OME-Zarr layer this viewer makes is multiscale, so the failure is
immediate and looks like a bug in the plugin:

```
TypeError: in method 'Median', argument 1 of type 'itk::simple::Image const &'
```

That one is `napari-simpleitk-image-processing`; `napari-segment-blobs-and-things`
and the rest of the assistant family behave the same way. Measured on this
plugin: a numpy array works, a dask array works, `MultiScaleData` raises — with
one level in it or five.

**Flatten for Plugins** (Ctrl+Shift+L) is the way round it. It copies the level
the viewer is currently drawing out of the selected pyramid layer and adds it as
an ordinary single-level layer, named `… [level 3]`. That copy is a plain dask
array, which those plugins handle. The scale is corrected for the level, so the
copy sits exactly on top of its parent and measurements taken on it stay in µm.

Zoom in before flattening and you get a finer level; zoom out and you get a
coarser, smaller one. Above 500 Mpixels it asks first, because a plugin handed
the copy reads all of it into memory — a plate mosaic level is tens of gigabytes.

For median filtering specifically there is no need for any of this: the
Segmentation panel does it natively, on the pyramid, as part of the run.

### Exports

**Export Snapshot** saves what is on the canvas at 2× oversampling, so it stays
sharp in PowerPoint. The scale bar is included. A `.tif` target writes lossless
pixels and tags the file with its on-screen resolution.

**Export Movie** writes a time series out as a video — see
[Time series](#time-series).

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
| **OME-Zarr / NGFF** | optional. Reads `multiscales`, the `axes` list and the `omero` rendering block directly, so no `ome-zarr` package is needed. A Zarr store is a *folder*, so open one by dropping it on the window or passing it on the command line — the file picker only lists files. Segmentations stored beside an image in a `labels` group come back as Labels layers on the same grid. |
| **OME-Zarr HCS plate** | a store whose root carries a `plate` block is assembled into one lazy mosaic per channel: wells laid out in their plate rows and columns, and the fields of each well tiled inside their cell. **Acquisitions are channels, not places** — a 4i plate lists its staining cycles inside the well exactly as several fields would be listed, and only the `acquisition` id tells them apart, so each cycle is assembled on the plate grid of its own and comes back as a further set of channels registered on top of the rest, named `… :: cycle 3 :: Red568-pSMAD1-5`. Segmentations in a `labels` group are assembled the same way. The mosaic covers the wells that are actually in the store, not the nominal plate — a 96-well layout holding one acquired well opens as that well, not as a mostly-blank plate — and the pyramid is preserved, so nothing is read until it is on screen. A single well or field folder can be opened on its own, and a folder with no NGFF metadata at all (a plate row, a converter's output directory) is descended into. |
| **OME-Zarr `bioformats2raw`** | a `bioformats2raw.layout` root loads every series as its own layer set, named from `OME/METADATA.ome.xml` when the converter wrote it. This is what `bioformats2raw` produces from `.czi`, `.lif`, `.nd2` and friends. |

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
    ome_zarr.py             OME-Zarr / NGFF, HCS plates, bioformats2raw series
  metadata.py               metadata model and the vendor-key synonym matching
  measurements.py           Shapes -> calibrated distances and areas
  intensity.py              ROI statistics, AUC / overlap; no Qt, runs off-thread
  registration.py           atlas registration and the per-region readout; no Qt
  segmentation.py           Cellpose segmentation, the GPU device, per-object stats; no Qt
  analysis.py               object tables read back in: label colouring, plot helpers; no Qt
  dashboard.py              the spatial dashboard's own analysis, and starting it; no Qt
  dashboard_app.py          the Streamlit page: run by streamlit, not imported by the viewer
  batch.py                  plate-wide segmentation: survey, run, NGFF label writing; no Qt
  explorer.py               browsing a plate: rows, miniatures, layer building, table folders; no Qt
  exports.py                snapshots and the Excel workbook
  slides.py                 channel/merge rendering and the PowerPoint slide; no Qt
  contrast.py               auto / reset contrast
  rendering.py              full-resolution 3D / MIP for multiscale layers
  gpu.py                    GPU limits and the 3D voxel budget
  volume_cache.py           local disk cache for volumes read from slow storage
  timeseries.py             the playback clock, and local caching of a time lapse
  movie.py                  canvas frames -> .mov / .mp4 / .gif; no Qt
  dragdrop.py               application-wide drop handling
  widgets/
    registry.py             the panel manifest
    toolbar.py
    metadata_widget.py
    measurements_widget.py
    intensity_comparison.py
    timeseries_widget.py    transport controls, the cache status, playback
    registration_widget.py  channel roles, the atlas, the region table
    segmentation_widget.py  channel, model and diameter; the object table
    plate_picker.py         the store box and the wells/cycles lists, shared by two panels
    explorer_widget.py      scan a plate, preview it, open the images you want
    batch_widget.py         wells, cycles and channels; the run and its results
    analysis_widget.py      the object table, the label colouring and the scatter plot
    slide_dialog.py         pick samples, name the stainings, write the .pptx
    movie_dialog.py         pick a range and a rate, render the frames
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
python tests/test_timeseries.py     # the playback clock, level budget, timeline cache — no Qt needed
python tests/test_movie.py          # frame selection, the overlay, the written movie — no Qt needed
python tests/test_slides.py         # channel/merge rendering and the .pptx table — no Qt needed
python tests/test_overview.py       # overview detection, stitching, the locator slide — no Qt needed
python tests/test_registration.py   # atlas registration engine — no Qt; ANTs checks skip without antspyx
python tests/test_segmentation.py   # segmentation engine, units, object table — no Qt; cellpose is never run
python tests/test_batch.py          # plate survey, channel matching, NGFF label writing — no Qt; cellpose is never run
python tests/test_analysis.py       # object tables, colour scales, label colouring — no Qt needed
python tests/test_explorer.py       # plate rows, miniatures, layer building, table folders — no Qt needed
python tests/test_dashboard.py      # dashboard slicing, grouping, squidpy calls — no Qt; the squidpy half skips without it
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
