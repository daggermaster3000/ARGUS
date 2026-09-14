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

### Startup

napari puts its main window on screen the instant the viewer object exists and
then spends several seconds building the docks — importing what each panel needs,
waking the GPU, reading settings. What is on screen for those seconds is an empty
white window, which looks like a hung application rather than a loading one.

So the window is kept hidden until it is built, and the animation in
`microscopy_viewer/resources/loading.gif` is shown in its place, with the step
currently running underneath it:
*Building the Segmentation panel…*, *Opening 3 file(s)…*, *Ready*. The window then
appears complete, with the files already open.

The animation only advances while something is processing events, and building a
viewer is one long blocking call, so the build drives it: each step calls the
splash, which is what both names the step and lets Qt paint a frame. That makes
the splash a progress report as much as a picture — if startup ever does hang, the
last line says where.

Nothing depends on it. A missing GIF, a Qt build without the image plugin, or any
other failure means the splash is skipped and the window opens the way it always
did. `--no-splash` does the same on purpose.

#### While the window is busy

The same animation comes back whenever the viewer stops responding for more than
about a second — a stack coming off the NAS, a Cellpose run, a deck being
written. It covers the window until the work finishes, counting the seconds, and
disappears the moment the window answers again.

It has to be **a second process**, and that is the whole design. Qt paints from
the main thread and nowhere else, so while that thread is inside a long call
there is nobody left in this process to draw a frame — which is exactly why a
frozen window goes white and Windows writes *(Not Responding)* on it. An
animation driven from the frozen process would be a frozen animation.

So there are two pieces, in `microscopy_viewer/busy.py`:

- A **heartbeat**: a timer on the main thread writing the time, and where the
  window is, into a small locked record. When the thread blocks the heartbeat
  stops — that is the signal — and the record still holds the last known geometry,
  which is where the animation has to go, because a stuck thread can no longer be
  asked.
- A **watchdog thread** comparing that timestamp against the clock and driving
  `microscopy_viewer/busy_window.py`, a child process whose only job is to play
  the GIF. It is spawned once at startup and kept hidden, so appearing during a
  freeze costs one line down a pipe rather than a process launch.

The watchdog thread touches only plain data and that pipe, never a Qt object:
calling into Qt from a second thread while the first is inside a long call is how
a hang turns into a crash.

- The overlay stays away when no window of the viewer's is focused, so alt-tabbing
  to something else while a long read finishes leaves the screen alone.
- The child exits when its stdin closes, which happens when the viewer exits —
  crash included. It cannot outlive the window it belongs to.
- `--no-busy-overlay` turns it off. So does anything going wrong with it: a child
  that will not start is logged once and the viewer carries on freezing the way
  every Qt application freezes.

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

**Sizes are in µm, not pixels.** Cellpose's `diameter` is in XY pixels and its
`anisotropy` is the Z/XY voxel ratio; both are derived from the calibrated voxel
size the reader already put on the layer. A 5 µm nucleus stays 5 µm whether the
stack is 0.13 µm/px or 0.6 µm/px, and it stays right after a large volume has been
decimated — the conversion happens after the decimation, not before.

| Setting | What it does |
|---|---|
| **Segment** | The channel that is labelled. A nuclear stain with the `nuclei` model is the reliable case. |
| **Measure** | The channel intensities are read from. Segment on DAPI, measure on the reporter, and every row is signal per nucleus. |
| **Mode** | `2D + stitch` segments each plane and joins masks in neighbouring planes whose overlap exceeds the stitch threshold, so one object spanning seven planes comes back as one label — faster than `3D`, and usually better on an anisotropic stack where a nucleus is four planes tall. `3D` computes flows in 3D. `2D per plane` leaves labels unconnected between planes. `2D on max projection` flattens the stack first — see below. |
| **Stitch gap** | How far, in µm of depth, stitching may reach over planes where the object was missed. Cellpose compares neighbouring planes only, so a nucleus absent from one plane comes back as two objects that no threshold can rejoin. On a 20-plane crop the default took 907 objects down to 602, against 551 for full `3D`. In µm rather than planes because a plane is not a fixed distance. Zero restores Cellpose's own behaviour. |
| **Diameter** | Expected object diameter in µm. The setting that matters most; automatic is worth overriding. |
| **Minimum diameter** | Smallest object to keep, in µm. Converted to Cellpose's pixel count against the voxel size, after decimation. Zero leaves Cellpose's own default. |
| **Maximum diameter** | Largest object to keep, in µm. Cellpose has no ceiling of its own — `min_size` has no counterpart — so this one is applied to the finished label map, which is also the only place it *can* be applied in `2D + stitch`, where an object does not exist until the planes are joined. Measured exactly as the table's *Equivalent diameter*, so a number read off the Objects tab means the same thing typed back in. The usual use is dropping a clump of touching nuclei that came back as one object. Survivors are renumbered from 1, and the panel says how many were removed. |
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
and mean / median / std / max / integrated intensity from the measure channel. It
exports through the same writer as the measurements workbook. Objects are measured
through the indices of the labelled voxels rather than a loop over labels, so the
memory it takes scales with the segmented fraction of the volume, not the volume.

The run is on a `thread_worker`, so the window stays usable while it goes.

#### Progress while it runs

A segmentation runs on a `thread_worker`, so the window stays usable — but a
sixteen-plane stack is a minute of nothing happening unless it says otherwise. The
panel counts the planes off as they go:

```
Segmenting dapi: plane 4 of 7…
Segmenting dapi: stitching 7 planes…
3 object(s) in 3 s (cpsam_v2, 2D + stitch, NVIDIA GeForce RTX 4090, diameter 17 px).
```

The text crosses threads on a Qt signal rather than being written to the label
directly. Touching a widget from the worker thread is how a slow segmentation turns
into a crash; a signal is queued and delivered on the GUI thread, so the label is
only ever written from the thread that owns it.

`2D + stitch` reports planes because the loop is **here** rather than inside
`model.eval`. Cellpose's own stitching path segments the planes internally, where
they cannot be counted, and it is slower: on a 16-plane crop of a 20x stack, 8 s
against 20–45 s. The planes are then joined by `segmentation.stitch_planes`, which
at a gap of one plane makes exactly the join cellpose makes — checked voxel for
voxel against `cellpose.utils.stitch3D` on a real 20-plane crop, identical
partition — and which can also reach further. See below. Planes must reach the
stitcher numbered from 1, which is why the per-plane label offsetting is switched
off on that path: offset first and nothing matches, leaving every plane's objects
separate.

#### One missed plane splits an object in two

This is the failure `2D + stitch` is most likely to produce, and it is worth
understanding because no amount of tuning the stitch threshold fixes it.

**Cellpose compares plane *i* with plane *i+1* and nothing else.** If a nucleus is
found in planes 0–3 and 6–9 but missed in 4 and 5, the two halves are never
compared with each other at all, so they come back as two objects. Lowering the
threshold cannot help: the comparison that would rejoin them does not happen.

On a 20-plane crop of a 20× confocal stack, measured against the same volume
segmented in full `3D` — the mode that *does* understand an object missing from a
plane — the effect is not subtle:

| Stitch gap | Planes | Objects | Split | Merged |
|---|---|---|---|---|
| off (what cellpose does) | 1 | 907 | 73 | 52 |
| 0.6 µm | 2 | 739 | 48 | 40 |
| **1.5 µm (the default)** | **5** | **602** | **26** | **34** |
| 3.0 µm | 10 | 562 | 24 | 31 |

*(3D found 551 objects. "Split" counts reference objects covered by two or more of
ours; "merged" counts ours covering two or more reference objects.)*

So **Stitch gap** lets a chain that has been broken reach further along Z, under
two rules that keep it from doing anything else:

- **Only broken chains bridge.** A mask that already matched in the next plane is
  not offered a longer jump, and neither is one that already has a predecessor.
  Bridging can only repair a gap, never reroute a match that worked.
- **Best overlap wins, one match each way.** Pairs are taken in descending order of
  IoU and each mask joins at most one mask per direction, so two nuclei that touch
  in a single plane do not collapse into one.

Both error directions improve together — splits fall from 73 to 26 *and* merges
from 52 to 34 — which is the evidence that this is repairing a real defect rather
than trading one error for another.

**The setting is in µm of depth, not planes**, like every other size in the panel.
Four planes is 1.2 µm on a 0.3 µm/plane stack and 8 µm on a 2 µm/plane one, and the
second would cheerfully bridge two different cells stacked above each other. At
2 µm/plane the 1.5 µm default converts to one plane, which is cellpose's own
behaviour — the coarser the stack, the less this does, which is the right way round.

Raise it if single nuclei come back split along Z; lower it if nuclei stacked above
one another are being merged. Zero restores cellpose's behaviour exactly, and the
summary line says what was used: `stitched over gaps up to 5 plane(s)`.

The gap costs almost nothing. The extra passes only compare masks whose chain is
broken, so the planes they look at are nearly empty by then: on that 20-plane crop
the whole stitch went from 0.07 s to 0.30 s, against 12 s for the segmentation
itself.

#### The 2D modes on a stack

Cellpose 4 accepts `z_axis` only when it is going to treat the array as a volume —
`do_3D`, or stitching with a threshold above zero. Hand it a stack in plain 2D mode
and it refuses:

```
ValueError: 2D image processing selected, but z_axis is not None.
            Set z_axis=None to process 2D images.
```

That caught `2D per plane` always, and `2D + stitch` whenever the stitch threshold
was wound down to 0 — which asks for exactly the same thing. Both now segment plane
by plane, one 2D call each, which is an honest reading of the rule rather than a way
around it: Cellpose is being asked for 2D segmentation, so it is given 2D images.

Labels are made unique across the stack. Restarting from 1 on every plane would
collide in the label map, so one object would appear to span planes it was never
found in — the exact thing the mode exists not to do. On a 24-plane DAPI crop of a
20× stack, `2D per plane` and `2D + stitch` at 0 both return 5329 objects, the same
number, because they are now the same operation.

#### Segmenting a maximum projection

`2D on max projection` collapses the stack with a maximum projection and segments
that single image. On the deconvolved 17 x 2040 x 2040 stacks it is the difference
between a coffee break and a keystroke — measured on an RTX 4090:

| Mode | Objects | Time |
|---|---|---|
| `2D on max projection` | 31 | **9 s** |
| `2D + stitch` | 140 | 78 s |
| `3D` | 296 | 2296 s (38 min) |

The object counts differ because the modes answer different questions, and the
projection answers the narrowest one: **anything overlapping along Z becomes one
object.** For sparse, well-separated objects in a thin stack that is exactly what
you want and the run says so in its warnings. For a dense nuclear stain it is not,
and `2D + stitch` is the mode to reach for.

What follows the projection through:

- **The Z voxel is dropped with the Z axis.** A diameter in µm converts against the
  XY voxel size only, so the same number keeps working.
- **The measured channel is projected too.** Segment on DAPI, measure on the
  reporter, and both are flattened the same way — 2D labels against a 3D signal
  would not line up at all.
- **The flattened image is added as a layer** beside the labels, named
  `<channel> [MIP]` and carrying the source channel's colormap and contrast. The
  labels are 2D, so without it there would be nothing 2D to check them against.
- **Sizes come back as areas.** The table and the export say `Area (µm²)` and
  `Pixels` rather than `Volume (µm³)` and `Voxels`. The numbers were always right —
  `object_table` multiplies by whatever voxel size it is handed — but the heading
  was not. The dimensionality is passed in, never guessed from the data: a real 3D
  run can have every object sitting at z=0.

Segmentation needs `cellpose`, which pulls in torch, so it is an optional extra
rather than a dependency:

```bash
# GPU: install a CUDA build of torch first, from https://pytorch.org
python -m pip install ".[segmentation]"
```

Without it the panel still appears and says exactly that, the way the atlas panel
reports a missing `antspyx`. `Backend` is a three-method interface here too, so
StarDist or micro-SAM can be added beside Cellpose.

### Brain regions

**Brain regions** answers the question that follows every segmentation of a whole
brain: *how many of those are in the cerebellum?* The label map knows where every
object is; what it does not know is what the parts of the specimen are called. So
the outlines are drawn by hand and each object is attributed to the region its
centroid falls in.

Press **Add region**, draw an outline, name it in the table — names are drawn on
the canvas and stored in the layer's `features`, so they are saved and reloaded
with the shapes rather than living only in this panel. **Suggest names** fills any
still-unnamed outline from `forebrain`, `midbrain`, `hindbrain`, `cerebellum left`,
`cerebellum right`. Then pick a Labels layer and press **Count objects**.

Three decisions are worth knowing:

- **Objects are counted by their centroid, not by overlap.** An object straddling a
  boundary belongs to exactly one region, so the per-region counts sum to the total
  and a nucleus is never counted twice. Overlap-weighted counting does not have
  that property.
- **Regions are tested in order and the first match wins.** Hand-drawn outlines
  overlap at every boundary; silently double-counting there would be worse than a
  rule that can be stated. Overlapping regions are detected and named in the status
  line, so it is visible when the rule is doing something.
- **Outlines are kept in world micrometres**, like the ROI comparison panel's. They
  are therefore independent of which layer they were drawn on, and the same set can
  be applied to a label map at a different pixel size — which is what lets them be
  written to a whole folder of samples from the *Experiment setup* panel.

The table reports, per region: objects, area in µm², **objects per mm²**, total
object volume, median diameter and mean intensity. Objects inside no region get
their own row rather than being dropped — a large count there means the outlines
missed something, and that is worth seeing. The export writes two sheets to one
workbook: the counts, and every object with the region it was assigned to, so a
surprising number can be traced to the rows that produced it.

**Stored regions come back when the sample is opened.** Outlines are written into
the `.ims` itself (see *Experiment setup* below), and opening that file again puts
them back on the canvas without being asked — having to remember a *Load from file*
button is how a saved annotation looks lost.

Two things it will not do:

- It will not replace outlines already on the canvas that came from somewhere else,
  since those may be unsaved work. It says the sample carries regions and leaves
  them; *Load from file* replaces them deliberately.
- When several samples opened at once carry regions it loads none of them. There is
  one region layer and the outlines are per-sample, so showing one sample's regions
  over another's image would be worse than showing none.

The region layer is deliberately kept out of the **Measurements** panel. That panel
renames every shape it finds to `ROI 1`, `ROI 2`, … on each recompute, which would
overwrite the anatomical names as fast as they were typed.

### Experiment setup

**Experiment setup** treats a folder as the unit of work. An acquisition session
leaves a folder of `.ims` files, and nearly everything done afterwards is done to
all of them: the same region outlined on every fish, the same channel segmented
with the same settings. Doing that through the viewer means opening thirty
datasets, and the layer list stops being usable around the fourth.

**Scanning adds nothing to the viewer.** Each file is opened far enough to read its
shape, channels and whatever this program has already stored in it, a thumbnail is
drawn, and the file is closed again. The preview comes off a middle pyramid level —
not the coarsest, which bottoms out near 64 px and tells you nothing about the
mount, and not the finest, which would be a full-resolution read per channel per
file. Tiles say what each file carries (`2 ROI(s), 1 label map(s)`) and unreadable
files — an aborted acquisition leaves `*_F0.ims` stubs — are listed in red with the
reason rather than filtered out.

**ROIs and label maps go inside the `.ims` file.** Both are written into one
top-level group, `/ARGUS`, beside Imaris's own `/DataSet` and `/DataSetInfo`.
Nothing Imaris wrote is read back, modified or deleted, and Imaris ignores groups
it does not recognise, so the file still opens and behaves normally. A sidecar
folder would work right up until the files are moved or sent to a collaborator, at
which point the derived data is orphaned in silence.

What this deliberately does **not** do is write Imaris Surfaces or Spots objects.
Those live in the undocumented `Scene8` structure; producing one Imaris will load
means guessing at a private format inside a file holding irreplaceable acquisition
data, and getting it wrong corrupts the file rather than failing. Regions saved
here are visible to this viewer and not to Imaris.

**A file being written to cannot be open in the viewer.** HDF5 refuses to open for
writing what is open for reading — in the same process as much as across
processes — and the reader deliberately holds its handle for the life of the
process, because the dask graphs read through it lazily. So the panel takes a
sample's layers off screen and releases the handle before writing into it, and
says how many of each it closed.

The workflow the panel is shaped around:

1. **Choose** the folder and **Scan** it.
2. **Open** one sample, draw the outlines on the `Brain regions` layer, then
   **Write to selected** — into that one sample, or into every sample selected at
   once. Writing to all of them is right when the samples are mounted and framed
   alike and wrong when they are not: the vertices are in micrometres from each
   image's own origin, so a fish sitting 200 µm further along its field gets an
   outline 200 µm out of place. **Load from sample** reads them back.
3. **Run on selected.** The batch takes its model, mode, diameters and device from
   the **Segmentation** panel rather than duplicating those controls — two sets of
   controls for one set of parameters is how a batch ends up run with settings
   nobody chose.

| Batch setting | What it does |
|---|---|
| **Segment channel** | Matched against the channel names in each file, so `dapi` finds it wherever it sits — the channel order is not the same in every acquisition. A bare number is an index instead. A name that matches nothing **skips that file and says so**, rather than quietly segmenting channel 0: over thirty files that would be an experiment's worth of wrong numbers. |
| **Measure channel** | Optional second channel the per-object intensities come from. |
| **Restrict** | Blanks everything outside each file's stored ROIs before segmenting. This is what drawing them was for — Cellpose has no idea the skin and the yolk are not brain, and finds plenty of objects in both. A file with no stored ROIs is segmented whole. The image is zeroed rather than cropped, so labels come back on the grid that went in. |
| **Results** | Write each label map into its own `.ims`, under `/ARGUS/Labels`, tagged with the model, mode, channel and diameters it was made with. |

A file that cannot be read, or whose channel cannot be found, is reported and
skipped — thirty files is long enough that aborting on the twenty-ninth because one
is a stub would be its own kind of failure. **Stop** ends the run after the file it
is on. The export writes two sheets: one row per sample, and every object from
every sample with the sample it came from.

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

*Manual limits* is the third option: **Min** and **Max** columns in the channel
table, one pair per channel, typed as raw intensities. Choosing it unlocks the
cells and seeds them with the range that channel is displayed at, so the numbers
start somewhere sensible and get nudged rather than invented.

The pair is keyed to the **channel column, not the sample** — which is the point.
Auto contrast stretches every panel to its own histogram, so a dim sample and a
bright one come out looking equally bright and the figure quietly lies. One typed
range across the row makes the panels comparable by eye, which is what a reviewer
assumes they already are.

- A channel left empty falls back to the range it is displayed at, so filling in
  one row and leaving the rest alone does what it looks like.
- A backwards or half-typed pair is ignored the same way, rather than stopping the
  export with a message box.
- What is typed is remembered while the dialog is open, so switching to *Auto per
  image* to see how it looks and back again does not lose it. Only *Manual limits*
  applies it.

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

#### Closeup slides

A folder is usually acquired by working inwards: a 10x of the fish, a 20x of the
region worth looking at, a 40x of the thing itself. In the table those are three
unrelated rows, and which part of the 10x the 40x came from is in the operator's
head.

Each of them gets its own slide instead: **the field it was taken from on the
left, with a red box around the part the closeup covers**, and the closeup's own
channels and merge beside it. A sample nothing else contains is shown against a
window of the overview rather than the whole mosaic, since the locator slide
already shows the whole mosaic and a box a hundredth of it wide points at
nothing.

The pairing is stage coordinates and nothing else — the same `ExtMin`/`ExtMax`
extents the overview is placed from. The objective a file names (`LensPower`) is
printed in the captions and never used to decide anything, because two
acquisitions are told apart by how much stage they cover, and that is recorded
even when the objective is not.

- The parent is the **smallest** acquisition that contains the closeup, so a 40x
  taken inside a 20x taken inside a 10x is shown against the 20x. The tightest
  context is the one that says where you are.
- A closeup may overhang its parent by 5% of its own width and still count: the
  stage repeats to a few micrometres and a field re-centred by eye can end up a
  hair over the edge. Half outside is not a closeup.
- A field has to be **at most 70%** of the one it sits in. Two acquisitions of the
  same region contain each other and neither is a closeup of the other; two
  images that merely start at the same corner are not related at all. Every real
  step down clears it — 40x inside 20x is 0.5, 60x inside 40x is 0.67.
- The box is a PowerPoint shape, drawn at a minimum size when the closeup is a
  small fraction of the field and grown about its own centre, so it stays
  findable without stopping pointing at the right place.
- The context picture is read at the resolution it is drawn at — about a third of
  the slide — which on a pyramid file is a coarser level and a fraction of the
  bytes.
- Untick **Closeups** in the dialog for the plain deck. With **Overview** unticked
  as well, only closeups that sit inside another sample get a slide.

On the real folder that is `ift88-curved_4` (40x, 311 µm) inside `ift88-curved_2`
(20x, 621 µm) inside `ift88-curved_1` (10x, 1243 µm), each one boxed on its
parent, and the 10x itself boxed on a 7.5 mm window of the 2x overview.

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
  registration.py           atlas registration and the per-region readout; no Qt
  segmentation.py           Cellpose segmentation, the GPU device, per-object stats; no Qt
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
python tests/test_busy.py           # the freeze watchdog and its animation process — no display needed
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

### The segmentation panel says CPU, or another account on the same machine has no GPU

Both are the same cause: **Python packages install per user, the GPU is shared.**
A machine where one account segments on a CUDA card and another does not is not a
driver problem — the second account simply has no `torch` with CUDA in *its*
site-packages. The panel is reporting truthfully.

Two things have to be true in each account that wants GPU segmentation:

```powershell
# 1. torch AND torchvision from the same CUDA index. Installing only torch leaves a
#    mismatched torchvision, which fails at "import cellpose.models" with
#    "RuntimeError: operator torchvision::nms does not exist" — not at import torch.
<python> -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision

# 2. cellpose itself
<python> -m pip install cellpose
```

`<python>` must be **the interpreter the viewer runs on**, which is the shortcut's
target — right-click the shortcut, *Properties*, and read *Target*. For a shared
miniconda whose `site-packages` is not writable, add `--user`; the per-user site is
on that interpreter's path automatically. Pick the CUDA index that matches the
driver (`nvidia-smi`): `cu130` for a CUDA 13 driver, `cu128` for CUDA 12.8.

Check it took:

```powershell
<python> -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`2.13.0+cu130 True` is what you want. A version ending `+cpu` is the default PyPI
wheel — that is the usual reason the panel says CPU, and reinstalling from the CUDA
index above is the fix. To give every account on a shared workstation the GPU with
one copy, install into a machine-wide environment as an administrator instead of
into each profile.

### "'NoneType' object has no attribute 'write'" during a segmentation

The desktop shortcut runs **`pythonw.exe`**, which gives the process no console at
all: `sys.stdout` and `sys.stderr` are both `None`. This project's own code checks
for that; libraries do not. Cellpose draws `tqdm` progress bars,
tqdm writes to `sys.stdout`, and the run dies with

```
AttributeError: 'NoneType' object has no attribute 'write'
```

after the slow part, with nothing to show for it. It first showed up in
`2D + stitch`, which at the time called `cellpose.utils.stitch3D` — the stitching
is now done here and that particular call is gone, but plenty of other cellpose
paths draw progress bars, so the guard stays. Running the same thing from a
console worked, which is what kept it hidden — a terminal supplies the streams.

`microscopy_viewer/__init__.py` now points both streams at the null device when
the interpreter supplied none, before anything can import cellpose. Progress-bar
redraws are all that gets discarded; real diagnostics go to the log file through
`setup_logging`. A stream the process actually has is never touched.

### The scale bar says "pixels" on a calibrated image

There are two ways this happens, and both are fixed.

The first is the layer never carrying a unit at all, described below.

The second is subtler: **one dimensionless layer takes the whole viewer down with
it.** napari compares units across layers right-aligned and by *dimensionality*,
and a layer added without units defaults to `pixel`, which is dimensionless. So a
single ROI, brain region, label map or projection layer left on the default makes
the entire list inconsistent — napari says

```
Inconsistent units across layers; units will not be used for rendering.
```

drops units from rendering, and the scale bar over your calibrated stack goes back
to counting pixels. Drawing a region should not change what the scale bar says.

Every layer this program creates now inherits the unit of the layer it came from,
through `loaders.layer_spec.units_like` (derived from one layer) and `world_units`
(matched to whatever calibrated layers are already open): ROI layers, the brain
region layer, label maps, maximum-projection layers and warped atlas volumes. The
region layer is also re-matched whenever the layer list changes, since it can be
created on an empty viewer before there is anything to match.


napari moved where the scale bar gets its unit. Up to 0.6 the overlay carried its
own `unit` field and `viewer.scale_bar.unit = "µm"` was the whole story. From
**napari 0.8 `ScaleBarOverlay` has no `unit` field at all** — the bar reads
`layer.units`, which defaults to `pixel`. Setting the old attribute does nothing,
silently, so a properly calibrated stack showed `25 pixels`.

The readers now pass `units` to the layer alongside `scale`, so the unit travels
with the data. Both are taken from the same file metadata, which is what stops
them disagreeing. Derived layers — the segmentation Labels layer, a `[MIP]`
projection — copy the units of the layer they came from, or selecting one would
put the bar back to pixels over an image measured in µm.

`viewer.scale_bar.unit` is still set where the field exists, so napari 0.6 keeps
working; the presence of the field is checked rather than the version.

If it still reads pixels, the file had no calibration to begin with — the metadata
panel says `Calibration: not in file` and measurements are reported in pixels
throughout, deliberately, rather than inventing a pixel size.

### "OMP: Error #15" kills the process

torch ships its own Intel OpenMP runtime (`libiomp5md.dll`) while numpy and scipy
use the copy from conda's MKL. When the second one initialises, Intel's runtime
prints *OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
already initialized* and calls `abort()` — a C-level kill, so no `try`/`except`
anywhere can catch it and the process simply disappears.

Whether it fires depends on import order. Importing napari first happens to
arrange the libraries acceptably, which is why the GUI escapes it, but a bare
`import microscopy_viewer.segmentation` — the test suite, or any script — aborted
reliably. `microscopy_viewer/__init__.py` now sets `KMP_DUPLICATE_LIB_OK=TRUE`
before anything can import torch, which is the supported switch for letting a
second copy load. Set the variable yourself to override it; an explicit value is
left alone.

The other cure is a single OpenMP build across the environment, which is a rebuild
rather than something an import can arrange.

### "NameError: name 'dinov3_vitb16' is not defined" when picking a cpdino model

Cellpose 4.2 lists `cpdino` and `cpdino-vitb` in its model zoo, but it does not
ship the DINOv3 backbone they are built on. Without it the import inside
`cellpose/vit.py` fails with a warning at import time and the model then fails with
a `NameError` at load time — a gigabyte of weights downloads first, which makes it
look like the download was the problem.

```bash
python -m pip install --user "git+https://github.com/facebookresearch/dinov3"
```

The weights themselves land in `~/.cellpose/models` on first use, and stay there.
`cpdino` is 1.2 GB and `cpdino-vitb` is 343 MB, so the first Segment with one of
them is a long wait with nothing on screen — the panel prefers an already-downloaded
model for its initial selection for exactly that reason.

### Cellpose 3 models (cyto, cyto2, cyto3, nuclei) do not appear

Cellpose 4 is one architecture and refuses them outright — *"This model does not
appear to be a CP4 model. CP3 models are not compatible with CP4"* — so they are
kept out of the model list rather than offered and then failing minutes into a run.
The panel names any it finds in `~/.cellpose/models` and says what to install.

The two model families cannot coexist in one environment. `python -m pip install
"cellpose<4"` swaps the zoo (and its size model, which is what makes an automatic
diameter meaningful there) in for `cpsam` and the `cpdino` models; upgrading again
swaps back. A second environment with its own shortcut is the way to keep both.

### Startup over Remote Desktop

Creating the window itself is the slow part in an RDP session, where OpenGL falls
back to software rendering: measured here at roughly three minutes from
double-click to a usable window, against about four seconds for everything else
(imports, viewer construction, reading the files). Locally it is a few seconds.
Image loading is unaffected — that is dask-backed and lazy either way.
