"""Minimal code for every step of the pipeline, as shown in pipeline_explainer.html.

Each block between ``# %% <step> | <title> | <where the real code is>`` lines is
shown in the matching section of the page. Each is the algorithm with the
bookkeeping taken out (no options, no special cases), short enough to read.
Every function says what it takes (Inputs) and what it gives back (Returns).

Conventions used throughout:

- an *image* or *map* is a 2-D numpy array of shape (rows, columns) on the
  overview grid, one value per overview pixel;
- a *mask* is such an array of True/False;
- ``um`` is the size of one overview pixel in µm (about 5);
- a *line* is an array of shape (n, 2) of points (row, column), in pixels
  unless the name ends in ``_um``.

This file is read, never run (some blocks use data defined elsewhere). The
junction steps (tissue to line) are checked against the package on a real
section, and every block is checked to compile, by::

    python notebooks/explainers/explainer_code_check.py
"""
# %% pipeline | The whole pipeline, end to end | microscopy_viewer/junction.py: find_junction; notebooks/dataset/run_dataset.py
# Every function called here is defined in the section of the page whose number is in the comment.


def junction_of(channels, nuclear_mean, valid, um, nuclear=0, label=1, other=2):
    """Find the junction of one section and the band beside it.

    Inputs
      channels      (C, rows, cols) array: the overview, one image per channel
      nuclear_mean  (rows, cols) array: the nuclear stain, each pixel the mean of its block
      valid         (rows, cols) mask: pixels the scanner actually imaged
      um            float: µm per overview pixel
      nuclear, label, other   int: which channel is the nuclear stain, AF488 and AF555
    Returns
      line          (n, 2) array: the junction, as points (row, col) in overview pixels
      band          (rows, cols) mask: where the cells are counted
    """
    others = [c for k, c in enumerate(channels) if k != nuclear]
    tissue, filled = tissue_mask([channels[nuclear], *others], valid, um)  # 3 tissue
    density = cell_density(channels[label], tissue, filled, um)          # 4 AF488 cells
    other_density = cell_density(channels[other], tissue, filled, um)    #   AF555 cells
    hot, cold = find_seeds(filled, density, other_density)               # 5 seeds
    cost = line_cost(nuclear_mean, filled, um)                           # 6 price
    halves = min_cut(filled, hot, cold, cost)                            # 7 minimum cut
    halves = straighten(filled, cost, hot, cold, um, bend=1000)          # 8 pay for turns
    line = trace_line(halves, um)                                        # 9 line ...
    return line, band(line, halves, um)                                  #   ... and band


rows = []
run, routed = select_sections(table)                                     # 1 pre-screen
for section in run.itertuples():
    stack, um, step = read(section.path)                                 # the scan: (C, Z, Y, X), µm per pixel
    channels, nuclear_mean = overview(stack, step, nuclear=0)            # 2 overview
    valid = (channels > 0).all(axis=0)                                  # imaged pixels (no-data pixels are exactly 0)
    line, band_mask = junction_of(channels, nuclear_mean, valid, um * step)
    if routed[section.Index] or failed(line):                            # 10 triage
        line, band_mask = hand_drawn(section.path)
    cells = measure_cells(stack[0].max(0), stack[1].max(0), um)          # 11 cells
    in_band = band_mask[tuple((cells["centre"] / step).astype(int).T)]   # cells whose centre is in the band
    positive = positive_cells(cells, in_band)                            # 12 positive
    contrast = cells["cell_mean"] / cells["ring_mean"]
    crossed = crossed_cells(cells["centre"] * um, contrast, positive,    # 13 crossed
                            line * um * step)
    rows.append({"name": section.name, "Genotype": section.Genotype,
                 "Experiment": section.Experiment, "slide": section.slide,
                 "crossed_per_mm": crossed.sum() / line_length_mm(line)})
fold, low, high = compare_genotypes(pd.DataFrame(rows))                  # 14 statistics

# %% prescreen | Which sections go on | apps/OASIS pre-screen; notebooks/dataset/run_dataset.py: selection
import pandas as pd


def select_sections(table):
    """Keep the sections the pipeline can use, and mark those whose junction is drawn by hand.

    Inputs
      table   DataFrame, one row per section, with the pre-screen's columns "Pass" (yes / no)
              and "Shape" (Fused pair / Damaged / Separated / ...)
    Returns
      run     the rows to analyse: usable (Pass = yes) and two organoids that touch (not Separated)
      routed  True/False per row of *run*: Damaged sections, whose junction is drawn by hand
    """
    run = table[(table["Pass"] == "yes") & (table["Shape"] != "Separated")]
    routed = run["Shape"] == "Damaged"
    return run, routed

# %% overview | The overview: flatten, shrink | microscopy_viewer/junction.py: build_overview
import numpy as np


def overview(stack, step, nuclear):
    """Make the small picture the junction is found on.

    Inputs
      stack    (C, Z, Y, X) array: the scan at full resolution (C channels, Z focal planes)
      step     int: how many full-resolution pixels become one overview pixel (about 5 µm worth)
      nuclear  int: which channel is the nuclear stain
    Returns
      small         (C, Y/step, X/step) array: every channel, keeping one pixel of each block
                    (so a bright cell keeps its full brightness)
      nuclear_mean  (Y/step, X/step) array: the nuclear stain, each pixel the mean of its block
                    (single nuclei average out; a faint dark line between organoids remains)
    """
    flat = stack.max(axis=1)                                  # each pixel: its brightest value over the Z planes
    small = flat[:, ::step, ::step]                           # one pixel out of every step x step block
    c, h, w = flat.shape
    nuc = flat[nuclear, : h - h % step, : w - w % step]
    nuclear_mean = nuc.reshape(h // step, step, w // step, step).mean(axis=(1, 3))   # mean of every block
    return small, nuclear_mean

# %% tissue | Where the tissue is | microscopy_viewer/junction.py: tissue_mask
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu


def disk(radius):
    """A round brush: (2r+1, 2r+1) mask, True within *radius* pixels of the centre."""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def above_slide(plane, valid, um):
    """Find where one channel is brighter than the bare slide.

    Inputs
      plane  (rows, cols) array: one channel of the overview
      valid  (rows, cols) mask: pixels the scanner actually imaged
      um     float: µm per pixel
    Returns
      (rows, cols) mask: True where the smoothed channel is above the slide's level
    """
    smooth = ndimage.gaussian_filter(plane.astype(np.float32), 10 / um)   # blur over 10 µm: nuclei merge into one signal
    values = smooth[valid]
    middle = threshold_otsu(values)                           # Otsu: the value that best splits slide from tissue
    background, tissue = np.median(values[values < middle]), np.median(values[values >= middle])
    return (smooth > background + 0.2 * (tissue - background)) & valid   # threshold one fifth of the way up from the slide


def tissue_mask(channels, valid, um):
    """Find the tissue: every pixel where any channel stands above the slide, tidied.

    Inputs
      channels  list of (rows, cols) arrays: the nuclear stain first, then the other channels
      valid     (rows, cols) mask: pixels the scanner actually imaged. A stitched scan fills
                areas it did not image with exact zeros; these are "no data", not slide, and
                are left out of the thresholds and of the tissue
      um        float: µm per pixel
    Returns
      tissue    (rows, cols) mask: the tissue, with its holes (tears) left open
      filled    (rows, cols) mask: the same with the holes filled; the area the junction is cut in
    """
    mask = np.zeros(valid.shape, bool)
    for plane in channels:                                    # any channel: an organoid faint in the nuclear stain is kept
        mask |= above_slide(plane, valid, um)
    mask = ndimage.binary_opening(mask, disk(max(1, int(round(10 / um)))))   # remove specks and thin spurs (< 10 µm)
    mask = ndimage.binary_closing(mask, disk(max(1, int(round(15 / um)))))   # close gaps narrower than 15 µm
    pieces, _ = ndimage.label(mask)
    sizes = np.bincount(pieces.ravel())[1:]
    mask = np.isin(pieces, np.flatnonzero(sizes >= 0.1 * sizes.max()) + 1)   # drop pieces under a tenth of the largest (debris)
    return mask, ndimage.binary_fill_holes(mask)

# %% labelled | Where the labelled cells are | microscopy_viewer/junction.py: cell_density
import numpy as np
from scipy import ndimage


def cell_density(label, tissue, filled, um, median_um=45, radius_um=100):
    """Turn a label channel into a smooth map of how densely labelled cells lie.

    Inputs
      label      (rows, cols) array: one label channel of the overview (AF488 or AF555)
      tissue     (rows, cols) mask: the tissue, holes open
      filled     (rows, cols) mask: the tissue, holes filled
      um         float: µm per pixel
      median_um  µm: size of the window whose median is taken as the local background
      radius_um  µm: how far the density is smoothed
    Returns
      (rows, cols) float array: labelled-cell density, about 0 (none) to 1 (densest); 0 off the tissue
    """
    size = max(3, int(round(median_um / um)) | 1)             # window size in pixels (odd)
    cells = np.clip(label - ndimage.median_filter(label, size), 0, None)   # what is brighter than its neighbourhood: cells, not haze
    cells = np.clip(cells / (np.percentile(cells[tissue], 99.5) + 1e-6), 0, 1)   # scale: the brightest 0.5 % become 1
    sigma = radius_um / um
    amount = ndimage.gaussian_filter(np.where(tissue, cells, 0).astype(np.float32), sigma)
    share = ndimage.gaussian_filter(tissue.astype(np.float32), sigma)       # how much of the window is tissue
    return np.where(filled, amount / np.maximum(share, 1e-3), 0)            # divided by it, so the edge of the section does not look empty

# %% seeds | One seed in each organoid | microscopy_viewer/junction.py: find_seeds
import numpy as np
from scipy import ndimage
from skimage.graph import MCP


def largest(mask):
    """The largest connected piece of *mask* (rows, cols), as a mask of the same shape."""
    pieces, n = ndimage.label(mask)
    return pieces == np.argmax(np.bincount(pieces.ravel())[1:]) + 1 if n else mask


def scaled(density, filled):
    """*density* rescaled over the tissue: 0 at its 10th percentile, 1 at its 98th (clipped)."""
    low, high = np.percentile(density[filled], (10, 98))
    return np.clip((density - low) / (high - low), 0, 1)


def through_tissue(filled, start):
    """How far each tissue pixel is from *start*, walking inside the tissue only.

    Inputs
      filled  (rows, cols) mask: the tissue
      start   (rows, cols) mask: where the walk starts
    Returns
      (rows, cols) float array: distance in pixels (a notch is walked round, not jumped); 0 off the tissue
    """
    costs, _ = MCP(np.where(filled, 1.0, 1e9), fully_connected=True).find_costs(
        [tuple(p) for p in np.argwhere(start)[::7]])
    return np.where(filled & (costs < 1e8), costs, 0.0)


def far_half(seed, distance):
    """The part of *seed* (mask) further than half its largest *distance* (map): its far half."""
    away = np.where(seed, distance, 0.0)
    return largest(seed & (away > 0.5 * away.max()))


def find_seeds(filled, density, other_density):
    """Place one seed deep inside each organoid.

    Inputs
      filled         (rows, cols) mask: the tissue
      density        (rows, cols) array: AF488 cell density (the labelled organoid)
      other_density  (rows, cols) array: AF555 cell density (the other organoid)
    Returns
      hot   (rows, cols) mask: the seed in the labelled organoid
      cold  (rows, cols) mask: the seed in the other organoid
    (The real code also has a fallback for sections without a usable AF555 core.)
    """
    labelled = scaled(density, filled)
    hot = largest((labelled > 0.5) & filled)                  # labelled core: the largest patch above half its range
    other = scaled(other_density, filled)
    core = largest((other > 0.5) & filled & ~hot & (other > labelled))   # other core: where AF555 is dense and outweighs AF488
    cold = far_half(core, through_tissue(filled, hot))        # keep each core's half furthest from the other seed,
    hot = far_half(hot, through_tissue(filled, cold))         # so both seeds start well inside their own organoid
    return hot, cold

# %% price | The price of cutting each pixel | microscopy_viewer/junction.py: line_cost
import numpy as np
from scipy import ndimage


def line_cost(nuclear_mean, filled, um, power=6):
    """Give every pixel a price for cutting through it: cheap on a dark line, 1 in plain tissue.

    Inputs
      nuclear_mean  (rows, cols) array: the nuclear stain of the overview (block means)
      filled        (rows, cols) mask: the tissue
      um            float: µm per pixel
      power         how strongly darkness lowers the price
    Returns
      (rows, cols) float array, 0.0007 to 1: the price; a pixel a fifth darker than its
      surroundings costs about a quarter
    """
    level = np.median(nuclear_mean[filled])
    data = np.where(filled, nuclear_mean, level)              # outside the tissue: the tissue's level, so the outline is not a "dark line"
    fine = ndimage.gaussian_filter(data, 8 / um)              # the stain, blurred just enough to merge single nuclei
    coarse = ndimage.gaussian_filter(data, 60 / um)           # its surroundings
    ratio = np.clip(fine / np.maximum(coarse, 1e-6), 0.3, 1.0)   # 1 = as bright as the surroundings, below 1 = darker
    return (ratio ** power).astype(np.float32)

# %% mincut | The minimum cut | microscopy_viewer/junction.py: _min_cut
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph


def min_cut(filled, hot, cold, cost):
    """Split the tissue in two between the seeds, along the cheapest possible cut.

    Every pixel is joined to its 8 neighbours; cutting a join costs the mean price of its
    two pixels. The cheapest set of joins whose removal separates the seeds is found as a
    maximum flow: each join is a pipe as wide as its price, water is pushed from one seed
    to the other until no route has room left, and the full pipes form the cut.

    Inputs
      filled  (rows, cols) mask: the tissue
      hot     (rows, cols) mask: the labelled seed
      cold    (rows, cols) mask: the other seed
      cost    (rows, cols) array: the price of each pixel (from line_cost)
    Returns
      (rows, cols) int array: 1 on the labelled seed's side, 2 on the other side, 0 off the tissue
    """
    index = np.full(filled.shape, -1)
    index[filled] = np.arange(int(filled.sum()))              # one numbered node per tissue pixel ...
    n = int(filled.sum())
    source, sink = n, n + 1                                   # ... plus one node for each seed
    H, W = filled.shape
    rows, cols, caps = [], [], []
    for dy, dx, weight in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 0.7071), (1, -1, 0.7071)):   # right, down and the two diagonals
        y0, x0 = np.mgrid[0:H - dy, max(0, -dx):W - max(0, dx)]
        a, b = index[y0, x0], index[y0 + dy, x0 + dx]
        ok = (a >= 0) & (b >= 0)
        width = (cost[y0, x0] + cost[y0 + dy, x0 + dx]) * 0.5 * weight * 1000      # pipe width = mean price (diagonals x 0.71)
        cap = np.maximum(width[ok].astype(np.int32), 1)
        rows += [a[ok], b[ok]]; cols += [b[ok], a[ok]]; caps += [cap, cap]
    for seed, node, outward in ((hot, source, True), (cold, sink, False)):          # seed pixels: joined to their node by pipes of no limit
        nodes = index[seed & filled]
        ends = np.full(nodes.size, node)
        rows.append(ends if outward else nodes); cols.append(nodes if outward else ends)
        caps.append(np.full(nodes.size, 2 ** 30, dtype=np.int32))
    pipes = sparse.csr_matrix((np.concatenate(caps), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(n + 2, n + 2), dtype=np.int32)
    flow = csgraph.maximum_flow(pipes, source, sink).flow     # push as much water as possible from seed to seed
    room = pipes - flow                                       # what each pipe can still carry
    room.data = np.maximum(room.data, 0); room.eliminate_zeros()
    reached = csgraph.breadth_first_order(room, source, return_predecessors=False)   # pixels water can still reach from the labelled seed
    side = np.zeros(n + 2, bool); side[reached] = True
    halves = np.zeros(filled.shape, np.uint8)
    halves[filled] = np.where(side[:n], 1, 2)                 # reached: labelled side; not reached: other side
    return halves

# %% bending | A price on bending: the dynamic programme | microscopy_viewer/junction.py: straighten_cut
import numpy as np
from scipy import ndimage


def straighten(filled, cost, hot, cold, um, bend=1000.0, station_um=20.0, step_um=5.0, max_steps=6):
    """Find the cheapest smooth curve between the seeds, paying for every turn it makes.

    The curve crosses the section one station at a time (a station every *station_um*
    across the section); at each station it sits at one position between the seeds. Its
    cost is the price of the pixels it crosses plus bend x (change of direction / 90°)² at
    every station. Dynamic programming finds the cheapest such curve exactly: station by
    station it keeps, for every position and direction, only the cheapest way to arrive.

    Inputs
      filled     (rows, cols) mask: the tissue
      cost       (rows, cols) array: the price of each pixel (from line_cost)
      hot, cold  (rows, cols) masks: the labelled seed and the other seed
      um         float: µm per pixel
      bend       µm: what a sharp 90° turn costs, as µm of plain tissue crossed
      station_um, step_um   µm between stations, and between positions at a station
      max_steps  int: how many positions the curve may move between two stations (sets the steepest direction)
    Returns
      (rows, cols) int array: 1 on the labelled seed's side of the curve, 2 on the other, 0 off the tissue
    """
    hot_at, cold_at = np.argwhere(hot & filled).mean(0), np.argwhere(cold & filled).mean(0)
    axis = cold_at - hot_at
    span = np.hypot(*axis)
    across = axis / span                                      # positions run along the seed-to-seed axis ...
    along = np.array([-across[1], across[0]])                 # ... and stations across it
    proj = (np.argwhere(filled) - hot_at) @ along
    station, step = station_um / um, step_um / um
    s = np.arange(proj.min() - station, proj.max() + 1.5 * station, station)   # station coordinates (pixels)
    o = np.arange(0.0, span + step / 2, step)                 # position coordinates (pixels from the labelled seed)

    price = np.where(filled, cost, 0.0).astype(np.float32)    # off the tissue: free, so the curve can end anywhere on the outline
    seeds = ((hot | cold) & filled).astype(np.float32)

    def sample(ss):                                           # the price at every (station, position)
        yx = (hot_at[:, None, None] + along[:, None, None] * ss[None, :, None]
              + across[:, None, None] * o[None, None, :]).reshape(2, -1)
        p = ndimage.map_coordinates(price, yx, order=1, cval=0.0).reshape(len(ss), len(o))
        on_seed = ndimage.map_coordinates(seeds, yx, order=0, cval=0.0).reshape(len(ss), len(o)) > 0
        return np.where(on_seed, 1e6, p)                      # crossing a seed: forbidden
    at, mid = sample(s), sample(s[:-1] + station / 2)         # at the stations, and halfway between them

    shifts = np.arange(-max_steps, max_steps + 1)             # the possible moves between stations (13 directions)
    length_um = np.hypot(station_um, shifts * step_um)        # µm of curve each move draws
    heading = np.arctan2(shifts * step_um, station_um)        # its direction
    turn = bend * ((heading[:, None] - heading[None, :]) / (np.pi / 2)) ** 2   # cost of turning: [arriving, leaving]

    value = np.zeros((len(o), len(shifts)))                   # cheapest cost so far into each (position, direction)
    came = []                                                 # for each, the direction it came from
    for i in range(len(s) - 1):
        totals = value[:, :, None] + (turn[None] if i else 0)              # + the cost of turning into each new direction
        choice = np.broadcast_to(totals.argmin(axis=1), (len(o), len(shifts)))
        leave = np.broadcast_to(totals.min(axis=1), (len(o), len(shifts)))  # cheapest way to leave each position in each direction
        new = np.full_like(value, np.inf)
        back = np.zeros(value.shape, int)
        for j, d in enumerate(shifts):
            k = np.arange(max(0, -d), min(len(o), len(o) - d))              # positions that can move d steps
            half = k + d / 2.0
            middle = 0.5 * (mid[i, np.floor(half).astype(int)] + mid[i, np.ceil(half).astype(int)])
            per_um = (at[i, k] + 2 * middle + at[i + 1, k + d]) / 4         # price under the stretch (both ends and middle)
            new[k + d, j] = leave[k, j] + per_um * length_um[j]             # + price x length of the stretch
            back[k + d, j] = choice[k, j]
        value = new
        came.append(back)

    k, j = np.unravel_index(np.argmin(value), value.shape)    # cheapest entry at the last station ...
    path = [k]
    for i in range(len(s) - 2, -1, -1):                       # ... followed back to the first station
        k, j = k - shifts[j], came[i][k, j]
        path.append(k)
    curve = o[np.array(path[::-1])]                           # the curve: one position per station

    rows, cols = np.nonzero(filled)                           # every tissue pixel: which side of the curve is it on?
    ps = (rows - hot_at[0]) * along[0] + (cols - hot_at[1]) * along[1]
    po = (rows - hot_at[0]) * across[0] + (cols - hot_at[1]) * across[1]
    labelled_side = po < np.interp(ps, s, curve)
    halves = np.zeros(filled.shape, np.uint8)
    halves[rows, cols] = np.where(labelled_side, 1, 2)
    return halves

# %% line_band | From the cut to a line, from the line to a band | microscopy_viewer/junction.py: trace_interface, _build_band
import numpy as np
from scipy import ndimage
from skimage.graph import MCP
from skimage.morphology import skeletonize


def trace_line(halves, um, smooth_um=150):
    """Turn the boundary between the two halves into an ordered, smooth line.

    Inputs
      halves     (rows, cols) int array: 1 labelled side, 2 other side (from straighten)
      um         float: µm per pixel
      smooth_um  µm: length of the running mean that smooths the line
    Returns
      (n, 2) float array: the line, as points (row, col) in pixels, from one end to the other
    """
    touching = (halves == 1) & ndimage.binary_dilation(halves == 2)   # labelled pixels next to the other side
    curve = skeletonize(ndimage.binary_dilation(touching))           # thinned to a line one pixel wide
    walk_cost = np.where(curve, 1.0, 1e9)

    def furthest(start):                                      # the point of the line furthest from *start*, walking along it
        walk = MCP(walk_cost, fully_connected=True)
        costs, _ = walk.find_costs([tuple(start)])
        costs = np.where(curve & (costs < 1e8), costs, -1.0)
        return np.unravel_index(int(np.argmax(costs)), costs.shape), walk
    one_end, _ = furthest(np.argwhere(curve)[0])
    other_end, walk = furthest(one_end)                       # the two ends: the pair of points furthest apart
    path = np.asarray(walk.traceback(other_end), float)       # every pixel from one end to the other, in order
    window = max(3, int(round(smooth_um / um)))
    return np.column_stack([ndimage.uniform_filter1d(path[:, a], window, mode="nearest") for a in (0, 1)])


def band(line_px, halves, um, width_um=300, gap_um=2):
    """Make the band: the other organoid's tissue within a set distance of the line.

    Inputs
      line_px   (n, 2) array: the junction (row, col) in pixels (from trace_line)
      halves    (rows, cols) int array: 1 labelled side, 2 other side
      um        float: µm per pixel
      width_um  µm: how far from the line the band reaches
      gap_um    µm: a thin strip right on the line left out
    Returns
      (rows, cols) mask: the band (the real code also rounds its corners and adds a thin rim outside the outline)
    """
    from skimage.draw import line as draw
    on_line = np.zeros(halves.shape, bool)
    for (y0, x0), (y1, x1) in zip(line_px[:-1].round().astype(int), line_px[1:].round().astype(int)):
        on_line[draw(y0, x0, y1, x1)] = True
    distance = ndimage.distance_transform_edt(~on_line) * um  # every pixel: its distance to the line in µm
    return (halves == 2) & (distance >= gap_um) & (distance <= gap_um + width_um)

# %% cells | Finding the cells | microscopy_viewer/segmentation.py; notebooks/validation/count_cells.py
import numpy as np
from cellpose import models
from scipy import ndimage
from skimage.segmentation import expand_labels


def measure_cells(dapi, af488, um):
    """Find every nucleus and measure the AF488 of the cell around it.

    Inputs
      dapi   (Y, X) array: the nuclear stain at full resolution (brightest Z plane per pixel)
      af488  (Y, X) array: the AF488 channel, same grid
      um     float: µm per full-resolution pixel
    Returns
      dict of 1-D arrays, one value per cell:
        centre         (n, 2) the nucleus centre (row, col) in pixels
        cell_mean      mean AF488 over the nucleus grown by 3 µm (the cell)
        ring_mean      mean AF488 in a ring 3–6 µm around the nucleus (the cell's local background)
        filtered_mean  mean AF488 over the cell after subtracting the 45 µm local median
    """
    model = models.CellposeModel(gpu=True)                    # Cellpose-SAM
    nuclei = model.eval(dapi)[0]                              # (Y, X) int: 0 background, 1..n one label per nucleus
    grown = expand_labels(nuclei, distance=round(3 / um))     # each nucleus grown by 3 µm: the cell
    outer = expand_labels(nuclei, distance=round(6 / um))
    ring = np.where(grown == 0, outer, 0)                     # the band between 3 and 6 µm out
    ids = np.unique(nuclei[nuclei > 0])
    filtered = af488 - ndimage.median_filter(af488, int(round(45 / um)) | 1)   # AF488 minus its local median (not clipped at 0)
    return {"centre": np.array(ndimage.center_of_mass(nuclei > 0, nuclei, ids)),
            "cell_mean": ndimage.mean(af488, grown, ids),
            "ring_mean": ndimage.mean(af488, ring, ids),
            "filtered_mean": ndimage.mean(filtered, grown, ids)}

# %% positivity | Calling a cell positive | notebooks/validation/count_common.py: rule_band_ksd, rule_contrast
import numpy as np


def robust(values):
    """Median of *values* (1-D array) and the robust standard deviation (1.4826 x the median absolute deviation)."""
    centre = np.median(values)
    return centre, 1.4826 * np.median(np.abs(values - centre))


def positive_cells(cells, in_band, k=7):
    """Call each cell AF488-positive or not.

    Inputs
      cells    dict from measure_cells (uses filtered_mean, cell_mean, ring_mean)
      in_band  1-D True/False array: which cells lie in the band
      k        how many robust SDs above the band's median a positive cell must be
    Returns
      1-D True/False array, one per cell: positive by the calibrated rule
      (the band's cells set the threshold: median + 7 robust SDs of their filtered AF488)
    """
    centre, sd = robust(cells["filtered_mean"][in_band])
    positive = cells["filtered_mean"] > centre + k * sd
    # The check used alongside: a cell at least 1.5 x as bright as its own ring.
    contrast = cells["cell_mean"] / cells["ring_mean"]
    positive_by_contrast = contrast >= 1.5
    return positive

# %% crossed | Cells that crossed the line | notebooks/validation/count_common.py: _side, home_is_band_side, region
import numpy as np
from scipy.spatial import cKDTree


def signed_distance(points, line):
    """Measure each point's distance to a line, and on which side it lies.

    Inputs
      points  (n, 2) array: positions (row, col) in µm
      line    (m, 2) array: the junction (row, col) in µm
    Returns
      distance  (n,) array: µm to the line, + on one side and − on the other
      beside    (n,) True/False: the point lies alongside the line, not beyond either of its ends
    """
    fine = np.vstack([np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / 2))) for a, b in zip(line[:-1], line[1:])])   # a point every 2 µm
    distance, nearest = cKDTree(fine).query(points)           # the nearest point of the line
    k = np.clip(nearest, 1, len(fine) - 1)
    tangent, to_point = fine[k] - fine[k - 1], points - fine[nearest]
    side = np.sign(tangent[:, 0] * to_point[:, 1] - tangent[:, 1] * to_point[:, 0])   # left or right of the line's direction
    return distance * side, (nearest > 0) & (nearest < len(fine) - 1)


def crossed_cells(cells_um, contrast, positive, line_um, reach_um=300, gap_um=2):
    """Find the positive cells that lie past the line, in the organoid AF488 does not come from.

    Inputs
      cells_um  (n, 2) array: cell centres (row, col) in µm
      contrast  (n,) array: each cell's AF488 over its ring (from measure_cells)
      positive  (n,) True/False: AF488-positive cells (from positive_cells)
      line_um   (m, 2) array: the junction in µm
      reach_um  µm: how far past the line a cell is still counted
    Returns
      (n,) True/False: crossed cells. Their number divided by the line's length gives "crossed per mm".
    """
    signed, beside = signed_distance(cells_um, line_um)
    near = beside & (np.abs(signed) <= reach_um)
    # The labelled organoid ("home") is the side whose brightest cells stand out most.
    home_on_plus = np.percentile(contrast[near & (signed > 0)], 95) > np.percentile(contrast[near & (signed < 0)], 95)
    away = -signed if home_on_plus else signed                # + = µm into the other organoid
    return positive & beside & (away > gap_um) & (away <= reach_um)

# %% counting | Counting fairly | notebooks/dataset/01_whole_dataset.ipynb
import numpy as np
import statsmodels.formula.api as smf


def compare_genotypes(sections):
    """Compare crossed cells per mm between genotypes, counting slides fairly.

    Inputs
      sections  DataFrame, one row per section: crossed_per_mm, Genotype (GD1 / WT3),
                Experiment (batch) and slide
    Returns
      fold       how many times more crossed cells per mm WT3 has than GD1, adjusted for experiment
      low, high  its 95 % confidence interval
    Sections cut from the same slide are not independent, so slide is a random effect.
    """
    fit = smf.mixedlm("np.log1p(crossed_per_mm) ~ C(Genotype) + C(Experiment)",
                      sections, groups=sections["slide"]).fit()
    fold = np.exp(fit.params["C(Genotype)[T.WT3]"])
    low, high = np.exp(fit.conf_int().loc["C(Genotype)[T.WT3]"])
    return fold, low, high
