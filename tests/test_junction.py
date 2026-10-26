"""Checks for the automatic junction and its quality flags.

No files and no GUI: the samples are drawn — two discs of "tissue" that overlap,
one of them sprinkled with bright points — so where the junction should be is
known, and so is which side the band belongs on.

Run with::

    python tests/test_junction.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import fusion as fu  # noqa: E402
from microscopy_viewer import junction as jn  # noqa: E402

_failures: list[str] = []

#: The drawn samples are made at the overview's own scale: 5 µm pixels.
UM = 5.0


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _disc(shape, centre, radius) -> np.ndarray:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    return (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= radius ** 2


def _assembloid(gap: int = 0, seed: int = 0, holes: bool = False, faint_right: bool = False,
                seam: bool = True):
    """Two organoids side by side; the left one carries the labelled cells.

    With *seam* the nuclear stain is darker along the junction, as where two
    real organoids are pressed together.

    With *faint_right* the right organoid's nuclear stain is barely above the
    slide, but its own label (red) is bright.

    Returns reader-like specs (red, GFP, DAPI) and the column the junction is at.
    """
    rng = np.random.default_rng(seed)
    shape = (240, 360)
    left = _disc(shape, (120, 110 - gap), 80)
    right = _disc(shape, (120, 250 + gap), 80)
    tissue = left | right
    if holes:
        tissue &= ~_disc(shape, (120, 250 + gap), 45)

    dapi = np.where(tissue, 3000.0, 200.0) + rng.normal(0, 30, shape)
    if seam:
        # Two organoids pressed together leave a thin dark line in the nuclear stain.
        cols = np.arange(shape[1])[None, :]
        dapi = np.where(tissue & (np.abs(cols - 180) <= 1), 0.5 * dapi, dapi)
    gfp = np.where(tissue, 400.0, 200.0) + rng.normal(0, 15, shape)
    cells = np.zeros(shape, dtype=bool)
    points = np.argwhere(left & ~right)
    chosen = points[rng.choice(len(points), 500, replace=False)]
    cells[chosen[:, 0], chosen[:, 1]] = True
    gfp[cells] += 4000.0
    red = np.where(tissue, 300.0, 200.0) + rng.normal(0, 15, shape)
    if faint_right:
        dapi = np.where(right & ~left, 260.0, dapi) + rng.normal(0, 30, shape) * (right & ~left)
        red = np.where(right & ~left, 1500.0, red)

    def spec(name, plane):
        data = np.clip(plane, 1, 65535).astype(np.uint16)[None]  # (Z=1, Y, X)
        return SimpleNamespace(data=data, scale=(4.0, UM, UM), multiscale=False,
                               channel_name=name, name=name)

    return [spec("AF555", red), spec("AF488", gfp), spec("AF405", dapi)], 180


def test_channels_are_found_by_name() -> None:
    print("picking the channels")
    names = ["AF555", "AF488", "AF405"]
    check(jn.pick_channel(names, "", jn.NUCLEAR_HINTS) == 2, "405 is the nuclear channel")
    check(jn.pick_channel(names, "", jn.LABEL_HINTS) == 1, "488 is the labelled-cell channel")
    check(jn.pick_channel(names, "555", jn.LABEL_HINTS) == 0, "a named channel wins over the hints")
    check(jn.pick_channel(["a", "b"], "", jn.NUCLEAR_HINTS) is None, "no match is no channel, not channel 0")
    try:
        jn.build_overview([SimpleNamespace(data=np.ones((1, 8, 8)), scale=(1, 1, 1),
                                           multiscale=False, channel_name="brightfield", name="x")])
        check(False, "a sample with no nuclear channel is refused")
    except ValueError as exc:
        check("nuclear" in str(exc), f"a sample with no nuclear channel is refused ({exc})")


def test_the_junction_is_found_between_the_organoids() -> None:
    print("finding the junction")
    specs, column = _assembloid()
    overview, result = jn.analyse_specs(specs)
    check(overview.step == 1 and overview.um == UM, "5 µm pixels are used as they are")
    check(result.tissue[120, 110] and result.tissue[120, 250] and not result.tissue[5, 5],
          "both organoids are tissue and the background is not")
    check(result.halves is not None and result.halves[120, 60] == 1 and result.halves[120, 300] == 2,
          "the half with the labelled cells is 1, the other 2")
    check(result.line_um is not None, "a junction comes back")
    x = result.line_um[:, 1] / UM
    y = result.line_um[:, 0] / UM
    check(abs(float(np.median(x)) - column) < 15, f"at the neck between them (x={np.median(x):.0f}, expected {column})")
    check(y.max() - y.min() > 40, "running across the neck rather than along it")
    check(result.status == jn.STATUS_OK and not result.flags, f"and nothing is flagged ({result.flags})")

    band = result.band
    cols = np.flatnonzero(band.any(axis=0))
    check(cols.min() > column - 15, "the band is on the organoid the cells migrate into")
    from scipy import ndimage

    outside = ndimage.distance_transform_edt(~result.filled) * UM
    check(float(outside[band].max()) <= fu.TISSUE_MARGIN_UM + UM,
          "and only on tissue, or within the margin past its edge")
    check((band & (outside > 0)).any(), "which it does reach, along the organoid's side")
    bare = jn.analyse_specs(specs, jn.JunctionOptions(margin_um=0.0))[1].band
    check(not (bare & ~result.filled).any() and band.sum() > bare.sum(),
          "with no margin it stops at the tissue's edge, and is smaller")
    # The band is the far organoid's own tissue within the band width, so its
    # ends follow that organoid's outline rather than being cut straight across.
    from scipy import ndimage

    curve = fu.fit_spline(result.line_um, UM) / UM
    on_line = jn._draw_curve(curve, band.shape)
    distance = ndimage.distance_transform_edt(~on_line) * UM
    check(float(distance[band].max()) <= 300 + 2 + 3 * UM, "nothing in it is further from the junction than the band width")
    reachable = (result.halves == 2) & (distance <= 300) & (distance >= 2 + 2 * jn.SMOOTH_UM)
    check((band & reachable).sum() >= 0.93 * reachable.sum(),
          "and all the organoid's tissue within that distance is in it, out to its outline")
    edge = band & ~ndimage.binary_erosion(band)
    # The outline, out to the margin the band reaches past it.
    reach = int(np.ceil(fu.TISSUE_MARGIN_UM / UM)) + 1
    outline = ndimage.binary_dilation(result.filled, iterations=reach) & ~ndimage.binary_erosion(result.filled, iterations=3)
    check((edge & outline).sum() > 40, "so its two ends run along the organoid's outline")
    check(ndimage.label(band)[1] == 1, "in one piece")

    # At full resolution the band is filled from its outline, not enlarged.
    overview, _ = jn.analyse_specs(specs)
    full = jn.full_resolution_band(result, overview)
    check(full.shape == overview.shape_yx and set(np.unique(full)) == {0, 1}, "the stored band is on the image's grid")
    both = (full > 0) | band
    check(((full > 0) & band).sum() / both.sum() > 0.93, "and is the same band")
    check(abs(result.metrics["Junction length (um)"] - 2 * 37 * UM) < 150,
          f"the junction length is in µm ({result.metrics['Junction length (um)']:.0f})")
    check(result.metrics["Band tissue area (mm2)"] > 0, "the band's tissue area is measured")
    check(result.metrics["Density contrast"] > 1.8, "the two sides differ in labelled cells")

    _overview, inside = jn.analyse_specs(specs, jn.JunctionOptions(into_unlabelled=False))
    check(np.flatnonzero(inside.band.any(axis=0)).max() < column + 15,
          "asking for the labelled side puts the band there instead")


def test_an_organoid_faint_in_the_nuclear_stain_is_still_tissue() -> None:
    print("an organoid faint in the nuclear stain")
    specs, column = _assembloid(faint_right=True)
    _overview, old = jn.analyse_specs(specs, jn.JunctionOptions(tissue_all_channels=False))
    check(not old.tissue[120, 300], "from the nuclear stain alone the faint organoid is lost")
    _overview, result = jn.analyse_specs(specs)
    check(result.tissue[120, 300] and result.tissue[120, 110] and not result.tissue[5, 5],
          "from every channel both organoids are tissue, and the slide is not")
    check(result.line_um is not None, "a junction comes back")
    if result.line_um is not None:
        x = result.line_um[:, 1] / UM
        y = result.line_um[:, 0] / UM
        check(abs(float(np.median(x)) - column) < 15, f"at the neck (x={np.median(x):.0f}, expected {column})")
        check(y.max() - y.min() > 40, f"and across it ({y.max() - y.min():.0f} px)")


def test_the_labelled_seed_can_cover_every_patch() -> None:
    print("a labelled organoid with two dense patches")
    filled = _disc((200, 300), (100, 150), 90)
    density = np.zeros(filled.shape, np.float32)
    density[_disc(filled.shape, (100, 85), 22)] = 1.0     # two patches of labelled cells, far apart ...
    density[_disc(filled.shape, (40, 150), 15)] = 0.9     # ... the second a little smaller
    other = np.zeros(filled.shape, np.float32)
    other[_disc(filled.shape, (150, 200), 25)] = 1.0      # the other organoid's own label
    core = jn.find_seeds(filled, density, other)
    both = jn.find_seeds(filled, density, other, patches="patches")
    pulled = jn.find_seeds(filled, density, other, patches="patches_pulled")
    check(not core[0][40, 150] and core[0].sum() > 0, "by default the labelled seed is one patch only")
    check(both[0][40, 150] and both[0][100, 85], "with all patches it covers both")
    check(not (pulled[0] & ~both[0]).any() and pulled[0][40, 150] and pulled[0][100, 85],
          "pulled back, each patch keeps (at least) its far part — here, far from the other seed, all of it")
    check(np.array_equal(core[1], both[1]), "the other seed is not changed")


def test_with_no_seam_the_seeds_can_come_from_the_labels() -> None:
    print("no dark line at the junction")
    specs, column = _assembloid()
    overview, _ = jn.analyse_specs(specs)
    channels = overview.channels
    valid = (channels > 0).all(axis=0)
    tissue = jn.tissue_mask(channels[overview.nuclear], valid, UM)
    filled = tissue.copy()
    density = jn.cell_density(channels[overview.label], tissue, filled, UM)
    found = jn.dominant_seeds(filled, density)
    check(found is not None, "dominant seeds are found")
    if found is not None:
        hot, cold, rule = found
        check(not (hot & cold).any() and rule == jn.SEED_DOMINANT, "two seeds that do not overlap")
        check(float(np.argwhere(hot)[:, 1].mean()) < column < float(np.argwhere(cold)[:, 1].mean()),
              "the labelled one left of the junction, the other right of it")
        core_hot, core_cold, _ = jn.find_seeds(filled, density)
        check(hot.sum() > core_hot.sum() and cold.sum() > core_cold.sum(), "each larger than the dense core")
    _overview, again = jn.analyse_specs(specs, jn.JunctionOptions(no_seam_price=0.01))
    check(again.metrics.get("No-seam fallback") == 1.0, "a cut dearer than the no-seam price is made again")
    if again.line_um is not None:
        x = again.line_um[:, 1] / UM
        check(abs(float(np.median(x)) - column) < 15, f"and still finds the neck (x={np.median(x):.0f})")
    check("no dark line at the junction" in jn.quality_flags({"Tissue area (mm2)": 1.0, "Junction length (um)": 500.0,
                                                               "Seam price": jn.NO_SEAM_PRICE + 0.1}),
          "a cut with no dark line under it is flagged")
    check("no dark line at the junction" not in jn.quality_flags({"Tissue area (mm2)": 1.0, "Junction length (um)": 500.0,
                                                                   "Seam price": 0.1}),
          "one along a dark line is not")


def test_a_drawn_line_gets_a_band_that_follows_the_tissue() -> None:
    print("a band for a line drawn by hand, following the tissue")
    specs, column = _assembloid()
    overview = jn.build_overview(specs)
    filled = jn.tissue_of(overview)
    check(filled[120, 110] and filled[120, 250] and not filled[5, 5], "the section's outline is found")
    # Drawn top to bottom down the neck, short of both edges of the tissue.
    line = np.array([[100.0, column], [140.0, column]]) * UM
    sides = {}
    for side in (fu.SIDE_LEFT, fu.SIDE_RIGHT):
        band = jn.band_along_tissue(overview, filled, line, side, width_um=200.0) > 0
        sides[side] = band
        # Out to the margin past the tissue's edge, and a pixel for filling its outline.
        from scipy import ndimage

        near = ndimage.binary_dilation(filled, iterations=int(np.ceil(fu.TISSUE_MARGIN_UM / UM)) + 2)
        check(band.any() and not (band & ~near).any(), f"{side}: a band, on tissue only")
        cols = np.flatnonzero(band.any(axis=0))
        on_right = cols.min() >= column - 3
        check(on_right or cols.max() <= column + 3, f"{side}: all on one side of the line")
        rows = np.flatnonzero(band.any(axis=1))
        check(rows.min() < 95 and rows.max() > 145,
              f"{side}: running past the drawn ends out to the outline (rows {rows.min()}–{rows.max()})")
    left_cols = np.flatnonzero(sides[fu.SIDE_LEFT].any(axis=0))
    right_cols = np.flatnonzero(sides[fu.SIDE_RIGHT].any(axis=0))
    check((left_cols.mean() - column) * (right_cols.mean() - column) < 0, "left and right are opposite sides")
    strip, _ = fu.fusion_region(line, filled.shape, (UM, UM), width_um=200.0, side=fu.SIDE_LEFT)
    check(((strip > 0) == sides[fu.SIDE_LEFT]).mean() < 1.0 and
          np.sign(np.flatnonzero((strip > 0).any(axis=0)).mean() - column) == np.sign(left_cols.mean() - column),
          "on the same side as the macro's strip for that setting")
    both = jn.band_along_tissue(overview, filled, line, fu.SIDE_BOTH, width_um=200.0)
    check(set(np.unique(both)) == {0, 1, 2}, "both sides: 1 on the left, 2 on the right")


def test_bending_costs_so_a_crack_round_a_corner_is_not_followed() -> None:
    print("a crack turning off the junction at a right angle")
    shape = (200, 300)
    filled = np.ones(shape, dtype=bool)
    hot = np.zeros(shape, dtype=bool)
    hot[:, :20] = True
    cold = np.zeros(shape, dtype=bool)
    cold[:80, 260:] = True
    cost = np.ones(shape, dtype=np.float32)
    cost[:100, 148:153] = 0.05   # the junction, clear in its upper half
    cost[100:, 148:153] = 0.3    # and faint in its lower half
    cost[98:103, 150:] = 0.05    # a crack turning off it at 90° to the edge
    density = np.zeros(shape, dtype=np.float32)

    free = jn.split_tissue(filled, density, UM, cost, seeds=(hot, cold), bend_um=0.0)
    check(free is not None and free[180, 250] == 1,
          "with bending free, the cut turns the corner along the crack")
    stiff = jn.split_tissue(filled, density, UM, cost, seeds=(hot, cold), bend_um=jn.BEND_UM)
    check(stiff is not None and stiff[180, 250] == 2 and stiff[180, 100] == 1,
          "with bending costed, it stays on the junction down to the edge")
    line = jn.trace_interface(stiff, UM)
    cols = line[:, 1]
    check(line is not None and float(np.abs(cols - 150).max()) < 12,
          f"and runs straight down it (columns {cols.min():.0f}–{cols.max():.0f})")


def test_a_tear_is_not_a_cheap_road_and_the_cut_sits_where_the_cells_change() -> None:
    print("a tear running from one organoid towards the other")
    shape = (200, 300)
    filled = np.ones(shape, dtype=bool)
    hot = np.zeros(shape, dtype=bool)
    hot[:, :20] = True
    cold = np.zeros(shape, dtype=bool)
    cold[:, 280:] = True
    # The stain: tissue at 1000, the junction a dim line at column 150 (0.7 of it),
    # and a tear — empty — from the top edge down to row 160 at column 100.
    stain = np.full(shape, 1000.0, dtype=np.float32)
    stain[:, 148:153] = 700.0
    stain[:160, 98:103] = 20.0
    tissue = stain > 100
    torn = jn.tear_mask(stain, tissue, filled, UM)
    check(torn[50, 100] and not torn[50, 150] and not torn[50, 200],
          "the tear is found; the junction line and plain tissue are not")
    # Labelled cells on the left of the junction, none on the right.
    density = np.where(np.arange(shape[1]) < 150, 1.0, 0.0)[None, :].repeat(shape[0], axis=0).astype(np.float32)
    from scipy import ndimage

    density = ndimage.gaussian_filter(density, 20)
    edge = jn.edge_cost(density, filled, hot, cold)
    check(edge[100, 150] < 0.1 and edge[100, 30] > 0.8 and edge[100, 270] > 0.8,
          "the edge cost is low where the cells change, high at either organoid")

    cost = jn.line_cost(stain, filled, UM)
    by_tear = jn.split_tissue(filled, density, UM, cost, seeds=(hot, cold))
    check(by_tear is not None and by_tear[100, 120] == 2,
          "with tears cheap, the cut runs down the tear (column 100), short of the junction")
    priced = cost.copy()
    priced[torn] = np.maximum(priced[torn], 1.0)
    priced += 0.5 * edge
    right = jn.split_tissue(filled, density, UM, priced, seeds=(hot, cold), bend_um=jn.BEND_UM)
    line = jn.trace_interface(right, UM) if right is not None else None
    check(line is not None and float(np.median(np.abs(line[:, 1] - 150))) < 6,
          "with the tear priced and the cells' edge costed, it runs down the junction")


def test_the_panel_opens_with_the_finders_own_settings() -> None:
    print("the Fusion region panel's defaults")
    from microscopy_viewer.widgets import fusion_widget as fw

    options = jn.JunctionOptions()
    check(fw.BEND_DEFAULT == options.bend_um, f"bending cost {fw.BEND_DEFAULT} = {options.bend_um}")
    check(fw.TEARS_DEFAULT == (options.tear_price > 0), f"tears {fw.TEARS_DEFAULT} = price {options.tear_price}")
    check(fw.EDGE_DEFAULT == options.edge_weight, f"cells' edge {fw.EDGE_DEFAULT} = {options.edge_weight}")
    check(fw.WORKERS_DEFAULT == jn.RUN_WORKERS, f"samples at once {fw.WORKERS_DEFAULT} = {jn.RUN_WORKERS}")
    check(fw.TISSUE_ALL_DEFAULT == options.tissue_all_channels,
          f"tissue in every channel {fw.TISSUE_ALL_DEFAULT} = {options.tissue_all_channels}")


def test_the_second_seed_comes_from_the_other_organoids_label() -> None:
    print("the second seed")
    names = ["AF555", "AF488", "AF405"]
    check(jn.pick_other_channel(names, "", 2, 1) == 0, "the other organoid's channel is found by name")
    check(jn.pick_other_channel(["Cy5", "AF488", "AF405"], "", 2, 1) == 0, "or is whichever channel is left")
    check(jn.pick_other_channel(["AF488", "AF405"], "", 1, 0) is None, "and is none when there are only two")
    check(jn.pick_other_channel(names, "405", 2, 1) is None, "the nuclear stain is never taken for it")

    specs, junction = _assembloid()
    overview, plain = jn.analyse_specs(specs)
    check(plain.second_seed == jn.SEED_FAR,
          "a channel with no cells in it is not used: its noise is not a dense core")
    tissue = plain.tissue
    check(not jn.has_cells(overview.channels[overview.other], tissue, overview.um)
          and jn.has_cells(overview.channels[overview.label], tissue, overview.um),
          "cells are told from an empty channel's own texture")

    # The right organoid labelled too, but only along its far edge.
    rng = np.random.default_rng(3)
    red = specs[0].data[0].astype(float)
    right = _disc(red.shape, (120, 250), 80) & (np.arange(red.shape[1])[None, :] > 270)
    points = np.argwhere(right)
    chosen = points[rng.choice(len(points), 250, replace=False)]
    red[chosen[:, 0], chosen[:, 1]] += 4000.0
    specs[0].data = np.clip(red, 1, 65535).astype(np.uint16)[None]

    overview, both = jn.analyse_specs(specs)
    check(both.second_seed == jn.SEED_CHANNEL, f"with cells in it, the second seed is their dense core ({both.second_seed})")
    labelled, other = both.seeds
    columns = np.nonzero(other)[1]
    check(columns.min() > junction, f"which lies in the other organoid (columns {columns.min()}-{columns.max()})")
    check(np.nonzero(labelled)[1].max() < junction and not (labelled & other).any(),
          "the labelled seed stays in its own, and the two do not overlap")
    x = both.line_um[:, 1] / UM
    check(abs(float(np.median(x)) - junction) <= 6, f"and the junction is still at the neck (x={np.median(x):.0f})")
    check(set(np.unique(both.halves[:, : junction - 20])) <= {0, 1}, "with the labelled organoid as half 1")

    _overview, off = jn.analyse_specs(specs, jn.JunctionOptions(seed_from_other_channel=False))
    check(off.second_seed == jn.SEED_FAR, "switched off, the second seed is the far tissue again")
    _overview, named = jn.analyse_specs(specs, jn.JunctionOptions(other_channel="555"))
    check(named.second_seed == jn.SEED_CHANNEL, "the channel can be named")

    # Labelled cells of both kinds in the same place say nothing about the other organoid.
    filled = plain.filled
    same = jn.find_seeds(filled, plain.density, plain.density)
    check(same is not None and same[2] == jn.SEED_FAR,
          "a label that sits where the first one does is not a second seed")


def test_the_junction_follows_the_dark_line_in_the_nuclear_stain() -> None:
    print("following the line between the organoids")
    # One block of tissue with no neck at all. The labelled cells stop at a
    # third of the way across; the dark line between the organoids is at 60 %.
    rng = np.random.default_rng(5)
    shape = (200, 300)
    tissue = np.zeros(shape, dtype=bool)
    tissue[30:170, 30:270] = True
    dapi = np.where(tissue, 3000.0, 200.0) + rng.normal(0, 30, shape)
    dapi[30:170, 172:176] = 1500.0  # the line: 20 µm wide, half as bright
    gfp = np.where(tissue, 400.0, 200.0) + rng.normal(0, 15, shape)
    points = np.argwhere(tissue & (np.arange(shape[1])[None, :] < 110))
    chosen = points[rng.choice(len(points), 400, replace=False)]
    gfp[chosen[:, 0], chosen[:, 1]] += 4000.0

    def spec(name, plane):
        return SimpleNamespace(data=np.clip(plane, 1, 65535).astype(np.uint16)[None],
                               scale=(4.0, UM, UM), multiscale=False, channel_name=name, name=name)

    specs = [spec("AF488", gfp), spec("AF405", dapi)]
    _overview, followed = jn.analyse_specs(specs)
    x = followed.line_um[:, 1] / UM
    check(followed.line_um is not None and abs(float(np.median(x)) - 174) <= 4,
          f"the junction is on the line, not where the cells thin out (x={np.median(x):.0f}, line at 174)")
    check(float(x.max() - x.min()) <= 8, "and stays on it from one side to the other")
    y = followed.line_um[:, 0] / UM
    check(y.max() - y.min() > 100, "across the whole section")

    cost = jn.line_cost(dapi, tissue, UM)
    check(float(cost[100, 174]) < 0.3 * float(cost[100, 100]),
          f"the line is several times cheaper to cut along than plain tissue ({cost[100, 174]:.2f})")
    dim = np.where(tissue, 3000.0, 200.0)
    dim[:, 150:] *= 0.5
    flat = jn.line_cost(dim, tissue, UM)
    check(abs(float(flat[100, 100]) - float(flat[100, 230])) < 0.05,
          "a dim organoid costs the same to cut as a bright one: only the line is cheap")


def test_aberrant_samples_are_flagged() -> None:
    print("quality flags")
    apart, _ = _assembloid(gap=30)
    _overview, result = jn.analyse_specs(apart)
    check("organoids not fused" in result.flags and result.status == jn.STATUS_FAIL,
          f"two organoids that do not touch fail ({result.flags})")

    torn, _ = _assembloid(holes=True)
    _overview, result = jn.analyse_specs(torn)
    check("large holes or torn" in result.flags, f"a section with a large hole is flagged ({result.flags})")
    check(result.status == jn.STATUS_CHECK, "as something to check, not a failure")

    empty = [SimpleNamespace(data=np.full((1, 60, 60), 200, np.uint16), scale=(1, UM, UM),
                             multiscale=False, channel_name=name, name=name)
             for name in ("AF488", "AF405")]
    _overview, result = jn.analyse_specs(empty)
    check(result.flags == ["no tissue found"] and result.status == jn.STATUS_FAIL,
          f"an empty field fails as having no tissue ({result.flags})")

    flags = jn.quality_flags({"Tissue area (mm2)": 1.0, "Junction length (um)": 900.0,
                              "Density contrast": 1.2, "Labelled share of tissue": 0.5,
                              "Band tissue area (mm2)": 0.2})
    check(flags == ["labelled cells on both sides (junction uncertain)"],
          "labelled cells on both sides makes the junction uncertain")
    flags = jn.quality_flags({"Tissue area (mm2)": 1.0, "Cut off by the scan": 0.2,
                              "Saturated fraction": 0.05, "Solidity": 0.5})
    check({"cut off by the scan edge", "saturated", "ragged outline", "no junction found"} <= set(flags),
          f"scan, saturation and outline problems are named ({flags})")


def test_outliers_are_judged_against_the_folder() -> None:
    print("cohort flags")

    def outcome(name, area, focus=1.0, stain=1000.0):
        return jn.JunctionOutcome(path=Path(name), name=name, status=jn.STATUS_OK, metrics={
            "Tissue area (mm2)": area, "Focus": focus, "Nuclear stain (mean)": stain})

    rng = np.random.default_rng(1)
    cohort = [outcome(f"s{i}", float(rng.normal(4.0, 0.3)), focus=float(rng.normal(1.0, 0.1)))
              for i in range(20)]
    cohort.append(outcome("tiny", 0.2))
    cohort.append(outcome("blurred", 4.0, focus=0.01))
    cohort.append(outcome("sharp", 4.0, focus=50.0))
    jn.flag_cohort(cohort)
    by_name = {o.name: o for o in cohort}
    check("size unlike the rest" in by_name["tiny"].flags, "a section a twentieth the size is an outlier")
    check("out of focus" in by_name["blurred"].flags, "so is one far blurrier than the rest")
    check("out of focus" not in by_name["sharp"].flags, "being sharper than the rest is not a fault")
    check(by_name["tiny"].status == jn.STATUS_CHECK and by_name["s0"].status == jn.STATUS_OK,
          "outliers are to check; the rest stay ok")
    normal = sum(1 for o in cohort[:20] if o.flags)
    check(normal <= 1, f"ordinary samples are left alone ({normal} of 20 flagged)")

    few = [outcome(f"s{i}", 4.0) for i in range(3)] + [outcome("tiny", 0.2)]
    jn.flag_cohort(few)
    check(not few[-1].flags, "a handful of samples is too few to call anything an outlier")


def test_a_folder_run_stores_the_band_and_writes_the_report() -> None:
    print("a run over files")
    import tifffile

    from microscopy_viewer import ims_store, loaders

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        paths = []
        for index, gap in enumerate((0, 0, 30)):
            specs, _column = _assembloid(gap=gap, seed=index)
            stack = np.stack([np.asarray(spec.data) for spec in specs])  # (C, Z, Y, X)
            path = root / f"assembloid_s{index}.ome.tiff"
            tifffile.imwrite(str(path), stack, ome=True, metadata={
                "axes": "CZYX", "Channel": {"Name": ["AF555", "AF488", "AF405"]},
                "PhysicalSizeX": UM, "PhysicalSizeY": UM, "PhysicalSizeZ": 4.0})
            paths.append(path)

        seen: list[str] = []
        options = jn.JunctionOptions(width_um=150.0, qc_dir=root / "junction_QC")
        outcomes = jn.run(paths, options, progress=seen.append)
        check([o.status for o in outcomes] == [jn.STATUS_OK, jn.STATUS_OK, jn.STATUS_FAIL],
              f"two good samples and one that never fused ({[o.status for o in outcomes]})")
        check(len(seen) == 3 and sum("fail" in m for m in seen) == 1, "progress names each sample and its status")
        one_by_one = jn.run(paths, jn.JunctionOptions(width_um=150.0, save=False), workers=1)
        two_at_once = jn.run(paths, jn.JunctionOptions(width_um=150.0, save=False), workers=2)
        check([o.name for o in two_at_once] == [o.name for o in one_by_one]
              and [o.status for o in two_at_once] == [o.status for o in one_by_one]
              and all((a.line_um is None and b.line_um is None) or np.allclose(a.line_um, b.line_um)
                      for a, b in zip(one_by_one, two_at_once)),
              "two at once, in their own processes, gives the same results in the same order")
        check(all(not loaders.is_open(path) for path in paths), "every file is given back")

        good, bad = paths[0], paths[2]
        check(ims_store.list_labels(good) == [fu.DEFAULT_KEY], "the band is stored as a label map")
        labels, attrs = ims_store.load_labels(good, fu.DEFAULT_KEY)
        check(labels.shape == (240, 360) and set(np.unique(labels)) == {0, 1},
              "on the image's own grid")
        check(np.asarray(attrs["line_yx_um"]).shape[1] == 2 and ims_store._text(attrs["status"]) == "ok",
              "with the line it was made from and its status")
        rois = ims_store.load_rois(good)
        check([roi.name for roi in rois] == [jn.BAND_ROI_NAME], "and as an outline the analysis can count inside")
        check(ims_store.list_labels(bad) == [] and not outcomes[2].saved,
              "nothing is stored for the sample with no junction")

        # A band stored again replaces the earlier one and keeps other outlines.
        other = ims_store.StoredRoi("drawn by hand", np.array([[0, 0], [0, 50], [50, 50.0]]))
        ims_store.save_rois(good, [*rois, other])
        jn.process_file(good, options)
        names = [roi.name for roi in ims_store.load_rois(good)]
        check(sorted(names) == sorted(["drawn by hand", jn.BAND_ROI_NAME]),
              f"a rerun keeps hand-drawn outlines and one band ({names})")
        # The band is not anatomy: the Brain regions layer leaves it out, and saving the
        # regions on screen (which replaces the stored set) does not delete it.
        check(jn.BAND_ROI_NAME in ims_store.OTHER_PANELS_ROIS, "the band is one of the other panels' outlines")
        check([roi.name for roi in ims_store.anatomy(ims_store.load_rois(good))] == ["drawn by hand"],
              "the Brain regions panel is shown only the hand-drawn region")
        ims_store.save_rois(good, [ims_store.StoredRoi("cortex", np.array([[0, 0], [0, 40], [40, 40.0]]))])
        names = sorted(roi.name for roi in ims_store.load_rois(good))
        check(names == sorted(["cortex", jn.BAND_ROI_NAME]),
              f"saving the regions replaces them and keeps the band ({names})")

        folder = root / "junction_QC"
        check((folder / "junction_QC.xlsx").is_file(), "the workbook is written")
        check(all((folder / status).is_dir() for status in jn.STATUSES), "with a folder per status")
        check(all((folder / "ok" / f"assembloid_s{i}.png").is_file() for i in range(2))
              and (folder / "fail" / "assembloid_s2.png").is_file(),
              "and each sample's picture in the folder of its status")
        check(len(list(folder.glob("*/*.png"))) == 3 and not list(folder.glob("assembloid*.png")),
              "one picture per sample, none loose")
        check(all(o.picture is not None and o.picture.parent.name == o.status for o in outcomes),
              "the outcome says where its picture is")

        # A status that changes — the folder-wide checks come last — moves the picture.
        (folder / "assembloid_s0.png").write_bytes((folder / "ok" / "assembloid_s0.png").read_bytes())
        outcomes[0].status, outcomes[0].flags = jn.STATUS_CHECK, ["unlike the rest"]
        jn._refile(outcomes, folder)
        check((folder / "check" / "assembloid_s0.png").is_file()
              and not (folder / "ok" / "assembloid_s0.png").exists()
              and not (folder / "assembloid_s0.png").exists(),
              "a picture follows its sample's status, and a loose one from before is cleared")
        moved = jn.file_picture(folder, "assembloid_s0", jn.STATUS_OK)
        check(moved.is_file() and not (folder / "check" / "assembloid_s0.png").exists(),
              "and can be moved back without drawing it again")

        missing = jn.process_file(root / "not_there.ome.tiff", options)
        check(missing.status == jn.STATUS_FAIL and missing.picture is not None
              and missing.picture.parent.name == "fail" and missing.picture.is_file(),
              "a file that cannot be read still gets a picture, in fail")
        check((folder / "junction_QC_sheet_01.png").is_file(), "and a contact sheet")
        frame = jn.qc_dataframe(outcomes)
        check(list(frame["Sample"]) == [f"assembloid_s{i}" for i in range(3)], "one row per sample, named without .ome")
        check({"Status", "Flags", "Tissue area (mm2)", "Junction length (um)"} <= set(frame.columns),
              "carrying the status, the flags and the figures behind them")
        check({"Assembloid", "Odd section", "Major axis (um)", "Shape 1", "Labelled area (mm2)"} <= set(frame.columns)
              and frame["Assembloid"].str.startswith("assembloid/").all(),
              f"the section's shape and its assembloid ({sorted(frame['Assembloid'].unique())})")


def main() -> int:
    for test in (
        test_channels_are_found_by_name,
        test_the_junction_is_found_between_the_organoids,
        test_an_organoid_faint_in_the_nuclear_stain_is_still_tissue,
        test_the_labelled_seed_can_cover_every_patch,
        test_with_no_seam_the_seeds_can_come_from_the_labels,
        test_a_drawn_line_gets_a_band_that_follows_the_tissue,
        test_bending_costs_so_a_crack_round_a_corner_is_not_followed,
        test_a_tear_is_not_a_cheap_road_and_the_cut_sits_where_the_cells_change,
        test_the_panel_opens_with_the_finders_own_settings,
        test_the_second_seed_comes_from_the_other_organoids_label,
        test_the_junction_follows_the_dark_line_in_the_nuclear_stain,
        test_aberrant_samples_are_flagged,
        test_outliers_are_judged_against_the_folder,
        test_a_folder_run_stores_the_band_and_writes_the_report,
    ):
        test()
        print()

    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
