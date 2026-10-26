"""Made-up assembloids, each built to show one way the junction finder goes wrong.

Used by ``build_explainer.py`` (``DATA.synthetic_failures``). Every sample is the
explainer's synthetic assembloid — two overlapping discs of nuclei, a dark seam
where they meet, labelled cells on one side — with one thing changed: a tear,
no seam, labels on both sides, organoids apart, a scan edge, a small
partner. The real finder (``junction.find_junction``) is run on each, with and
without its price on bending, and the line it finds is compared with the true
junction the sample was drawn with.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from microscopy_viewer import fusion as fu
from microscopy_viewer import junction as jn

PX = 2.0                     # µm per full-resolution pixel
#: How dark the seam is in the tear case: faint, as where those failures happen.
SEAM_FAINT = 0.35
TEAR_STOP_UM = 380.0
H, W = 240, 360              # 480 × 720 µm

#: id → (title, what was changed, why the finder fails or not)
CASES = {
    "clean": ("A clean pair (reference)",
              "Two organoids pressed together, a faint dark seam where they meet, labelled cells on one side.",
              "Everything the finder relies on is there: two seeds and a dark seam. The line follows the true "
              "junction, with or without the price on bending."),
    "tear": ("A tear, and a faint seam",
             "A gap runs down through the labelled organoid, and the seam between the organoids is only faintly "
             "darker than the tissue.",
             "Empty tissue is the cheapest thing to cut. Without the price on bending (dotted grey) the cut runs "
             "partly down the tear. With it, the cut no longer turns into the tear, but with the seam this faint it "
             "takes the narrowest neck instead (amber) — and the status still says ok. The finder cannot see this "
             "failure; only a look at the picture does."),
    "no_seam": ("No visible seam",
                "The organoids are fused so smoothly that the nuclear stain has no dark line between them.",
                "With nothing dark to follow, the cheapest cut is the shortest way across — the narrowest neck. That is "
                "the junction only if the organoids happen to meet straight across. Here the true junction slants, "
                "the line does not, and the status says ok."),
    "both_labelled": ("Labelled cells on both sides",
                      "The label is in both organoids (or bleeds through), so it no longer tells them apart.",
                      "The seeds are placed where the labelled cells are densest; with labelled cells everywhere, that "
                      "says little about which organoid is which, and the cut drifts. The finder flags it ('labelled "
                      "cells on both sides')."),
    "apart": ("Organoids that never fused",
              "Two organoids lying side by side with slide between them.",
              "There is no junction. The finder still finds a sliver to cut and flags it ('junction very short', "
              "'hardly any tissue in the band'). The pre-screen calls these Separated, and they are left out."),
    "scan_edge": ("Cut off by the edge of the scan",
                  "The scanned area stops before the end of the second organoid.",
                  "Part of the tissue is missing, so the outline and the neck are wrong and the line stops where the "
                  "scan does. The finder flags it ('cut off by the scan edge')."),
    "small_partner": ("One organoid much smaller",
                      "The second organoid is a small bud on the labelled one.",
                      "The line can be right, but it is short and the band holds little tissue, so counts per mm of "
                      "junction are noisy. The finder flags it."),
}


#: id → the real section (a junction-gallery id) showing the same thing, and how close the match is;
#: or None with the reason there is none among the 30 annotated sections.
REAL = {
    "clean": ("055-054_s15", "a clean junction"),
    "tear": ("033-032_s12", "the same failure: a tear taken for the junction"),
    "no_seam": ("034-033_s17", "the same symptom: status ok, line wrong"),
    "both_labelled": (None, "no clean real example: this flag fires on most sections, good and bad alike "
                            "(it was on 12 good lines and 3 bad ones in the validation)"),
    "apart": ("052-051_s23", "the same failure: the organoids never fused"),
    "scan_edge": ("052-051_s14", "the same failure: cut off by the scan edge"),
    "small_partner": (None, "none among the 30 annotated sections"),
}

def _sample(case: str, seed: int = 7):
    """Full-resolution planes of one made-up assembloid, and its true junction (µm) or None."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W] * PX
    c2, r2 = (240, 480), 160
    if case == "apart":
        c2 = (240, 590)
    if case == "small_partner":
        c2, r2 = (250, 400), 85
    first = (yy - 240) ** 2 + (xx - 220) ** 2 < 160 ** 2
    second = (yy - c2[0]) ** 2 + (xx - c2[1]) ** 2 < r2 ** 2
    tissue = first | second
    junction_x = 350 + 0.45 * (yy - 240) + 0.002 * (yy - 240) ** 2
    if case == "small_partner":
        junction_x = 335 + 0.25 * (yy - 250)
    if case == "apart":
        labelled, other = first, second
    else:
        labelled, other = tissue & (xx < junction_x), tissue & (xx >= junction_x)

    gap = np.zeros_like(tissue)
    if case == "tear":
        # A smooth tear from the top outline down through the labelled organoid, stopping short of the bottom.
        tear_x = 270 + 0.35 * (yy - 240)
        gap = (np.abs(xx - tear_x) < 8) & (yy < TEAR_STOP_UM) & first
    labelled, other = labelled & ~gap, other & ~gap

    def spots(where, count, sigma_um, brightness, weight=None):
        p = (where * (1.0 if weight is None else weight)).ravel().astype(float)
        if p.sum() == 0:
            return np.zeros((H, W))
        picked = rng.choice(p.size, count, p=p / p.sum())
        image = np.bincount(picked, minlength=p.size).reshape(H, W).astype(float)
        s = sigma_um / PX
        return ndimage.gaussian_filter(image, s) * 2 * np.pi * s * s * brightness

    across = xx - junction_x
    n_other = 9000 if case != "small_partner" else 2400
    nuclei = spots(labelled, 9000, 3, 700) + spots(other, n_other, 3, 520)
    seam = {"no_seam": 0.0, "apart": 0.0, "tear": SEAM_FAINT}.get(case, 0.7)
    nuclei = ndimage.gaussian_filter(nuclei, 1.5) * (1 - seam * np.exp(-(across / 12) ** 2))
    nuclei *= ~ndimage.binary_dilation(gap, iterations=1)
    dapi = 80 + nuclei
    patchy = np.exp(-((xx - 150) ** 2 + (yy - 200) ** 2) / (2 * 110 ** 2)) + 0.3
    gfp = 100 + spots(labelled, 260, 5, 3000, patchy)
    if case == "both_labelled":
        gfp = gfp + spots(other, 260, 5, 3000)
    else:
        gfp = gfp + spots(other & (across < 60), 12, 5, 3000)
    mcherry = 90 + spots(other, 220, 5, 2200)
    planes = []
    for plane in (dapi, gfp, mcherry):
        plane = rng.poisson(np.clip(plane, 0, None)) + rng.normal(0, 8, plane.shape)
        plane = np.clip(plane, 1, 65535).astype(np.uint16)
        if case == "scan_edge":
            plane[:, int(560 / PX):] = 0          # the scan stops 560 µm in
        planes.append(plane)
    truth = None
    if case != "apart":
        rows = np.arange(0, H, 4)
        cols = np.clip((junction_x[rows, 0] / PX).astype(int), 0, W - 1)
        on = tissue[rows, cols] & ~gap[rows, cols]
        if case == "scan_edge":
            on &= junction_x[rows, 0] < 560
        truth = np.column_stack([rows[on] * PX, junction_x[rows[on], 0]])
    return planes, truth


def _distance(line_a, line_b):
    """Mean symmetric distance between two lines (µm), both densely resampled."""
    if line_a is None or line_b is None or len(line_a) < 2 or len(line_b) < 2:
        return None
    a, b = fu.fit_spline(np.asarray(line_a, float), 2.0), fu.fit_spline(np.asarray(line_b, float), 2.0)
    d1, _ = cKDTree(b).query(a)
    d2, _ = cKDTree(a).query(b)
    return float(np.concatenate([d1, d2]).mean())


def synthetic_failures(helpers) -> list[dict]:
    """One entry per case: picture, true line, the finder's line with and without bending, status, flags."""
    out = []
    for case, (title, change, why) in CASES.items():
        planes, truth = _sample(case)
        specs = [SimpleNamespace(data=p[None], scale=(1.0, PX, PX), channel_name=n, multiscale=False)
                 for p, n in zip(planes, ("DAPI", "GFP", "mCherry"))]
        lines, statuses = {}, {}
        result = None
        for bend in (0, int(jn.BEND_UM)):
            options = jn.JunctionOptions(width_um=100.0, gap_um=10.0, save=False, bend_um=float(bend))
            ov = jn.build_overview(specs, options, target_um=10.0)
            found = jn.find_junction(ov, options)
            lines[str(bend)] = helpers.points(found.line_um) if found.line_um is not None else None
            statuses[str(bend)] = found.status
            if bend == int(jn.BEND_UM):
                result, overview = found, ov
        error = _distance(result.line_um, truth)
        out.append({
            "id": case, "title": title, "change": change, "why": why,
            "real": {"id": REAL[case][0], "match": REAL[case][1]},
            "um": float(overview.um), "shape": [int(n) for n in overview.channels.shape[1:]],
            "img": helpers.data_url(helpers.overview_composite(overview), "PNG"),
            "truth": helpers.points(truth) if truth is not None else None,
            "lines": lines,
            "band_outline": helpers.points(result.band_outline_um) if result.band_outline_um is not None else None,
            "status": result.status, "flags": list(result.flags),
            "error_um": None if error is None else round(error, 1),
            "error_um_no_bending": (None if lines["0"] is None or truth is None
                                    else round(_distance(np.asarray(lines["0"]), truth), 1)),
        })
        helpers.log(f"  synthetic failure {case}: {result.status} {result.flags} error {out[-1]['error_um']} µm "
                    f"(no bending {out[-1]['error_um_no_bending']})")
    return out
