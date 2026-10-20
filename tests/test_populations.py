"""Checks for the population engine: samples, gates, thresholds, the summary, the file.

No Qt and no browser — everything here is the logic the napari panel and the
dashboard page both call, so a threshold that is wrong is wrong once rather than
twice. Needs anndata and pandas; skips the parts that need a written file when
anndata is not installed.

Run with::

    python tests/test_populations.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import analysis  # noqa: E402
from microscopy_viewer import dashboard as db  # noqa: E402
from microscopy_viewer import populations as pop  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _has(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


#: The plate every test is run against: two wells, two cycles each.
COMPONENTS = ("B/02/0", "B/02/1", "C/03/0", "C/03/1")

#: Objects per image that are debris — a speck of dirt a hundredth the size of a
#: nucleus and twenty times as bright — and objects that are two nuclei fused into
#: one. The gates exist for these, and they are rare, which is what makes the
#: suggested percentile bounds able to exclude them.
DEBRIS = 1
FUSED = 1

#: Share of the objects in each image that are genuinely bright in the marker.
#: A twentieth: ``median + 3 sd`` is a rule for a rare population, and on a well
#: that is a third positive the positives themselves push the threshold past
#: their own mean.
POSITIVE_SHARE = 0.05


def _plate_file(
    directory: Path,
    components: tuple[str, ...] = COMPONENTS,
    per_image: int = 300,
    without_marker: tuple[str, ...] = (),
) -> Path:
    """A combined file shaped like the real one, with a population to find in it.

    Each image holds mostly dim objects, a twentieth that are bright in
    ``Marker``, one speck of debris and one fused pair for the gates to remove.

    Each image also has its own **gain** — the same cells imaged at a different
    exposure, which is what a plate actually looks like. Without it every scope
    gives the same threshold and the choice between them could not be tested; with
    it a plate-wide threshold in raw grey levels is unfair to the dim images,
    which is the thing normalising exists to fix.

    An image named in *without_marker* has no marker column at all, which is what
    a 4i cycle that did not measure this stain looks like after the outer join.
    """
    import pandas as pd

    rng = np.random.default_rng(7)
    tables: list[Path] = []
    for index, component in enumerate(components):
        n = per_image
        bright = rng.random(n) < POSITIVE_SHARE
        voxels = rng.normal(500, 50, n)
        voxels[:DEBRIS] = rng.normal(5, 1, DEBRIS)
        voxels[DEBRIS : DEBRIS + FUSED] = rng.normal(4000, 100, FUSED)
        frame = pd.DataFrame(
            {
                "Label": np.arange(1, n + 1),
                "Voxels": voxels,
                "Solidity": np.clip(rng.normal(0.9, 0.03, n), 0.0, 1.0),
                "Centroid Z (µm)": np.zeros(n),
                "Centroid Y (µm)": rng.uniform(0, 400, n),
                "Centroid X (µm)": rng.uniform(0, 400, n),
            }
        )
        if component not in without_marker:
            marker = np.where(bright, rng.normal(3000, 200, n), rng.normal(400, 60, n))
            # Debris is dirt: tiny, and far brighter than any cell. It is there so
            # that gating can be shown to change the threshold it sets.
            marker[:DEBRIS] = rng.normal(20_000, 500, DEBRIS)
            frame["Marker"] = marker * (1.0 + 0.4 * index)
        path = directory / f"{component.replace('/', '_')}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        tables.append(path)
    return analysis.write_combined_anndata(tables, directory / "plate.h5ad")


# ---------------------------------------------------------------------------


def test_scope_groups_speak_the_existing_vocabulary() -> None:
    print("the pools a threshold is computed over")

    check(
        all(scope in analysis.SCOPES for scope in (pop.PositiveRule("x").scope,)),
        f"the default scope is one of the application's own ({pop.PositiveRule('x').scope})",
    )

    names = list(COMPONENTS)
    plate = pop.scope_groups(names, analysis.SCOPE_PLATE)
    check(len(plate) == 1, f"the whole plate is one pool ({list(plate)})")
    check(
        sorted(next(iter(plate.values()))) == sorted(names),
        "and every image is in it",
    )

    per_image = pop.scope_groups(names, analysis.SCOPE_IMAGE)
    check(len(per_image) == len(names), f"per image is one pool each ({len(per_image)})")

    wells = pop.scope_groups(names, analysis.SCOPE_WELL)
    check(sorted(wells) == ["B/02", "C/03"], f"per well pools the cycles of a well: {sorted(wells)}")
    check(
        wells["B/02"] == ("B/02/0", "B/02/1"),
        f"both cycles of B/02 in one pool: {wells['B/02']}",
    )

    cycles = pop.scope_groups(names, analysis.SCOPE_CYCLE)
    check(sorted(cycles) == ["cycle 0", "cycle 1"], f"per cycle pools the wells: {sorted(cycles)}")
    check(
        cycles["cycle 0"] == ("B/02/0", "C/03/0"),
        f"one staining round across the plate: {cycles['cycle 0']}",
    )


def test_the_rule_is_a_rule() -> None:
    print("the positive rule, on its own")

    rule = pop.PositiveRule("Marker", k=2.5, scope=analysis.SCOPE_WELL)
    check(
        pop.PositiveRule.from_attrs(rule.as_attrs()) == rule,
        "the rule round-trips, which is what lets a stored one be compared",
    )
    check(
        rule != pop.PositiveRule("Marker", k=3.0, scope=analysis.SCOPE_WELL),
        "and a changed multiplier is a different rule",
    )
    check(
        pop.PositiveRule("Marker").statistic == pop.STAT_MEDIAN_SD
        and pop.PositiveRule("Marker").k == 3.0,
        "the default is median + 3 sd, which is the rule this exists to apply",
    )
    check(
        "median + 3 sd" in pop.PositiveRule("Marker").describe(),
        f"and it says so in words: {pop.PositiveRule('Marker').describe()!r}",
    )
    check(
        all(name in pop.STATISTIC_LABELS for name in pop.STATISTICS),
        "every statistic offered has a label the panel can show",
    )

    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    centre, spread, threshold = pop.threshold_of_pool(values, pop.PositiveRule("x", k=1.0))
    check(
        np.isclose(centre, 3.0) and np.isclose(threshold, 3.0 + np.std(values)),
        f"median + k sd is the median and the population sd ({centre:g}, {threshold:.4g})",
    )
    _, _, mean_threshold = pop.threshold_of_pool(values, pop.PositiveRule("x", pop.STAT_MEAN_SD, k=1.0))
    check(
        mean_threshold > threshold,
        "the mean sits above the median on a skewed column, which is why median is the default",
    )
    _, spread, flat = pop.threshold_of_pool(
        values, pop.PositiveRule("x", pop.STAT_PERCENTILE, percentile=50.0)
    )
    check(np.isclose(flat, 3.0) and not np.isfinite(spread), "a percentile is a percentile, no sd")
    check(
        not np.isfinite(pop.threshold_of_pool([np.nan, np.nan], pop.PositiveRule("x"))[2]),
        "a pool with no numbers in it has no threshold rather than a zero",
    )


def test_samples_and_gates(directory: Path) -> None:
    print("choosing samples, and gating out what is not a cell")

    if not _has("anndata"):
        print("  skip anndata is not installed")
        return

    source = db.read(_plate_file(directory))
    check(
        pop.sample_labels(source) == list(COMPONENTS),
        f"the samples are the images, in plate order: {pop.sample_labels(source)}",
    )
    check(
        "Marker" in pop.measurable_names(source) and "Voxels" in pop.measurable_names(source),
        "the features are offered to gate on",
    )
    check(
        any(name.startswith("centroid") for name in pop.measurable_names(source)),
        "and so are the numeric obs columns, for gating on position",
    )

    everything = pop.sample_mask(source, None)
    check(everything.all(), "no choice yet means every object")
    two = pop.sample_mask(source, ["B/02/0", "B/02/1"])
    check(
        int(two.sum()) == 600,
        f"choosing two of four samples takes their objects and no others ({int(two.sum())})",
    )
    check(
        not pop.sample_mask(source, []).any(),
        "and unticking everything selects nothing, rather than quietly selecting all",
    )

    debris = pop.Gate("Voxels", low=50.0, high=2000.0)
    check(debris.describe() == "Voxels in [50, 2000]", f"a gate says what it is: {debris.describe()!r}")
    report = pop.gate_report(source, [debris])
    expected = len(COMPONENTS) * (DEBRIS + FUSED)
    check(
        report.n_removed == expected,
        f"one size gate removes the debris and the fused objects ({report.n_removed} of {expected})",
    )
    check(
        report.n_kept + report.n_removed == report.n_objects,
        "and the two accounts add up",
    )
    check(
        report.costs[0].n_removed == expected,
        "the per-gate cost is what that gate removed on its own",
    )

    loose = pop.Gate("Solidity", low=0.0, high=1.0)
    both = pop.gate_report(source, [debris, loose])
    check(
        both.costs[1].n_removed == 0,
        "a gate that removes nothing is reported as costing nothing, not as cheap",
    )
    check(both.n_kept == report.n_kept, "and the surviving count does not change")

    off = pop.gate_report(source, [pop.Gate("Voxels", 50.0, 2000.0, enabled=False)])
    check(off.n_removed == 0, "a gate switched off removes nothing")
    check("(off)" in off.costs[0].gate.describe(), "and says it is off")

    unknown = pop.gate_report(source, [pop.Gate("Nothing measured this", 0.0, 1.0)])
    check(
        unknown.missing == ("Nothing measured this",),
        f"a gate on a feature the file lacks is named rather than applied: {unknown.missing}",
    )
    check(unknown.n_kept == unknown.n_objects, "and removes nothing while it is named")
    check(
        "does not have" in unknown.describe(),
        f"which the one-line summary mentions: {unknown.describe()!r}",
    )

    inside = pop.gate_mask(source, [debris], within=two)
    check(
        int(inside.sum()) == int(two.sum()) - 2 * (DEBRIS + FUSED),
        f"gating inside a selection stays inside it ({int(inside.sum())})",
    )

    # Blanks: a feature a sample never measured must not empty that sample.
    patchy = db.read(_plate_file(directory, without_marker=("C/03/1",)))
    marker_gate = pop.Gate("Marker", low=0.0, high=1e9)
    blanks = pop.gate_report(patchy, [marker_gate])
    labels = pop.sample_of(patchy).astype(str)
    kept = pop.gate_mask(patchy, [marker_gate])
    check(
        int(kept[labels == "C/03/1"].sum()) == int((labels == "C/03/1").sum()),
        "a sample that never measured the gated feature keeps all its objects",
    )
    check(
        blanks.costs[0].n_blank == int((labels == "C/03/1").sum()),
        f"and the blanks are counted and reported ({blanks.costs[0].n_blank})",
    )


def test_the_histogram(directory: Path) -> None:
    print("the histogram a threshold is set on")

    if not _has("anndata"):
        print("  skip anndata is not installed")
        return

    source = db.read(_plate_file(directory))
    histogram = pop.feature_histogram(source, "Voxels")
    check(
        histogram.counts.size == pop.DEFAULT_BINS
        and histogram.edges.size == pop.DEFAULT_BINS + 1,
        f"one count per bin and one more edge than bins ({histogram.counts.size})",
    )
    check(
        int(histogram.counts.sum()) == histogram.n_finite,
        "every object lands in a bin, the ones outside the range included",
    )
    check(
        histogram.centres.size == histogram.counts.size,
        "the bin centres line up with the counts, so the widget just plots them",
    )
    check(
        histogram.suggested_low > 5.0 and histogram.suggested_high < 4000.0,
        f"the suggested bounds cut the debris and the fused objects "
        f"({histogram.suggested_low:.0f} to {histogram.suggested_high:.0f})",
    )
    check(
        histogram.suggested_gate() == pop.Gate("Voxels", histogram.suggested_low, histogram.suggested_high),
        "and they come back as a gate, which is what the panel puts in its boxes",
    )

    one = pop.feature_histogram(source, "Voxels", within=pop.sample_mask(source, ["B/02/0"]))
    check(one.n_finite == 300, f"a histogram of one sample covers that sample ({one.n_finite})")

    patchy = db.read(_plate_file(directory, without_marker=("C/03/1",)))
    blank = pop.feature_histogram(patchy, "Marker", within=pop.sample_mask(patchy, ["C/03/1"]))
    check(
        blank.n_finite == 0 and blank.n_blank == 300,
        f"a sample with no values says so rather than drawing a flat line "
        f"({blank.n_finite} finite, {blank.n_blank} blank)",
    )


def test_calling_positives(directory: Path) -> None:
    print("what counts as positive")

    if not _has("anndata"):
        print("  skip anndata is not installed")
        return

    source = db.read(_plate_file(directory))
    gates = [pop.Gate("Voxels", 50.0, 2000.0)]
    keep = pop.gate_mask(source, gates)

    rule = pop.PositiveRule("Marker", scope=analysis.SCOPE_IMAGE)
    calls = pop.call_positives(source, rule, keep=keep)
    share = 100.0 * calls.n_positive / int(keep.sum())
    check(
        3.0 < share < 9.0,
        f"median + 3 sd finds the bright twentieth of the objects ({share:.1f} %)",
    )
    check(
        not (calls.positive & ~keep).any(),
        "nothing outside the gates is called positive",
    )

    check(
        len(calls.thresholds) == len(COMPONENTS),
        f"one threshold row per sample ({len(calls.thresholds)})",
    )
    check(
        calls.thresholds["scope group"].nunique() == len(COMPONENTS),
        "per image, every sample is its own pool",
    )
    signal = np.asarray(calls.values, dtype=float)
    labels = pop.sample_of(source).astype(str)
    mine = (labels == "B/02/0") & keep
    expected = np.median(signal[mine]) + 3.0 * np.std(signal[mine])
    check(
        np.isclose(calls.threshold_of("B/02/0"), expected),
        f"and it really is the median plus three deviations of that pool "
        f"({calls.threshold_of('B/02/0'):.1f} against {expected:.1f})",
    )

    pooled = pop.call_positives(source, pop.PositiveRule("Marker", scope=analysis.SCOPE_PLATE), keep=keep)
    check(
        pooled.thresholds["scope group"].nunique() == 1,
        "over the whole plate there is one threshold",
    )
    check(
        pooled.thresholds["threshold"].nunique() == 1,
        "and every sample is compared with it",
    )
    check(
        pooled.n_positive != calls.n_positive,
        f"which is not the same answer as per image ({pooled.n_positive} against "
        f"{calls.n_positive}) — the scope is a decision, not a detail",
    )

    ungated = pop.call_positives(source, rule)
    check(
        ungated.threshold_of("B/02/0") > calls.threshold_of("B/02/0"),
        "leaving the debris in raises the threshold, which is why gating comes first",
    )

    strict = pop.call_positives(source, pop.PositiveRule("Marker", k=6.0, scope=analysis.SCOPE_IMAGE), keep=keep)
    check(strict.n_positive < calls.n_positive, "a larger k calls fewer objects positive")

    # Raw against normalised, over the same scope: the same cells, read differently.
    normalised = pop.call_positives(
        source,
        pop.PositiveRule(
            "Marker", scope=analysis.SCOPE_IMAGE, normalised=True, normalise_scope=analysis.SCOPE_IMAGE
        ),
        keep=keep,
    )
    agree = int(((normalised.positive == calls.positive) & keep).sum())
    check(
        agree > 0.97 * int(keep.sum()),
        f"normalising over the scope the threshold is pooled over calls the same objects "
        f"({100.0 * agree / int(keep.sum()):.1f} % agree)",
    )
    check(
        0.0 <= float(normalised.thresholds["threshold"].max()) <= 1.0,
        f"but the threshold is now on a 0-1 scale, comparable between samples "
        f"({float(normalised.thresholds['threshold'].max()):.3f})",
    )
    check(
        any("0-1 scale" in note for note in normalised.notes),
        f"and the result says that is what normalising did: {normalised.notes}",
    )

    wide = pop.call_positives(
        source,
        pop.PositiveRule(
            "Marker", scope=analysis.SCOPE_PLATE, normalised=True, normalise_scope=analysis.SCOPE_IMAGE
        ),
        keep=keep,
    )
    check(
        wide.n_positive != pooled.n_positive,
        f"normalising per image before a plate-wide threshold does change the answer "
        f"({wide.n_positive} against {pooled.n_positive}) — that is what it is for",
    )

    patchy = db.read(_plate_file(directory, without_marker=("C/03/1",)))
    missing = pop.call_positives(patchy, rule)
    check(
        missing.missing_samples == ("C/03/1",),
        f"a sample that lacks the feature is named in the result: {missing.missing_samples}",
    )
    check(
        any("carry no Marker" in note for note in missing.notes),
        f"in words the panel can show: {missing.notes}",
    )
    check(
        not np.isfinite(missing.threshold_of("C/03/1")),
        "it gets no threshold rather than a zero that reads as an answer",
    )


def test_the_summary(directory: Path) -> None:
    print("the answer, per sample and per well")

    if not _has("anndata"):
        print("  skip anndata is not installed")
        return

    source = db.read(_plate_file(directory))
    gates = [pop.Gate("Voxels", 50.0, 2000.0)]
    keep = pop.gate_mask(source, gates)
    calls = pop.call_positives(source, pop.PositiveRule("Marker", scope=analysis.SCOPE_IMAGE), keep=keep)
    summary = pop.sample_summary(source, calls, keep=keep)

    check(len(summary) == len(COMPONENTS), f"one row per sample ({len(summary)})")
    check(
        list(summary["objects"]) == [300] * len(COMPONENTS),
        "objects counts everything in the sample",
    )
    check(
        (summary["kept"] == 300 - DEBRIS - FUSED).all(),
        f"kept counts what survived the gates ({int(summary['kept'].iloc[0])})",
    )
    check(
        (summary["objects"] == summary["kept"] + summary["excluded"]).all(),
        "and the denominator is printed beside the percentage rather than implied",
    )
    check(
        int(summary["positive"].sum()) == calls.n_positive,
        "the positives in the table are the positives that were called",
    )
    check(
        np.allclose(summary["percent positive"], 100.0 * summary["positive"] / summary["kept"]),
        "the percentage is of the kept objects, which is what the gates mean",
    )
    check(
        list(summary["well"]) == ["B/02", "B/02", "C/03", "C/03"],
        f"the well and the cycle are split out, ready to plot: {list(summary['well'])}",
    )

    wells = pop.well_summary(summary)
    check(len(wells) == 2, f"pooled to one row per well, the cycles added ({len(wells)})")
    check(
        int(wells["kept"].sum()) == int(summary["kept"].sum()),
        "with nothing lost in the pooling",
    )
    check(
        np.allclose(wells["percent positive"], 100.0 * wells["positive"] / wells["kept"]),
        "and the percentage recomputed from the sums, not averaged over cycles",
    )
    check(len(pop.well_summary(None)) == 0, "no summary yet is an empty table, not an exception")


def test_it_goes_back_onto_the_file(directory: Path) -> None:
    print("writing the gates, the rule and the calls into the .h5ad")

    if not _has("anndata"):
        print("  skip anndata is not installed")
        return

    path = _plate_file(directory)
    source = db.read(path)
    check(pop.load_populations(source) == {}, "a fresh file has no population analysis")

    gates = [pop.Gate("Voxels", 50.0, 2000.0), pop.Gate("Solidity", 0.5, 1.0, enabled=False)]
    rule = pop.PositiveRule("Marker", k=3.0, scope=analysis.SCOPE_CYCLE)
    samples = ["B/02/0", "B/02/1", "C/03/0"]
    record = pop.run_populations(source, gates, rule, samples=samples)

    check(
        all(column in source.obs for column in (pop.KEPT_COLUMN, pop.POSITIVE_COLUMN, pop.SIGNAL_COLUMN)),
        "every object carries whether it was kept, whether it is positive, and its signal",
    )
    check(
        int(source.obs[pop.KEPT_COLUMN].sum()) == record["n_kept"],
        "the column and the record agree about how many survived",
    )
    check(
        record["n_selected"] == 900 and record["n_objects"] == 1200,
        f"the record says what was selected out of what ({record['n_selected']} of "
        f"{record['n_objects']})",
    )
    check(
        not source.obs[pop.POSITIVE_COLUMN].to_numpy()[~source.obs[pop.KEPT_COLUMN].to_numpy()].any(),
        "and nothing outside the selection is marked positive",
    )
    check(
        np.isclose(record["percent_positive"], 100.0 * record["n_positive"] / record["n_kept"]),
        f"the headline number is in the record ({record['percent_positive']:.1f} %)",
    )
    check(
        list(record["gate_removed"])[0] > 0 and list(record["gate_removed"])[1] == 0,
        f"with what each gate cost, the disabled one included: {list(record['gate_removed'])}",
    )

    target = pop.save_populations(directory / "populations.h5ad", source)
    back = pop.load_populations(db.read(target))
    check(back["rule"] == rule, "the rule survives the file")
    check(back["gates"] == gates, "and so do the gates, bounds and all")
    check(
        back["samples"] == samples,
        f"and which samples were included: {back['samples']}",
    )
    check(
        back["n_positive"] == record["n_positive"] and back["n_kept"] == record["n_kept"],
        "and the counts, so a reopened file does not have to be recomputed to be read",
    )
    check(bool(back["created"]), f"with when it was run ({back['created']})")

    # Re-applying a stored rule to the file it came from is the same answer.
    reopened = db.read(target)
    again = pop.run_populations(reopened, back["gates"], back["rule"], samples=back["samples"])
    check(
        again["n_positive"] == record["n_positive"],
        "and re-running the stored definition reproduces the result exactly",
    )

    empty = db.read(path)
    pop.run_populations(empty, [], pop.PositiveRule("Marker"))
    check(
        pop.load_populations(empty)["gates"] == [],
        "a run with no gates at all stores and reads back as no gates",
    )


def test_finding_the_files(directory: Path) -> None:
    print("finding the combined files beside a plate")

    plate = directory / "experiment.zarr"
    plate.mkdir()
    folder = directory / "experiment_nuclei_objects"
    folder.mkdir()
    (folder / "B_02_0.csv").write_text("Label,Voxels\n1,10\n", encoding="utf-8")
    inside = folder / "nuclei_objects.h5ad"
    inside.write_bytes(b"not really an h5ad")
    beside = directory / "experiment_nuclei_objects.h5ad"
    beside.write_bytes(b"not really an h5ad either")

    found = pop.population_files(plate)
    check(
        set(found) == {inside, beside},
        f"both places a run writes one are found: {[path.name for path in found]}",
    )
    check(
        pop.population_files(directory / "no_such_plate.zarr") == [],
        "and a plate that is not there offers nothing rather than a neighbour's files",
    )


def test_gates_combine_with_and_or_xor() -> None:
    """The three joins, on four objects that make the difference visible.

    Four objects: one in both windows, one in each, one in neither. Any pair of
    gates that keeps the same set under all three operators is a pair that does
    not test them.
    """
    print("combining gates with and / or / xor")

    if not _has("anndata"):
        print("  skip anndata is not installed")
        return

    import anndata as ad

    matrix = np.array(
        [
            [1.0, 90.0],   # both
            [1.0, 10.0],   # only A
            [9.0, 90.0],   # only B
            [9.0, 10.0],   # neither
        ]
    )
    source = ad.AnnData(X=matrix)
    source.var_names = ["A", "B"]
    source.obs_names = ["both", "onlyA", "onlyB", "neither"]

    low_a = pop.Gate("A", 0.0, 3.0)

    def kept(join: str) -> list[str]:
        mask = pop.gate_mask(source, [low_a, pop.Gate("B", 50.0, np.inf, join=join)])
        return [str(name) for name, keep in zip(source.obs_names, mask) if keep]

    check(kept(pop.JOIN_AND) == ["both"], f"and keeps what passes both: {kept(pop.JOIN_AND)}")
    check(
        kept(pop.JOIN_OR) == ["both", "onlyA", "onlyB"],
        f"or keeps what passes either: {kept(pop.JOIN_OR)}",
    )
    check(
        kept(pop.JOIN_XOR) == ["onlyA", "onlyB"],
        f"xor keeps what passes exactly one, and drops the one that passes both: "
        f"{kept(pop.JOIN_XOR)}",
    )

    # Chained xor is parity, which is what "an odd number of them" means.
    parity = pop.gate_mask(
        source,
        [low_a, pop.Gate("B", 50.0, np.inf, join=pop.JOIN_XOR),
         pop.Gate("A", 0.0, 3.0, join=pop.JOIN_XOR)],
    )
    alone = pop.gate_mask(source, [pop.Gate("B", 50.0, np.inf)])
    check(
        bool(np.array_equal(parity, alone)),
        "A xor B xor A is B on its own, so chaining really is parity",
    )

    # No precedence: the order on screen is the order applied.
    first = pop.gate_mask(
        source,
        [low_a, pop.Gate("B", 50.0, np.inf, join=pop.JOIN_OR),
         pop.Gate("A", 0.0, 3.0, join=pop.JOIN_AND)],
    )
    check(
        [str(n) for n, k in zip(source.obs_names, first) if k] == ["both", "onlyA"],
        "(A or B) and A is applied left to right, not as boolean precedence would",
    )
    check(
        pop.gate_expression(
            [low_a, pop.Gate("B", 50.0, np.inf, join=pop.JOIN_OR),
             pop.Gate("A", 0.0, 3.0, join=pop.JOIN_AND)]
        ).startswith("((A in [0, 3]) or B in [50, inf]) and"),
        "and the written rule brackets it the way it is applied",
    )

    # A switched-off gate is out of the rule entirely, join and all.
    off = [low_a, pop.Gate("B", 50.0, np.inf, enabled=False, join=pop.JOIN_OR)]
    check(
        [str(n) for n, k in zip(source.obs_names, pop.gate_mask(source, off)) if k]
        == ["both", "onlyA"],
        "switching a gate off removes its join too, rather than leaving an or behind",
    )
    check(
        pop.gate_expression(off) == low_a.describe(),
        f"and the rule says so: {pop.gate_expression(off)!r}",
    )
    check(
        pop.gate_expression([]) == "everything (no gate is on)",
        "no gates is not an empty rule, it is everything",
    )


def test_a_blank_abstains_rather_than_voting() -> None:
    """An object a gate cannot judge leaves the running answer alone.

    This is the one subtlety in combining gates, and getting it wrong is silent.
    A blank — a 4i cycle that never measured this feature — used to count as a
    pass, which is right for ``and`` and was all that was needed when ``and`` was
    the only option. Under ``or`` a pass is a vote: one unmeasured feature and the
    union keeps the whole plate.
    """
    print("a blank abstains rather than voting")

    if not _has("anndata"):
        print("  skip anndata is not installed")
        return

    import anndata as ad

    source = ad.AnnData(X=np.array([[1.0, 10.0], [9.0, np.nan]]))
    source.var_names = ["A", "B"]
    source.obs_names = ["measured", "blank_in_B"]

    passes_a = pop.Gate("A", 0.0, 3.0)        # keeps "measured" only
    needs_b = pop.Gate("B", 50.0, np.inf)     # "blank_in_B" has no value

    anded = pop.gate_mask(source, [passes_a, pop.Gate("B", 50.0, np.inf, join=pop.JOIN_AND)])
    check(
        bool(anded[0]) is False and bool(anded[1]) is False,
        f"under and, the blank is not kept by the gate before it failing: {anded}",
    )
    ored = pop.gate_mask(source, [passes_a, pop.Gate("B", 50.0, np.inf, join=pop.JOIN_OR)])
    check(
        bool(ored[1]) is False,
        "under or, a blank does not keep an object the other gates rejected",
    )
    check(
        bool(ored[0]) is True,
        "while an object a gate did judge is still kept",
    )

    # On its own, though, a gate must not throw away what was never measured:
    # that is the whole reason blanks are not simply failed.
    only_b = pop.gate_mask(source, [needs_b])
    check(
        bool(only_b[1]) is True,
        "one gate on its own keeps the object it cannot judge",
    )
    check(
        bool(np.array_equal(needs_b.mask(np.asarray([np.nan, 90.0, 10.0])), [True, True, False])),
        "which is what Gate.mask means by keeping blanks",
    )

    inside, blank = needs_b.decide(np.asarray([np.nan, 90.0]))
    check(
        not bool(inside[0]) and bool(blank[0]) and bool(inside[1]) and not bool(blank[1]),
        "decide() keeps the two apart, which is what lets the join choose",
    )
    check(
        pop.JOIN_IDENTITY[pop.JOIN_AND] is True
        and pop.JOIN_IDENTITY[pop.JOIN_OR] is False
        and pop.JOIN_IDENTITY[pop.JOIN_XOR] is False,
        "a blank takes the operator's identity, which is what abstaining is",
    )


def test_the_report_owns_up_to_or(directory: Path) -> None:
    print("what the report says about a rule that is not a plain and")

    if not _has("anndata"):
        print("  skip anndata is not installed")
        return

    source = db.read(_plate_file(directory))
    feature = "Voxels"
    values = np.asarray(pop.values_of(source, feature), dtype=float)
    low, high = (float(v) for v in np.percentile(values[np.isfinite(values)], [10, 90]))

    tails = [
        pop.Gate(feature, float("-inf"), low),
        pop.Gate(feature, high, float("inf"), join=pop.JOIN_OR),
    ]
    report = pop.gate_report(source, tails)
    check(report.combined, "the report says the rule is not a plain and")
    check(
        pop.JOIN_OR in report.expression,
        f"and carries the rule itself: {report.expression}",
    )
    check(
        "do not add up" in report.describe(),
        "and warns that the per-gate counts do not add up to the total",
    )
    check(
        0 < report.n_kept < report.n_objects,
        f"two opposite tails keep some of the objects, not none and not all "
        f"({report.n_kept} of {report.n_objects})",
    )

    # The same two windows ANDed are disjoint, so they keep nothing -- and that
    # is exactly why a second window on one feature needed `or` to be useful.
    anded = pop.gate_report(source, [tails[0], pop.Gate(feature, high, float("inf"))])
    check(
        anded.n_kept == 0 and not anded.combined,
        f"ANDed, the same two windows keep nothing ({anded.n_kept})",
    )

    plain = pop.gate_report(source, [tails[0]])
    check(
        "do not add up" not in plain.describe(),
        "a plain and is not warned about",
    )


def test_the_two_feature_view(directory: Path) -> None:
    """Two features, the gates' bounds on them, and the rule in the colour.

    A bound chosen on a histogram can sit in a dip of one feature and straight
    through a cloud in two. This is the view that shows that, so what it has to
    get right is: the bounds belong to the axes they are drawn on, the colour is
    the whole rule and not just the two gates in the picture, and anything the
    picture cannot show is said out loud.
    """
    print("two features, with the bounds on them")

    if not _has("anndata"):
        print("  skip anndata is not installed")
        return

    import anndata as ad

    matrix = np.array(
        [
            [1.0, 90.0],      # in the x window, in the y window
            [1.0, 10.0],      # in x, out of y
            [9.0, 90.0],      # out of x, in y
            [9.0, 10.0],      # out of both
            [np.nan, 50.0],   # no x at all
        ]
    )
    source = ad.AnnData(X=matrix)
    source.var_names = ["A", "B"]
    source.obs_names = ["in_both", "in_x", "in_y", "in_neither", "no_x"]

    gates = [pop.Gate("A", 0.0, 3.0), pop.Gate("B", 50.0, np.inf)]
    view = pop.feature_scatter(source, "A", "B", gates)

    check(view is not None, "a view is built for two features the file has")
    check(
        view.x_feature == "A" and view.y_feature == "B",
        "it knows which feature is on which axis",
    )
    check(
        view.x_bounds == ((0.0, 3.0),) and view.y_bounds == ((50.0, np.inf),),
        f"each gate's bounds go to the axis its feature is on: "
        f"x={view.x_bounds} y={view.y_bounds}",
    )
    check(
        [str(name) for name in source.obs_names[view.rows]]
        == ["in_both", "in_x", "in_y", "in_neither"],
        f"an object with no value for an axis is not drawn: "
        f"{[str(n) for n in source.obs_names[view.rows]]}",
    )
    check(view.n_blank == 1, f"and is counted as undrawable rather than lost ({view.n_blank})")
    check(
        [bool(k) for k in view.kept] == [True, False, False, False],
        f"the colour is the whole rule, so only the object in both windows is kept: "
        f"{[bool(k) for k in view.kept]}",
    )
    check(view.n_kept == 1, f"and the count agrees ({view.n_kept})")

    # A gate on neither axis still gates, and the view says so.
    third = gates + [pop.Gate("A", -np.inf, np.inf)]
    plain = pop.feature_scatter(source, "A", "B", third)
    check(
        plain.elsewhere == (),
        "a third gate on a feature already on an axis is not called elsewhere",
    )
    source.obs["extra"] = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    hidden = pop.feature_scatter(
        source, "A", "B", gates + [pop.Gate("extra", 5.0, 6.0)]
    )
    check(
        hidden.elsewhere == ("extra",),
        f"a gate on neither axis is named: {hidden.elsewhere}",
    )
    check(
        "which these axes do not show" in hidden.describe(),
        f"and the line under the plot says so: {hidden.describe()}",
    )
    check(
        not any(hidden.kept),
        "while still being applied — it excludes everything here, and the colour "
        "shows that even though the bound is off the picture",
    )

    check(
        pop.feature_scatter(source, "A", "not measured", gates) is None,
        "a feature the file does not have gives no view, rather than empty axes",
    )


def test_it_filters_the_way_the_analysis_panel_does(directory: Path) -> None:
    """The shared seams, checked rather than assumed.

    Two panels drawing the same measurements have to agree about what "all the
    points" and "the range of this column" mean, or the same data reads two ways
    in one application. These are the three places that could drift.
    """
    print("filtering the way the measurement analysis panel does")

    if not _has("anndata"):
        print("  skip anndata is not installed")
        return

    source = db.read(_plate_file(directory))
    feature = "Voxels"

    # 1. The same cap, limit and seed as the Measurement analysis scatter.
    small = pop.feature_scatter(source, feature, "Solidity", [], limit=7)
    check(
        len(small.x) == 7 and small.n_undrawn == source.n_obs - 7,
        f"the plot is capped and says how much it dropped "
        f"({len(small.x)} drawn, {small.n_undrawn} not)",
    )
    again = pop.feature_scatter(source, feature, "Solidity", [], limit=7)
    check(
        bool(np.array_equal(small.rows, again.rows)),
        "the sample is seeded, so the same data draws the same picture twice",
    )
    check(
        bool(np.array_equal(
            np.sort(small.rows),
            np.sort(np.flatnonzero(np.ones(source.n_obs, dtype=bool))[
                analysis.scatter_sample(source.n_obs, limit=7)
            ]),
        )),
        "and it is analysis.scatter_sample that chose them, not a second sampler",
    )
    check(
        "random sample" in small.describe() and "gating is over all of them" in small.describe(),
        f"the note says it is a sample and that the gating is not: {small.describe()}",
    )

    # 2. Capping the drawing must not change a single number.
    gates = [pop.Gate(feature, 300.0, 900.0)]
    full = pop.gate_report(source, gates)
    check(
        pop.feature_scatter(source, feature, "Solidity", gates, limit=5).n_selected
        == full.n_objects,
        "the view reports the whole selection, not the sample it drew",
    )
    check(
        int(pop.gate_mask(source, gates).sum()) == full.n_kept,
        "and the gating itself is over every object either way",
    )

    # 3. One answer to "the range of this column".
    values = np.asarray(pop.values_of(source, feature), dtype=float)
    low, high = analysis.value_range(
        values, pop.SUGGEST_LOW_PERCENTILE, pop.SUGGEST_HIGH_PERCENTILE
    )
    histogram = pop.feature_histogram(source, feature)
    check(
        abs(histogram.suggested_low - low) < 1e-9
        and abs(histogram.suggested_high - high) < 1e-9,
        f"the suggested bounds are analysis.value_range at this panel's percentiles "
        f"({histogram.suggested_low:.4g}, {histogram.suggested_high:.4g} vs "
        f"{low:.4g}, {high:.4g})",
    )
    check(
        pop.SUGGEST_LOW_PERCENTILE < analysis.DEFAULT_LOW_PERCENTILE
        and pop.SUGGEST_HIGH_PERCENTILE > analysis.DEFAULT_HIGH_PERCENTILE,
        f"deliberately tighter than a colour scale's "
        f"({pop.SUGGEST_LOW_PERCENTILE}/{pop.SUGGEST_HIGH_PERCENTILE} against "
        f"{analysis.DEFAULT_LOW_PERCENTILE}/{analysis.DEFAULT_HIGH_PERCENTILE}): a "
        "colour scale clips the tails, a gate is chosen by looking at them",
    )

    # 4. Blanks: the same test for "no value", in both modules.
    blanks = np.asarray([1.0, np.nan, np.inf, 3.0])
    check(
        bool(np.array_equal(
            np.isfinite(blanks),
            ~pop.Gate("x", -np.inf, np.inf).decide(blanks)[1],
        )),
        "a blank is np.isfinite in both, so neither counts an infinity as a value",
    )
    check(
        analysis.value_range(blanks)[1] > analysis.value_range(blanks)[0],
        "and value_range survives a column with blanks and infinities in it",
    )


def test_a_gate_can_keep_the_outside() -> None:
    """Exclude: a window round what you do not want.

    Not redundant with include. A single window can be inverted by hand -- two
    gates, below and above, ORed -- but that is three decisions to express one,
    and on a bounded measurement the outside of a window is not a window.
    """
    print("keeping what is outside the bounds")

    if not _has("anndata"):
        print("  skip anndata is not installed")
        return

    import anndata as ad

    source = ad.AnnData(X=np.array([[1.0], [5.0], [9.0], [np.nan]]))
    source.var_names = ["A"]
    source.obs_names = ["low", "middle", "high", "blank"]

    inside = pop.Gate("A", 3.0, 7.0)
    outside = pop.Gate("A", 3.0, 7.0, mode=pop.MODE_EXCLUDE)

    def kept(gate):
        mask = pop.gate_mask(source, [gate])
        return [str(name) for name, keep in zip(source.obs_names, mask) if keep]

    check(
        kept(inside) == ["middle", "blank"],
        f"inside keeps the objects in the window: {kept(inside)}",
    )
    check(
        kept(outside) == ["low", "high", "blank"],
        f"outside keeps the objects beyond it: {kept(outside)}",
    )
    check(
        not inside.excluding and outside.excluding,
        "and each knows which way round it is",
    )
    check(
        outside.describe() == "exclude A in [3, 7]" and inside.describe() == "A in [3, 7]",
        f"and says it in terms of the objects it points at: "
        f"{outside.describe()!r} against {inside.describe()!r}",
    )
    check(
        "not in" not in outside.describe(),
        "not as a negation, which reads as a claim about everything it is not "
        "pointing at",
    )
    check(
        pop.MODE_WORDS[pop.MODE_INCLUDE] == "keep"
        and pop.MODE_WORDS[pop.MODE_EXCLUDE] == "exclude",
        "with one word each for a table column",
    )
    check(
        "in the bounds" in pop.MODE_LABELS[pop.MODE_INCLUDE]
        and "in the bounds" in pop.MODE_LABELS[pop.MODE_EXCLUDE],
        f"and both labels name the same set, the objects in the bounds: "
        f"{tuple(pop.MODE_LABELS.values())}",
    )

    # The blank is the subtle one. Inverting it along with the values is the
    # obvious way to write exclude, and it makes an exclude gate keep every
    # unmeasured object -- on a 4i plate, most of them.
    check(
        "blank" in kept(outside) and "blank" in kept(inside),
        "a blank abstains under both, because it is no more outside the bounds "
        "than inside them",
    )
    _passes, blank = outside.decide(np.asarray([np.nan, 5.0]))
    check(
        bool(blank[0]) and not bool(blank[1]),
        "decide() still reports it as unjudged rather than as a pass",
    )

    # An exclude gate is exactly the two-window OR it saves you writing.
    by_hand = pop.gate_mask(
        source,
        [pop.Gate("A", -np.inf, 3.0), pop.Gate("A", 7.0, np.inf, join=pop.JOIN_OR)],
    )
    # ... bar the bounds themselves, which are inclusive on both sides.
    check(
        [str(n) for n, k in zip(source.obs_names, by_hand) if k] == kept(outside),
        "and it agrees with writing that OR out by hand",
    )

    histogram = pop.feature_histogram(source, "A")
    check(
        histogram.suggested_gate().mode == pop.MODE_INCLUDE
        and histogram.suggested_gate(pop.MODE_EXCLUDE).excluding,
        "the suggested bounds come back either way round, as asked",
    )


def main() -> int:
    directory = Path(tempfile.mkdtemp(prefix="mv-populations-"))
    try:
        test_scope_groups_speak_the_existing_vocabulary()
        print()
        test_the_rule_is_a_rule()
        print()
        test_gates_combine_with_and_or_xor()
        print()
        test_a_blank_abstains_rather_than_voting()
        print()
        test_a_gate_can_keep_the_outside()
        print()
        for name, test in (
            ("gates", test_samples_and_gates),
            ("joins", test_the_report_owns_up_to_or),
            ("scatter", test_the_two_feature_view),
            ("shared", test_it_filters_the_way_the_analysis_panel_does),
            ("histogram", test_the_histogram),
            ("positives", test_calling_positives),
            ("summary", test_the_summary),
            ("file", test_it_goes_back_onto_the_file),
            ("files", test_finding_the_files),
        ):
            case = directory / name
            case.mkdir()
            test(case)
            print()
    finally:
        shutil.rmtree(directory, ignore_errors=True)

    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
