"""Populations across samples: which objects count, which are positive, and how many.

A plate ends as one ``.h5ad`` with every object in it, and the question asked of
that file is not spatial. It is: *of the cells in these wells, what fraction is
positive for this marker, and is that fraction different between conditions?*
Answering it takes three decisions, and every one of them is a decision someone
has to be able to see and change:

**Which samples.** A plate has wells that failed, wells that are controls and
wells that belong to a different experiment, and a percentage pooled over all of
them is a percentage of nothing. :func:`sample_mask` is the choice.

**Which objects are objects.** Segmentation produces debris, two nuclei fused
into one, and slices of nucleus at the edge of the field. :class:`Gate` is one
low-and-high bound on one feature; :func:`gate_report` says what each gate threw
away *on its own* and what survives all of them together, because a filter whose
cost is invisible is a filter that quietly removes the population of interest.

**What positive means.** The headline rule is ``median + 3 sd`` of a chosen
measurement, computed over a pool of samples rather than over one — see
:class:`PositiveRule`. It is a rule, not a number, so the same definition can be
applied to a plate that was acquired next week.

**Nothing here knows about Qt or Streamlit.** The napari panel and the dashboard
page are two front ends over these functions, and a threshold that means one
thing in the panel and another on the page is worse than having only one of them.

Three choices in here are not obvious and are worth stating.

*A blank value does not fail a gate.* On a 4i plate the cycles measure different
stains and the combined file is an outer join, so a feature measured in cycle 3
is blank for every object of cycle 1. Treating blank as "outside the bounds"
would empty those samples completely. Blanks are kept and counted separately;
:class:`GateReport` reports them.

*Gated-out objects do not set the threshold.* ``median + 3 sd`` is computed over
the objects that survived gating, because the debris the gates exist to remove is
exactly the thing with the extreme intensities that would move the standard
deviation.

*Normalising does not, by itself, move the call.* Mapping a measurement onto the
1-99 % range of its scope is an affine rescaling, and ``median + k sd`` is affine
too, so within one pool the same objects come out positive either way — bar those
in the clipped tails. What it changes is that the threshold is then reported on a
0-1 scale comparable between pools, and that a threshold pooled over a wide scope
stops being dominated by whichever image was brightest. The clipping is also the
one way normalising can mislead — the top of the range is a ceiling a whole
percent of objects share — so :func:`call_positives` compares at or above the
threshold and says in its notes what the toggle did, rather than letting someone
believe it did something it did not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import analysis
from .dashboard import (
    IMAGE_KEY,
    feature_names,
    images,
    obs_columns,
    row_column,
    values_of,
    well_of,
)
from .utils import get_logger

logger = get_logger("populations")

#: Stands in for the sample name when a file holds a single image and so has no
#: ``obs["image"]`` to name it by.
WHOLE_FILE = "(whole file)"

#: Bins in a gating histogram. Enough that a second mode a few hundred objects
#: wide is visible, few enough that a well of two hundred objects is not drawn as
#: a comb of ones.
DEFAULT_BINS = 60

#: Percentiles a gate's default bounds are suggested at. Tight enough to cut the
#: handful of fused objects that stretch every axis, wide enough that a real
#: bright population is inside them rather than gated away before it is seen.
SUGGEST_LOW_PERCENTILE = 0.5
SUGGEST_HIGH_PERCENTILE = 99.5


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


def sample_labels(adata) -> list[str]:
    """The samples in a file, in plate order. One ``obs["image"]`` is one sample.

    A single-image file has no such column; it is one sample called
    :data:`WHOLE_FILE`, so everything downstream can group by sample without a
    special case for the file that has only one.
    """
    found = images(adata)
    return found if found else [WHOLE_FILE]


def sample_of(adata) -> np.ndarray:
    """The sample each object belongs to, as an array of names."""
    if IMAGE_KEY in adata.obs:
        return adata.obs[IMAGE_KEY].astype(str).to_numpy()
    return np.full(int(adata.n_obs), WHOLE_FILE, dtype=object)


def sample_mask(adata, samples: Sequence[str] | None) -> np.ndarray:
    """Objects belonging to *samples*; everything when *samples* is None.

    An empty selection selects nothing rather than everything. "I unticked every
    well" and "I have not chosen yet" are different states, and collapsing them
    hands a plate-wide percentage to someone who just emptied the list.
    """
    if samples is None:
        return np.ones(int(adata.n_obs), dtype=bool)
    wanted = [str(name) for name in samples]
    if not wanted:
        return np.zeros(int(adata.n_obs), dtype=bool)
    return np.isin(sample_of(adata).astype(str), wanted)


def measurable_names(adata) -> list[str]:
    """Everything that can be gated or thresholded: the features, then numeric ``obs``.

    Features first because they are the measurements. The numeric ``obs`` columns
    are there because a centroid is occasionally what you want to gate on — the
    top edge of a field, say, where the nuclei are cut in half.
    """
    names = feature_names(adata)
    seen = set(names)
    return names + [
        name for name in obs_columns(adata, numeric_only=True) if name not in seen
    ]


def values_by_sample(adata, feature: str) -> dict[str, np.ndarray]:
    """``{sample: values}`` for one feature, which is what the scope machinery takes.

    The same shape :func:`microscopy_viewer.analysis.column_by_component` builds
    from a folder of CSVs, so :func:`~microscopy_viewer.analysis.scope_range` does
    not care whether the numbers came from a plate of tables or from one ``.h5ad``.
    """
    column = np.asarray(values_of(adata, feature), dtype=float)
    labels = sample_of(adata).astype(str)
    return {name: column[labels == name] for name in sample_labels(adata)}


def scope_groups(samples: Sequence[str], scope: str) -> dict[str, tuple[str, ...]]:
    """``{group: samples}`` — the pools a statistic of *scope* is computed over.

    Built through :func:`~microscopy_viewer.analysis.scope_components` so that
    "this cycle, every well" means here exactly what it means when the viewer
    paints a colour scale. One row per pool rather than one per sample: the
    samples of a pool share a threshold, and a table with the same number repeated
    forty-four times hides that they do.
    """
    names = [str(name) for name in samples]
    pools: dict[tuple[str, ...], list[str]] = {}
    for name in names:
        pooled = tuple(sorted(analysis.scope_components(name, names, scope)))
        pools.setdefault(pooled, []).append(name)
    return {
        _group_label(members[0], scope): tuple(pooled)
        for pooled, members in pools.items()
    }


def _group_label(sample: str, scope: str) -> str:
    """A name for the pool *sample* falls in, readable in a table of thresholds."""
    name = str(sample).strip("/")
    if scope == analysis.SCOPE_PLATE:
        return "whole plate"
    if scope == analysis.SCOPE_WELL:
        return well_of(name)
    if scope == analysis.SCOPE_CYCLE and "/" in name:
        return f"cycle {name.rsplit('/', 1)[-1]}"
    return str(sample)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


#: How a gate combines with everything before it.
#:
#: ``and`` narrows — an object has to pass this gate as well. ``or`` widens — it
#: may pass this one instead. ``xor`` keeps the objects that pass an odd number of
#: the gates so far, which over two gates is "one or the other but not both": the
#: way to ask for the objects two markers disagree about.
JOIN_AND = "and"
JOIN_OR = "or"
JOIN_XOR = "xor"
JOINS = (JOIN_AND, JOIN_OR, JOIN_XOR)

JOIN_LABELS = {
    JOIN_AND: "and — must also pass this",
    JOIN_OR: "or — may pass this instead",
    JOIN_XOR: "xor — passes an odd number of the gates",
}

#: What an object a gate cannot judge contributes to each join.
#:
#: This is the one subtle thing about combining gates, and getting it wrong is
#: silent. A blank — a 4i cycle that never measured this feature — means the gate
#: has no opinion, so it must contribute the **identity** of the operator it is
#: being combined with: ``True`` for ``and``, ``False`` for ``or`` and ``xor``.
#: Then an object the gate cannot judge leaves the running answer exactly as it
#: was.
#:
#: Treating a blank as a pass regardless — which is right for ``and``, and was
#: all this needed when ``and`` was the only option — makes ``or`` keep
#: everything: one unmeasured feature and the union is the whole plate.
JOIN_IDENTITY = {JOIN_AND: True, JOIN_OR: False, JOIN_XOR: False}


def _combine(running: np.ndarray, contribution: np.ndarray, join: str) -> np.ndarray:
    if join == JOIN_OR:
        return running | contribution
    if join == JOIN_XOR:
        return running ^ contribution
    return running & contribution


@dataclass(frozen=True)
class Gate:
    """One low-and-high bound on one feature, and how it joins the ones before it.

    Frozen so a gate can be compared with the one stored on a file: change a bound
    and it is a different gate, and a panel showing the old count beside the new
    bound would be lying.

    The bounds are inclusive and infinite by default, so a gate that has been added
    but not yet set removes nothing.

    :attr:`join` says how this gate combines with the result of every gate before
    it — see :data:`JOINS`. It is ignored on the first enabled gate, which has
    nothing to combine with.
    """

    feature: str
    low: float = float("-inf")
    high: float = float("inf")
    enabled: bool = True
    join: str = JOIN_AND

    def describe(self) -> str:
        low = f"{self.low:.4g}" if np.isfinite(self.low) else "-inf"
        high = f"{self.high:.4g}" if np.isfinite(self.high) else "inf"
        return f"{self.feature} in [{low}, {high}]" + ("" if self.enabled else " (off)")

    def decide(self, values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """``(inside, blank)`` — what this gate judges, and what it cannot judge.

        Kept apart because what to do with a blank depends on the operator it is
        about to meet; see :data:`JOIN_IDENTITY`.
        """
        array = np.asarray(values, dtype=float)
        blank = ~np.isfinite(array)
        inside = (array >= float(self.low)) & (array <= float(self.high))
        return inside & ~blank, blank

    def contribution(self, values: Sequence[float], join: str = JOIN_AND) -> np.ndarray:
        """:meth:`decide`, with blanks filled in with *join*'s identity."""
        inside, blank = self.decide(values)
        return np.where(blank, bool(JOIN_IDENTITY.get(join, True)), inside)

    def mask(self, values: Sequence[float]) -> np.ndarray:
        """Objects this gate keeps on its own. Blanks are kept.

        On its own means as an ``and``: a gate asked in isolation whether to keep
        an object it cannot judge should say yes. This is what :func:`gate_report`
        measures each gate's cost with.
        """
        return self.contribution(values, JOIN_AND)


@dataclass(frozen=True)
class GateCost:
    """What one gate removed, applied on its own to the whole selection.

    On its own rather than in sequence: in sequence the second gate looks cheap
    because the first already removed what they share, and the order the gates
    happen to be listed in is not information about the data.
    """

    gate: Gate
    n_removed: int
    n_blank: int
    missing: bool = False


@dataclass(frozen=True)
class GateReport:
    """What the artifact filter cost: per gate, and all of them together."""

    n_objects: int
    n_kept: int
    costs: tuple[GateCost, ...]
    missing: tuple[str, ...] = ()
    #: The rule the gates add up to, from :func:`gate_expression`.
    expression: str = ""
    #: Whether anything other than ``and`` is in it. With a plain ``and`` chain
    #: the per-gate costs are at least an upper bound on what each one did; with
    #: ``or`` or ``xor`` in the rule they do not add up to the total at all, and
    #: saying so is the difference between a table that informs and one that
    #: misleads.
    combined: bool = False

    @property
    def n_removed(self) -> int:
        return int(self.n_objects - self.n_kept)

    def describe(self) -> str:
        if not self.n_objects:
            return "no objects in the selection"
        percent = 100.0 * self.n_kept / self.n_objects
        text = f"{self.n_kept:,} of {self.n_objects:,} objects kept ({percent:.1f} %)"
        if self.expression:
            text += f"; rule: {self.expression}"
        if self.combined:
            text += (
                "; the per-gate counts are each gate on its own and do not add up to "
                "this, because the rule is not a plain and"
            )
        if self.missing:
            text += f"; {len(self.missing)} gate(s) name a feature this file does not have"
        return text


def applicable(adata, gates: Sequence[Gate]) -> list[Gate]:
    """The gates that will actually be applied: enabled, and on a feature we have."""
    known = set(measurable_names(adata))
    return [gate for gate in gates if gate.enabled and gate.feature in known]


def gate_expression(gates: Sequence[Gate]) -> str:
    """The rule the gates add up to, as it is applied: left to right.

    Written out because it cannot be inferred from a list of rows. There is no
    operator precedence and there are no brackets — ``A or B and C`` here means
    ``(A or B) and C``, not what boolean algebra would make of it. Precedence
    would be indefensible in a table of rows whose order is the only thing on
    screen; left to right is at least what the reader sees.
    """
    applied = [gate for gate in gates if gate.enabled]
    if not applied:
        return "everything (no gate is on)"
    text = applied[0].describe()
    for gate in applied[1:]:
        text = f"({text}) {gate.join} {gate.describe()}"
    return text


def gate_mask(
    adata, gates: Sequence[Gate], within: np.ndarray | None = None
) -> np.ndarray:
    """Objects the gates keep, inside *within*.

    The gates are folded left to right, each combining with the running answer
    through its own :attr:`Gate.join` — see :func:`gate_expression` for what that
    means and why there is no precedence.

    The first enabled gate has nothing to combine with, so its own join is
    ignored and an object it cannot judge is kept: one gate on its own should not
    throw away what was never measured.

    A gate naming a feature the file does not carry is skipped rather than failing
    the whole run — but not silently: :func:`gate_report` lists it, and the panel
    is expected to show that list.
    """
    selection = (
        np.ones(int(adata.n_obs), dtype=bool)
        if within is None
        else np.asarray(within, dtype=bool)
    )
    applied = applicable(adata, gates)
    if not applied:
        return selection.copy()

    keep = applied[0].contribution(values_of(adata, applied[0].feature), JOIN_AND)
    for gate in applied[1:]:
        keep = _combine(
            keep,
            gate.contribution(values_of(adata, gate.feature), gate.join),
            gate.join,
        )
    return keep & selection


def gate_report(
    adata, gates: Sequence[Gate], within: np.ndarray | None = None
) -> GateReport:
    """:func:`gate_mask`, with the cost of each gate alongside it."""
    selection = (
        np.ones(int(adata.n_obs), dtype=bool)
        if within is None
        else np.asarray(within, dtype=bool)
    )
    n_objects = int(selection.sum())
    known = set(measurable_names(adata))

    costs: list[GateCost] = []
    missing: list[str] = []
    for gate in gates:
        if gate.feature not in known:
            missing.append(str(gate.feature))
            costs.append(GateCost(gate, 0, 0, missing=True))
            continue
        values = np.asarray(values_of(adata, gate.feature), dtype=float)
        blank = (~np.isfinite(values)) & selection
        removed = (
            n_objects - int((gate.mask(values) & selection).sum()) if gate.enabled else 0
        )
        costs.append(GateCost(gate, int(removed), int(blank.sum())))

    kept = gate_mask(adata, gates, within=selection)
    applied = applicable(adata, gates)
    report = GateReport(
        n_objects,
        int(kept.sum()),
        tuple(costs),
        tuple(missing),
        expression=gate_expression(gates),
        combined=any(gate.join != JOIN_AND for gate in applied[1:]),
    )
    logger.info("gating: %s", report.describe())
    return report


# ---------------------------------------------------------------------------
# The histogram a threshold is set on
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureHistogram:
    """Everything needed to draw one feature's distribution, and nothing else.

    The widget draws what it is handed: no percentile, no bin edge and no suggested
    bound is computed inside a paint method, so the same numbers can be checked
    headless and the Streamlit page draws the same picture from the same call.
    """

    feature: str
    counts: np.ndarray
    edges: np.ndarray
    n_finite: int
    n_blank: int
    suggested_low: float
    suggested_high: float

    @property
    def centres(self) -> np.ndarray:
        """Bin midpoints, for a plot that wants one x per count."""
        return (self.edges[:-1] + self.edges[1:]) / 2.0

    def suggested_gate(self) -> Gate:
        """A gate at the suggested bounds, as a starting point to be moved."""
        return Gate(self.feature, self.suggested_low, self.suggested_high)


def feature_histogram(
    adata,
    feature: str,
    within: np.ndarray | None = None,
    bins: int = DEFAULT_BINS,
    low_percentile: float = SUGGEST_LOW_PERCENTILE,
    high_percentile: float = SUGGEST_HIGH_PERCENTILE,
) -> FeatureHistogram:
    """One feature's distribution over the chosen objects, with default bounds.

    Binned over the suggested range rather than the full one, with the objects
    outside it folded into the end bins: a single fused object a hundred times the
    size of the rest puts every real object in the first bin, and a histogram with
    one visible bar is a histogram no threshold can be set on.
    """
    values = np.asarray(values_of(adata, feature), dtype=float)
    if within is not None:
        values = values[np.asarray(within, dtype=bool)]
    finite = values[np.isfinite(values)]
    n_blank = int(values.size - finite.size)
    if finite.size == 0:
        edges = np.linspace(0.0, 1.0, int(max(2, bins)) + 1)
        return FeatureHistogram(
            str(feature), np.zeros(edges.size - 1, dtype=int), edges, 0, n_blank, 0.0, 1.0
        )

    low, high = (float(value) for value in np.percentile(finite, [low_percentile, high_percentile]))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:  # every object has the same value
        high = low + 1.0
    counts, edges = np.histogram(
        np.clip(finite, low, high), bins=int(max(2, bins)), range=(low, high)
    )
    return FeatureHistogram(
        str(feature), counts, edges, int(finite.size), n_blank, low, high
    )


# ---------------------------------------------------------------------------
# What counts as positive
# ---------------------------------------------------------------------------

#: How the threshold is computed.
#:
#: ``median+ksd``  the headline rule: the median of the gated objects plus *k*
#:                 standard deviations. The median rather than the mean because
#:                 the positives are themselves inside the distribution the
#:                 threshold is derived from, and they drag a mean upwards — on a
#:                 well that is 30 % positive, far enough to miss half of them.
#: ``mean+ksd``    the same with a mean, for matching a protocol written that way.
#: ``percentile``  a flat percentile, which assumes the positive fraction instead
#:                 of measuring it. A sanity check, not a result.
STAT_MEDIAN_SD = "median+ksd"
STAT_MEAN_SD = "mean+ksd"
STAT_PERCENTILE = "percentile"

STATISTICS = (STAT_MEDIAN_SD, STAT_MEAN_SD, STAT_PERCENTILE)

STATISTIC_LABELS = {
    STAT_MEDIAN_SD: "median + k x sd",
    STAT_MEAN_SD: "mean + k x sd",
    STAT_PERCENTILE: "percentile",
}

#: Standard deviations above the centre unless told otherwise. Three, which is the
#: rule this feature exists to apply.
DEFAULT_K = 3.0

#: Percentile :data:`STAT_PERCENTILE` uses unless told otherwise.
DEFAULT_PERCENTILE = 99.0


@dataclass(frozen=True)
class PositiveRule:
    """What makes an object positive, as a rule that can be stored and re-applied.

    Frozen and compared as a whole, the way
    :class:`~microscopy_viewer.dashboard.ClusteringParams` is: change any field and
    the populations are different populations.

    *scope* is what the statistic is pooled over, from
    :data:`microscopy_viewer.analysis.SCOPES`. It is the decision that matters
    most. A threshold per image lets every well define its own normality, which
    hides exactly the difference between wells a plate was run to find; a threshold
    per cycle gives one staining round one threshold across the plate, which is how
    a plate is usually read.

    *normalised* puts the measurement on the 0-1 range of *normalise_scope* first,
    through :func:`~microscopy_viewer.analysis.scope_range` — the same "normalise
    over" the viewer's colour scales use. See the module docstring for what that
    does and does not change.
    """

    feature: str
    statistic: str = STAT_MEDIAN_SD
    k: float = DEFAULT_K
    percentile: float = DEFAULT_PERCENTILE
    scope: str = analysis.DEFAULT_SCOPE
    normalised: bool = False
    normalise_scope: str = analysis.SCOPE_IMAGE

    def as_attrs(self) -> dict[str, Any]:
        return {
            "feature": str(self.feature),
            "statistic": str(self.statistic),
            "k": float(self.k),
            "percentile": float(self.percentile),
            "scope": str(self.scope),
            "normalised": bool(self.normalised),
            "normalise_scope": str(self.normalise_scope),
        }

    @classmethod
    def from_attrs(cls, block: Mapping) -> "PositiveRule":
        return cls(
            feature=str(block.get("feature", "")),
            statistic=str(block.get("statistic", STAT_MEDIAN_SD)),
            k=float(block.get("k", DEFAULT_K)),
            percentile=float(block.get("percentile", DEFAULT_PERCENTILE)),
            scope=str(block.get("scope", analysis.DEFAULT_SCOPE)),
            normalised=bool(block.get("normalised", False)),
            normalise_scope=str(block.get("normalise_scope", analysis.SCOPE_IMAGE)),
        )

    def describe(self) -> str:
        if self.statistic == STAT_PERCENTILE:
            rule = f"{self.percentile:g}th percentile"
        else:
            centre = "median" if self.statistic == STAT_MEDIAN_SD else "mean"
            rule = f"{centre} + {self.k:g} sd"
        signal = (
            "normalised over "
            + analysis.SCOPE_LABELS.get(self.normalise_scope, self.normalise_scope)
            if self.normalised
            else "raw signal"
        )
        where = analysis.SCOPE_LABELS.get(self.scope, self.scope)
        return f"{self.feature} > {rule}, pooled over {where}, {signal}"


@dataclass(frozen=True)
class PositiveCalls:
    """A rule's outcome: the signal, the thresholds it was compared with, the calls.

    *missing_samples* names the samples that carry no value for the feature at all
    — a 4i cycle that did not measure this stain. Those samples get no threshold
    and no positives, and saying which they are is the difference between a zero
    that is an answer and a zero that is a bug.
    """

    rule: PositiveRule
    values: np.ndarray
    positive: np.ndarray
    thresholds: Any
    missing_samples: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def n_positive(self) -> int:
        return int(np.count_nonzero(self.positive))

    def threshold_of(self, sample: str) -> float:
        """The threshold applied to *sample*, or NaN when it had none."""
        frame = self.thresholds
        rows = frame[frame["sample"].astype(str) == str(sample)]
        return float(rows["threshold"].iloc[0]) if len(rows) else float("nan")


def signal_values(adata, rule: PositiveRule) -> tuple[np.ndarray, tuple[str, ...]]:
    """The numbers a rule is applied to, and the samples that have none of them.

    Raw, or each sample's objects mapped onto the 1-99 % range of its
    *normalise_scope* pool by :func:`~microscopy_viewer.analysis.scope_range` —
    the one piece of normalisation arithmetic in the application, reused rather
    than reimplemented so "normalised" cannot come to mean two things.
    """
    raw = np.asarray(values_of(adata, rule.feature), dtype=float)
    labels = sample_of(adata).astype(str)
    pools = values_by_sample(adata, rule.feature)
    absent = tuple(
        name
        for name, values in pools.items()
        if not np.isfinite(np.asarray(values, dtype=float)).any()
    )
    if not rule.normalised:
        return raw, absent

    out = np.full(raw.shape, np.nan, dtype=float)
    for name in pools:
        where = labels == name
        if not where.any():
            continue
        span = analysis.scope_range(pools, name, rule.normalise_scope)
        out[where] = analysis.normalise(raw[where], span.low, span.high)
    return out, absent


def threshold_of_pool(
    values: Sequence[float], rule: PositiveRule
) -> tuple[float, float, float]:
    """``(centre, spread, threshold)`` for one pool, or NaNs when it is empty."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    if rule.statistic == STAT_PERCENTILE:
        centre = float(np.percentile(finite, float(rule.percentile)))
        return centre, float("nan"), centre
    centre = float(np.mean(finite) if rule.statistic == STAT_MEAN_SD else np.median(finite))
    # The population deviation, not the sample-corrected one. On tens of thousands
    # of objects the difference is in the fifth decimal, but it has to be the same
    # fifth decimal every time: a threshold that moves when the plate is re-exported
    # is a threshold nobody can explain.
    spread = float(np.std(finite))
    return centre, spread, centre + float(rule.k) * spread


def call_positives(
    adata, rule: PositiveRule, keep: np.ndarray | None = None
) -> PositiveCalls:
    """Apply *rule*, pooling its statistic over each scope group of the kept objects.

    *keep* is what came out of the sample selection and the gates. Objects outside
    it are neither counted towards a threshold nor called positive: debris is the
    thing with the extreme intensities, and leaving it in the pool moves the
    threshold by exactly the amount the gates were drawn to prevent.
    """
    import pandas as pd

    n_obs = int(adata.n_obs)
    kept = np.ones(n_obs, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    signal, absent = signal_values(adata, rule)
    labels = sample_of(adata).astype(str)
    surviving = set(labels[kept].tolist())
    present = [name for name in sample_labels(adata) if name in surviving]

    notes: list[str] = []
    if rule.normalised and rule.normalise_scope == rule.scope:
        notes.append(
            "the signal is normalised over the same scope the threshold is pooled over, so the "
            "calls are the raw ones bar the clipped tails; what changes is that the threshold is "
            "read on a 0-1 scale"
        )
    if rule.normalised and rule.statistic == STAT_PERCENTILE and rule.percentile >= 99.0:
        notes.append(
            "a normalised signal is clipped at the 1-99 % of its pool, so a threshold this high "
            "lands on the ceiling and selects the whole clipped tail rather than a fraction of it"
        )

    rows: list[dict[str, Any]] = []
    positive = np.zeros(n_obs, dtype=bool)
    for group, pooled in scope_groups(present, rule.scope).items():
        in_pool = np.isin(labels, list(pooled)) & kept
        centre, spread, threshold = threshold_of_pool(signal[in_pool], rule)
        if np.isfinite(threshold):
            # At or above, not strictly above. A normalised signal is clipped at
            # the top of its range, so a threshold that lands on that ceiling is
            # shared by every object in the clipped tail; a strict comparison
            # calls none of them positive and reports zero where the answer is
            # "the brightest one per cent". On unclipped values the two agree.
            positive |= in_pool & (signal >= threshold)
        for name in pooled:
            rows.append(
                {
                    "sample": name,
                    "scope group": group,
                    "samples pooled": len(pooled),
                    "objects pooled": int(in_pool.sum()),
                    "centre": centre,
                    "sd": spread,
                    "threshold": threshold,
                    "objects": int(((labels == name) & kept).sum()),
                }
            )

    thresholds = pd.DataFrame(
        rows,
        columns=[
            "sample",
            "scope group",
            "samples pooled",
            "objects pooled",
            "centre",
            "sd",
            "threshold",
            "objects",
        ],
    ).sort_values("sample", kind="stable", ignore_index=True)

    missing = tuple(name for name in absent if name in surviving)
    if missing:
        notes.append(
            f"{len(missing)} sample(s) carry no {rule.feature} at all: "
            + ", ".join(missing[:6])
        )
    calls = PositiveCalls(rule, signal, positive, thresholds, missing, tuple(notes))
    logger.info(
        "positives: %s -> %d of %d kept object(s) in %d group(s)",
        rule.describe(),
        calls.n_positive,
        int(kept.sum()),
        thresholds["scope group"].nunique() if len(thresholds) else 0,
    )
    return calls


# ---------------------------------------------------------------------------
# The answer, per sample
# ---------------------------------------------------------------------------


def sample_summary(adata, calls: PositiveCalls, keep: np.ndarray | None = None):
    """One row per sample: objects, objects kept, positives, percent positive.

    The table the plots are drawn from and the one that gets exported, so the
    numbers in a figure and the numbers in the CSV beside it cannot disagree.

    ``objects`` counts everything in the sample and ``kept`` what survived the
    gates, both of them: a percentage computed over a tenth of a well is a
    percentage that needs its denominator printed next to it.
    """
    import pandas as pd

    n_obs = int(adata.n_obs)
    kept = np.ones(n_obs, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    labels = sample_of(adata).astype(str)
    positive = np.asarray(calls.positive, dtype=bool)
    signal = np.asarray(calls.values, dtype=float)

    rows: list[dict[str, Any]] = []
    for name in sample_labels(adata):
        mine = labels == name
        if not mine.any():
            continue
        survived = mine & kept
        n_kept = int(survived.sum())
        n_positive = int(np.count_nonzero(positive & survived))
        row, column = row_column(name)
        values = signal[survived]
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "sample": name,
                "well": well_of(name),
                "row": row,
                "column": column,
                "objects": int(mine.sum()),
                "kept": n_kept,
                "excluded": int(mine.sum()) - n_kept,
                "positive": n_positive,
                "percent positive": 100.0 * n_positive / n_kept if n_kept else float("nan"),
                "median signal": float(np.median(finite)) if finite.size else float("nan"),
                "threshold": calls.threshold_of(name),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "sample",
            "well",
            "row",
            "column",
            "objects",
            "kept",
            "excluded",
            "positive",
            "percent positive",
            "median signal",
            "threshold",
        ],
    )


def well_summary(summary):
    """:func:`sample_summary` pooled to one row per well, the cycles added together.

    A 4i well is imaged seven times and those are the same cells, so a bar chart
    per sample draws one condition seven times. The counts are summed and the
    percentage recomputed from the sums rather than averaged, because the cycles
    hold different numbers of objects.
    """
    import pandas as pd

    columns = ["well", "objects", "kept", "positive", "percent positive"]
    if summary is None or len(summary) == 0:
        return pd.DataFrame(columns=columns)
    grouped = summary.groupby("well", as_index=False)[["objects", "kept", "positive"]].sum()
    kept = grouped["kept"].replace(0, np.nan)
    grouped["percent positive"] = 100.0 * grouped["positive"] / kept
    return grouped[columns]


# ---------------------------------------------------------------------------
# Back onto the file
# ---------------------------------------------------------------------------

#: Where a run lives inside the ``.h5ad``, next to ``uns["clustering"]``.
POPULATIONS_KEY = "populations"

#: Columns written onto every object. ``population_`` in front of each, so a file
#: carrying both a clustering and a population analysis has no two columns that
#: could be mistaken for one another.
KEPT_COLUMN = "population_kept"
POSITIVE_COLUMN = "population_positive"
SIGNAL_COLUMN = "population_signal"


def run_populations(
    source,
    gates: Sequence[Gate],
    rule: PositiveRule,
    samples: Sequence[str] | None = None,
) -> dict:
    """Gate, threshold, and write the result into *source*. Returns the record.

    Mirrors :func:`~microscopy_viewer.dashboard.run_clustering`: the per-object
    answers go in ``obs`` and the definition goes in ``uns``, so a file re-opened
    next month still says which objects were excluded, what positive meant, and
    over what the threshold was pooled. A percentage without those three is not a
    result anybody can defend.
    """
    chosen = sample_mask(source, samples)
    report = gate_report(source, gates, within=chosen)
    keep = gate_mask(source, gates, within=chosen)
    calls = call_positives(source, rule, keep=keep)
    summary = sample_summary(source, calls, keep=keep)

    source.obs[KEPT_COLUMN] = keep
    source.obs[POSITIVE_COLUMN] = np.asarray(calls.positive, dtype=bool)
    source.obs[SIGNAL_COLUMN] = np.asarray(calls.values, dtype=float)

    n_kept = int(keep.sum())
    record = {
        **rule.as_attrs(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": [
            str(name)
            for name in (samples if samples is not None else sample_labels(source))
        ],
        "gate_features": [str(gate.feature) for gate in gates],
        "gate_low": np.asarray([float(gate.low) for gate in gates], dtype=float),
        "gate_high": np.asarray([float(gate.high) for gate in gates], dtype=float),
        "gate_enabled": np.asarray([bool(gate.enabled) for gate in gates], dtype=bool),
        "gate_join": [str(gate.join) for gate in gates],
        "gate_expression": gate_expression(gates),
        "gate_removed": np.asarray(
            [int(cost.n_removed) for cost in report.costs], dtype=int
        ),
        "n_objects": int(source.n_obs),
        "n_selected": int(chosen.sum()),
        "n_kept": n_kept,
        "n_positive": calls.n_positive,
        "percent_positive": 100.0 * calls.n_positive / n_kept if n_kept else float("nan"),
        "threshold_samples": [str(name) for name in summary["sample"]],
        "thresholds": np.asarray(summary["threshold"].to_numpy(), dtype=float),
        "missing_samples": [str(name) for name in calls.missing_samples],
        "notes": [str(note) for note in calls.notes],
    }
    source.uns[POPULATIONS_KEY] = record
    logger.info(
        "populations: %s; %d of %d kept object(s) positive",
        rule.describe(),
        calls.n_positive,
        n_kept,
    )
    return record


def load_populations(source) -> dict:
    """The stored run, or ``{}`` when the file has none.

    The gates and the rule come back as the objects they were rather than the
    arrays they were stored as, so a panel can put the user's own thresholds back
    in its boxes instead of offering them defaults over their own saved work.
    """
    block = source.uns.get(POPULATIONS_KEY)
    if not isinstance(block, (dict, Mapping)) or "feature" not in block:
        return {}
    if POSITIVE_COLUMN not in source.obs:
        return {}

    features = [str(name) for name in np.atleast_1d(block.get("gate_features", []))]
    lows = np.atleast_1d(np.asarray(block.get("gate_low", []), dtype=float))
    highs = np.atleast_1d(np.asarray(block.get("gate_high", []), dtype=float))
    enabled = np.atleast_1d(np.asarray(block.get("gate_enabled", []), dtype=bool))
    # Defaulting to "and" rather than failing: a file written before gates could
    # be combined has no join on them, and every one of those rules was an and.
    joins = [str(name) for name in np.atleast_1d(block.get("gate_join", []))]
    gates = [
        Gate(
            feature,
            float(lows[index]) if index < lows.size else float("-inf"),
            float(highs[index]) if index < highs.size else float("inf"),
            bool(enabled[index]) if index < enabled.size else True,
            joins[index] if index < len(joins) and joins[index] in JOINS else JOIN_AND,
        )
        for index, feature in enumerate(features)
    ]
    return {
        "rule": PositiveRule.from_attrs(block),
        "gates": gates,
        "samples": [str(name) for name in np.atleast_1d(block.get("samples", []))],
        "created": str(block.get("created", "")),
        "n_objects": int(block.get("n_objects", 0) or 0),
        "n_selected": int(block.get("n_selected", 0) or 0),
        "n_kept": int(block.get("n_kept", 0) or 0),
        "n_positive": int(block.get("n_positive", 0) or 0),
        "percent_positive": float(block.get("percent_positive", float("nan"))),
        "missing_samples": [
            str(name) for name in np.atleast_1d(block.get("missing_samples", []))
        ],
        "notes": [str(note) for note in np.atleast_1d(block.get("notes", []))],
    }


def save_populations(path: str | Path, source) -> Path:
    """Write the file back with the gates, the rule and the calls on it."""
    from .dashboard import save_h5ad

    return save_h5ad(path, source)


# ---------------------------------------------------------------------------
# Finding the files
# ---------------------------------------------------------------------------


def population_files(plate_path: str | Path) -> list[Path]:
    """The combined ``.h5ad`` files written beside a plate, newest first.

    Both places a run puts one: inside the analysis folder, where
    :meth:`microscopy_viewer.batch.BatchSettings.anndata_path` writes it, and
    beside that folder, where the analysis panel's *Combine* button writes it.
    Looked up rather than typed in, because what someone opening this panel has in
    hand is the plate they just scanned, not a path.
    """
    from . import explorer

    plate = Path(str(plate_path))
    found: list[Path] = []
    for folder in explorer.analysis_folders(plate):
        beside = folder.parent / f"{folder.name}.h5ad"
        if beside.exists() and beside not in found:
            found.append(beside)
        try:
            inside = sorted(folder.glob("*.h5ad"))
        except OSError:  # pragma: no cover - a folder that vanished mid-scan
            inside = []
        for path in inside:
            if path not in found:
                found.append(path)
    found.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    logger.info("found %d combined file(s) beside %s", len(found), plate.name)
    return found
