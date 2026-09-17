"""Paint a clustering back onto the nuclei it came from, inside the plate.

A dashboard run ends with a phenotype per object — a row in a table saying that
label 4021 of well G/07 belongs to cluster 5. That is the wrong shape for looking
at: to see whether a phenotype sits at the rim of an organoid or through the
middle of it, the cluster has to be *in the image*, on the same grid as the
nuclei, in the viewer.

This module does that mapping. For each image it reads the nuclei mask the
clustering was computed from, replaces every label with its cluster id, and writes
the result into the plate as another NGFF ``labels`` group beside the original.
Nothing is overwritten: the nuclei stay exactly as they were, and the clustering
is a second label set that can be deleted without touching them.

**The cluster ids are not the object ids.** A cluster map has a handful of values
where the nuclei mask has tens of thousands. :func:`paint_clusters` is where that
mapping happens and it is the only arithmetic here worth checking.

**An object with no entry is painted zero**, the same as background. Whether that
happens to many objects or none is the caller's decision, not this module's: a
clustering runs on a sample, and
:func:`microscopy_viewer.dashboard.assign_all` is what gives the rest of the plate
a cluster before it gets here. Passing only the sampled objects leaves a label
layer that is almost entirely empty, which is why that is not the default.

**What produced it travels with it.** The method, the resolution, the features it
was computed from, the label set it was painted onto and how many objects were
assigned are written into the label group's attributes, so a clustering found in
a plate months later can say where it came from. :func:`describe` reads it back.

No Qt and no napari: the dashboard writes these from its own process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .utils import get_logger

logger = get_logger("clusters")

#: Attribute block a clustering is recorded under, inside its label group. The
#: presence of this key is also what marks a label set as a clustering rather than
#: a segmentation, which is how "replace every clustering" knows what to remove.
CLUSTER_KEY = "microscopy_viewer_clustering"

#: Default name for the label set written.
DEFAULT_NAME = "clusters"

#: Where the object id and the cluster live in a table of assignments.
LABEL_COLUMN = "label"
CLUSTER_COLUMN = "cluster"


@dataclass(frozen=True)
class ClusterRun:
    """What produced a clustering, written beside it and read back with it."""

    method: str = ""
    #: Cluster names in id order: ``names[0]`` is written as 1, and so on. Zero is
    #: kept for "no cluster", which is what an unassigned object gets.
    names: tuple[str, ...] = ()
    source: str = ""
    features: tuple[str, ...] = ()
    resolution: float | None = None
    n_objects: int = 0
    #: How many of :attr:`n_objects` had their cluster computed, and how many were
    #: predicted from their neighbours because the clustering never sampled them.
    #: Kept apart because they are not the same claim: one is a measurement of this
    #: object, the other is what the objects nearest it in feature space are.
    n_clustered: int = 0
    n_assigned: int = 0
    colors: tuple[str, ...] = ()
    created: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_attrs(self) -> dict[str, Any]:
        return {
            "method": str(self.method),
            "clusters": [str(name) for name in self.names],
            "source_labels": str(self.source),
            "features": [str(name) for name in self.features],
            "resolution": None if self.resolution is None else float(self.resolution),
            "n_objects": int(self.n_objects),
            "n_clustered": int(self.n_clustered),
            "n_assigned": int(self.n_assigned),
            "colors": [str(colour) for colour in self.colors],
            "created": self.created or time.strftime("%Y-%m-%d %H:%M:%S"),
            **self.extra,
        }

    def describe(self) -> str:
        parts = [self.method or "clustering", f"{len(self.names)} cluster(s)"]
        if self.source:
            parts.append(f"on {self.source}")
        if self.n_objects:
            parts.append(f"{self.n_objects:,} objects")
        if self.n_assigned:
            parts.append(
                f"{self.n_clustered:,} clustered, {self.n_assigned:,} assigned by neighbours"
            )
        return ", ".join(parts)


def paint_clusters(
    masks: np.ndarray,
    assignment: Mapping[int, int],
    dtype=np.uint16,
) -> tuple[np.ndarray, int]:
    """Replace each label in *masks* with its cluster id. Returns ``(painted, n)``.

    *assignment* maps an object's label to a cluster id counting from 1. A label
    with no entry is painted 0 — the same value as background — because an object
    the clustering did not see has no phenotype, and painting it into some
    "other" cluster would invent one. ``n`` is how many objects were assigned.

    Built as a lookup table rather than a loop over clusters: a well has tens of
    thousands of labels, and one pass over the mask through a table is the
    difference between a second and several minutes.
    """
    labels = np.asarray(masks)
    if labels.size == 0:
        return labels.astype(dtype, copy=True), 0
    highest = int(labels.max())
    table = np.zeros(highest + 1, dtype=dtype)
    assigned = 0
    for label, cluster in assignment.items():
        index = int(label)
        if 0 < index <= highest:
            table[index] = int(cluster)
            assigned += 1
    painted = table[np.clip(labels, 0, highest)]
    painted[labels == 0] = 0
    return painted, assigned


def assignments_from_frame(frame, component: str | None = None) -> dict[str, dict[int, int]]:
    """``{component: {label: cluster_id}}`` from a table of assignments.

    The table needs a label column and a cluster column; an ``image`` column
    splits it per image, which is what a plate-wide clustering has. Cluster ids
    are assigned from the categories in order, counting from 1, so they line up
    with :attr:`ClusterRun.names`.
    """
    label_column = _first_column(frame, (LABEL_COLUMN, "Label", "label_id", "ObjectNumber"))
    cluster_column = _first_column(frame, (CLUSTER_COLUMN, "Cluster", "leiden", "phenotype"))
    if label_column is None or cluster_column is None:
        raise ValueError(
            "the table needs a label column and a cluster column to be painted back"
        )

    names = cluster_names(frame[cluster_column])
    order = {name: index + 1 for index, name in enumerate(names)}

    image_column = _first_column(frame, ("image", "Image", "component"))
    out: dict[str, dict[int, int]] = {}
    if image_column is None:
        key = str(component or "")
        out[key] = {
            int(label): order[str(cluster)]
            for label, cluster in zip(frame[label_column], frame[cluster_column])
        }
        return out

    for image, label, cluster in zip(
        frame[image_column], frame[label_column], frame[cluster_column]
    ):
        out.setdefault(str(image), {})[int(label)] = order[str(cluster)]
    return out


def cluster_names(values) -> list[str]:
    """The cluster names in a stable order, categories first when there are any."""
    categories = getattr(getattr(values, "cat", None), "categories", None)
    if categories is not None:
        return [str(name) for name in categories]
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(str(value), None)
    return sorted(seen, key=_sort_key)


def _sort_key(name: str):
    """Sort ``2`` before ``10``, which a plain string sort does not."""
    text = str(name)
    return (0, int(text), "") if text.isdigit() else (1, 0, text)


def _first_column(frame, candidates: Sequence[str]) -> str | None:
    lowered = {str(name).lower(): str(name) for name in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return str(candidate)
        if str(candidate).lower() in lowered:
            return lowered[str(candidate).lower()]
    return None


# ---------------------------------------------------------------------------
# Into the plate
# ---------------------------------------------------------------------------


def _hex_to_rgba(colour: str, alpha: int = 255) -> list[int]:
    text = str(colour).lstrip("#")
    if len(text) != 6:
        return [128, 128, 128, alpha]
    return [int(text[i : i + 2], 16) for i in (0, 2, 4)] + [alpha]


def write_clusters(
    job,
    assignment: Mapping[int, int],
    run: ClusterRun,
    name: str = DEFAULT_NAME,
    source_labels: str = "",
    level: int = 0,
    overwrite: bool = True,
) -> tuple[Path, int]:
    """Paint one image's clustering into the plate. Returns ``(path, n_assigned)``.

    *job* is a :class:`~microscopy_viewer.batch.ImageJob`; *source_labels* names
    the label set the clustering was computed from, which is the one that gets
    painted. Overwriting is the default here and not in the segmentation writer:
    a clustering is cheap to recompute and is the sort of thing that gets redone
    at three resolutions in a row, while a segmentation is an hour of GPU.
    """
    import zarr

    from .batch import LABELS_GROUP, write_labels

    source = source_labels or run.source
    if not source:
        raise ValueError("which label set the clustering was computed on must be given")
    masks_path = job.path / LABELS_GROUP / source
    if not masks_path.is_dir():
        raise FileNotFoundError(f"{job.component} has no label set called {source!r}")

    store = zarr.open_group(str(masks_path), mode="r")
    level_path = _level_path(store, level)
    masks = np.asarray(store[level_path])
    painted, assigned = paint_clusters(masks, assignment)

    target = write_labels(job, name, painted, level=level, overwrite=overwrite)

    # The provenance, and the colours, written where a reader will find them.
    group = zarr.open_group(str(target), mode="a")
    group.attrs[CLUSTER_KEY] = {**run.as_attrs(), "source_labels": source}
    marker = dict(group.attrs.get("image-label") or {})
    if run.colors:
        marker["colors"] = [
            {"label-value": index + 1, "rgba": _hex_to_rgba(colour)}
            for index, colour in enumerate(run.colors)
        ]
    if run.names:
        marker["properties"] = [
            {"label-value": index + 1, "cluster": str(cluster)}
            for index, cluster in enumerate(run.names)
        ]
    group.attrs["image-label"] = marker

    logger.info(
        "%s: wrote %r — %d of %d objects assigned",
        job.component,
        name,
        assigned,
        int(masks.max()),
    )
    return target, assigned


def _level_path(store, level: int) -> str:
    """The dataset path for *level*, clamped to what the pyramid actually has."""
    entry = (store.attrs.get("multiscales") or [{}])[0]
    datasets = entry.get("datasets") or []
    paths = [str(dataset.get("path", index)) for index, dataset in enumerate(datasets)]
    if not paths:
        paths = sorted(key for key, _value in store.arrays())
    if not paths:
        raise ValueError("the label set has no arrays in it")
    return paths[max(0, min(int(level), len(paths) - 1))]


def write_plate_clusters(
    survey,
    assignments: Mapping[str, Mapping[int, int]],
    run: ClusterRun,
    name: str = DEFAULT_NAME,
    source_labels: str = "",
    level: int = 0,
    overwrite: bool = True,
    replace_existing: bool = False,
    progress=None,
) -> list[tuple[str, Path | None, int, str]]:
    """Paint a whole plate's clustering in. One row per image: ``(component, path, n, note)``.

    Never raises for one bad image: a plate of forty-four wells should not lose
    forty-three of them because one has no matching label set.
    """
    if replace_existing:
        removed = remove_clusterings(survey)
        logger.info("removed %d clustering(s) before writing", len(removed))

    rows: list[tuple[str, Path | None, int, str]] = []
    by_component = {job.component: job for job in survey.jobs}
    for component, assignment in assignments.items():
        job = by_component.get(str(component))
        if job is None:
            rows.append((str(component), None, 0, "no such image in this plate"))
            continue
        if progress is not None:
            progress(f"{component}: painting {len(assignment)} object(s)")
        try:
            path, assigned = write_clusters(
                job, assignment, run, name=name, source_labels=source_labels,
                level=level, overwrite=overwrite,
            )
        except Exception as exc:  # noqa: BLE001 - reported per image
            logger.exception("%s: could not write the clustering", component)
            rows.append((str(component), None, 0, f"{type(exc).__name__}: {exc}"))
            continue
        rows.append((str(component), path, assigned, ""))
    return rows


# ---------------------------------------------------------------------------
# Finding and removing them again
# ---------------------------------------------------------------------------


def read_run(group) -> ClusterRun | None:
    """The :class:`ClusterRun` recorded on a label group, or None if it is not one."""
    from .loaders.ome_zarr import ngff_attrs

    block = ngff_attrs(group).get(CLUSTER_KEY)
    if not isinstance(block, dict):
        return None
    return ClusterRun(
        method=str(block.get("method", "")),
        names=tuple(str(name) for name in (block.get("clusters") or [])),
        source=str(block.get("source_labels", "")),
        features=tuple(str(name) for name in (block.get("features") or [])),
        resolution=block.get("resolution"),
        n_objects=int(block.get("n_objects") or 0),
        n_clustered=int(block.get("n_clustered") or 0),
        n_assigned=int(block.get("n_assigned") or 0),
        colors=tuple(str(colour) for colour in (block.get("colors") or [])),
        created=str(block.get("created", "")),
    )


def clusterings_of(job) -> dict[str, ClusterRun]:
    """``{label set: run}`` for the clusterings stored inside one image."""
    from .loaders.ome_zarr import LABELS_GROUP, _label_names, child_group, open_group

    try:
        group = open_group(job.path)
    except Exception:  # pragma: no cover - an image listed but unreadable
        return {}
    if child_group(group, LABELS_GROUP) is None:
        return {}
    found: dict[str, ClusterRun] = {}
    for name in _label_names(group):
        node = child_group(child_group(group, LABELS_GROUP), name)
        if node is None:
            continue
        run = read_run(node)
        if run is not None:
            found[str(name)] = run
    return found


def remove_clusterings(survey, name: str | None = None) -> list[str]:
    """Delete every clustering in the plate, or every one called *name*.

    Only label sets carrying the clustering marker are touched — a segmentation
    is an hour of GPU and must not be swept away by a tidy-up meant for something
    that takes forty seconds to recompute.
    """
    import shutil

    import zarr

    from .loaders.ome_zarr import LABELS_GROUP, ngff_attrs

    removed: list[str] = []
    for job in survey.jobs:
        found = clusterings_of(job)
        for label_name in found:
            if name is not None and label_name != name:
                continue
            target = job.path / LABELS_GROUP / label_name
            try:
                shutil.rmtree(target)
            except OSError:
                logger.exception("could not remove %s", target)
                continue
            removed.append(f"{job.component}/{label_name}")
            # Take it out of the listing too, or the reader goes looking for it.
            try:
                labels_group = zarr.open_group(str(job.path / LABELS_GROUP), mode="a")
                listed = ngff_attrs(labels_group).get(LABELS_GROUP)
                if isinstance(listed, list):
                    labels_group.attrs[LABELS_GROUP] = [
                        str(entry) for entry in listed if str(entry) != label_name
                    ]
            except Exception:  # pragma: no cover - a store that refuses the edit
                logger.exception("could not update the labels listing in %s", job.component)
    return removed
