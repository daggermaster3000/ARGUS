"""Read an object table back in, and paint the labels it describes.

A segmentation run writes one row per object — size, shape, intensity — and then
the table and the image go their separate ways: the numbers end up in Excel and
the masks stay in the viewer, with nothing tying a row to the object it came
from. This module is the way back. Give it the table a run wrote and it will

* say which column holds the label id, and which columns are worth plotting;
* turn any numeric column into a colour per label, so the segmentation can be
  looked at *as* the measurement — nuclei shaded by area, by mean intensity, by
  solidity — instead of as 18 000 arbitrary colours;
* guess which layer in the viewer the table belongs to, from the file name a
  batch run gave it;
* find one object in the image, from its centroid;
* hand the whole table to `AnnData <https://anndata.readthedocs.io>`_, which is
  what squidpy, scanpy and the rest of the single-cell stack read -- one file per
  image, or one for a whole plate with the well and cycle kept in ``obs``.

No Qt and no napari here: the widget is a thin layer over these functions, and
everything below is testable headless.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .utils import get_logger

logger = get_logger("analysis")

#: Column names that hold the object id, in the order they are looked for. The
#: first is what this viewer's own export writes; the rest are what the tables
#: from scikit-image, CellProfiler and Fractal call it.
LABEL_COLUMNS = ("Label", "label", "label_id", "ObjectNumber", "object_id", "id")

#: Colormaps offered for label colouring. Perceptually uniform ones first: a
#: measurement painted in jet is a measurement misread.
COLORMAPS = ("viridis", "magma", "plasma", "cividis", "turbo", "coolwarm", "Greys")

#: Percentile range the colour scale is stretched over by default. Object tables
#: routinely have a handful of enormous outliers — two nuclei merged into one —
#: and scaling to the true maximum leaves everything else the same dark blue.
DEFAULT_LOW_PERCENTILE = 1.0
DEFAULT_HIGH_PERCENTILE = 99.0

#: Points above which the scatter plot subsamples. Matplotlib will draw a million
#: points, but it will take seconds to do it and the result is a solid block.
MAX_SCATTER_POINTS = 100_000

TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".xlsx", ".xls")


def read_table(path: str | Path):
    """Read a measurement table into a DataFrame, from CSV or Excel."""
    import pandas as pd

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist")
    suffix = source.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        frame = pd.read_excel(source)
    else:
        separator = "\t" if suffix == ".tsv" else None
        # sep=None asks the sniffer to work it out, which handles the semicolon
        # CSVs a European Excel writes without the user having to know that.
        frame = pd.read_csv(source, sep=separator, engine="python" if separator is None else "c")
    if frame.empty:
        raise ValueError(f"{source.name} has no rows")
    frame.columns = [str(name).strip() for name in frame.columns]
    logger.info("read %s: %d row(s), %d column(s)", source.name, len(frame), len(frame.columns))
    return frame


def label_column(frame) -> str | None:
    """The column holding the object id, or None when the table has no such thing.

    A table without one can still be plotted; it just cannot colour anything, and
    saying so is better than colouring by row number and being quietly wrong.
    """
    for name in LABEL_COLUMNS:
        if name in frame.columns:
            return name
    for name in frame.columns:
        if str(name).strip().lower() in {"label", "labels", "id"}:
            return str(name)
    return None


def numeric_columns(frame, exclude: Sequence[str] = ()) -> list[str]:
    """Columns that can be coloured by or plotted, in table order."""
    import pandas as pd

    skip = {str(name) for name in exclude}
    return [
        str(name)
        for name in frame.columns
        if str(name) not in skip and pd.api.types.is_numeric_dtype(frame[name])
    ]


def value_range(
    values: Sequence[float],
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
) -> tuple[float, float]:
    """The range the colour scale covers, ignoring NaNs and the extreme tails."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.percentile(finite, max(0.0, min(100.0, low_percentile))))
    high = float(np.percentile(finite, max(0.0, min(100.0, high_percentile))))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:  # every object has the same value
        high = low + 1.0
    return low, high


def normalise(values: Sequence[float], low: float, high: float) -> np.ndarray:
    """Values mapped onto 0-1 and clipped, with NaN left as NaN."""
    array = np.asarray(values, dtype=float)
    span = float(high) - float(low)
    if span <= 0:
        span = 1.0
    scaled = (array - float(low)) / span
    return np.clip(scaled, 0.0, 1.0, out=scaled, where=np.isfinite(scaled))


def colormap_colors(fractions: Sequence[float], name: str = "viridis") -> np.ndarray:
    """RGBA for each 0-1 fraction; NaN comes back fully transparent."""
    from matplotlib import colormaps

    values = np.asarray(fractions, dtype=float)
    try:
        table = colormaps[name]
    except KeyError:
        logger.warning("unknown colormap %r; using viridis", name)
        table = colormaps["viridis"]
    colors = np.asarray(table(np.nan_to_num(values, nan=0.0)), dtype=float)
    colors[~np.isfinite(values)] = (0.0, 0.0, 0.0, 0.0)
    return colors


def label_colors(
    labels: Sequence[int],
    values: Sequence[float],
    colormap: str = "viridis",
    low: float | None = None,
    high: float | None = None,
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
) -> tuple[dict, tuple[float, float]]:
    """``({label: rgba}, (low, high))`` for a napari ``DirectLabelColormap``.

    Background and any label the table does not mention map to ``None``, which
    napari draws as transparent — so an object with no row simply is not painted,
    rather than being painted the colour of zero.
    """
    ids = np.asarray(labels)
    numbers = np.asarray(values, dtype=float)
    if ids.size != numbers.size:
        raise ValueError(
            f"{ids.size} label(s) and {numbers.size} value(s); they come from the same table"
        )
    if low is None or high is None:
        low, high = value_range(numbers, low_percentile, high_percentile)

    colors = colormap_colors(normalise(numbers, low, high), colormap)
    mapping: dict[Any, tuple[float, float, float, float]] = {
        None: (0.0, 0.0, 0.0, 0.0),  # background, and every label without a row
        0: (0.0, 0.0, 0.0, 0.0),
    }
    for label, color in zip(ids, colors):
        try:
            key = int(label)
        except (TypeError, ValueError):
            continue
        if key <= 0:
            continue
        mapping[key] = tuple(float(c) for c in color)
    return mapping, (float(low), float(high))


#: Alpha given to objects outside a selection. Not zero: the point of selecting a
#: cluster in the plot is to see where it *is* in the image, and that needs the
#: rest of the field still faintly there to place it against.
DIM_ALPHA = 0.12


def points_in_rectangle(
    x: Sequence[float], y: Sequence[float], x0: float, x1: float, y0: float, y1: float
) -> np.ndarray:
    """Boolean mask of the points inside a rectangle, in either drag direction."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    left, right = (x0, x1) if x0 <= x1 else (x1, x0)
    bottom, top = (y0, y1) if y0 <= y1 else (y1, y0)
    return (xs >= left) & (xs <= right) & (ys >= bottom) & (ys <= top)


def points_in_polygon(
    x: Sequence[float], y: Sequence[float], vertices: Sequence[Sequence[float]]
) -> np.ndarray:
    """Boolean mask of the points inside a lassoed polygon.

    Uses matplotlib's own point-in-path test, which is the same code that decides
    whether a click landed on a patch, so a lasso selects exactly what it looks
    like it encloses.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    points = np.column_stack([xs, ys])
    if len(vertices) < 3 or points.size == 0:
        return np.zeros(xs.shape, dtype=bool)
    from matplotlib.path import Path as MplPath

    return MplPath(np.asarray(vertices, dtype=float)).contains_points(points)


def dim_unselected(
    mapping: dict, selected_labels: Sequence[int], alpha: float = DIM_ALPHA
) -> dict:
    """A copy of a label colour mapping with everything outside the selection faded.

    The selected objects keep the colour their measurement gave them rather than
    turning some highlight colour: the question being asked is "where are the
    objects in that cluster", and the answer is easier to read when they still
    carry the value that put them in it.
    """
    chosen = {int(label) for label in selected_labels}
    if not chosen:
        return dict(mapping)
    faded = {}
    for key, colour in mapping.items():
        if key is None or not isinstance(key, (int, np.integer)) or int(key) <= 0:
            faded[key] = colour
            continue
        red, green, blue, opacity = (float(c) for c in colour)
        faded[key] = (red, green, blue, opacity if int(key) in chosen else opacity * float(alpha))
    return faded


#: Centroid columns as this viewer's exporter writes them, in ``(z, y, x)`` order.
#: These are already in micrometres, which is the world unit napari puts a
#: calibrated layer in, so a centroid is a viewer coordinate without conversion.
CENTROID_COLUMNS = ("Centroid Z (µm)", "Centroid Y (µm)", "Centroid X (µm)")

#: Fallbacks for tables written by something else. Matched case-insensitively on
#: a name with the units and separators stripped.
CENTROID_KEYS = {
    "z": ("centroidz", "z", "centroid0", "centerz"),
    "y": ("centroidy", "y", "centroid1", "centery"),
    "x": ("centroidx", "x", "centroid2", "centerx"),
}

#: How much of the shorter canvas edge one object should take up when the viewer
#: is sent to it. Not the whole canvas: an object with nothing around it is an
#: object you cannot place, and the neighbours are usually the reason you looked.
ZOOM_FILL = 0.25

#: Zoom used when the table says nothing about how big the object is.
FALLBACK_DIAMETER_UM = 20.0


def _squashed(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower().replace("µm", "").replace("um", ""))


def centroid_columns(frame) -> dict[str, str]:
    """``{axis: column}`` for whichever of z, y and x the table carries."""
    found: dict[str, str] = {}
    exact = {name: axis for name, axis in zip(CENTROID_COLUMNS, "zyx")}
    for name in frame.columns:
        axis = exact.get(str(name))
        if axis is not None:
            found[axis] = str(name)
    if len(found) >= 2:
        return found

    squashed = {_squashed(name): str(name) for name in frame.columns}
    for axis, keys in CENTROID_KEYS.items():
        if axis in found:
            continue
        for key in keys:
            if key in squashed:
                found[axis] = squashed[key]
                break
    return found


def object_centroid(frame, row: int) -> tuple[float, ...] | None:
    """The ``(z, y, x)`` centroid of one row in micrometres, or None.

    Z is dropped when the table has no Z column; a 2D table gives ``(y, x)``, and
    the caller pads it against the viewer's own dimensionality rather than
    guessing a plane here.
    """
    columns = centroid_columns(frame)
    if "y" not in columns or "x" not in columns:
        return None
    try:
        values = []
        for axis in ("z", "y", "x"):
            if axis not in columns:
                continue
            value = float(frame[columns[axis]].iat[int(row)])
            if not np.isfinite(value):
                return None
            values.append(value)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return tuple(values)


def object_diameter(frame, row: int) -> float:
    """How wide the object at *row* is, in micrometres, for choosing a zoom."""
    for name in frame.columns:
        if _squashed(name) in ("equivalentdiameter", "diameter", "equivdiameter"):
            try:
                value = float(frame[name].iat[int(row)])
            except (IndexError, TypeError, ValueError):
                break
            if np.isfinite(value) and value > 0:
                return value
            break
    return FALLBACK_DIAMETER_UM


def zoom_for(diameter_um: float, canvas_px: float, fill: float = ZOOM_FILL) -> float:
    """Canvas pixels per micrometre that puts an object of *diameter_um* at *fill*.

    napari's ``camera.zoom`` is exactly that ratio, so this is the whole of the
    arithmetic: how many screen pixels one micrometre should be worth for a
    nucleus to take up a quarter of the window.
    """
    diameter = float(diameter_um) if float(diameter_um) > 0 else FALLBACK_DIAMETER_UM
    span = max(1.0, float(canvas_px)) * float(fill)
    return float(span / diameter)


# ---------------------------------------------------------------------------
# Out to the single-cell stack
# ---------------------------------------------------------------------------

#: Columns kept out of the feature matrix: an identifier and three coordinates are
#: not measurements, and leaving them in ``X`` means every clustering in squidpy
#: is partly a clustering on position.
NON_FEATURE_COLUMNS = frozenset({"label", "labels", "label_id", "objectnumber", "object_id", "id"})


def feature_columns(frame, label_column: str | None = None) -> list[str]:
    """Numeric columns that are measurements, in table order."""
    centroids = set(centroid_columns(frame).values())
    skip = {str(label_column)} if label_column else set()
    return [
        name
        for name in numeric_columns(frame, exclude=skip)
        if name not in centroids and _squashed(name) not in NON_FEATURE_COLUMNS
    ]


def to_anndata(frame, label_column: str | None = None, source: str | Path | None = None):
    """The object table as an ``AnnData``, laid out the way squidpy expects.

    * ``X`` is the measurements, one row per object and one column per feature.
    * ``obs`` carries the label id, the centroids and any text column, indexed by
      the label so a result can be joined back to the mask it came from.
    * ``obsm["spatial"]`` is the centroid as ``(x, y)`` — or ``(x, y, z)`` when the
      table describes a volume — which is the array
      :func:`squidpy.gr.spatial_neighbors` builds its graph from.

    The label is deliberately *not* a feature. It is an identifier; clustering on
    it would be clustering on the order Cellpose happened to number things in.
    """
    import anndata as ad
    import pandas as pd

    if label_column is None:
        label_column = label_column_of(frame)
    features = feature_columns(frame, label_column)
    if not features:
        raise ValueError("the table has no numeric measurement columns to export")

    matrix = frame[features].to_numpy(dtype=np.float32, copy=True)

    observations = pd.DataFrame(index=_obs_index(frame, label_column))
    if label_column:
        observations["label"] = frame[label_column].to_numpy()
    centroids = centroid_columns(frame)
    for axis in ("z", "y", "x"):
        if axis in centroids:
            observations[f"centroid_{axis}_um"] = frame[centroids[axis]].to_numpy(dtype=float)
    for name in frame.columns:
        if name not in features and name != label_column and name not in centroids.values():
            observations[str(name)] = frame[name].to_numpy()

    adata = ad.AnnData(
        X=matrix,
        obs=observations,
        var=pd.DataFrame(index=pd.Index([str(name) for name in features], name="feature")),
    )

    spatial = _spatial_matrix(frame, centroids)
    if spatial is not None:
        adata.obsm["spatial"] = spatial
    else:
        logger.warning("the table has no centroid columns; obsm['spatial'] was not written")

    adata.uns["microscopy_viewer"] = {
        "source": "" if source is None else str(source),
        "image": component_from_name(Path(str(source)).stem) if source else "",
        "label_column": str(label_column or ""),
        "spatial_units": "micrometer",
        "spatial_axes": "xy" if spatial is not None and spatial.shape[1] == 2 else "xyz",
    }
    logger.info(
        "AnnData: %d object(s) x %d feature(s)%s",
        adata.n_obs,
        adata.n_vars,
        "" if spatial is None else f", spatial {spatial.shape[1]}D",
    )
    return adata


def _obs_index(frame, label_column: str | None):
    """Observation names: the label ids as strings, or the row numbers without one.

    Named ``object`` rather than ``label``: ``obs`` also carries the label as an
    integer column, and h5ad refuses an index whose name is a column holding
    different values — which strings and integers are.
    """
    import pandas as pd

    if label_column and label_column in frame.columns:
        return pd.Index([str(value) for value in frame[label_column]], name="object")
    return pd.Index([str(i) for i in range(len(frame))], name="object")


def _spatial_matrix(frame, centroids: dict[str, str]) -> np.ndarray | None:
    """``obsm['spatial']`` as ``(x, y)`` or ``(x, y, z)``, or None when there is none.

    X before Y because that is the order squidpy's plotting reads, and Z only when
    it varies: a plate image is one plane deep, and a column of zeros would make
    every neighbour graph a 3D one built on a degenerate axis.
    """
    if "y" not in centroids or "x" not in centroids:
        return None
    columns = [centroids["x"], centroids["y"]]
    if "z" in centroids:
        z = frame[centroids["z"]].to_numpy(dtype=float)
        if np.nanmax(z) - np.nanmin(z) > 0:
            columns.append(centroids["z"])
    return np.ascontiguousarray(frame[columns].to_numpy(dtype=float))


def label_column_of(frame) -> str | None:
    """:func:`label_column`, under a name that does not shadow a local variable."""
    return label_column(frame)


def anndata_path(table: str | Path) -> Path:
    """Where the ``.h5ad`` for a table goes: beside it, same stem."""
    source = Path(table)
    return source.with_suffix(".h5ad")


def write_anndata(
    frame, path: str | Path, label_column: str | None = None, source: str | Path | None = None
) -> Path:
    """Write the table as ``.h5ad`` and return where it went."""
    adata = to_anndata(frame, label_column=label_column, source=source)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(target)
    logger.info("wrote %s (%.1f kB)", target, target.stat().st_size / 1024)
    return target


#: Column the image is recorded under when tables are combined. ``image`` rather
#: than ``batch``: it is a well and a cycle, and calling it a batch invites it to
#: be handed to a batch-correction routine that has no business running here.
IMAGE_KEY = "image"


def combine_anndata(
    sources: Sequence[str | Path],
    label_column: str | None = None,
    keys: Sequence[str] | None = None,
    join: str = "outer",
):
    """One ``AnnData`` from several object tables, with the image kept in ``obs``.

    Each table becomes a block of observations tagged with the image it came from
    in ``obs[IMAGE_KEY]``, and the observation names are made unique by appending
    it — ``100-G/09/0`` — because label 100 exists in every well and concatenating
    without that would give forty-four objects the same name.

    *join* is ``outer`` on purpose. A 4i plate names its stains differently in
    every cycle, so two images can measure different columns; the inner join that
    is usual for single-cell data would silently drop every column they did not
    share, which here is most of them. Blanks in the result mean "this image did
    not measure that", which is the truth and is visible.
    """
    import anndata as ad

    paths = [Path(str(source)) for source in sources]
    if not paths:
        raise ValueError("no tables to combine")
    names = list(keys) if keys is not None else [component_from_name(p.stem) or p.stem for p in paths]
    if len(names) != len(paths):
        raise ValueError(f"{len(paths)} table(s) and {len(names)} name(s)")

    blocks: dict[str, Any] = {}
    columns: dict[str, set] = {}
    for path, name in zip(paths, names):
        frame = read_table(path)
        block = to_anndata(frame, label_column=label_column, source=path)
        blocks[name] = block
        columns[name] = set(block.var_names)

    shared = set.intersection(*columns.values()) if columns else set()
    everything = set().union(*columns.values()) if columns else set()
    if shared != everything:
        logger.warning(
            "the tables do not all measure the same columns: %d shared of %d in total; "
            "the missing ones are blank in the result",
            len(shared),
            len(everything),
        )

    combined = ad.concat(blocks, label=IMAGE_KEY, index_unique="-", join=join, merge="unique")
    combined.uns["microscopy_viewer"] = {
        "sources": [str(path) for path in paths],
        "images": names,
        "n_tables": len(paths),
        "spatial_units": "micrometer",
        "join": str(join),
    }
    logger.info(
        "combined %d table(s): %d object(s) x %d feature(s)",
        len(paths),
        combined.n_obs,
        combined.n_vars,
    )
    return combined


def write_combined_anndata(
    sources: Sequence[str | Path],
    path: str | Path,
    label_column: str | None = None,
    keys: Sequence[str] | None = None,
) -> Path:
    """Combine several object tables into one ``.h5ad`` and return where it went."""
    combined = combine_anndata(sources, label_column=label_column, keys=keys)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(target)
    logger.info("wrote %s (%.1f MB)", target, target.stat().st_size / 1024 / 1024)
    return target


def combine_frames(blocks: "Sequence[tuple[str, Any]]", label_column: str | None = None):
    """:func:`combine_anndata` for tables already in memory.

    ``blocks`` is ``[(image, frame), ...]``. Used by a batch run, which has the
    measured numbers in hand and should not write them to a CSV and read them back
    to build the combined file — a float that has been through a text file is not
    the float that was measured.
    """
    import anndata as ad

    if not blocks:
        raise ValueError("no tables to combine")
    parts = {
        str(name): to_anndata(frame, label_column=label_column, source=f"{name}.csv")
        for name, frame in blocks
    }
    combined = ad.concat(parts, label=IMAGE_KEY, index_unique="-", join="outer", merge="unique")
    combined.uns["microscopy_viewer"] = {
        "images": list(parts),
        "n_tables": len(parts),
        "spatial_units": "micrometer",
        "join": "outer",
    }
    logger.info(
        "combined %d table(s) in memory: %d object(s) x %d feature(s)",
        len(parts),
        combined.n_obs,
        combined.n_vars,
    )
    return combined


def scatter_sample(count: int, limit: int = MAX_SCATTER_POINTS, seed: int = 0) -> np.ndarray | None:
    """Indices to plot when there are too many points, or None to plot them all.

    Random rather than the first N: a table is written in label order, which on a
    plate runs roughly top-left to bottom-right, so the first N points would be a
    corner of the well rather than a sample of it.
    """
    if count <= limit:
        return None
    return np.random.default_rng(seed).choice(count, size=limit, replace=False)


def describe_column(values: Sequence[float]) -> str:
    """One line of distribution, for under the plot."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return "no numeric values"
    low, median, high = np.percentile(finite, [0, 50, 100])
    return (
        f"n = {finite.size}, min {low:.4g}, median {median:.4g}, max {high:.4g}"
        + ("" if finite.size == array.size else f" ({array.size - finite.size} blank)")
    )


def component_from_name(stem: str) -> str:
    """``G_07_0`` -> ``G/07/0``: the image a batch table was written for.

    :func:`microscopy_viewer.batch.run_batch` names each table after the path of
    the image inside the store with the separators flattened, so the path can be
    read back out of the file name and used to find the layer.
    """
    text = str(stem).strip()
    match = re.search(r"([A-Za-z]{1,2})[_\-/](\d{1,3})(?:[_\-/](\d{1,3}))?$", text)
    if not match:
        return ""
    parts = [part for part in match.groups() if part]
    return "/".join(parts)


def match_layer(stem: str, names: Sequence[str]) -> str | None:
    """The layer a table most likely belongs to, or None when nothing fits.

    Matched on the well and field in the file name rather than on the whole name,
    because the layer is called something like ``G/07 :: cycle 1 :: nuclei`` and
    the table ``G_07_0.csv``; what they have in common is the well.
    """
    candidates = [str(name) for name in names]
    if not candidates:
        return None
    component = component_from_name(Path(str(stem)).stem)
    if not component:
        return None

    parts = component.split("/")
    well = "/".join(parts[:2])
    scored: list[tuple[int, str]] = []
    for name in candidates:
        score = 0
        if well and well in name:
            score += 2
        if component in name:
            score += 3
        if score:
            scored.append((score, name))
    if not scored:
        return None
    best = max(score for score, _name in scored)
    return next(name for score, name in scored if score == best)
