"""Reading the slide-scanner sections: OME-TIFF, one 2-D plane per channel and z position.

The scans' OME-XML carries a few stray bytes after its closing ``</OME>``, so tifffile does not recognise them as
OME-TIFF and returns a bare stack of pages. The XML itself is intact: it is cut at its closing tag and read here for
the channel names, the pixel size and the order of the pages (``DimensionOrder``). A channel is then its pages, one
per z position; :func:`plane` is a channel's maximum over z, inside an optional window.

``load_path`` / ``release`` / ``plane`` stand where the analysis code expects a reader.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ChannelSpec:
    """One channel of a scan: its name, pixel size and pages (one per z, in z order)."""

    path: Path
    channel_name: str
    name: str
    #: (z, y, x) µm
    scale: tuple[float, float, float]
    shape_yx: tuple[int, int]
    pages: list[int] = field(default_factory=list)
    multiscale: bool = False

    @property
    def data(self):
        """The channel's (z, y, x) shape, for code that asks ``spec.data.shape``; read planes with :func:`plane`."""
        from types import SimpleNamespace

        return SimpleNamespace(shape=(len(self.pages), *self.shape_yx))


def _strip(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _ome(tif) -> ET.Element:
    raw = tif.pages[0].tags["ImageDescription"].value
    data = raw if isinstance(raw, bytes) else str(raw).encode("latin-1", "replace")
    start, stop = data.find(b"<OME"), data.rfind(b"</OME>")
    if start < 0 or stop < 0:
        raise ValueError("no OME-XML in the first page")
    declaration = data.find(b"<?xml")
    begin = declaration if 0 <= declaration < start else start
    return ET.fromstring(data[begin: stop + len(b"</OME>")].decode("utf-8", "replace"))


def load_path(path: str | Path) -> list[ChannelSpec]:
    """The channels of the scan at *path*."""
    import tifffile

    path = Path(path)
    with tifffile.TiffFile(str(path)) as tif:
        root = _ome(tif)
        n_pages = len(tif.pages)
        page_shape = tuple(int(n) for n in tif.pages[0].shape)
    pixels = next(e for e in root.iter() if _strip(e.tag) == "Pixels")
    size = {a: int(pixels.get(f"Size{a}", 1)) for a in "XYZCT"}
    order = str(pixels.get("DimensionOrder", "XYZCT")).upper()        # fastest first
    if size["Z"] * size["C"] * size["T"] != n_pages or page_shape != (size["Y"], size["X"]):
        raise ValueError(f"{path.name}: the OME header does not match the pages")
    px = float(pixels.get("PhysicalSizeX") or 1.0)
    py = float(pixels.get("PhysicalSizeY") or px)
    pz = float(pixels.get("PhysicalSizeZ") or 1.0)
    names = [(c.get("Name") or c.get("Fluor") or f"channel {i}").strip()
             for i, c in enumerate(e for e in root.iter() if _strip(e.tag) == "Channel")]
    names += [f"channel {i}" for i in range(len(names), size["C"])]
    # page index of (c, z, t): the axes of DimensionOrder after X and Y, fastest first
    strides, step = {}, 1
    for axis in order[2:]:
        strides[axis] = step
        step *= size[axis]
    specs = []
    for c in range(size["C"]):
        pages = [c * strides["C"] + z * strides["Z"] for z in range(size["Z"])]   # first time point
        specs.append(ChannelSpec(path=path, channel_name=names[c], name=f"{path.stem} :: {names[c]}",
                                 scale=(pz, py, px), shape_yx=(size["Y"], size["X"]), pages=pages))
    return specs


def release(path: str | Path, own_thread_only: bool = False) -> int:
    """Nothing is kept open between reads; here for the code that releases a scan when done."""
    return 0


def plane(spec: ChannelSpec, window: tuple[slice, slice] | None = None) -> np.ndarray:
    """*spec*'s channel, maximum over z, inside *window* (the whole plane when None)."""
    import tifffile

    best = None
    with tifffile.TiffFile(str(spec.path)) as tif:
        for index in spec.pages:
            page = tif.pages[index].asarray()
            page = page if window is None else page[window]
            best = page if best is None else np.maximum(best, page)
    return best


#: The name the analysis code reads planes by.
_plane = plane
