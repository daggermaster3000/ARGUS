"""Finding the overviews in a scanned experiment folder, and the samples on them."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microscopy_viewer import overview as ov  # noqa: E402


def entry(name, x, y, size, depth=1):
    return SimpleNamespace(name=name, path=Path(f"/exp/{name}.ims"), shape=(depth, 64, 64),
                           stage_extent=(x, x + size, y, y + size, 0.0, 1.0))


def test_a_field_series_is_an_overview_with_the_samples_inside_it():
    fields = [entry(f"slide_F{i:04d}", x, y, 1000) for i, (x, y) in
              enumerate([(0, 0), (900, 0), (0, 900), (900, 900)])]
    inside, outside = entry("fish1", 200, 200, 200, depth=30), entry("fish2", 9000, 0, 200, depth=30)
    samples, overviews = ov.find_overviews([fields[2], inside, *fields[:2], outside, fields[3]])
    assert [s.name for s in samples] == ["fish1", "fish2"]
    [found] = overviews
    assert found.is_mosaic and [f.name for f in found.fields] == [f.name for f in fields]
    assert found.samples == [inside]


def test_a_flat_image_that_contains_samples_is_an_overview():
    overview = entry("5x_map", 0, 0, 5000)
    fish = entry("fish1", 1000, 1000, 300, depth=40)
    samples, [found] = ov.find_overviews([overview, fish])
    assert samples == [fish] and found.fields == [overview] and not found.is_mosaic


def test_a_stack_with_a_closeup_inside_stays_a_sample():
    wide, close = entry("fish1_20x", 0, 0, 1000, depth=40), entry("fish1_40x", 200, 200, 400, depth=40)
    samples, overviews = ov.find_overviews([wide, close])
    assert samples == [wide, close] and overviews == []


def test_a_flat_image_named_overview_is_one_even_alone():
    samples, [found] = ov.find_overviews([entry("Overview_slide2", 0, 0, 3000)])
    assert samples == [] and found.samples == []


def test_no_stage_position_nothing_is_taken():
    blind = entry("slide_F0000", 0, 0, 1000)
    blind.stage_extent = None
    other = entry("slide_F0001", 900, 0, 1000)
    other.stage_extent = None
    samples, overviews = ov.find_overviews([blind, other])
    assert samples == [blind, other] and overviews == []


def test_the_overview_picture_places_fields_by_stage_position():
    fields = [entry(f"s_F{i:04d}", 1000 * i, 0, 1000) for i in range(2)]
    [found] = ov.find_overviews(fields)[1]
    mosaic = ov.render_overview(found, lambda f, size: np.full((32, 32), 100.0 * (1 + ov.field_index(f.name))))
    assert mosaic is not None and mosaic.image.shape[1] > mosaic.image.shape[0]
    left, right = mosaic.image[:, : mosaic.image.shape[1] // 4], mosaic.image[:, -mosaic.image.shape[1] // 4:]
    assert right.mean() > left.mean()
