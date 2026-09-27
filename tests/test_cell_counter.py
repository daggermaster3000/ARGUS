"""Counting cells by hand: dots as a label map."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microscopy_viewer import cell_counter as cc  # noqa: E402


def test_the_map_covers_the_image_in_yx_or_zyx():
    flat = cc.geometry_of((3, 8, 128, 160), (1, 2.0, 0.5, 0.5), (0, 0, 10.0, 20.0), "TZYX")
    assert flat.shape == (128, 160) and flat.scale == (0.5, 0.5) and flat.translate == (10.0, 20.0)
    deep = cc.geometry_of((3, 8, 128, 160), (1, 2.0, 0.5, 0.5), (0, 0, 0, 0), "TZYX", three_d=True)
    assert deep.shape == (8, 128, 160) and deep.scale == (2.0, 0.5, 0.5)
    with pytest.raises(ValueError):
        cc.geometry_of((1, 64, 64), (1, 1, 1), (), "TYX", three_d=True)


def test_a_dot_is_a_disc_of_the_asked_diameter_and_leaves_other_dots_alone():
    geometry = cc.Geometry(shape=(50, 50), scale=(0.5, 0.5), translate=(0.0, 0.0))
    data = np.zeros(geometry.shape, np.uint32)
    first = cc.dot_indices((20, 20), 5.0, geometry, data)
    data[first] = 1
    # 5 µm across at 0.5 µm/px: a disc of radius 5 px, about 80 pixels.
    assert 60 < len(first[0]) < 100
    second = cc.dot_indices((24, 20), 5.0, geometry, data)
    assert len(second[0]) and not (data[second] != 0).any(), "overlap is not painted over"


def test_a_dot_near_the_edge_is_clipped_and_a_tiny_one_still_marks_a_pixel():
    geometry = cc.Geometry(shape=(10, 10), scale=(1.0, 1.0), translate=(0.0, 0.0))
    edge = cc.dot_indices((0, 0), 6.0, geometry)
    assert all((i >= 0).all() and (i < 10).all() for i in edge)
    tiny = cc.dot_indices((4.2, 6.7), 0.2, geometry)
    assert len(tiny[0]) == 1


def test_positions_map_from_world_to_pixels():
    geometry = cc.Geometry(shape=(100, 100), scale=(0.5, 0.25), translate=(10.0, 0.0))
    assert np.allclose(cc.to_data((7.0, 15.0, 5.0), geometry), [10.0, 20.0])


def test_the_dot_under_a_click_is_found_and_the_cells_counted():
    data = np.zeros((40, 40), np.uint32)
    data[10:13, 10:13] = 4
    data[30:33, 30:33] = 9
    assert cc.label_near(data, (11, 11)) == 4
    assert cc.label_near(data, (14, 11), radius_px=3) == 4
    assert cc.label_near(data, (20, 20), radius_px=3) == 0
    where = cc.indices_of(data, 9, (31, 31), reach_px=5)
    assert len(where[0]) == 9
    assert cc.count(data) == 2
