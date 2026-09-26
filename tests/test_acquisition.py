"""The acquisition movie: a z-stack's projection building up, channel by channel."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microscopy_viewer import acquisition as aq  # noqa: E402
from microscopy_viewer.slides import ChannelView  # noqa: E402


def _stack(depth=12, size=64, spot_every=3):
    """A ZYX stack with one bright spot per few planes, each somewhere new."""
    data = np.zeros((depth, size, size), dtype=np.uint16)
    for z in range(0, depth, spot_every):
        y = x = 4 + z * 4
        data[z, y:y + 4, x:x + 4] = 1000
    data += 10
    return data


def _channel(label, color, data):
    return ChannelView(label=label, color=color, layer_name=label, data=data, axes="ZYX",
                       levels=[data], contrast_limits=(0.0, 1000.0))


def _spec(**kwargs):
    channels = [_channel("DAPI", (0.0, 0.4, 1.0), _stack()),
                _channel("GFP", (0.0, 1.0, 0.0), _stack()[::-1].copy())]
    defaults = dict(channels=channels, seconds_per_channel=1.0, fps=6.0, hold_s=0.5,
                    panel_pixels=64, labels=False, scale_bar=False)
    return aq.AcquisitionSpec(**{**defaults, **kwargs})


def test_the_projection_fills_in_plane_by_plane():
    stack = aq.stack_of(_channel("c", (1, 1, 1), _stack()), 64)
    snapshots, reached = aq.sweep(stack, 4, 64)
    assert reached == [2, 5, 8, 11]
    bright = [int((s > 500).sum()) for s in snapshots]
    assert bright == sorted(bright) and bright[0] < bright[-1]
    assert np.array_equal(snapshots[-1], _stack().max(axis=0).astype(np.float32))


def test_a_shallow_stack_repeats_frames_so_every_channel_takes_as_long():
    stack = aq.stack_of(_channel("c", (1, 1, 1), _stack(depth=3)), 64)
    snapshots, reached = aq.sweep(stack, 6, 64)
    assert len(snapshots) == 6 and reached == [0, 0, 1, 1, 2, 2]


def test_the_movie_has_a_panel_per_channel_and_a_merge_that_gains_each_one():
    spec = _spec()
    frames = list(aq.frames(spec))
    assert len(frames) == spec.total_frames() == 6 + 6 + 3
    height, width, _ = frames[0].shape
    assert width == 3 * 64 + 2 * aq.GAP and height == 64

    merge = slice(2 * (64 + aq.GAP), None)
    gfp = slice(64 + aq.GAP, 2 * 64 + aq.GAP)
    during_dapi = frames[5]
    assert during_dapi[:, gfp].max() == 0, "GFP is not acquired yet"
    assert during_dapi[:, merge, 1].max() < during_dapi[:, merge, 2].max()
    end = frames[-1]
    assert end[:, merge, 1].max() > 200 and end[:, merge, 2].max() > 200


def test_merge_only_is_one_panel():
    frame = next(iter(aq.frames(_spec(layout=aq.LAYOUT_MERGE))))
    assert frame.shape[:2] == (64, 64)


def test_labels_and_scale_bar_are_drawn():
    plain = list(aq.frames(_spec()))[-1]
    marked = list(aq.frames(_spec(labels=True, scale_bar=True, pixel_size_um=0.5)))[-1]
    assert marked.shape == plain.shape and not np.array_equal(marked, plain)


def test_stopping_raises_cancelled():
    with pytest.raises(aq.Cancelled):
        list(aq.frames(_spec(), should_cancel=lambda: True))


def test_it_writes_an_mp4(tmp_path):
    from microscopy_viewer.movie import encoder_available

    if not encoder_available()[0]:
        pytest.skip("no ffmpeg")
    written = aq.export(_spec(labels=True), tmp_path / "acq.mp4")
    assert written.exists() and written.stat().st_size > 1000
