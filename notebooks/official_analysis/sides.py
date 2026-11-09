"""Which side of the junction line a cell is on: home (the AF488 organoid) or crossed (the other organoid).

The definitions of the official analysis (5 Nov 2026):

* **crossed** — every AF488⁺ cell on the far side of the junction line, in the organoid where AF488 is not at
  home, however far past the line and whether or not it lies in the junction region;
* **home** — every AF488⁺ cell that has not crossed.

The line is the run's junction line (``RUN/cells/<name>.json`` ``line``, µm, (y, x)), **extended straight past
both ends** (along its end segments) so that every point of the section lies on one side or the other, not only
the points beside the line. The AF488 organoid — home — is the side holding more of the AF488 mask objects' area:
those of the counting method, the Gaussian 20 µm mask at median + 8 robust SD (``kmad8``; the 5 SD mask, used
before 9 Nov 2026, gives the same side in all 614 sections).

The side is worked out here from the line itself, for every table the same way. (The stored distance columns of
the run's cells and of ``af488_masks.py``'s objects each take their sign from their own points through
``junction_analysis.place_cells``, and in 4 of the 614 sections the two disagree.) No Qt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from microscopy_viewer import fusion as fu  # noqa: E402

#: How far the line is extended past each end (µm): beyond any section.
REACH_UM = 20000.0
#: The last stretch of the line whose direction the extension follows (µm), so one kinked vertex does not steer it.
END_UM = 60.0


def _end_direction(points: np.ndarray) -> np.ndarray:
    """Unit direction leaving the line at points[0], from the first END_UM of the line."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    reach = np.searchsorted(np.cumsum(steps), END_UM) + 1
    v = points[0] - points[min(reach, len(points) - 1)]
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([0.0, 0.0])


def extended_line(line_um) -> np.ndarray:
    """The line (y, x µm) with a straight piece of REACH_UM added at each end."""
    line = np.asarray(line_um, float).reshape(-1, 2)
    if len(line) < 2:
        return line
    head = line[0] + _end_direction(line) * REACH_UM
    tail = line[-1] + _end_direction(line[::-1]) * REACH_UM
    return np.vstack([head, line, tail])


def raw_side(yx_um, line_um) -> np.ndarray:
    """Signed distance (µm) of each point to the extended line; the sign's orientation is arbitrary but fixed."""
    yx = np.asarray(yx_um, float).reshape(-1, 2)
    if not len(yx):
        return np.zeros(0)
    raw, _beside, _length = fu.signed_distance(yx, extended_line(line_um))
    return np.asarray(raw, float)


def home_sign(raw: np.ndarray, weight) -> float:
    """Multiplier making ``away = raw × sign`` negative at home: home is the side with more *weight* (AF488 area)."""
    weight = np.asarray(weight, float)
    plus, minus = weight[raw > 0].sum(), weight[raw < 0].sum()
    return -1.0 if plus > minus else 1.0


def away(yx_um, line_um, sign: float) -> np.ndarray:
    """Distance past the junction line (µm): > 0 crossed (in the other organoid), < 0 at home."""
    return raw_side(yx_um, line_um) * sign
