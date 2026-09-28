"""Who drew each brain region, from the outline in the file to the workbook."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_sample_data as msd  # noqa: E402
from microscopy_viewer import analysis as an  # noqa: E402
from microscopy_viewer import ims_store, user  # noqa: E402


def _square(y, x, side):
    return np.array([[y, x], [y, x + side], [y + side, x + side], [y + side, x]], dtype=float)


def test_the_name_is_remembered_and_falls_back_to_the_account(tmp_path, monkeypatch):
    monkeypatch.setattr(user, "_file", lambda: tmp_path / "user.json")
    assert user.user_name() == user.account_name()
    user.set_user_name("  Ada Lovelace ")
    assert user.user_name() == "Ada Lovelace"
    assert user.credit() == {user.DRAWN_BY: "Ada Lovelace"}
    user.set_user_name("")
    assert user.user_name() == user.account_name()


def test_regions_carry_who_drew_them_into_every_region_sheet(tmp_path):
    sample = msd.write_ims(tmp_path / "fish1.ims", shape=(1, 4, 64, 64), n_channels=1)
    ims_store.save_rois(sample, [
        ims_store.StoredRoi("Tel", _square(2, 2, 3), attrs={user.DRAWN_BY: "Ada"}),
        ims_store.StoredRoi("OT", _square(4, 4, 3), attrs={user.DRAWN_BY: "Bob"}),
        ims_store.StoredRoi("OT", _square(0, 5, 2), attrs={user.DRAWN_BY: "Ada"}),
        ims_store.StoredRoi("Cb", _square(1, 1, 2)),  # saved before names were kept
    ])
    stored = ims_store.load_rois(sample)
    assert stored[0].attrs == {user.DRAWN_BY: "Ada"}

    sheets = an.workbook_sheets(an.analyse([sample], an.AnalysisOptions()))
    for name in an.REGION_SHEETS:
        frame = sheets[name]
        columns = list(frame.columns)
        assert columns[columns.index("Region") + 1] == an.DRAWN_BY_COLUMN, name
    features = sheets["Region features"].set_index("Region")[an.DRAWN_BY_COLUMN]
    assert features["Tel"] == "Ada" and features["OT"] == "Bob, Ada" and features["Cb"] == ""
    everything = sheets["Region intensities"]
    assert set(everything.loc[everything["Region"] == an.ALL_REGIONS, an.DRAWN_BY_COLUMN]) == {"Ada, Bob"}
