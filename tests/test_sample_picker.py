"""Picking samples off a table: categories, matching, keep and draw."""

from pathlib import Path

import numpy as np
import pandas as pd

from microscopy_viewer import sample_picker as sp


def _table(n=60):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "object_id": [f"exp_s{i}" for i in range(n)],
        "file": [f"exp_s{i}.ome.tiff" for i in range(n)],
        "Ripped": rng.choice(["yes", "no", "review"], n),
        "Ripped confidence": rng.random(n),
        "Shape": rng.choice(["Fused pair", "Damaged", "Separated"], n),
        "Pass": ["no" if i % 6 == 5 else "yes" for i in range(n)],
        "notes": [np.nan] * n,
    })


def test_categories_skip_ids_scores_and_notes():
    assert sp.category_columns(_table()) == ["Ripped", "Shape", "Pass"]


def test_rows_match_files_by_name(tmp_path):
    table = _table(5)
    paths = [tmp_path / f"exp_s{i}.ome.tiff" for i in (0, 2, 4)]
    matched = sp.match_paths(table, paths)
    assert list(matched.notna()) == [True, False, True, False, True]
    assert matched[2] == paths[1]


def test_rows_match_on_stems_when_there_is_no_file_column(tmp_path):
    table = _table(3).drop(columns=["file"])
    matched = sp.match_paths(table, [tmp_path / "exp_s1.ome.tiff"])
    assert matched[1] == tmp_path / "exp_s1.ome.tiff"


def test_keep_and_draw_fill_every_group_reproducibly():
    table = _table()
    eligible = sp.keep_mask(table, {"Pass": ["yes"]})
    assert eligible.sum() == 50
    groups = [("Ripped", "yes"), ("Shape", "Damaged"), ("Shape", "Separated")]
    picks = sp.draw(table, eligible, groups, 5, seed=3)
    assert all(table.at[i, "Pass"] == "yes" for i in picks.rows)
    assert len(set(picks.rows)) == len(picks.rows)
    # Each group its own five: 3 × 5 distinct samples, each drawn for one group.
    assert len(picks.rows) == 15
    assert all(got == 5 for got, _have in picks.filled.values())
    assert all(len(names) == 1 for names in picks.groups.values())
    again = sp.draw(table, eligible, groups, 5, seed=3)
    assert again.rows == picks.rows
    assert sp.draw(table, eligible, groups, 5, seed=4).rows != picks.rows


def test_shared_picks_cover_every_group_with_fewer_samples():
    table = _table()
    eligible = sp.keep_mask(table, {"Pass": ["yes"]})
    # Every sample has a Ripped and a Shape: overlapping groups.
    groups = [("Ripped", "yes"), ("Ripped", "no"), ("Shape", "Damaged"), ("Shape", "Fused pair")]
    shared = sp.draw(table, eligible, groups, 5, seed=1, share=True)
    for name, (got, have) in shared.filled.items():
        assert got >= min(5, have), name
    assert len(shared.rows) < 20
    assert len(sp.draw(table, eligible, groups, 5, seed=1).rows) == 20


def test_small_group_gives_what_it_has_and_no_groups_gives_all():
    table = _table()
    eligible = sp.keep_mask(table, {"Pass": ["no"]})
    picks = sp.draw(table, eligible, [("Pass", "no")], 50)
    assert picks.filled["Pass: no"] == (10, 10)
    assert sp.draw(table, eligible, [], 5).rows == list(table.index[eligible])


def test_draw_all_takes_every_kept_sample_of_the_group():
    table = _table()
    eligible = sp.keep_mask(table, {"Shape": ["Fused pair", "Damaged"]})
    picks = sp.draw(table, eligible, [("Pass", "yes")], 0, seed=5)
    expected = [i for i in table.index[eligible] if table.at[i, "Pass"] == "yes"]
    assert picks.rows == expected            # all of them, in table order, whatever the seed
    assert picks.filled["Pass: yes"] == (len(expected), len(expected))
    assert sp.draw(table, eligible, [("Pass", "yes")], 0, seed=9).rows == expected


def test_saved_list_has_path_groups_and_seed(tmp_path):
    table = _table(20)
    matched = sp.match_paths(table, [tmp_path / f for f in table["file"]])
    picks = sp.draw(table, matched.notna(), [("Ripped", "yes")], 3, seed=7)
    out = sp.picks_table(table, picks, matched)
    assert list(out.columns[:2]) == [sp.PATH_COLUMN, sp.GROUPS_COLUMN]
    assert (out["seed"] == 7).all()
    assert all(Path(p).parent == tmp_path for p in out[sp.PATH_COLUMN])
    assert out[sp.GROUPS_COLUMN].str.contains("Ripped: yes").all()


def _file_list(n=60):
    # As Image_File_List.csv: its own name for the file column, numbers as numbers.
    return pd.DataFrame({
        "File_Name": [f"exp_s{i}.ome.tiff" for i in range(n)],
        "Genotype": ["WT3" if i % 2 else "GD1" for i in range(n)],
        "Experiment": [8 + i % 3 for i in range(n)],
        "Slide": [1 + i // 20 for i in range(n)],
    })


def test_file_list_adds_genotype_by_file_name():
    table = _table(10).drop(columns=["file"])          # matched on object_id, the stem
    files = _file_list(8).iloc[::-1]                   # other order, two files missing
    joined, found = sp.join_file_list(table, files)
    assert found == 8
    assert list(joined.columns[: len(table.columns)]) == list(table.columns)
    assert joined.at[3, "Genotype"] == "WT3" and joined.at[4, "Genotype"] == "GD1"
    assert joined.at[3, "Experiment"] == 8 and sp._text(joined.at[4, "Experiment"]) == "9"
    assert sp._text(joined.at[9, "Genotype"]) == sp.MISSING     # not in the list
    assert {"Genotype", "Experiment", "Slide"} <= set(sp.category_columns(joined))
    # Choosing by genotype works as for any column.
    eligible = sp.keep_mask(joined, {"Genotype": ["WT3"]})
    assert list(joined.index[eligible]) == [1, 3, 5, 7]
    picks = sp.draw(joined, sp.keep_mask(joined, {"Genotype": ["WT3", "GD1"]}),
                    [("Genotype", "WT3"), ("Genotype", "GD1")], 2, seed=0)
    assert picks.filled == {"Genotype: WT3": (2, 4), "Genotype: GD1": (2, 4)}


def test_file_list_fills_only_the_blanks_of_a_column_the_table_has():
    table = _table(4)
    table["genotype"] = ["KO", np.nan, "KO", np.nan]
    joined, found = sp.join_file_list(table, _file_list(4))
    assert found == 4
    assert "Genotype" not in joined.columns
    assert list(joined["genotype"]) == ["KO", "WT3", "KO", "WT3"]


def test_file_list_matching_nothing_changes_nothing():
    table = _table(4)
    joined, found = sp.join_file_list(table, _file_list(4).assign(File_Name=lambda f: "other_" + f["File_Name"]))
    assert found == 0
    assert list(joined.columns) == list(table.columns)


def test_reads_csv_and_excel(tmp_path):
    table = _table(4)
    table.to_csv(tmp_path / "t.csv", index=False)
    table.to_excel(tmp_path / "t.xlsx", index=False)
    for name in ("t.csv", "t.xlsx"):
        back = sp.read_table(tmp_path / name)
        assert list(back.columns) == list(table.columns)
        assert len(back) == 4
