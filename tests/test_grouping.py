"""Group columns read off file names and folders, and unique sample names."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microscopy_viewer import grouping as gr  # noqa: E402

ROOT = Path("/data/experiment")


def paths(*relative):
    return [ROOT / r for r in relative]


def test_by_default_the_genotype_comes_from_the_file_name_as_before():
    labels = gr.label_samples(paths("fish01_mut_20x.ims", "fish02_wt_20x.ims"), root=ROOT)
    assert [(l.name, l.genotype, l.conditions) for l in labels] == [
        ("fish01_mut_20x", "mut", {}), ("fish02_wt_20x", "wt", {}),
    ]


def test_columns_can_be_read_from_the_folder_or_a_numbered_part():
    rules = [
        gr.ColumnRule(gr.GENOTYPE, source=gr.FOLDER, rule=gr.GENOTYPE_WORDS),
        gr.ColumnRule("Treatment", source=gr.FOLDER_ABOVE, rule=gr.WHOLE),
        gr.ColumnRule("Stage", source=gr.FILE_NAME, rule=gr.FIELD, argument="2"),
        gr.ColumnRule("Last", source=gr.FILE_NAME, rule=gr.FIELD, argument="-1"),
    ]
    [label] = gr.label_samples(paths("DMSO/mutant/fish1_5dpf_x20.ims"), rules, ROOT)
    assert label.genotype == "mut"
    assert label.conditions == {"Treatment": "DMSO", "Stage": "5dpf", "Last": "x20"}


def test_one_of_matches_whole_words_and_returns_them_as_typed():
    rule = gr.ColumnRule("Drug", source=gr.WHOLE_PATH, rule=gr.ONE_OF, argument="DMSO, drug")
    assert rule.value(ROOT / "batch2" / "fish_Drug_1.ims", ROOT) == "drug"
    assert rule.value(ROOT / "batch2" / "fish_nodrug_1.ims", ROOT) == ""
    assert rule.value(ROOT / "dmso" / "fish1.ims", ROOT) == "DMSO"


def test_a_pattern_takes_its_group_and_a_bad_one_finds_nothing():
    rule = gr.ColumnRule("hpf", rule=gr.PATTERN, argument=r"(\d+)hpf")
    assert rule.value(ROOT / "fish_48hpf_2.ims") == "48"
    assert gr.ColumnRule("x", rule=gr.PATTERN, argument="(").value(ROOT / "a.ims") == ""


def test_duplicate_names_get_what_tells_them_apart():
    rules = [gr.ColumnRule(gr.GENOTYPE), gr.ColumnRule("Treatment", source=gr.FOLDER, rule=gr.WHOLE)]
    labels = gr.label_samples(
        paths("DMSO/fish1_wt.ims", "drug/fish1_wt.ims", "drug/fish2_wt.ims"), rules, ROOT
    )
    assert [l.name for l in labels] == ["fish1_wt (DMSO)", "fish1_wt (drug)", "fish2_wt"]


def test_duplicates_no_column_separates_fall_back_to_the_folder_then_a_number():
    labels = gr.label_samples(paths("day1/fish1.ims", "day2/fish1.ims"), root=ROOT)
    assert [l.name for l in labels] == ["fish1 (day1)", "fish1 (day2)"]
    names = gr.unique_names([Path("/a/f.ims"), Path("/a/f.ims")], [{}, {}], None)
    assert names == ["f (a)", "f (a) #2"]


def test_headers_are_never_blank_or_repeated():
    rules = [gr.ColumnRule("anything"), gr.ColumnRule(""), gr.ColumnRule("Genotype"),
             gr.ColumnRule("Dose"), gr.ColumnRule("Dose")]
    assert gr.column_names(rules) == ["Genotype", "Condition 1", "Genotype 2", "Dose", "Dose 2"]


def test_rules_are_remembered(tmp_path):
    rules = [gr.ColumnRule(gr.GENOTYPE, source=gr.FOLDER),
             gr.ColumnRule("Drug", rule=gr.ONE_OF, argument="a, b")]
    gr.save_rules(rules, tmp_path / "rules.json")
    assert gr.load_rules(tmp_path / "rules.json") == rules
    assert gr.load_rules(tmp_path / "missing.json") == gr.default_rules()


def test_the_workbook_carries_the_columns_after_the_genotype(tmp_path):
    import numpy as np
    import tifffile

    from microscopy_viewer import analysis as an

    for folder in ("DMSO", "drug"):
        (tmp_path / folder).mkdir()
        tifffile.imwrite(tmp_path / folder / "fish1_wt.tif",
                         (np.random.default_rng(0).random((32, 32)) * 100).astype("uint16"))
    files = sorted(tmp_path.glob("*/fish1_wt.tif"))
    rules = [gr.ColumnRule(gr.GENOTYPE), gr.ColumnRule("Treatment", source=gr.FOLDER, rule=gr.WHOLE)]

    outcomes = an.analyse(files, an.AnalysisOptions())
    gr.apply(outcomes, gr.label_samples(files, rules, tmp_path))
    sheets = an.workbook_sheets(outcomes)

    samples = sheets["Samples"]
    assert list(samples.columns[:3]) == ["Sample", "Genotype", "Treatment"]
    assert list(samples["Sample"]) == ["fish1_wt (DMSO)", "fish1_wt (drug)"]
    assert list(samples["Treatment"]) == ["DMSO", "drug"]
    features = sheets["Region features"]
    assert set(features["Treatment"]) == {"DMSO", "drug"}
