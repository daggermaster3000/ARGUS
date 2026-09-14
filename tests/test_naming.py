"""What the file names of an experiment can be read to say."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microscopy_viewer import naming  # noqa: E402


def test_the_genotype_field_is_read_out_of_the_name():
    assert naming.genotype_from_name("fish01_mut_20x.ims") == "mut"
    assert naming.genotype_from_name("fish02_wt_20x.ims") == "wt"
    assert naming.genotype_from_name("sample_HET_01.ims") == "het"


def test_spellings_of_one_genotype_collapse_onto_one_value():
    """Otherwise a bar chart of a folder grows three columns for one group."""
    assert naming.genotype_from_name("a_WT_1.ims") == "wt"
    assert naming.genotype_from_name("a_wildtype_1.ims") == "wt"
    assert naming.genotype_from_name("a_wild-type_1.ims") == "wt"
    assert naming.genotype_from_name("a_mutant_1.ims") == naming.genotype_from_name("a_mut_1.ims")


def test_a_genotype_word_inside_a_field_still_counts():
    """The real names have it inside a longer field, not alone in one."""
    assert naming.genotype_from_name("TUNEL_ift88 mut_Confocal - Blue_2025-04-08.ims") == "mut"
    assert naming.genotype_from_name("NOTEB-CTL.ims") == "ctl"


def test_a_name_saying_nothing_gives_nothing():
    """A blank column gets noticed and fixed; a wrong one becomes a result."""
    assert naming.genotype_from_name("TEB-BIOTIN_4_F1.ims") == ""
    assert naming.genotype_from_name("20250408_stack_003.ims") == ""


def test_a_genotype_word_is_not_found_inside_another_word():
    """``ki`` is a knock-in; ``kidney`` is not."""
    assert naming.genotype_from_name("kidney_sample_3.ims") == ""
    assert naming.genotype_from_name("mothership_2.ims") == ""


def test_an_explicit_pattern_wins_over_the_vocabulary():
    name = "exp_A12_geno-DMSO_mut.ims"
    assert naming.genotype_from_name(name) == "mut"
    assert naming.genotype_from_name(name, r"geno-([A-Za-z0-9]+)") == "DMSO"
    assert naming.genotype_from_name(name, r"geno-(?P<genotype>[A-Za-z0-9]+)") == "DMSO"


def test_a_pattern_that_does_not_match_falls_back_rather_than_failing():
    """One odd file in a folder must not empty the column for the rest."""
    assert naming.genotype_from_name("fish_mut_1.ims", r"geno-(\w+)") == "mut"
    assert naming.genotype_from_name("fish_mut_1.ims", r"geno-(\w+") == "mut"  # invalid regex


def test_genotypes_reads_a_whole_folder():
    names = ["a_wt_1.ims", "b_mut_2.ims", "c_3.ims"]
    assert naming.genotypes(names) == ["wt", "mut", ""]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
