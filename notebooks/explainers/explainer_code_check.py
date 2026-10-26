"""Check the minimal code shown in the explainer (explainer_code.py) against the package.

Every block must compile; the junction steps, run on the section that worked
(034-033_s13), must give what microscopy_viewer.junction gives.

    python notebooks/explainers/explainer_code_check.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE / "explainer_code.py"


def blocks(path: Path = CODE) -> dict:
    """``{step: {"title", "source", "code"}}`` from the ``# %%`` markers of explainer_code.py."""
    text = path.read_text(encoding="utf-8")
    out = {}
    for chunk in text.split("\n# %% ")[1:]:
        head, _, body = chunk.partition("\n")
        step, title, source = (part.strip() for part in head.split("|", 2))
        out[step] = {"title": title, "source": source, "code": body.strip("\n") + "\n"}
    return out


def main() -> int:
    sys.path.insert(0, str(HERE.parents[1]))
    sys.path.insert(0, str(HERE.parent / "validation"))
    import junction_tuning as jt
    from microscopy_viewer import junction as jn

    failures = []

    def ok(cond, what):
        print(("  ok   " if cond else "  FAIL ") + what)
        if not cond:
            failures.append(what)

    found = blocks()
    for step, block in found.items():
        try:
            ast.parse(block["code"])
            ok(True, f"{step}: compiles")
        except SyntaxError as exc:
            ok(False, f"{step}: {exc}")
    namespace: dict = {}
    for step in ("tissue", "labelled", "seeds", "price", "mincut", "bending", "line_band"):
        exec(found[step]["code"], namespace)
    g = type("blocks", (), {k: staticmethod(v) if callable(v) else v for k, v in namespace.items()})

    name = "2025_09_26_Asssembloid_034-033_s13"
    ov, _band, _line, _base = jt.load_case(name)
    um, ch = ov.um, ov.channels
    valid = (ch > 0).all(axis=0)
    others = jn._others(ch, ov.nuclear, True)
    tissue, filled = g.tissue_mask([ch[ov.nuclear], *others], valid, um)
    ok(np.array_equal(tissue, jn.tissue_mask(ch[ov.nuclear], valid, um, others)), "tissue_mask = junction.tissue_mask")
    density = g.cell_density(ch[ov.label].astype(np.float32), tissue, filled, um)
    ok(np.allclose(density, jn.cell_density(ch[ov.label], tissue, filled, um), atol=1e-5),
       "cell_density = junction.cell_density")
    other_density = jn.cell_density(ch[ov.other], tissue, filled, um)
    hot, cold = g.find_seeds(filled, density, other_density)
    real_hot, real_cold, _ = jn.find_seeds(filled, density, other_density)
    ok(np.array_equal(hot, real_hot) and np.array_equal(cold, real_cold), "find_seeds = junction.find_seeds")
    cost = g.line_cost(ov.nuclear_mean, filled, um)
    ok(np.allclose(cost, jn.line_cost(ov.nuclear_mean, filled, um)), "line_cost = junction.line_cost")
    halves = g.min_cut(filled, hot, cold, cost)
    ok(np.array_equal(halves == 1, jn._min_cut(filled, hot, cold, cost) & filled),
       "min_cut = junction._min_cut (the labelled side)")
    bent = g.straighten(filled, cost, hot, cold, um, bend=jn.BEND_UM)
    real_halves = jn.split_tissue(filled, density, um, cost, seeds=(hot, cold), bend_um=jn.BEND_UM)
    agree = float(((bent == 1) == (real_halves == 1))[filled].mean())
    ok(agree > 0.995, f"straighten = junction.straighten_cut (pixels on the same side: {agree:.2%})")
    line = g.trace_line(real_halves, um)
    real_line = jn.trace_interface(real_halves, um)
    ok(np.allclose(line[:: max(1, int(round(40.0 / um)))], real_line), "trace_line = junction.trace_interface")

    # The whole pipeline block: junction_of, end to end, against junction.find_junction.
    exec(found["pipeline"]["code"].split("\nrows = []")[0], namespace)
    whole_line, whole_band = namespace["junction_of"](ov.channels, ov.nuclear_mean, valid, um,
                                                      nuclear=ov.nuclear, label=ov.label, other=ov.other)
    result = jn.find_junction(ov, jn.JunctionOptions(save=False))
    ok(np.allclose(whole_line[:: max(1, int(round(40.0 / um)))] * um, result.line_um),
       "junction_of: the same line as junction.find_junction")
    on_tissue = result.filled
    a, b = whole_band & on_tissue, result.band & on_tissue
    dice = 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
    ok(dice > 0.9, f"junction_of: the band agrees with find_junction's on the tissue (Dice {dice:.3f}; "
                   "the real one also rounds its corners and adds a rim past the outline)")
    print("all checks passed" if not failures else f"{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
