"""Re-run the OASIS paper's analysis from the raw scans and the stored junctions.

    python run.py                          # counting → sizes → cellpose → stats → browser
    python run.py --steps cellpose,stats   # the paper's counts and primary outcome from the shipped masks (~2 min)
    python run.py --infer                  # first re-make the Cellpose masks with the shipped model (GPU, hours)
    python run.py --segment                # first segment every section's nuclei again with Cellpose (GPU, ~1 day)
    python run.py --workers 8              # sections processed at a time (each ~2 GB of memory)
    python check.py                        # compare the re-run with the shipped results

Steps (each skips what it has already done, so an interrupted run continues):

  segment    optional; nuclei segmentation of every section (oasis/segment.py) → rerun/segmentation/
  counting   the AF488 mask (its noise SD, the home side, the section outline) (oasis/counting.py) → rerun/counting/
  sizes      tissue area, assembloid length, the slim cell cache (oasis/sizes.py) → rerun/derived/
  infer      optional; the self-trained Cellpose masks of every section (oasis/cpself.py, GPU) → rerun/counting/cellpose/
  cellpose   the paper's counts: Cellpose masks at 3, 5, 8 noise SD, home / crossed / junction region (oasis/cpself.py)
             → rerun/counting/cellpose_counts.csv
  stats      the primary outcome, WT3 ÷ GD1 of crossed cells per mm, and its sensitivity analyses (oasis/stats.py)
             → rerun/statistics/primary_outcome.csv
  browser    the interactive section browser, with full-resolution tiles and outlines
             → rerun/results/section_browser/ (~2–3 h)
  notebooks  optional, when notebooks/ is present (the paper package): 01_validation, 02_biology

Everything is written under ``rerun/`` beside ``data/`` (see oasis/paths.py); nothing shipped is changed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from oasis import paths  # noqa: E402

STEPS = ("counting", "sizes", "cellpose", "stats", "browser")
NOTEBOOKS = ("01_validation.ipynb", "02_biology.ipynb")


def run_notebooks() -> int:
    paths.NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for name in NOTEBOOKS:
        print(f"notebook {name} …", flush=True)
        started = time.perf_counter()
        code = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                               "--ExecutePreprocessor.timeout=-1", "--output-dir", str(paths.NOTEBOOKS),
                               str(HERE / "notebooks" / name)], cwd=HERE / "notebooks").returncode
        if code:
            print(f"notebook {name} FAILED", flush=True)
            return code
        print(f"notebook {name}: {time.perf_counter() - started:.0f} s → {paths.NOTEBOOKS / name}", flush=True)
    return 0


def run_browser(workers: int, limit: int = 0) -> int:
    from oasis.browser import browser_outlines, browser_small, browser_tiles, section_browser

    for module in (section_browser, browser_tiles, browser_outlines, browser_small):
        module.WORKERS = workers
    argv = [str(limit)] if limit else []
    code = section_browser.main(argv)
    code |= browser_tiles.main(argv)
    code |= browser_outlines.main(argv)
    code |= browser_small.main(argv)
    return code


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", action="store_true", help="segment every section's nuclei again with Cellpose first")
    parser.add_argument("--infer", action="store_true", help="re-make the AF488 Cellpose masks with the shipped model (GPU)")
    parser.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated, of: segment, infer, {', '.join(STEPS)}, notebooks")
    parser.add_argument("--workers", type=int, default=4, help="sections processed at a time")
    parser.add_argument("--sections", type=int, default=0, help="only the first N sections (to try the pipeline)")
    args = parser.parse_args(argv)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    if args.segment and "segment" not in steps:
        steps.insert(0, "segment")
    if args.infer and "infer" not in steps:
        steps.insert(steps.index("cellpose") if "cellpose" in steps else len(steps), "infer")
    names = paths.section_names()[: args.sections] if args.sections else None
    print(f"package: {paths.ROOT}\noutputs: {paths.OUT}\nsteps:   {', '.join(steps)}", flush=True)
    for step in steps:
        started = time.perf_counter()
        print(f"\n=== {step}", flush=True)
        if step == "segment":
            from oasis import segment
            code = segment.main(names)
        elif step == "counting":
            from oasis import counting
            code = counting.main(names, args.workers)
        elif step == "sizes":
            from oasis import sizes
            code = sizes.main(names, args.workers)
        elif step == "infer":
            from oasis import cpself
            code = cpself.infer(names)
        elif step == "cellpose":
            from oasis import cpself
            code = cpself.count(names)
        elif step == "stats":
            from oasis import stats
            code = stats.main()
        elif step == "notebooks":
            if not (HERE / "notebooks").is_dir():
                print("no notebooks/ here (they ship with the paper package only)")
                code = 2
            else:
                code = run_notebooks()
        elif step == "browser":
            code = run_browser(args.workers, args.sections)
        else:
            print(f"unknown step {step!r}")
            return 2
        print(f"=== {step}: {'done' if not code else 'FAILED'} in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
