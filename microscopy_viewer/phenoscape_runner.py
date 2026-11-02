"""Run phenoscapes steps for the Phenoscape panel. Started as a child process.

    python phenoscape_runner.py <request.json>

Runs in whichever Python has phenoscapes, which is usually not napari's, so it
imports nothing from ARGUS: the standard library, yaml, pandas and phenoscapes,
and that is all. :mod:`microscopy_viewer.phenoscape` writes the request and
reads what this prints.

The steps are called the way ``phenoscapes.cli.process_sample`` calls them — the
same arguments, the same folders — so a run started here and a run started from
the original command line produce the same tree. Two differences, both because
this runs under a panel rather than in a terminal:

* **Only the steps asked for run.** The config's ``run`` flags still say what
  the pipeline *is* — whether the refined mask stands in for the simple one,
  whether background subtraction applies speckle masks — and the request says
  which of those steps to do now. Rerunning segmentation does not realign.
* **Output is not swallowed.** The original points ``sys.stdout`` at a log file
  for good; here every line goes to the panel *and* to ``logs/<sample>.log``.

Lines starting with ``@@phenoscape`` are for the panel; the rest is the
pipeline talking.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

MARK = "@@phenoscape"


class _Tee:
    """Write to the real stream and to a log file, line for line."""

    def __init__(self, stream, log):
        self._stream = stream
        self._log = log

    def write(self, text):
        self._stream.write(text)
        self._stream.flush()
        if self._log is not None:
            self._log.write(text)
            self._log.flush()
        return len(text)

    def flush(self):
        self._stream.flush()
        if self._log is not None:
            self._log.flush()

    def isatty(self):
        return False


def mark(*fields) -> None:
    sys.__stdout__.write(MARK + " " + "\t".join(str(f) for f in fields) + "\n")
    sys.__stdout__.flush()


def _dirs(dir_input: str, dir_output: str, config: dict) -> dict:
    out = Path(dir_output)
    return {
        "stitched": Path(dir_input, "stitched"),
        "masks": out / "masks",
        "aligned": out / "aligned",
        "check_aligned": out / "check_aligned",
        "masks_refined": out / "masks_refined" if config["refined_masking"]["run"] else out / "masks",
        "speckle_masks": out / "speckle_masks",
        "denoised": out / "denoised",
        "bg_subtracted": out / "bg_subtracted",
        "segmented": out / "segmented",
        "feature_tables": out / "feature_tables",
        "feature_tables_morphometrics": out / "feature_tables_morphometrics",
        "label_overlay": out / "label_overlay",
        "montages": out / "montages",
        "anndata": out / "anndata",
        "plots_overview": out / "plots_overview",
        "avg_nuclei": out / "avg_nuclei",
    }


def _step(key: str, sample: str, d: dict, config: dict, df_stains) -> None:
    """One step, called as ``process_sample`` calls it."""
    if key == "simple_masking":
        from phenoscapes.masking import simple_mask

        simple_mask(sample=sample, cycle=config["reference_cycle"],
                    channels=config["simple_masking"]["channels"],
                    sigma=config["simple_masking"]["sigma"],
                    n_binary=config["simple_masking"]["n_binary_operations"],
                    dir_input=d["stitched"], dir_output=d["masks"], config=config)
    elif key == "alignment":
        from phenoscapes.registration import run_elastix

        run_elastix(sample=sample, dir_input=d["stitched"], dir_output=d["aligned"],
                    config=config,
                    dir_masks=d["masks"] if config["alignment"]["mask"] else None)
    elif key == "alignment_check":
        from phenoscapes.registration import run_checks

        run_checks(sample=sample, dir_input=d["aligned"], dir_output=d["check_aligned"],
                   df_stains=df_stains, config=config)
    elif key == "refined_masking":
        from phenoscapes.masking import refined_mask

        refined_mask(sample=sample, dir_input=d["aligned"], dir_output=d["masks_refined"],
                     sigma=config["refined_masking"]["sigma"],
                     n_binary=config["refined_masking"]["n_binary_operations"],
                     use_only_dapi=config["refined_masking"]["use_only_dapi"],
                     df_stains=df_stains, config=config)
    elif key == "speckle_removal":
        from phenoscapes.speckle_removal import run_speckle_removal

        run_speckle_removal(sample=sample, dir_input=d["aligned"], dir_masks=d["masks_refined"],
                            dir_output_masks=d["speckle_masks"], config=config)
    elif key == "cropping_denoising":
        from phenoscapes.cropping_denoising import run_cropping_and_denoising

        run_cropping_and_denoising(sample=sample, dir_input=d["aligned"],
                                   dir_masks=d["masks_refined"], dir_output=d["denoised"],
                                   dir_output_masks=d["masks_refined"],
                                   dir_avg_nuclei=d["avg_nuclei"], config=config)
    elif key == "bg_subtraction":
        from phenoscapes.bg_subtraction import run_bg_subtraction

        run_bg_subtraction(sample=sample, dir_input=d["denoised"], dir_masks=d["masks_refined"],
                           dir_speckle_masks=d["speckle_masks"], dir_output=d["bg_subtracted"],
                           df_stains=df_stains, config=config)
    elif key == "cellpose":
        from phenoscapes.segmentation import run_cellpose

        c = config["cellpose"]
        run_cellpose(sample=sample, dir_images=d["bg_subtracted"], dir_output=d["segmented"],
                     dir_avg_nuclei=d["avg_nuclei"], diameter=c["diameter"],
                     model_type=c["model"], flow_threshold=c["flow_threshold"],
                     cellprob_threshold=c["cellprob_threshold"],
                     pretrained_model=c["pretrained_model"], config=config)
    elif key == "feature_extraction":
        from phenoscapes.feature_extraction import extract_features

        extract_features(sample=sample, dir_images=d["bg_subtracted"],
                         dir_output=d["feature_tables"], dir_segmented=d["segmented"],
                         df_stains=df_stains, config=config)
    elif key == "feature_morphology_extraction":
        from phenoscapes.morphometrics import run_extract_morphology_features

        run_extract_morphology_features(sample=sample,
                                        dir_output=d["feature_tables_morphometrics"],
                                        dir_segmented=d["segmented"],
                                        dir_speckle_masks=d["speckle_masks"], config=config)
    elif key == "to_h5ad":
        from phenoscapes.sc import convert_to_h5ad

        convert_to_h5ad(sample=sample, dir_input=d["feature_tables"], dir_output=d["anndata"],
                        stat=config["analysis_pipeline"]["stat"])
    elif key == "label_overlay":
        from phenoscapes.segmentation import generate_label_overlay

        generate_label_overlay(sample=sample, dir_images=d["bg_subtracted"],
                               dir_segmented=d["segmented"], dir_output=d["label_overlay"],
                               config=config)
    elif key == "overview_montage":
        from phenoscapes.montage import generate_overview_montage

        generate_overview_montage(sample=sample, dir_images=d["bg_subtracted"],
                                  dir_masks=d["masks_refined"], dir_output=d["montages"],
                                  df_stains=df_stains,
                                  slice_step=config["analysis_pipeline"]["slice_step"],
                                  config=config)
    elif key == "overview_plots":
        from phenoscapes.sc import plot_summary

        plot_summary(sample=sample, dir_input=d["anndata"], dir_output=d["plots_overview"])
    else:
        raise ValueError(f"Unknown step: {key}")


def run(request: dict) -> int:
    import pandas as pd
    import yaml

    config = request["config"]
    config["metadata"] = request["metadata_file"]
    dir_output = Path(request["dir_output"])
    dir_output.mkdir(parents=True, exist_ok=True)
    (dir_output / "logs").mkdir(exist_ok=True)
    # The config the run used, beside what it made, as the original does.
    with open(dir_output / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.dump(config, handle, sort_keys=False)

    df_stains = pd.read_csv(request["metadata_file"])
    d = _dirs(request["dir_input"], str(dir_output), config)
    failed = 0
    for sample in request["samples"]:
        log = open(dir_output / "logs" / f"{sample}.log", "a", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, log)
        sys.stderr = _Tee(sys.__stderr__, log)
        mark("sample-start", sample)
        print(f"=== {sample} · {time.strftime('%Y-%m-%d %H:%M:%S')} · "
              f"steps: {', '.join(request['steps'])}")
        try:
            for key in request["steps"]:
                mark("step-start", sample, key)
                started = time.perf_counter()
                try:
                    _step(key, sample, d, config, df_stains)
                except Exception as exc:  # noqa: BLE001 - reported, then the sample stops
                    traceback.print_exc()
                    mark("step-failed", sample, key, f"{type(exc).__name__}: {exc}".splitlines()[0])
                    failed += 1
                    # Every later step reads what this one should have written.
                    break
                mark("step-done", sample, key, f"{time.perf_counter() - started:.1f}")
        finally:
            mark("sample-done", sample)
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            log.close()
    mark("finished", failed)
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    return run(request)


# The guard is not optional: crop-and-denoise starts a multiprocessing pool, and
# on Windows every worker imports this file again.
if __name__ == "__main__":
    sys.exit(main(sys.argv))
