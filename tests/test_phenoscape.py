"""Checks for the phenoscapes driver: the config, the folders, and the runner.

The real pipeline needs elastix, Cellpose and numpy 1.26, none of which a test
should wait on. So the runner is run for real, as the panel runs it — a child
process, reading a request file — against a stand-in ``phenoscapes`` package
whose functions only write down how they were called. What is checked is the
wiring: the right function, the right folders, the right order, and the marks the
panel reads.

Run with::

    python tests/test_phenoscape.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import phenoscape as ph  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


#: Every function the runner calls, by module, written so each records its
#: keyword arguments and touches the file the real one would write. The calls
#: log is named by an environment variable so one stand-in serves every test.
_FAKE = {
    "masking": ("simple_mask", "refined_mask"),
    "registration": ("run_elastix", "run_checks"),
    "speckle_removal": ("run_speckle_removal",),
    "cropping_denoising": ("run_cropping_and_denoising",),
    "bg_subtraction": ("run_bg_subtraction",),
    "segmentation": ("run_cellpose", "generate_label_overlay"),
    "feature_extraction": ("extract_features",),
    "morphometrics": ("run_extract_morphology_features",),
    "montage": ("generate_overview_montage",),
    "sc": ("convert_to_h5ad", "plot_summary"),
}

_FAKE_BODY = '''
import json, os
from pathlib import Path


def _record(name, kwargs):
    with open(os.environ["FAKE_PHENOSCAPES_LOG"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"call": name, "kwargs": {
            k: (str(v) if not isinstance(v, (int, float, bool, type(None), list, dict)) else v)
            for k, v in kwargs.items() if k not in ("df_stains", "config")
        }}) + "\\n")
    if os.environ.get("FAKE_PHENOSCAPES_FAIL") == name:
        raise RuntimeError("told to fail")
    print("working on", name)
'''


def _fake_package(root: Path) -> Path:
    package = root / "phenoscapes"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    for module, functions in _FAKE.items():
        body = _FAKE_BODY
        for name in functions:
            body += textwrap.dedent(f'''

                def {name}(**kwargs):
                    _record("{name}", kwargs)
                    if "{name}" == "run_cellpose":
                        Path(kwargs["dir_output"]).mkdir(parents=True, exist_ok=True)
                        (Path(kwargs["dir_output"]) / (kwargs["sample"] + ".tif")).write_bytes(b"x")
            ''')
        (package / f"{module}.py").write_text(body, encoding="utf-8")
    return root


def _input_tree(root: Path) -> tuple[Path, Path]:
    folder = root / "input"
    sample = folder / "stitched" / "org1"
    sample.mkdir(parents=True)
    for cycle in (1, 2):
        for channel in (0, 1):
            name = f"x_c{cycle:03d}_B02_r001_t001_z000_c0_stitch_ch{channel:02d}.tif"
            (sample / name).write_bytes(b"")
    (sample / "notes.tif.txt").write_bytes(b"")
    (folder / "stitched" / "empty").mkdir()
    stains = root / "stains.csv"
    stains.write_text("cycle_id,channel_id,stain\n1,1,SOX2\n2,1,\n", encoding="utf-8")
    return folder, stains


def test_the_config_matches_the_original() -> None:
    print("the config, key for key")

    config = ph.default_config()
    check(config["cellpose"]["model"] == "cyto2" and config["reference_cycle"] == 1,
          "the defaults are phenoscapes' own")
    check(ph.enabled_steps(config) == [
        "simple_masking", "alignment", "alignment_check", "refined_masking",
        "cropping_denoising", "bg_subtraction", "cellpose", "feature_extraction",
        "to_h5ad", "label_overlay", "overview_montage", "overview_plots"],
        "the default pipeline is the original's: no speckles, no morphometrics")

    old = ph.complete_config({"reference_cycle": 3, "cellpose": {"diameter": 35}})
    check(old["reference_cycle"] == 3 and old["cellpose"]["diameter"] == 35,
          "a partial config keeps what it says")
    check(old["cellpose"]["model"] == "cyto2" and "speckle_removal" in old,
          "and gets every key it does not say from the defaults")

    ph.set_enabled(config, ph.get_step("label_overlay"), False)
    check(config["analysis_pipeline"]["run"], "one analysis step off leaves the block on")
    for key in ("to_h5ad", "overview_montage", "overview_plots"):
        ph.set_enabled(config, ph.get_step(key), False)
    check(not config["analysis_pipeline"]["run"], "all four off turns the block off")

    text = ph.config_text(ph.default_config())
    check(ph.parse_config_text(text) == ph.default_config(), "YAML there and back is the same config")
    try:
        ph.parse_config_text("- a list\n- not a mapping\n")
        check(False, "a YAML list is refused")
    except ValueError:
        check(True, "a YAML list is refused")


def test_the_folders(directory: Path) -> None:
    print("finding samples, and where each step writes")

    folder, stains = _input_tree(directory)
    check(ph.list_samples(folder) == ["org1"], "a sample is a folder in 'stitched' with TIFFs")
    check(ph.list_samples(directory / "nowhere") == [], "no 'stitched', no samples")
    total, readable = ph.count_images(folder, "org1")
    check((total, readable) == (5, 4), f"the default regex reads the stitching's names ({readable} of {total})")
    check(ph.channel_of("x_c002_B02_r001_t001_z000_c0_stitch_ch01.tif") == (2, 1),
          "cycle and channel come out of a name")
    check(ph.stain_names(stains) == {(1, 1): "SOX2"}, "an empty stain is a background cycle, not a name")
    check(ph.check_metadata(stains) is None, "a good stain table passes")
    bad = directory / "bad.csv"
    bad.write_text("cycle,stain\n1,a\n", encoding="utf-8")
    check("channel_id" in (ph.check_metadata(bad) or ""), "a table without channel_id says so")

    config = ph.default_config()
    out = directory / "out"
    check(ph.output_dirs(out, config)["masks_refined"] == out / "masks_refined",
          "with refined masking on, the refined masks are what later steps read")
    ph.set_enabled(config, ph.get_step("refined_masking"), False)
    check(ph.output_dirs(out, config)["masks_refined"] == out / "masks",
          "with it off, the simple mask stands in, as in the original")

    request = ph.Request(("org1",), str(folder), str(out), str(stains), config, ("refined_masking",))
    check(any("Switched off" in p for p in request.problems()),
          "a step the config switches off cannot be run on its own")
    request = ph.Request((), str(directory), "", "", ph.default_config(), ())
    problems = " ".join(request.problems())
    check(all(word in problems for word in ("sample", "stitched", "output", "Stain", "step")),
          "an empty request names everything it lacks")


def _run(directory: Path, steps, fail: str | None = None) -> tuple[int, list[str], list[dict]]:
    folder, stains = _input_tree(directory)
    code = _fake_package(directory / "code")
    out = directory / "out"
    log = directory / "calls.jsonl"
    request = ph.Request(("org1",), str(folder), str(out), str(stains),
                         ph.default_config(), tuple(steps))
    check(not request.problems(), "the request is complete")
    env = ph.child_environment(code)
    env["FAKE_PHENOSCAPES_LOG"] = str(log)
    if fail:
        env["FAKE_PHENOSCAPES_FAIL"] = fail
    done = subprocess.run(ph.run_command(sys.executable, ph.write_request(request, out)),
                          env=env, capture_output=True, text=True, timeout=120)
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    if done.returncode not in (0, 1):
        print(done.stdout, done.stderr)
    return done.returncode, done.stdout.splitlines(), calls


def test_the_runner_calls_the_pipeline(directory: Path) -> None:
    print("the runner, in a child process, against a stand-in phenoscapes")

    code, lines, calls = _run(directory, ph.enabled_steps(ph.default_config()))
    out = directory / "out"
    check(code == 0, "a clean run exits 0")
    names = [c["call"] for c in calls]
    check(names == ["simple_mask", "run_elastix", "run_checks", "refined_mask",
                    "run_cropping_and_denoising", "run_bg_subtraction", "run_cellpose",
                    "extract_features", "convert_to_h5ad", "generate_label_overlay",
                    "generate_overview_montage", "plot_summary"],
          "every enabled step, in process_sample's order")
    by = {c["call"]: c["kwargs"] for c in calls}
    check(Path(by["run_elastix"]["dir_input"]) == directory / "input" / "stitched"
          and Path(by["run_elastix"]["dir_output"]) == out / "aligned",
          "alignment reads 'stitched' and writes 'aligned'")
    check(by["run_elastix"]["dir_masks"] is None, "without alignment.mask, no mask goes to elastix")
    check(Path(by["run_bg_subtraction"]["dir_input"]) == out / "denoised"
          and Path(by["run_bg_subtraction"]["dir_masks"]) == out / "masks_refined",
          "background subtraction reads the denoised images and the refined mask")
    check(by["run_cellpose"]["model_type"] == "cyto2" and by["run_cellpose"]["diameter"] == 20,
          "Cellpose gets the config's model and diameter")
    check(Path(by["plot_summary"]["dir_input"]) == out / "anndata", "the plots read the AnnData")

    marks = [ph.parse_mark(line) for line in lines if ph.parse_mark(line)]
    kinds = [m[0] for m in marks]
    check(kinds[0] == "sample-start" and kinds[-1] == "finished" and kinds.count("step-done") == 12,
          "the panel is told when each step starts and ends")
    check(any(line == "working on run_cellpose" for line in lines),
          "the pipeline's own output comes through")
    check((out / "config.yaml").is_file() and (out / "logs" / "org1.log").is_file(),
          "the config and a log are left beside the results")
    check("working on simple_mask" in (out / "logs" / "org1.log").read_text(encoding="utf-8"),
          "the log holds what was printed")

    outputs = ph.sample_outputs(out, "org1", ph.default_config())
    check([o.step for o in outputs] == ["cellpose"] and outputs[0].kind == "labels",
          "what a step wrote is found, as labels for the segmentation")


def test_a_failure_stops_the_sample(directory: Path) -> None:
    print("a step that fails")

    code, lines, calls = _run(directory, ["simple_masking", "alignment", "cellpose"],
                              fail="run_elastix")
    check(code == 1, "the run exits 1")
    check([c["call"] for c in calls] == ["simple_mask", "run_elastix"],
          "nothing after the failed step runs: it would read what was never written")
    failed = [ph.parse_mark(l) for l in lines if l.startswith(ph.MARK + " step-failed")]
    check(len(failed) == 1 and failed[0][1][1] == "alignment" and "told to fail" in failed[0][1][2],
          "the panel is told which step failed, and why")


def test_the_check_command() -> None:
    print("asking a Python what it has")

    done = subprocess.run(ph.check_command(sys.executable), capture_output=True, text=True, timeout=300)
    ok, missing, version = ph.parse_check(done.stdout)
    check(version == sys.version.split()[0], "the version is reported")
    check(len(ok) + len(missing) == len(ph.REQUIRED_MODULES), "every module is accounted for")
    check("yaml" in ok, "what is there is reported as there")


def main() -> int:
    directory = Path(tempfile.mkdtemp(prefix="mv-phenoscape-"))
    try:
        test_the_config_matches_the_original()
        test_the_folders(directory / "folders")
        test_the_runner_calls_the_pipeline(directory / "run")
        test_a_failure_stops_the_sample(directory / "fail")
        test_the_check_command()
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
