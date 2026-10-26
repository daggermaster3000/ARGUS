"""The junction finder alone, with the current defaults, on the sections of the whole-dataset run.

No cells are segmented and nothing is stored in the sample files: one row per
section (status, flags, figures, the line) and a QC picture per section, in a
new folder, so the result can be compared with the 21 October run
(``dataset_analysis_20261021``) before anything downstream is redone.

Sections run in parallel processes; the run appends as it goes and skips
sections already done, so it can be stopped and started again.

    python notebooks/dataset/run_junctions.py            # run
    python notebooks/dataset/run_junctions.py --report   # tables only
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from microscopy_viewer import assembloids as asm  # noqa: E402
from microscopy_viewer import junction as jn  # noqa: E402
from microscopy_viewer import loaders  # noqa: E402

DATA = Path(r"D:\Slidescanner extracted")
PREVIOUS = DATA / "dataset_analysis_20261021"
OUT_DIR = DATA / "junction_rerun_20261022"
ROWS = OUT_DIR / "junctions.jsonl"
CLASSES = ["Shape", "Ripped", "Folded", "Quality", "Genotype", "Experiment"]
WORKERS = 12


def one(row: dict) -> dict:
    """One section: the finder with today's defaults, its QC picture filed by status."""
    path = Path(row["path"])
    out = {"name": row["name"], "error": ""}
    started = time.perf_counter()
    try:
        options = jn.JunctionOptions(save=False)
        specs = loaders.load_path(path)
        try:
            overview, result = jn.analyse_specs(specs, options)
        finally:
            loaders.release(path)
        out.update(status=result.status, flags="; ".join(result.flags),
                   **{k: float(v) for k, v in result.metrics.items()})
        if result.line_um is not None:
            out["line_yx_um"] = np.round(np.asarray(result.line_um), 1).tolist()
        if result.signature is not None:
            out["signature"] = np.packbits(result.signature).tobytes().hex()
        jn.file_picture(OUT_DIR / "junction_QC", row["name"], result.status,
                        jn.qc_picture(overview, result, row["name"]))
    except Exception as exc:  # one bad file must not stop the run
        out.update(status=jn.STATUS_FAIL, error=f"crashed: {type(exc).__name__}: {exc}")
    out["seconds"] = time.perf_counter() - started
    return out


def done() -> set[str]:
    if not ROWS.exists():
        return set()
    names = set()
    for line in ROWS.read_text(encoding="utf-8").splitlines():
        try:
            names.add(json.loads(line)["name"])
        except (json.JSONDecodeError, KeyError):
            pass
    return names


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(PREVIOUS / "samples.csv", encoding="utf-8-sig")
    samples = samples[samples["exists"]] if "exists" in samples else samples
    todo = [r for r in samples.to_dict("records") if r["name"] not in done()]
    print(f"{len(samples)} sections, {len(todo)} to run, {WORKERS} processes", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool, ROWS.open("a", encoding="utf-8") as f:
        futures = [pool.submit(one, r) for r in todo]
        for k, future in enumerate(as_completed(futures), 1):
            f.write(json.dumps(future.result()) + "\n")
            f.flush()
            if k % 50 == 0 or k == len(futures):
                print(f"{k}/{len(futures)} in {time.perf_counter() - started:.0f} s", flush=True)
    report()


def report() -> None:
    samples = pd.read_csv(PREVIOUS / "samples.csv", encoding="utf-8-sig")
    new = pd.DataFrame([json.loads(l) for l in ROWS.read_text(encoding="utf-8").splitlines() if l.strip()])
    new = new.drop_duplicates("name", keep="last")
    old = pd.read_csv(PREVIOUS / "summary.csv")[["name", "status", "flags"]].rename(
        columns={"status": "status before", "flags": "flags before"})
    table = samples[["name", *CLASSES]].merge(new.drop(columns=["line_yx_um"], errors="ignore"), on="name") \
        .merge(old, on="name", how="left")
    table = asm.group_sections(table, series=["Genotype", "Experiment"], signature_column="signature")
    table.drop(columns=["signature"], errors="ignore").to_csv(OUT_DIR / "junctions.csv", index=False)
    strips(table)
    lines = [f"{len(table)} sections; crashed {int(table['error'].fillna('').str.startswith('crashed').sum())}", ""]
    lines.append("status before (21 Oct) → now:\n" + pd.crosstab(table["status before"], table["status"], margins=True).to_string())
    for c in CLASSES:
        t = pd.crosstab(table[c].fillna("(none)"), table["status"], margins=True, margins_name="total")
        before = pd.crosstab(table[c].fillna("(none)"), table["status before"], margins=True, margins_name="total")
        for s in ("ok", "fail"):
            t[f"{s} % now"] = (100 * t.get(s, 0) / t["total"]).round(0)
            t[f"{s} % before"] = (100 * before.get(s, 0) / before["total"]).round(0)
        lines += ["", f"== {c}", t.to_string()]
    per_slide = table.groupby("Slide")["Assembloid"].nunique()
    lines += ["", f"assemblies: {table['Assembloid'].nunique()} on {table['Slide'].nunique()} slides; "
              f"slides with more than one: {int((per_slide > 1).sum())}; running on onto the next slide: "
              f"{table.loc[table['Continues from'] != '', 'Slide'].nunique()}; odd sections: {int(table['Odd section'].sum())}",
              "slide patterns: " + table.groupby("Slide")["Pattern"].first().value_counts().to_string().replace("\n", ", "),
              "sections per assembloid: " + table.groupby("Assembloid").size().describe()[["min", "50%", "max"]].to_string().replace("\n", ", ")]
    flags = table["flags"].fillna("").str.split("; ").explode()
    lines += ["", "flags now:", flags[flags != ""].value_counts().to_string()]
    text = "\n".join(lines)
    (OUT_DIR / "report.txt").write_text(text, encoding="utf-8")
    print(text)


def strips(table, thumb: int = 150) -> None:
    """One picture per slide: its sections' QC pictures in scene order, framed by assembloid.

    Each frame's colour is the assembloid; the caption gives the scene, the
    assembloid's letter and how unlike the previous section of the same
    assembloid it is (``step``, 0 = the same outline); odd sections are marked.
    """
    from PIL import Image, ImageDraw

    folder = OUT_DIR / "assembloids"
    folder.mkdir(parents=True, exist_ok=True)
    palette = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178), (213, 94, 0), (204, 121, 167)]
    pictures = {p.stem: p for p in (OUT_DIR / "junction_QC").glob("*/*.png")}
    for slide, part in table.groupby("Slide"):
        part = part.sort_values("Scene")
        colours = {a: palette[k % len(palette)] for k, a in enumerate(dict.fromkeys(part["Assembloid"]))}
        sheet = Image.new("RGB", (len(part) * (thumb + 6) + 6, thumb + 46), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        for k, (_, row) in enumerate(part.iterrows()):
            x = 6 + k * (thumb + 6)
            draw.rectangle([x - 3, 3, x + thumb + 2, thumb + 8], fill=colours[row["Assembloid"]])
            if row["name"] in pictures:
                im = Image.open(pictures[row["name"]]).convert("RGB")
                im.thumbnail((thumb, thumb))
                sheet.paste(im, (x, 6))
            draw.text((x, thumb + 12), f"s{row['Scene']} {row['Assembloid'].split('/')[-1]}", fill=(255, 255, 255))
            if row["Odd section"]:
                draw.text((x + thumb - 30, thumb + 12), "odd", fill=(255, 90, 90))
            if np.isfinite(row["Step"]):
                draw.text((x, thumb + 26), f"step {row['Step']:.2f}", fill=(190, 190, 190))
        sheet.save(folder / f"{slide}.png")


if __name__ == "__main__":
    report() if "--report" in sys.argv else run()
