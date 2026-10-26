"""Inline a DATA json into explainer_template.html -> _build/preview.html (front-end development only).

    python notebooks/explainers/_build/assemble.py [data.json] [out.html]

Without arguments: the newer of _build/explainer_data.json (real) and _build/mock_data.json.

Same replacement rule as build_explainer.py: the single line
``const DATA = /*__DATA__*/ null;`` becomes ``const DATA = <json>;``.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "explainer_template.html"
MARKER = "const DATA = /*__DATA__*/ null;"


def assemble(data_path: Path, out_path: Path) -> Path:
    html = TEMPLATE.read_text(encoding="utf-8")
    if html.count(MARKER) != 1:
        raise SystemExit(f"template must contain the marker exactly once: {MARKER!r}")
    text = data_path.read_text(encoding="utf-8").strip()
    # A JSON string may hold "</script>" (it never does here, but be safe) and U+2028/9.
    text = text.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    out_path.write_text(html.replace(MARKER, f"const DATA = {text};"), encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB) from {data_path.name}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data = Path(sys.argv[1])
    else:  # the newest of the real builder's json and the mock
        found = [p for p in (HERE / "explainer_data.json", HERE / "mock_data.json") if p.exists()]
        data = max(found, key=lambda p: p.stat().st_mtime)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "preview.html"
    assemble(data, out)
