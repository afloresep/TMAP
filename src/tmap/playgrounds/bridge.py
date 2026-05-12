"""Inject the cross-frame bridge into a generated TmapViz HTML."""
from __future__ import annotations

from pathlib import Path

_BRIDGE_DIR = Path(__file__).parent / "static"
BRIDGE_TAG = '<script src="/static/playground-bridge.js"></script>'


def inject_bridge(html_path: Path) -> None:
    text = html_path.read_text(encoding="utf-8")
    text = text.replace(
        "const scatterplot = createScatterplot({",
        "const scatterplot = window._tmap_scatterplot = createScatterplot({",
    )
    text = text.replace("</body>", BRIDGE_TAG + "\n</body>")
    html_path.write_text(text, encoding="utf-8")


def bridge_js_path() -> Path:
    return _BRIDGE_DIR / "playground-bridge.js"
