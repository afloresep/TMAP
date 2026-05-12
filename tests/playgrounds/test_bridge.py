from pathlib import Path

from tmap.playgrounds.bridge import inject_bridge


def test_inject_bridge_replaces_scatterplot_and_appends_script(tmp_path: Path):
    html = (
        "<html><body>"
        "<script>const scatterplot = createScatterplot({foo:1});</script>"
        "</body></html>"
    )
    file = tmp_path / "index.html"
    file.write_text(html)
    inject_bridge(file)
    out = file.read_text()
    assert "window._tmap_scatterplot" in out
    assert "playground-bridge.js" in out
