"""Build the words playground: regenerate TmapViz HTML under playgrounds/static/words/."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from tmap import TMAP
from tmap.playgrounds.bridge import inject_bridge

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "examples" / "data" / "word50k_cache"
OUT = ROOT / "src" / "tmap" / "playgrounds" / "static" / "words"


def main():
    model = TMAP.load(CACHE / "word_tmap.model")
    words = np.load(CACHE / "word_list.npy", allow_pickle=True)
    cats = np.load(CACHE / "word_categories.npy", allow_pickle=True)

    viz = model.to_tmapviz()
    viz.title = f"Word Embeddings — {len(words):,}"
    viz.add_label("Word", words.tolist())
    viz.add_color_layout("Category", cats.tolist(), categorical=True, color="tab20")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    viz.write_static(OUT)
    inject_bridge(OUT / "index.html")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
