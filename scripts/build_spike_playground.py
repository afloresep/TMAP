"""Build the spike playground.

Fits TMAP on aligned spike sequences, saves model + metadata + static HTML.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from tmap import TMAP
from tmap.playgrounds.bridge import inject_bridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "examples"))
import sars_cov2_spike_tmap as spike  # noqa: E402

DATA = HERE.parent / "examples" / "data" / "sars_cov2_spike"
OUT_DATA = DATA / "playground"
OUT_STATIC = HERE.parent / "src" / "tmap" / "playgrounds" / "static" / "spike"


def main(n: int = 4000):
    print(f"Loading and subsampling {n:,} sequences from {DATA}...", flush=True)
    df, sequences = spike.load_subsample(DATA, n_target=n, seed=42)
    print(f"  retained {len(df):,} sequences (incl. Wuhan-Hu-1 reference)", flush=True)

    print("Computing k-mer shingles...", flush=True)
    shingles = spike.shingles_to_minhash(sequences, k=6, n_perm=128)

    print("Fitting TMAP (metric=jaccard, n_neighbors=20)...", flush=True)
    model = TMAP(metric="jaccard", n_neighbors=20, seed=42, store_index=True).fit(shingles)
    print("  fit complete", flush=True)

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    model.save(OUT_DATA / "spike.tmap")
    print(f"  saved model -> {OUT_DATA / 'spike.tmap'}", flush=True)

    df = df.copy()
    df["sequence"] = sequences
    df.to_parquet(OUT_DATA / "spike_meta.parquet")
    print(f"  saved metadata -> {OUT_DATA / 'spike_meta.parquet'}", flush=True)

    viz = model.to_tmapviz()
    viz.title = f"SARS-CoV-2 Spike — {len(df):,} sequences"
    viz.add_label("Strain", df["strain"].tolist())
    viz.add_color_layout("Clade", df["clade"].tolist(), categorical=True, color="tab20")
    viz.add_color_layout("Date", df["date_numeric"].tolist(), color="viridis")
    viz.add_color_layout(
        "Country",
        df["country"].fillna("?").tolist(),
        categorical=True,
        color="tab20",
    )

    if OUT_STATIC.exists():
        shutil.rmtree(OUT_STATIC)
    OUT_STATIC.mkdir(parents=True, exist_ok=True)
    viz.write_static(OUT_STATIC)
    inject_bridge(OUT_STATIC / "index.html")
    print(f"Wrote {OUT_DATA} and {OUT_STATIC}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4000)
    main(**vars(p.parse_args()))
