# scripts/build_endosymbiosis_playground.py
"""Build the cross-kingdom protein playground from cached ESM-2 embeddings."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from tmap import TMAP
from tmap.playgrounds.bridge import inject_bridge

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "examples" / "data"
END = DATA / "endosymbiosis"
EMB_NPZ = DATA / "embeddings.npz"
OUT_DATA = END / "playground"
OUT_STATIC = ROOT / "src" / "tmap" / "playgrounds" / "static" / "proteins"


def main():
    d = np.load(EMB_NPZ, allow_pickle=True)
    emb = d["embeddings"].astype("float32")
    accs = np.asarray(d["accessions"])
    meta = pd.read_csv(END / "dataset_metadata.tsv", sep="\t")
    # Order metadata to match embedding row order
    meta = meta.set_index("accession").loc[accs].reset_index()
    model = TMAP(metric="cosine", n_neighbors=15, seed=42, store_index=True).fit(emb)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    model.save(OUT_DATA / "proteins.tmap")
    # Save the input embeddings alongside the model so the playground module
    # can use them for accession-based queries (no need to re-encode).
    np.save(OUT_DATA / "embeddings.npy", emb)
    meta.to_parquet(OUT_DATA / "proteins_meta.parquet")

    viz = model.to_tmapviz()
    viz.title = f"Cross-kingdom proteins — {len(accs):,}"
    viz.add_label("Accession", meta["accession"].tolist())
    viz.add_label("Organism", meta["organism"].tolist())
    viz.add_color_layout("Domain", meta["domain"].tolist(), categorical=True, color="tab10")
    viz.add_color_layout(
        "Compartment",
        meta["compartment"].fillna("-").tolist(),
        categorical=True,
        color="tab20",
    )
    viz.configure_card(
        title_column="Accession",
        links=[
            {"label": "UniProt", "url": "https://www.uniprot.org/uniprotkb/{Accession}"},
            {"label": "AlphaFold", "url": "https://alphafold.ebi.ac.uk/entry/{Accession}"},
        ],
    )
    if OUT_STATIC.exists():
        shutil.rmtree(OUT_STATIC)
    OUT_STATIC.mkdir(parents=True, exist_ok=True)
    viz.write_static(OUT_STATIC)
    inject_bridge(OUT_STATIC / "index.html")
    print(f"Wrote {OUT_DATA} and {OUT_STATIC}")


if __name__ == "__main__":
    main()
