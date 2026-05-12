"""Build the ChEMBL playground: write TmapViz HTML under playgrounds/static/chembl/."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from tmap import TMAP
from tmap.playgrounds.bridge import inject_bridge

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "chembl"
OUT = ROOT / "src" / "tmap" / "playgrounds" / "static" / "chembl"


def main():
    model = TMAP.load(DATA / "chembl_full.tmap")
    df = pd.read_parquet(DATA / "chembl_full_meta.parquet")
    viz = model.to_tmapviz()
    viz.title = f"ChEMBL 36 — {len(df):,} molecules"
    viz.add_smiles(df["canonical_smiles"].tolist())
    viz.add_label("ChEMBL ID", df["chembl_id"].tolist())
    viz.add_label("Scaffold", df["scaffold"].fillna("").tolist())
    viz.add_color_layout("MW (log)", np.log1p(df["mw"].fillna(0)).tolist(), color="viridis")
    viz.add_color_layout("LogP", df["logp"].fillna(0).tolist(), color="plasma")
    viz.add_color_layout("QED", df["qed"].fillna(0).tolist(), color="magma")
    viz.add_color_layout("TPSA (log)", np.log1p(df["tpsa"].fillna(0)).tolist(), color="cividis")
    viz.add_color_layout(
        "Aromatic Rings",
        df["n_aromatic_rings"].fillna(0).astype(int).tolist(),
        categorical=True,
        color="tab10",
    )
    viz.add_color_layout(
        "Natural Product",
        df["natural_product"].fillna(0).astype(int).tolist(),
        categorical=True,
        color="Set1",
    )
    viz.add_color_layout(
        "Best pChEMBL",
        np.log1p(df["best_pchembl"].fillna(0)).tolist(),
        color="inferno",
    )
    if "target_class" in df.columns:
        viz.add_color_layout(
            "Target Class",
            df["target_class"].fillna("Unknown").tolist(),
            categorical=True,
            color="tab20",
        )
    viz.configure_card(
        title_column="ChEMBL ID",
        links=[
            {
                "label": "ChEMBL",
                "url": "https://www.ebi.ac.uk/chembl/explore/compound/{ChEMBL ID}",
            }
        ],
    )

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    viz.write_static(OUT)
    inject_bridge(OUT / "index.html")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
