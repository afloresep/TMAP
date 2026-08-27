"""molecules_tmap.py variant with the adaptive layout + untangle post-pass OFF.

Usage:
    python examples/chemistry/molecules_tmap_legacy.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tmap import TMAP
from tmap.layout import LayoutConfig
from tmap.utils import fingerprints_from_smiles, molecular_properties, murcko_scaffolds

# Scripts live in a subfolder; data, caches and outputs stay at examples/.
EXAMPLES_DIR = Path(__file__).resolve().parents[1]


DATA_PATH = EXAMPLES_DIR / "cluster_65053.csv"
OUTPUT = EXAMPLES_DIR / "cluster_65053_legacy.html"


def main() -> None:
    smiles = pd.read_csv(DATA_PATH)["smiles"].tolist()
    print(f"Loaded {len(smiles):,} molecules from {DATA_PATH.name}")

    print("Computing Morgan fingerprints...")
    fps, valid = fingerprints_from_smiles(
        smiles, fp_type="morgan", radius=2, n_bits=2048, return_valid=True
    )

    # Drop the SMILES that could not be read, so everything computed below
    # still belongs to the right molecule.
    if not valid.all():
        print(f"  Dropping {int((~valid).sum()):,} unparseable SMILES")
        smiles = [smi for smi, ok in zip(smiles, valid) if ok]

    print("Computing molecular properties...")
    props = molecular_properties(smiles, properties=["mw", "logp", "n_rings", "qed"])

    print("Computing Murcko scaffolds...")
    scaffolds = murcko_scaffolds(smiles)

    cfg = LayoutConfig()
    cfg.adaptive = False
    cfg.untangle = False
    cfg.deterministic = True
    cfg.seed = 42
    cfg.fme_iterations = 1000

    print("Fitting TMAP (adaptive=False, untangle=False)...")
    model = TMAP(
        metric="jaccard",
        n_neighbors=20,
        n_permutations=512,
        kc=50,
        seed=42,
        layout_config=cfg,
    ).fit(fps)

    viz = model.to_tmapviz()
    viz.title = "Cluster 65053 (legacy layout)"
    viz.add_smiles(smiles)
    viz.add_color_layout("MW", props["mw"].tolist(), color="viridis")
    viz.add_color_layout("LogP", props["logp"].tolist(), color="plasma")
    viz.add_color_layout("Ring Count", props["n_rings"].tolist(), categorical=True, color="tab10")
    viz.add_color_layout("QED", props["qed"].tolist(), color="magma")
    viz.add_label("Murcko Scaffold", scaffolds.tolist())

    output_path = viz.write_html(OUTPUT)
    print(f"Saved HTML to {output_path}")


if __name__ == "__main__":
    main()
