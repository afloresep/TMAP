"""Low-level TMAP pipeline from SMILES strings.

Demonstrates each step of the pipeline explicitly:
    SMILES → Morgan fingerprints → MinHash → LSHForest → OGDF layout → TmapViz

For a simpler high-level approach, see molecules_tmap.py which uses the
TMAP estimator to do all of this in a few lines.

Requirements:
    pip install rdkit

Usage:
    python examples/chemistry/smiles_tmap.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tmap import LSHForest, MinHash, fingerprints_from_smiles, molecular_properties
from tmap.layout import LayoutConfig, ScalingType, layout_from_lsh_forest
from tmap.visualization import TmapViz

# Scripts live in a subfolder; data, caches and outputs stay at examples/.
EXAMPLES_DIR = Path(__file__).resolve().parents[1]


DATA_PATH = EXAMPLES_DIR / "cluster_65053.csv"


def main() -> None:
    # 1. Load SMILES and compute fingerprints
    df = pd.read_csv(DATA_PATH)
    smiles = df["smiles"].tolist()
    print(f"Loaded {len(smiles):,} molecules")

    fps = fingerprints_from_smiles(smiles, fp_type="morgan", radius=2, n_bits=2048)
    n = fps.shape[0]
    print(f"  Valid fingerprints: {n} x {fps.shape[1]}")

    # 2. MinHash encoding
    #    Converts binary fingerprints into compact hash signatures for
    #    fast approximate Jaccard similarity search.
    num_perm = 128
    mh = MinHash(num_perm=num_perm, seed=42)
    signatures = mh.batch_from_binary_array(fps)
    print(f"  MinHash signatures: {signatures.shape}")

    # 3. Build LSH Forest index
    #    Locality-sensitive hashing index for fast approximate kNN queries.
    #    d = number of permutations, l = number of hash tables.
    lsh = LSHForest(d=num_perm, l=64)
    lsh.batch_add(signatures)
    lsh.index()
    print(f"  LSH Forest indexed ({lsh.size()} points)")

    # 4. Compute OGDF layout
    #    layout_from_lsh_forest queries the kNN graph internally,
    #    computes an MST, and runs the OGDF tree layout algorithm.
    #    Returns (x, y, edge_sources, edge_targets).
    cfg = LayoutConfig()
    cfg.k = 20
    cfg.kc = 50
    cfg.node_size = 1 / 30
    cfg.mmm_repeats = 2
    cfg.sl_extra_scaling_steps = 10
    cfg.sl_scaling_type = ScalingType.RelativeToDrawing
    cfg.fme_iterations = 1000
    cfg.deterministic = True
    cfg.seed = 42

    x, y, s, t = layout_from_lsh_forest(lsh, cfg)
    print(f"  Layout: {len(x)} nodes, {len(s)} edges")

    # 5. Build visualization
    #    TmapViz takes pre-computed coordinates and edges directly.
    props = molecular_properties(smiles, properties=["mw", "logp", "n_rings"])

    viz = TmapViz()
    viz.title = "Molecular TMAP (low-level pipeline)"
    viz.background_color = "#FFFFFF"
    viz.set_points(x, y)
    viz.set_edges(s, t)

    viz.add_smiles(smiles)
    viz.add_color_layout("MW", props["mw"].tolist(), color="viridis")
    viz.add_color_layout("LogP", props["logp"].tolist(), color="plasma")
    viz.add_color_layout(
        "Ring Count",
        props["n_rings"].tolist(),
        categorical=True,
        color="tab10",
    )

    out = viz.write_html(str(DATA_PATH.with_suffix(".html")))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
