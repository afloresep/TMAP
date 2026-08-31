"""MNIST digits → TMAP with cosine metric.

Demonstrates cosine metric support on a classic dataset.
70,000 handwritten digits (784D pixel vectors) embedded as an explorable tree.

Usage:
    python examples/images/mnist_cosine_tmap.py

Output:
    examples/mnist_tmap.html  — interactive TMAP visualization
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml

from tmap import TMAP
from tmap.layout import LayoutConfig, ScalingType

# Scripts live in a subfolder; data, caches and outputs stay at examples/.
EXAMPLES_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    # 1. Load MNIST (70k samples, 784D)
    print("Loading MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32)
    labels = mnist.target.astype(int)
    print(f"  Shape: {X.shape}, labels: {np.unique(labels)}")

    # 2. Fit TMAP with cosine metric
    cfg = LayoutConfig()
    cfg.k = 20
    cfg.kc = 200
    cfg.node_size = 1 / 30
    cfg.mmm_repeats = 2
    cfg.sl_extra_scaling_steps = 10
    cfg.sl_scaling_type = ScalingType.RelativeToDrawing
    cfg.fme_iterations = 1000
    cfg.deterministic = True
    cfg.seed = 42

    print("\nFitting TMAP (metric='cosine', n_neighbors=20)...")
    t0 = time.perf_counter()
    model = TMAP(
        metric="cosine", n_neighbors=20, seed=42, layout_iterations=1000, layout_config=cfg
    ).fit(X)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Embedding: {model.embedding_.shape}")
    print(f"  Tree edges: {model.tree_.edges.shape[0]}")

    # ------------------------------------------------------------------
    # 3. Export interactive HTML
    # ------------------------------------------------------------------
    viz = model.to_tmapviz()
    viz.background_color = "#FFFFFF"

    viz.add_color_layout("digit", labels.tolist(), categorical=True, color="tab10")
    out_path = viz.write_html(EXAMPLES_DIR / "mnist_tmap.html")
    print(f"\n  Saved: {out_path}")

    # ------------------------------------------------------------------
    # 4. Trace paths between morphologically similar digits
    # ------------------------------------------------------------------
    pairs = [(3, 8), (4, 9), (7, 1), (5, 6)]
    for a, b in pairs:
        idx_a = int(np.where(labels == a)[0][0])
        idx_b = int(np.where(labels == b)[0][0])
        path = model.path(idx_a, idx_b)
        path_labels = labels[path]
        unique, counts = np.unique(path_labels, return_counts=True)
        digit_counts = " ".join(f"{d}x{c}" for d, c in zip(unique, counts))
        print(f"\n  Path {a}→{b}: {len(path)} nodes [{digit_counts}]")


if __name__ == "__main__":
    main()
