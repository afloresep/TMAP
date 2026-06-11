"""Shows the effect of the crossing-reduction (untangle) post-pass.

Usage:
    python examples/untangle_demo.py                # 3000 molecules (default)
    python examples/untangle_demo.py --nrows 6000
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from tmap import TMAP
from tmap.layout import LayoutConfig
from tmap.utils import fingerprints_from_smiles

DATA_PATH = Path(__file__).with_name("cluster_65053.csv")
OUTPUT = Path(__file__).with_name("untangle_demo.png")


def _orient(ax, ay, bx, by, cx, cy) -> int:
    v = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return int(v > 0.0) - int(v < 0.0)


def _intersection(ax, ay, bx, by, cx, cy, dx, dy):
    """Intersection point of two segments known to cross properly."""
    r_x, r_y = bx - ax, by - ay
    s_x, s_y = dx - cx, dy - cy
    denom = r_x * s_y - r_y * s_x
    if denom == 0.0:
        return None
    t = ((cx - ax) * s_y - (cy - ay) * s_x) / denom
    return ax + t * r_x, ay + t * r_y


def count_crossings(x, y, s, t):
    """Count proper edge crossings via a uniform spatial grid (over-approx AABB
    bucketing, exact crossing test). Returns (count, intersection_points)."""
    ex0, ey0 = x[s], y[s]
    ex1, ey1 = x[t], y[t]
    seglen = np.hypot(ex1 - ex0, ey1 - ey0)
    keep = seglen > 0.0
    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        return 0, np.empty((0, 2))

    med = float(np.median(seglen[keep]))
    cell = max(med * 2.0, 1e-9)
    x0, y0 = float(np.min(x)), float(np.min(y))

    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for e in idx:
        cxa, cxb = int((min(ex0[e], ex1[e]) - x0) // cell), int((max(ex0[e], ex1[e]) - x0) // cell)
        cya, cyb = int((min(ey0[e], ey1[e]) - y0) // cell), int((max(ey0[e], ey1[e]) - y0) // cell)
        for ix in range(cxa, cxb + 1):
            for iy in range(cya, cyb + 1):
                grid[(ix, iy)].append(e)

    seen: set[tuple[int, int]] = set()
    points: list[tuple[float, float]] = []
    su, sv = s, t
    for bucket in grid.values():
        nb = len(bucket)
        for a in range(nb):
            i = bucket[a]
            for b in range(a + 1, nb):
                j = bucket[b]
                key = (i, j) if i < j else (j, i)
                if key in seen:
                    continue
                # shared endpoint -> not a proper crossing
                if su[i] == su[j] or su[i] == sv[j] or sv[i] == su[j] or sv[i] == sv[j]:
                    continue
                ax, ay, bx, by = ex0[i], ey0[i], ex1[i], ey1[i]
                cx, cy, dxp, dyp = ex0[j], ey0[j], ex1[j], ey1[j]
                if (
                    _orient(ax, ay, bx, by, cx, cy) != _orient(ax, ay, bx, by, dxp, dyp)
                    and _orient(cx, cy, dxp, dyp, ax, ay) != _orient(cx, cy, dxp, dyp, bx, by)
                ):
                    seen.add(key)
                    p = _intersection(ax, ay, bx, by, cx, cy, dxp, dyp)
                    if p is not None:
                        points.append(p)
    return len(seen), np.array(points) if points else np.empty((0, 2))


def layout(fps, *, untangle: bool):
    cfg = LayoutConfig()
    cfg.adaptive = True  # same adaptive layout in both runs
    cfg.untangle = untangle  # the only thing that changes
    cfg.deterministic = True
    cfg.seed = 42
    cfg.fme_iterations = 1000
    model = TMAP(
        metric="jaccard",
        n_neighbors=20,
        n_permutations=512,
        kc=50,
        seed=42,
        layout_config=cfg,
    )
    x, y, s, t = model.fit_transform(fps)
    return (
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(s, dtype=np.int64),
        np.asarray(t, dtype=np.int64),
    )


def draw(ax, x, y, s, t, points, title):
    segs = np.stack([np.column_stack([x[s], y[s]]), np.column_stack([x[t], y[t]])], axis=1)
    ax.add_collection(LineCollection(segs, colors="#3b6ea5", linewidths=0.35, alpha=0.8))
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], s=14, c="#e02424", zorder=3, label="crossing")
    ax.set_title(title, fontsize=13)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale_view()


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo of the untangle post-pass effect.")
    parser.add_argument("--nrows", type=int, default=3000, help="Molecules to load (0 = all).")
    args = parser.parse_args()

    read_rows = None if args.nrows == 0 else args.nrows
    smiles = pd.read_csv(DATA_PATH, nrows=read_rows)["smiles"].tolist()
    print(f"Loaded {len(smiles):,} molecules from {DATA_PATH.name}")

    print("Computing Morgan fingerprints...")
    fps = fingerprints_from_smiles(smiles, fp_type="morgan", radius=2, n_bits=2048)

    print("Layout WITHOUT untangle...")
    x0, y0, s0, t0 = layout(fps, untangle=False)
    c0, p0 = count_crossings(x0, y0, s0, t0)
    print(f"  crossings: {c0}")

    print("Layout WITH untangle...")
    x1, y1, s1, t1 = layout(fps, untangle=True)
    c1, p1 = count_crossings(x1, y1, s1, t1)
    print(f"  crossings: {c1}")

    reduction = "100%" if c0 == 0 else f"{100 * (c0 - c1) / c0:.1f}%"
    print(f"Crossings {c0} -> {c1}  ({reduction} reduction)")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    draw(axes[0], x0, y0, s0, t0, p0, f"untangle = False\n{c0} edge crossings")
    draw(axes[1], x1, y1, s1, t1, p1, f"untangle = True\n{c1} edge crossings")
    fig.suptitle(
        f"Untangle post-pass on cluster_65053 ({len(smiles):,} molecules) — "
        f"crossings {c0} → {c1} ({reduction})",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT, dpi=140)
    print(f"Saved figure to {OUTPUT}")


if __name__ == "__main__":
    main()
