"""Three-panel interactive 8x8 explorer: 2k / 64k / 200k, ns_cap = infinity.

Data is examples/chembl_200k.csv (200k ChEMBL canonical SMILES; subsets used for the
smaller panels).

Usage:
    python examples/adaptive_coef_exp_3panel.py
    python examples/adaptive_coef_exp_3panel.py --probe   # time risky 200k corners only
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from tmap.graph.connect import connect_knn_components
from tmap.index.usearch_index import USearchIndex
from tmap.layout import LayoutConfig, layout_from_knn_graph
from tmap.utils import fingerprints_from_smiles

DATA_PATH = Path(__file__).with_name("chembl_200k.csv")
OUTPUT = Path(__file__).with_name("adaptive_coef_exp_3panel.html")

PANELS = [("2k", 2000, 0.35), ("64k", 64000, 0.12), ("200k", 200000, 0.07)]

COEFS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 14.0, 20.0]
EXPS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.70, 0.90]
G = 8


def fill_fraction(xn, yn, grid=48) -> float:
    ix = np.clip((xn * (grid - 1)).astype(int), 0, grid - 1)
    iy = np.clip((yn * (grid - 1)).astype(int), 0, grid - 1)
    occ = np.zeros((grid, grid), dtype=bool)
    occ[ix, iy] = True
    return float(occ.mean())


def run_layout(knn, coef, exp):
    cfg = LayoutConfig()
    cfg.adaptive = True
    cfg.untangle = False
    cfg.deterministic = False
    cfg.seed = 42
    cfg.fme_iterations = 1000
    cfg.ns_cap = math.inf
    cfg.ns_coef = coef
    cfg.ns_exp = exp
    x, y, s, t = layout_from_knn_graph(knn, config=cfg, create_mst=True)
    return (
        np.nan_to_num(np.asarray(x, dtype=np.float64)),
        np.nan_to_num(np.asarray(y, dtype=np.float64)),
        np.asarray(s, dtype=np.int64),
        np.asarray(t, dtype=np.int64),
    )


def render_tile(x, y, s, t, lw, px=680):
    minx, miny = x.min(), y.min()
    span = max(x.max() - minx, y.max() - miny, 1e-12)
    xn = np.clip((x - minx) / span, 0, 1)
    yn = np.clip((y - miny) / span, 0, 1)
    segs = np.stack([np.column_stack([xn[s], yn[s]]),
                     np.column_stack([xn[t], yn[t]])], axis=1)
    dpi = 120
    fig, ax = plt.subplots(figsize=(px / dpi, px / dpi))
    # transparent tile + bright edges so it sits on the dark Beer CSS surface
    ax.add_collection(LineCollection(segs, colors="#74b9ff", linewidths=lw, alpha=0.9))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii"), fill_fraction(xn, yn)


def build_knn(nrows):
    """Build a CONNECTED binary-Jaccard kNN graph (USearch HNSW + component bridging).

    Builds the HNSW index directly (no wasted TMAP.fit layout) and then runs the
    connect_components fix so the returned graph is a single connected component --
    no forest blobs in the panels, even where a low k would otherwise fragment it.
    """
    smiles = pd.read_csv(DATA_PATH, nrows=nrows)["smiles"].tolist()
    fps = fingerprints_from_smiles(smiles, fp_type="morgan", radius=2, n_bits=2048)
    index = USearchIndex(seed=42, expansion_search=512)
    index.build_from_binary(fps)
    knn = index.query_knn(k=20)
    knn, n_comp, n_bridges = connect_knn_components(knn, requery=lambda kk: index.query_knn(k=kk))
    if n_bridges:
        print(f"  connected {n_comp} components with {n_bridges} bridge edge(s)", flush=True)
    else:
        print("  graph already connected (1 component)", flush=True)
    return knn


def compute_grid(knn, lw, label):
    tiles = [None] * (G * G)
    fills = [0.0] * (G * G)
    n = 0
    for ci in range(G):
        for cj in range(G):
            t = time.perf_counter()
            x, y, s, tt = run_layout(knn, COEFS[ci], EXPS[cj])
            png, fill = render_tile(x, y, s, tt, lw)
            idx = ci * G + cj
            tiles[idx] = png
            fills[idx] = round(fill, 4)
            n += 1
            print(f"  [{label} {n}/64] coef={COEFS[ci]:5} exp={EXPS[cj]:4} "
                  f"{time.perf_counter() - t:5.1f}s fill={fill:.2f}", flush=True)
    return tiles, fills


def probe():
    t0 = time.perf_counter()
    print("PROBE: building 200k kNN...", flush=True)
    knn = build_knn(200000)
    print(f"  200k kNN ready in {time.perf_counter() - t0:.1f}s, {knn.n_nodes:,} nodes", flush=True)
    # worst (low coef, high exp) first -- this is the stall risk
    for coef, exp in [(0.25, 0.90), (0.25, 0.70), (0.25, 0.45), (20.0, 0.05)]:
        t = time.perf_counter()
        x, y, s, tt = run_layout(knn, coef, exp)
        dt = time.perf_counter() - t
        finite = np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        print(f"  coef={coef:5} exp={exp:4} layout={dt:6.1f}s finite={finite}", flush=True)
    print("PROBE done.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="3-panel 2k/64k/200k explorer (cap=inf).")
    parser.add_argument("--probe", action="store_true", help="Time risky 200k corners, then exit.")
    args = parser.parse_args()

    if args.probe:
        probe()
        return

    t0 = time.perf_counter()
    panel_data = []
    for label, nrows, lw in PANELS:
        print(f"Building {label} kNN ({nrows:,})...", flush=True)
        knn = build_knn(nrows)
        print(f"  {label}: {knn.n_nodes:,} nodes", flush=True)
        print(f"Computing {label} grid...", flush=True)
        tiles, fills = compute_grid(knn, lw, label)
        panel_data.append((label, knn.n_nodes, tiles, fills))

    def uris(tiles):
        return ["data:image/png;base64," + t for t in tiles]

    html = HTML_TEMPLATE
    for i, (label, nnodes, tiles, fills) in enumerate(panel_data):
        html = (html
                .replace(f"__TILES_{i}__", json.dumps(uris(tiles)))
                .replace(f"__FILLS_{i}__", json.dumps(fills))
                .replace(f"__NAME_{i}__", label)
                .replace(f"__NN_{i}__", f"{nnodes:,}"))
    html = (html
            .replace("__COEFS__", json.dumps([f"{c:g}" for c in COEFS]))
            .replace("__EXPS__", json.dumps([f"{e:g}" for e in EXPS]))
            .replace("__G__", str(G))
            .replace("__GMAX__", str(G - 1)))
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Saved 3-panel explorer to {OUTPUT} "
          f"({OUTPUT.stat().st_size / 1e6:.1f} MB, total {time.perf_counter() - t0:.0f}s)", flush=True)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TMAP adaptive ns_coef x ns_exp: 2k vs 64k vs 200k (cap = infinity)</title>
<link href="https://cdn.jsdelivr.net/npm/beercss@4.0.21/dist/cdn/beer.min.css" rel="stylesheet">
<style>
  body { padding: 22px; }
  .wrap { max-width: 1360px; margin: 0 auto; }
  .head-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .images { display: flex; gap: 16px; }
  .fig { flex: 1; padding: 14px; }
  .fig img { display: block; width: 100%; aspect-ratio: 1 / 1; border-radius: 0.6rem; }
  .cap { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
  .cap .name { font-weight: 700; font-size: 15px; }
  .cap .nn { opacity: 0.6; font-size: 12px; }
  .cap .fill b { color: var(--primary); }
  .controls { margin-top: 18px; padding: 18px; display: flex; flex-wrap: wrap;
              gap: 24px 40px; align-items: flex-start; }
  .ctrl label { display: flex; justify-content: space-between; font-weight: 600; font-size: 14px;
                width: 340px; }
  .ctrl .val { color: var(--primary); font-variant-numeric: tabular-nums; }
  input[type=range] { width: 340px; margin-top: 6px; accent-color: var(--primary); }
  .ticks { display: flex; justify-content: space-between; width: 340px;
           font-size: 10px; opacity: 0.55; margin-top: 2px; }
  .formula { font-family: ui-monospace, monospace; padding: 8px 10px; border-radius: 6px;
             font-size: 13px; background: var(--surface-container-highest); }
  .hint { opacity: 0.62; font-size: 12px; margin-top: 12px; line-height: 1.5; max-width: 520px; }
</style>
</head>
<body class="dark">
<div class="wrap">
  <div class="head-row">
    <h5 class="no-margin">Adaptive layout: ns_coef &times; ns_exp &mdash; 2k vs 64k vs 200k</h5>
    <span class="chip primary small round">ns_cap = &infin; (no cap)</span>
  </div>
  <p class="small-text" style="opacity:0.6; margin-top:4px">ChEMBL subsets (same chemistry, only N differs) &middot; components bridged into one tree &middot; same two sliders drive all three &middot; 8&times;8 precomputed (untangle off)</p>

  <div class="images">
    <article class="fig round surface-container">
      <div class="cap"><span class="name">__NAME_0__</span><span class="nn">__NN_0__ nodes</span>
        <span class="fill">fill <b id="fill0"></b></span></div>
      <img id="tile0" alt="__NAME_0__ layout">
    </article>
    <article class="fig round surface-container">
      <div class="cap"><span class="name">__NAME_1__</span><span class="nn">__NN_1__ nodes</span>
        <span class="fill">fill <b id="fill1"></b></span></div>
      <img id="tile1" alt="__NAME_1__ layout">
    </article>
    <article class="fig round surface-container">
      <div class="cap"><span class="name">__NAME_2__</span><span class="nn">__NN_2__ nodes</span>
        <span class="fill">fill <b id="fill2"></b></span></div>
      <img id="tile2" alt="__NAME_2__ layout">
    </article>
  </div>

  <article class="controls round surface-container">
    <div class="ctrl">
      <label>ns_coef <span class="val" id="vcoef"></span></label>
      <input type="range" id="scoef" min="0" max="__GMAX__" step="1">
      <div class="ticks" id="tcoef"></div>
    </div>
    <div class="ctrl">
      <label>ns_exp <span class="val" id="vexp"></span></label>
      <input type="range" id="sexp" min="0" max="__GMAX__" step="1">
      <div class="ticks" id="texp"></div>
    </div>
    <div>
      <div class="formula">ns_cap = &infin; &nbsp;&rArr;&nbsp; nodeSize(n) = ns_coef / n<sup>ns_exp</sup></div>
      <div class="hint">Same (ns_coef, ns_exp) on all three panels &mdash; only the dataset size differs.
        Because node size depends on the per-level node count n, the larger sets reach a given spread
        at different settings: watch the three fill values diverge as you move ns_exp, then re-converge
        as the trees fully collapse.</div>
    </div>
  </article>
</div>

<script>
const G = __G__;
const TILES = [__TILES_0__, __TILES_1__, __TILES_2__];
const FILLS = [__FILLS_0__, __FILLS_1__, __FILLS_2__];
const COEFS = __COEFS__, EXPS = __EXPS__;
function fillTicks(id, vals) {
  document.getElementById(id).innerHTML = vals.map(v => "<span>" + v + "</span>").join("");
}
fillTicks("tcoef", COEFS); fillTicks("texp", EXPS);
const scoef = document.getElementById("scoef"), sexp = document.getElementById("sexp");
scoef.value = 3; sexp.value = 2;
function update() {
  const ci = +scoef.value, cj = +sexp.value, idx = ci * G + cj;
  for (let p = 0; p < 3; p++) {
    document.getElementById("tile" + p).src = TILES[p][idx];
    document.getElementById("fill" + p).textContent = (FILLS[p][idx] * 100).toFixed(0) + "%";
  }
  document.getElementById("vcoef").textContent = COEFS[ci];
  document.getElementById("vexp").textContent = EXPS[cj];
}
[scoef, sexp].forEach(s => s.addEventListener("input", update));
update();
</script>
<script type="module" src="https://cdn.jsdelivr.net/npm/beercss@4.0.21/dist/cdn/beer.min.js"></script>
<script type="module" src="https://cdn.jsdelivr.net/npm/material-dynamic-colors@1.1.4/dist/cdn/material-dynamic-colors.min.js"></script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
