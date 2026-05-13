"""Pass B: compare a TMAP H3N2 HA tree against the Nextstrain ML phylogeny.

Uses the Nextstrain Auspice JSON for the seasonal H3N2 HA 12-year build,
which contains the published ML tree for ~1598 representative strains plus
their metadata. Sequences are reconstructed from the root sequence and the
per-branch AA mutations stored in the JSON (no separate FASTA required).

We fit TMAP on the HA1 amino-acid k-mer Jaccard fingerprints of those exact
strains, then measure pairwise-distance agreement:

    Pairwise-distance Spearman: for a random sample of strain pairs, compute
    the hop distance in TMAP tree and the hop distance in the Nextstrain ML
    tree, then correlate.

Note: Robinson-Foulds is omitted. Flu Auspice internal-node labels do not
align cleanly with TMAP internal nodes, so RF would be an apples-to-oranges
comparison. Spearman pairwise-hop is the relevant metric here.

Outputs:
    paper/images/flu_h3n2_ha_compare.png
    examples/flu_h3n2_ha_compare_report.txt
"""
from __future__ import annotations

import argparse
import sys as _sys
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

HERE = Path(__file__).parent
_sys.path.insert(0, str(HERE))
import flu_h3n2_ha_tmap as base  # noqa: E402

IMG_PATH = HERE.parent / "paper" / "images" / "flu_h3n2_ha_compare.png"
REPORT_PATH = HERE / "flu_h3n2_ha_compare_report.txt"


# ---------------------------------------------------------------------------
# Dataset-agnostic helper functions (verbatim-compatible with SARS compare)
# ---------------------------------------------------------------------------

def auspice_adjacency(tree_root) -> tuple[dict[str, int], dict[int, list[int]]]:
    """Walk the Auspice tree and return name->idx + adjacency for ALL nodes."""
    name_to_idx: dict[str, int] = {}
    adj: dict[int, list[int]] = {}

    counter = [0]

    def visit(node, parent_idx):
        name = node.get("name") or f"__internal_{counter[0]}"
        counter[0] += 1
        idx = len(name_to_idx)
        name_to_idx[name] = idx
        adj[idx] = []
        if parent_idx is not None:
            adj[parent_idx].append(idx)
            adj[idx].append(parent_idx)
        for c in node.get("children") or []:
            visit(c, idx)

    visit(tree_root, None)
    return name_to_idx, adj


def tmap_adjacency(tree) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {i: [] for i in range(tree.n_nodes)}
    for src, tgt in tree.edges:
        adj[int(src)].append(int(tgt))
        adj[int(tgt)].append(int(src))
    return adj


def plot_pairwise(
    tmap_hops_flat: NDArray,
    ns_hops_flat: NDArray,
    rho: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    n_pairs = len(tmap_hops_flat)
    sample = np.random.default_rng(0).choice(n_pairs, size=min(20000, n_pairs), replace=False)
    ax.hexbin(
        tmap_hops_flat[sample], ns_hops_flat[sample],
        gridsize=50, cmap="viridis", mincnt=1,
    )
    ax.set_xlabel("TMAP tree hop distance")
    ax.set_ylabel("Nextstrain ML tree hop distance")
    ax.set_title(f"Pairwise tree-distance agreement (Spearman={rho:.3f})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=5, help="AA k-mer size")
    p.add_argument("--n-neighbors", type=int, default=60)
    p.add_argument("--n-permutations", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-pairs", type=int, default=200_000,
        help="Subsample pair count for the Spearman correlation.",
    )
    p.add_argument(
        "--max-tips", type=int, default=0,
        help="If >0, subsample this many tips uniformly at random (for fast runs).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    from tmap import TMAP

    base.ensure_auspice_json()
    base.ensure_root_sequence_json()
    print("Loading Auspice JSON ...", flush=True)
    tree, root_seq, tip_meta = base.load_auspice_h3n2()
    print(f"  {len(tip_meta):,} tips, root HA1 length {len(root_seq)}", flush=True)

    seqs_by_strain = base.reconstruct_tip_sequences(tree, root_seq, gene="HA1")

    # Optional subsampling for fast runs.
    if args.max_tips and len(tip_meta) > args.max_tips:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(tip_meta), size=args.max_tips, replace=False)
        tip_meta = [tip_meta[i] for i in idx]

    atlas = base.build_atlas(tip_meta, seqs_by_strain, k=args.k)
    print(f"  atlas size: {len(atlas.names):,}", flush=True)

    print("Fitting TMAP...", flush=True)
    model = TMAP(
        metric="jaccard",
        n_neighbors=args.n_neighbors,
        n_permutations=args.n_permutations,
        kc=80,
        seed=args.seed,
        minhash_seed=args.seed,
    ).fit(atlas.kmers)

    print("Building Nextstrain adjacency...", flush=True)
    ns_name_to_idx, ns_adj = auspice_adjacency(tree)
    print("Building TMAP adjacency...", flush=True)
    tmap_adj = tmap_adjacency(model.tree_)

    common = [n for n in atlas.names if n in ns_name_to_idx]
    print(f"Common strains for distance comparison: {len(common):,}", flush=True)
    tmap_idx_of = {name: i for i, name in enumerate(atlas.names)}

    rng = np.random.default_rng(args.seed)
    if len(common) < 3:
        raise RuntimeError("Need at least 3 common strains for comparison.")
    pair_count = min(args.max_pairs, len(common) * (len(common) - 1) // 2)
    print(f"Sampling {pair_count:,} pairs for distance correlation...", flush=True)
    a_idx = rng.integers(0, len(common), size=pair_count)
    b_idx = rng.integers(0, len(common), size=pair_count)
    mask = a_idx != b_idx
    a_idx, b_idx = a_idx[mask], b_idx[mask]
    pair_count = len(a_idx)

    print("Computing pairwise hops on sampled pairs...", flush=True)
    tmap_hops = np.empty(pair_count, dtype=np.int32)
    bfs_cache: dict[int, NDArray[np.int32]] = {}

    def tmap_bfs(src: int) -> NDArray[np.int32]:
        if src in bfs_cache:
            return bfs_cache[src]
        dist = np.full(model.tree_.n_nodes, -1, dtype=np.int32)
        dist[src] = 0
        queue: deque[int] = deque([src])
        while queue:
            node = queue.popleft()
            for nbr in tmap_adj[node]:
                if dist[nbr] < 0:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        bfs_cache[src] = dist
        return dist

    ns_bfs_cache: dict[int, NDArray[np.int32]] = {}

    def ns_bfs(src: int) -> NDArray[np.int32]:
        if src in ns_bfs_cache:
            return ns_bfs_cache[src]
        n = len(ns_adj)
        dist = np.full(n, -1, dtype=np.int32)
        dist[src] = 0
        queue: deque[int] = deque([src])
        while queue:
            node = queue.popleft()
            for nbr in ns_adj[node]:
                if dist[nbr] < 0:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        ns_bfs_cache[src] = dist
        return dist

    ns_hops = np.empty(pair_count, dtype=np.int32)
    for i in range(pair_count):
        name_a = common[int(a_idx[i])]
        name_b = common[int(b_idx[i])]
        ta = tmap_idx_of[name_a]
        tb = tmap_idx_of[name_b]
        tmap_hops[i] = tmap_bfs(ta)[tb]
        na = ns_name_to_idx[name_a]
        nb = ns_name_to_idx[name_b]
        ns_hops[i] = ns_bfs(na)[nb]
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1:,}/{pair_count:,}", flush=True)

    valid = (tmap_hops >= 0) & (ns_hops >= 0)
    rho = float(spearmanr(tmap_hops[valid], ns_hops[valid]).statistic)
    print(f"Pairwise-distance Spearman: {rho:.3f}", flush=True)

    plot_pairwise(tmap_hops[valid], ns_hops[valid], rho, IMG_PATH)

    lines = [
        "Flu H3N2 HA phylogeny comparison (Pass B)",
        f"Nextstrain JSON tips: {len(tip_meta):,}",
        f"TMAP atlas size: {len(atlas.names):,}",
        f"Common strains: {len(common):,}",
        f"k-mer size (AA): {args.k}",
        (
            f"TMAP: metric=jaccard, n_neighbors={args.n_neighbors},"
            f" n_permutations={args.n_permutations}, kc=80"
        ),
        "",
        "Pairwise tree-distance agreement:",
        f"  Spearman(TMAP hops, Nextstrain hops) on {pair_count:,} pairs = {rho:.3f}",
        "",
        f"PNG: {IMG_PATH}",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
