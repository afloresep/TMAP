"""Reproduce the Shahan et al. 2022 Arabidopsis root ground-tissue atlas with TMAP2.

Source paper: https://doi.org/10.1016/j.devcel.2022.01.008 (Shahan et al. 2022,
Developmental Cell). Data: GEO GSE152766, ground-tissue sub-atlas.

Run once:
    Rscript examples/data/shahan_root/prepare.R

Then:
    python examples/arabidopsis_root_ground_tissue_tmap.py
    python examples/arabidopsis_root_ground_tissue_tmap.py --validate
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from tmap import TMAP
from tmap.graph.analysis import edge_delta

HERE = Path(__file__).parent
DATA_PATH = HERE / "data" / "shahan_root" / "ground_tissue.h5ad"
IMG_DIR = HERE.parent / "paper" / "images"


@dataclass
class ShahanAtlas:
    """Typed view into the published ground-tissue sub-atlas."""
    X_pca: NDArray[np.float32]
    X_umap: NDArray[np.float32]
    obs: pd.DataFrame
    var_names: NDArray         # gene symbols
    expression: NDArray        # (n_cells, n_genes) dense log-normalized counts

    @property
    def n_cells(self) -> int:
        return self.X_pca.shape[0]


def load_shahan_h5ad(path: Path | str) -> ShahanAtlas:
    """Load the h5ad produced by `examples/data/shahan_root/prepare.R`.

    Raises KeyError if X_pca is missing (signals a bad conversion).
    """
    adata = ad.read_h5ad(path)
    if "X_pca" not in adata.obsm:
        raise KeyError(
            f"X_pca missing from {path}. Re-run the R conversion script; "
            f"available obsm keys: {list(adata.obsm.keys())}"
        )
    X_pca = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    X_umap = np.asarray(adata.obsm.get("X_umap", np.zeros((adata.n_obs, 2))), dtype=np.float32)
    expression = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    var_names = np.asarray(adata.var_names)
    return ShahanAtlas(
        X_pca=X_pca, X_umap=X_umap, obs=adata.obs.copy(),
        var_names=var_names, expression=expression,
    )


def pick_root_cell(X: NDArray, cell_types: NDArray, *, label: str) -> int:
    """Return the index of the cell closest to the centroid of cells with
    cell_type==label.

    Raises ValueError if no cells carry that label.
    """
    mask = cell_types == label
    if not mask.any():
        unique = sorted(set(cell_types.tolist()))
        raise ValueError(f"No cells with cell_type={label!r}. Available: {unique}")
    pool = np.where(mask)[0]
    centroid = X[pool].mean(axis=0)
    local = int(np.argmin(np.linalg.norm(X[pool] - centroid, axis=1)))
    return int(pool[local])


def pick_target_cell(
    pseudotime: NDArray, cell_types: NDArray, *, label: str,
) -> int:
    """Return the index of the highest-pseudotime cell with cell_type==label."""
    mask = cell_types == label
    if not mask.any():
        unique = sorted(set(cell_types.tolist()))
        raise ValueError(f"No cells with cell_type={label!r}. Available: {unique}")
    pool = np.where(mask)[0]
    return int(pool[np.argmax(pseudotime[pool])])


def fit_tmap_with_pseudotime(
    *,
    X_pca: NDArray,
    cell_types: NDArray,
    consensus_time: NDArray,
    n_neighbors: int = 30,
    seed: int = 42,
) -> tuple[TMAP, NDArray[np.float32], float]:
    """Fit TMAP on X_pca, root at a QC-centroid cell, return (model, tree pseudotime, Spearman).

    n_neighbors=30 matches the Shahan et al. RunUMAP setting.
    """
    model = TMAP(
        metric="cosine",
        n_neighbors=n_neighbors,
        seed=seed,
        store_index=True,
    ).fit(X_pca.astype(np.float32, copy=False))

    root = pick_root_cell(X_pca, cell_types, label="QC")
    tmap_pt = np.asarray(model.distances_from(root), dtype=np.float32)

    finite = np.isfinite(consensus_time) & np.isfinite(tmap_pt)
    rho = float(spearmanr(tmap_pt[finite], consensus_time[finite]).statistic)
    return model, tmap_pt, rho


def plot_atlas_side_by_side(
    *,
    X_umap: NDArray,
    tmap_layout: NDArray,
    cell_types: NDArray,
    out_path: Path,
) -> None:
    """Two panels: Shahan's published UMAP vs our TMAP, same cells, same colors."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)

    unique = sorted(set(cell_types.tolist()))
    if len(unique) <= 10:
        cmap = plt.get_cmap("tab10")
    elif len(unique) <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        raise ValueError(
            f"plot_atlas_side_by_side has no safe palette for {len(unique)} "
            f"cell types (>20). Pass a subset or map labels to coarser groups."
        )
    color_by = {label: cmap(i) for i, label in enumerate(unique)}
    colors = np.array([color_by[c] for c in cell_types])

    for ax, coords, title in (
        (axes[0], X_umap, "Shahan et al. UMAP (published)"),
        (axes[1], tmap_layout, "TMAP2 (this work)"),
    ):
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=3, linewidths=0, alpha=0.7)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")

    handles = [plt.scatter([], [], c=[color_by[label]], s=24, label=label) for label in unique]
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   fontsize=8, frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def marker_expression_along_path(
    *,
    expression: NDArray,
    gene_names: NDArray,
    markers: tuple[str, ...],
    path: NDArray,
) -> dict[str, NDArray]:
    """Extract expression of each marker along the ordered cell path."""
    name_to_col = {str(g): i for i, g in enumerate(gene_names)}
    out: dict[str, NDArray] = {}
    for m in markers:
        if m not in name_to_col:
            continue
        col = name_to_col[m]
        values = np.asarray(expression[path, col]).ravel().astype(np.float32)
        out[m] = values
    return out


def plot_path_killshot(
    *,
    model: TMAP,
    root: int,
    target: int,
    expression: NDArray,
    gene_names: NDArray,
    consensus_time: NDArray,
    markers: tuple[str, ...] = ("SCR", "MYB36", "CASP1"),
    out_path: Path,
) -> dict[str, object]:
    """Three panels: path on tree, marker sweep, edge_delta histogram.

    Returns a dict with 'path', 'hops', 'markers' keys for downstream reporting.
    """
    tree = model.tree_
    path = np.asarray(tree.path(root, target), dtype=np.int64)
    hops = len(path)

    markers_along = marker_expression_along_path(
        expression=expression, gene_names=gene_names, markers=markers, path=path,
    )
    deltas = edge_delta(tree, consensus_time)

    layout = model.embedding_

    fig = plt.figure(figsize=(14, 4.2), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.3, 1.2, 1.0))
    ax_tree, ax_markers, ax_hist = (fig.add_subplot(gs[0, i]) for i in range(3))

    ax_tree.scatter(layout[:, 0], layout[:, 1], c="lightgray", s=2, linewidths=0)
    path_xy = layout[path]
    ax_tree.plot(path_xy[:, 0], path_xy[:, 1], "-", lw=1.2, color="#222")
    ax_tree.scatter(path_xy[:, 0], path_xy[:, 1],
                    c=np.arange(hops), cmap="magma", s=14, zorder=3)
    ax_tree.set_title(f"Tree path (QC → mature endodermis, {hops} hops)")
    ax_tree.set_xticks([])
    ax_tree.set_yticks([])
    ax_tree.set_aspect("equal", adjustable="datalim")

    for name, values in markers_along.items():
        ax_markers.plot(np.arange(hops), values, marker="o", ms=3, label=name, lw=1.0)
    ax_markers.set_xlabel("Hop index along path")
    ax_markers.set_ylabel("Expression")
    ax_markers.set_title("Marker expression along tree path")
    ax_markers.legend(fontsize=8, frameon=False)

    ax_hist.hist(np.abs(deltas), bins=40, color="#4c72b0")
    ax_hist.set_title("|Δ consensus pseudotime| per tree edge")
    ax_hist.set_xlabel("Absolute pseudotime delta per edge")
    ax_hist.set_ylabel("Edges")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "path": path.tolist(),
        "hops": hops,
        "markers": {k: v.tolist() for k, v in markers_along.items()},
        "edge_delta_mean": float(np.mean(np.abs(deltas))),
    }


def plot_mutant_projection(
    *,
    model: TMAP,
    X_pca_mutant: NDArray,
    X_pca_wt: NDArray,
    cell_types_wt: NDArray,
    out_path: Path,
) -> dict[str, float]:
    """Project mutant cells onto the fitted WT tree via TMAP.transform() and
    report per-cell-type enrichment.

    Each mutant cell is assigned to its nearest WT cell in the 2-D MST layout
    (disjoint; one label per mutant). Enrichment is the Haldane-Anscombe-
    corrected log-odds of WT vs mutant inside each label's subtree relative
    to the rest of the tree. Positive = WT-enriched (mutant-depleted) in that
    cell type.

    Writes `out_path` as a PNG and returns the per-label enrichment dict.
    """
    coords_mutant = model.transform(X_pca_mutant.astype(np.float32, copy=False))
    coords_wt = model.embedding_

    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=150)
    ax.scatter(coords_wt[:, 0], coords_wt[:, 1],
               c="lightgray", s=2, linewidths=0, label=f"WT (n={len(coords_wt)})")
    ax.scatter(coords_mutant[:, 0], coords_mutant[:, 1],
               c="#d62728", s=5, linewidths=0, alpha=0.8,
               label=f"scr-4 (n={len(coords_mutant)})")
    ax.set_title("scarecrow-4 mutant cells projected onto WT tree")
    ax.legend(fontsize=9, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # Assign each mutant cell to its nearest WT cell in the MST-derived 2-D
    # layout — this is disjoint (one label per mutant) and uses tree structure
    # via the layout coordinates (not an arbitrary Euclidean disk).
    kd = cKDTree(coords_wt)
    _, nearest_wt = kd.query(coords_mutant, k=1)
    mutant_labels = np.asarray(cell_types_wt)[nearest_wt]

    is_mutant = np.concatenate([np.zeros(len(coords_wt), dtype=bool),
                                np.ones(len(coords_mutant), dtype=bool)])
    enrichment: dict[str, float] = {}
    for label in sorted(set(cell_types_wt.tolist())):
        in_sub_wt  = cell_types_wt == label
        in_sub_mut = mutant_labels == label
        in_subtree = np.concatenate([in_sub_wt, in_sub_mut])
        enrichment[label] = subtree_enrichment(in_subtree=in_subtree, is_mutant=is_mutant)
    return enrichment


def subtree_enrichment(*, in_subtree: NDArray, is_mutant: NDArray) -> float:
    """Log-odds of WT-vs-mutant inside a subtree relative to outside.

    Positive = WT-enriched (equivalently, mutant-depleted) inside the subtree.
    Uses Haldane-Anscombe correction (+0.5 to every cell) for zero-safety.
    """
    a = float(((in_subtree) & (~is_mutant)).sum()) + 0.5   # WT inside
    b = float(((in_subtree) & ( is_mutant)).sum()) + 0.5   # mutant inside
    c = float(((~in_subtree) & (~is_mutant)).sum()) + 0.5  # WT outside
    d = float(((~in_subtree) & ( is_mutant)).sum()) + 0.5  # mutant outside
    return float(np.log((a / b) / (c / d)))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Arabidopsis root ground-tissue TMAP reproduction.")
    p.add_argument("--path", type=Path, default=DATA_PATH,
                   help="Path to the h5ad produced by data/shahan_root/prepare.R.")
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mutant-sample-col", type=str, default="sample")
    p.add_argument("--mutant-value", type=str, default="scr4",
                   help="Substring matched against obs[mutant_sample_col] to select mutant cells.")
    p.add_argument("--root-label", type=str, default="QC")
    p.add_argument("--target-label", type=str, default="Endodermis")
    p.add_argument("--validate", action="store_true",
                   help="Exit non-zero if success criteria fail.")
    return p


def _criterion(name: str, ok: bool, detail: str) -> tuple[str, bool, str]:
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name}: {detail}")
    return (name, ok, detail)


def _is_monotone_non_decreasing(xs) -> bool:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size <= 1:
        return True
    return bool(np.all(np.diff(arr) >= -1e-6))


def main() -> None:
    args = _build_parser().parse_args()
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading atlas from {args.path} …")
    atlas = load_shahan_h5ad(args.path)
    print(f"  {atlas.n_cells:,} cells, obs columns: {list(atlas.obs.columns)[:10]}")

    sample_col = atlas.obs[args.mutant_sample_col].astype(str).to_numpy()
    is_mutant = np.array([args.mutant_value in s for s in sample_col])
    wt_mask = ~is_mutant
    print(f"  WT cells: {int(wt_mask.sum()):,}   mutant cells: {int(is_mutant.sum()):,}")

    cell_types = atlas.obs["celltype.anno"].astype(str).to_numpy() \
        if "celltype.anno" in atlas.obs.columns \
        else atlas.obs["cell_type"].astype(str).to_numpy()
    consensus_time = atlas.obs["consensus_time"].astype(np.float32).to_numpy()

    X_pca_wt = atlas.X_pca[wt_mask]
    X_pca_mut = atlas.X_pca[is_mutant] if is_mutant.any() else None
    cell_types_wt = cell_types[wt_mask]
    consensus_time_wt = consensus_time[wt_mask]

    model, tmap_pt, rho = fit_tmap_with_pseudotime(
        X_pca=X_pca_wt,
        cell_types=cell_types_wt,
        consensus_time=consensus_time_wt,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
    )
    print(f"  Spearman(tree_pseudotime, consensus_time) = {rho:.3f}")

    # Figure 1.
    plot_atlas_side_by_side(
        X_umap=atlas.X_umap[wt_mask],
        tmap_layout=model.embedding_,
        cell_types=cell_types_wt,
        out_path=IMG_DIR / "arabidopsis_atlas_umap_vs_tmap.png",
    )
    print("  Figure 1 written.")

    # Figure 2. Expression matrix is already loaded in the atlas.
    gene_names = atlas.var_names
    X_expr_wt = atlas.expression[wt_mask]

    root = pick_root_cell(X_pca_wt, cell_types_wt, label=args.root_label)
    target = pick_target_cell(tmap_pt, cell_types_wt, label=args.target_label)
    path_report = plot_path_killshot(
        model=model,
        root=root,
        target=target,
        expression=X_expr_wt,
        gene_names=gene_names,
        consensus_time=consensus_time_wt,
        out_path=IMG_DIR / "arabidopsis_path_killshot.png",
    )
    print(f"  Figure 2 written. Path: {path_report['hops']} hops.")

    # Figure 3.
    enrichment: dict[str, float] = {}
    if X_pca_mut is not None:
        enrichment = plot_mutant_projection(
            model=model,
            X_pca_mutant=X_pca_mut,
            X_pca_wt=X_pca_wt,
            cell_types_wt=cell_types_wt,
            out_path=IMG_DIR / "arabidopsis_scr_mutant.png",
        )
        print("  Figure 3 written.")
        for k, v in enrichment.items():
            print(f"    subtree log-odds WT/mutant  [{k}] = {v:+.2f}")
    else:
        print("  Figure 3 skipped — no mutant cells found in this atlas.")

    # Validation.
    if args.validate:
        print("\nValidation:")
        results = []
        results.append(_criterion(
            "Spearman(tree_pseudotime, consensus_time) >= 0.85",
            rho >= 0.85,
            f"rho={rho:.3f}",
        ))

        markers = path_report["markers"]
        scr_values = markers.get("SCR", [])
        scr_found = len(scr_values) > 0
        scr_monotone = scr_found and _is_monotone_non_decreasing(scr_values)
        results.append(_criterion(
            "SCR expression is non-decreasing along QC→Endodermis path",
            scr_monotone,
            f"len={len(scr_values)}" + (
                f", min={min(scr_values):.2f}, max={max(scr_values):.2f}"
                if scr_found else " (SCR not found in gene_names!)"
            ),
        ))

        if enrichment:
            endodermis_wt_enriched = enrichment.get("Endodermis", 0.0) > 0
            results.append(_criterion(
                "Endodermis subtree is WT-enriched in scr-4 projection",
                endodermis_wt_enriched,
                f"log-odds={enrichment.get('Endodermis', 0.0):+.2f}",
            ))

        failures = [name for (name, ok, _) in results if not ok]
        if failures:
            print(f"\n{len(failures)}/{len(results)} criteria failed: {failures}")
            raise SystemExit(1)
        print(f"\nAll {len(results)} criteria passed.")


if __name__ == "__main__":
    main()
