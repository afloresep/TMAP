"""Reproduce the Shahan et al. 2022 Arabidopsis root ground-tissue atlas with TMAP2.

Source paper: https://doi.org/10.1016/j.devcel.2022.01.008 (Shahan et al. 2022,
Developmental Cell). Data: GEO GSE152766, ground-tissue sub-atlas.

Run once:
    Rscript examples/data/shahan_root/prepare.R

Then:
    python examples/arabidopsis_root_ground_tissue_tmap.py
    python examples/arabidopsis_root_ground_tissue_tmap.py --validate
    python examples/arabidopsis_root_ground_tissue_tmap.py --serve
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
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
    return ShahanAtlas(X_pca=X_pca, X_umap=X_umap, obs=adata.obs.copy())


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
    report per-cell-type enrichment (log-odds of WT vs mutant in each type).

    Returns the enrichment dict for downstream validation.
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

    # Enrichment per WT cell-type (coarse version — no subtree walk).
    is_mutant = np.concatenate([np.zeros(len(coords_wt), dtype=bool),
                                np.ones(len(coords_mutant), dtype=bool)])
    enrichment: dict[str, float] = {}
    for label in sorted(set(cell_types_wt.tolist())):
        # A "subtree" here = all cells of this WT label + mutants projected
        # near them. We use nearest-WT-label as a proxy.
        in_sub_wt = cell_types_wt == label
        dists = np.linalg.norm(
            coords_mutant[:, None, :] - coords_wt[None, in_sub_wt, :], axis=2
        ).min(axis=1)
        threshold = np.median(np.linalg.norm(coords_wt - coords_wt.mean(axis=0), axis=1))
        in_sub_mut = dists < threshold
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


def main() -> None:
    raise NotImplementedError("Filled in later tasks.")


if __name__ == "__main__":
    main()
