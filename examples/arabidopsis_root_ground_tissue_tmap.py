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


def main() -> None:
    raise NotImplementedError("Filled in later tasks.")


if __name__ == "__main__":
    main()
