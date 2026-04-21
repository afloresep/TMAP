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
import numpy as np
import pandas as pd
from numpy.typing import NDArray

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


def main() -> None:
    raise NotImplementedError("Filled in later tasks.")


if __name__ == "__main__":
    main()
