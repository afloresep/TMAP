"""Unit tests for the Arabidopsis root TMAP example helpers.

These tests use synthetic AnnData objects; the real ground-tissue h5ad is
not committed and would require an R toolchain to regenerate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))

from arabidopsis_root_ground_tissue_tmap import (  # noqa: E402
    ShahanAtlas,
    load_shahan_h5ad,
)


def _make_synthetic_atlas(n_cells: int = 200, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X_pca = rng.standard_normal((n_cells, 50)).astype(np.float32)
    X_umap = rng.standard_normal((n_cells, 2)).astype(np.float32)
    obs = pd.DataFrame({
        "cell_type": rng.choice(["QC", "Cortex", "Endodermis"], n_cells),
        "celltype.anno": rng.choice(["QC", "Cortex", "Endodermis"], n_cells),
        "consensus_time": rng.uniform(0, 1, n_cells),
        "sample": rng.choice(["WT_rep1", "WT_rep2", "scr4_rep1"], n_cells),
    })
    adata = ad.AnnData(X=rng.standard_normal((n_cells, 10)).astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = X_pca
    adata.obsm["X_umap"] = X_umap
    return adata


def test_load_shahan_h5ad_returns_shahan_atlas(tmp_path):
    adata = _make_synthetic_atlas()
    path = tmp_path / "ground_tissue.h5ad"
    adata.write_h5ad(path)

    atlas = load_shahan_h5ad(path)

    assert isinstance(atlas, ShahanAtlas)
    assert atlas.X_pca.shape == (200, 50)
    assert atlas.X_umap.shape == (200, 2)
    assert "consensus_time" in atlas.obs.columns
    assert atlas.n_cells == 200


def test_load_shahan_h5ad_missing_pca_raises(tmp_path):
    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=rng.standard_normal((50, 5)).astype(np.float32))
    path = tmp_path / "bad.h5ad"
    adata.write_h5ad(path)

    with pytest.raises(KeyError, match="X_pca"):
        load_shahan_h5ad(path)


def test_load_shahan_h5ad_missing_umap_falls_back_to_zeros(tmp_path):
    adata = _make_synthetic_atlas(n_cells=30)
    del adata.obsm["X_umap"]
    path = tmp_path / "no_umap.h5ad"
    adata.write_h5ad(path)

    atlas = load_shahan_h5ad(path)

    assert atlas.X_umap.shape == (30, 2)
    assert np.all(atlas.X_umap == 0)
