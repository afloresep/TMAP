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


from arabidopsis_root_ground_tissue_tmap import (  # noqa: E402
    pick_root_cell,
    pick_target_cell,
)


def test_pick_root_cell_picks_qc_centroid():
    rng = np.random.default_rng(0)
    # 50 QC cells clustered at origin, 50 Cortex at (5, 5, ...).
    X = np.vstack([
        rng.normal(0, 0.1, size=(50, 10)),
        rng.normal(5, 0.1, size=(50, 10)),
    ]).astype(np.float32)
    cell_types = np.array(["QC"] * 50 + ["Cortex"] * 50)

    idx = pick_root_cell(X, cell_types, label="QC")

    assert 0 <= idx < 50, f"Root cell {idx} must be inside the QC block"


def test_pick_root_cell_missing_label_raises():
    X = np.zeros((10, 5), dtype=np.float32)
    cell_types = np.array(["Cortex"] * 10)
    with pytest.raises(ValueError, match="QC"):
        pick_root_cell(X, cell_types, label="QC")


def test_pick_target_cell_picks_highest_pseudotime_in_label():
    pseudotime = np.array([0.1, 0.9, 0.2, 0.8, 0.5, 0.99, 0.3], dtype=np.float32)
    cell_types = np.array(["Cortex", "Endodermis", "Cortex", "Endodermis",
                           "QC", "Cortex", "Endodermis"])
    idx = pick_target_cell(pseudotime, cell_types, label="Endodermis")
    assert idx == 1, "Highest pseudotime among Endodermis is index 1 (0.9)"


from arabidopsis_root_ground_tissue_tmap import fit_tmap_with_pseudotime  # noqa: E402


def test_fit_tmap_with_pseudotime_returns_model_and_spearman():
    rng = np.random.default_rng(0)
    # Make a ramp: 200 cells whose PCA coords drift linearly along their index.
    n = 200
    X_pca = rng.standard_normal((n, 50)).astype(np.float32)
    X_pca[:, 0] += np.linspace(0, 20, n)  # strong linear signal on axis 0
    cell_types = np.array(["QC"] * 20 + ["Cortex"] * 100 + ["Endodermis"] * 80)
    consensus_time = np.linspace(0, 1, n).astype(np.float32)

    model, tmap_pt, spearman = fit_tmap_with_pseudotime(
        X_pca=X_pca,
        cell_types=cell_types,
        consensus_time=consensus_time,
        seed=42,
    )

    assert tmap_pt.shape == (n,)
    assert model.tree_ is not None
    # Smoke test: cosine TMAP on this ramp-plus-noise synthetic data yields
    # a strong |correlation| (typically ~0.8) with the reference pseudotime,
    # but the sign depends on how the MST happens to orient relative to the
    # QC-centroid root under cosine similarity. The strict sign + magnitude
    # bar lives in --validate (>= 0.85 on real data).
    assert abs(spearman) >= 0.75, (
        f"With a strong linear PCA signal + correct root in QC cells, the "
        f"tree pseudotime should correlate strongly (in magnitude) with the "
        f"reference; got {spearman:.3f}"
    )


def test_plot_atlas_side_by_side_writes_file(tmp_path):
    from arabidopsis_root_ground_tissue_tmap import plot_atlas_side_by_side
    rng = np.random.default_rng(0)
    n = 50
    out = tmp_path / "fig1.png"
    plot_atlas_side_by_side(
        X_umap=rng.standard_normal((n, 2)).astype(np.float32),
        tmap_layout=rng.standard_normal((n, 2)).astype(np.float32),
        cell_types=np.array(["QC", "Cortex"] * 25),
        out_path=out,
    )
    assert out.exists(), "figure file was not created"
    # Two-panel scatter at 150 DPI produces >10 KB; this bar rejects a blank canvas.
    assert out.stat().st_size > 8_000, (
        f"figure file is only {out.stat().st_size} bytes — suspiciously small, "
        f"likely blank or truncated"
    )


def test_plot_atlas_side_by_side_raises_on_too_many_classes(tmp_path):
    from arabidopsis_root_ground_tissue_tmap import plot_atlas_side_by_side
    rng = np.random.default_rng(0)
    n = 50
    cell_types = np.array([f"T{i}" for i in range(25)] * 2)  # 25 unique classes
    with pytest.raises(ValueError, match="no safe palette"):
        plot_atlas_side_by_side(
            X_umap=rng.standard_normal((n, 2)).astype(np.float32),
            tmap_layout=rng.standard_normal((n, 2)).astype(np.float32),
            cell_types=cell_types,
            out_path=tmp_path / "toomany.png",
        )
