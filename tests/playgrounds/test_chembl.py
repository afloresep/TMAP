import pandas as pd
import pytest

from tmap import TMAP
from tmap.playgrounds.chembl import ChemblPlayground


@pytest.fixture
def tiny_chembl(tmp_path):
    smiles = ["CCO", "CCN", "CCC", "c1ccccc1"]
    df = pd.DataFrame({
        "canonical_smiles": smiles,
        "chembl_id": [f"CHEMBL{i}" for i in range(len(smiles))],
        "scaffold": smiles,
        "mw": [46, 45, 44, 78],
        "logp": [-0.1, -0.2, 1.8, 2.1],
        "qed": [0.4, 0.4, 0.3, 0.6],
    })
    from tmap.utils import fingerprints_from_smiles
    fps = fingerprints_from_smiles(smiles, fp_type="morgan", radius=2, n_bits=512)
    # store_index=True is required for kneighbors on binary jaccard (USearch path)
    model = TMAP(metric="jaccard", n_neighbors=3, seed=0, store_index=True).fit(fps)
    model_path = tmp_path / "chembl.tmap"
    meta_path = tmp_path / "meta.parquet"
    model.save(model_path)
    df.to_parquet(meta_path)
    # Pass matching fp params so _encode uses the same bit size as the fitted model
    return ChemblPlayground(model_path, meta_path, fp_radius=2, fp_n_bits=512)


def test_query_smiles(tiny_chembl):
    rs = tiny_chembl.query("CCO", k=3)
    assert len(rs) == 3
    assert any(r.extra["chembl_id"] == "CHEMBL0" for r in rs)


def test_query_invalid_raises(tiny_chembl):
    with pytest.raises(ValueError):
        tiny_chembl.query("not-a-smiles!", k=3)


def test_path(tiny_chembl):
    pr = tiny_chembl.path("CCO", "c1ccccc1")
    assert pr.nodes[0].label == "CHEMBL0"
    assert pr.nodes[-1].label == "CHEMBL3"
