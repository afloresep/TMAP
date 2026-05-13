import numpy as np
import pandas as pd
import pytest

from tmap import TMAP
from tmap.playgrounds.protein import ProteinPlayground


@pytest.fixture
def tiny_protein(tmp_path):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(6, 16)).astype("float32")
    accs = [f"P{i:03d}" for i in range(6)]
    meta = pd.DataFrame({
        "accession": accs,
        "organism": ["O1"] * 3 + ["O2"] * 3,
        "domain": ["Bacteria-alpha"] * 3 + ["Eukaryota-mito"] * 3,
        "compartment": ["-"] * 3 + ["mito"] * 3,
    })
    model = TMAP(metric="cosine", n_neighbors=3, seed=0, store_index=True).fit(emb)
    base = tmp_path / "pg"
    base.mkdir()
    model.save(base / "proteins.tmap")
    np.save(base / "embeddings.npy", emb)
    meta.to_parquet(base / "proteins_meta.parquet")

    def fake_encode(seq: str) -> np.ndarray:
        h = abs(hash(seq))
        rng2 = np.random.default_rng(h)
        return rng2.normal(size=16).astype("float32")

    return ProteinPlayground(
        base / "proteins.tmap",
        base / "proteins_meta.parquet",
        encode_fn=fake_encode,
    )


def test_query_accession(tiny_protein):
    rs = tiny_protein.query("P000", k=3)
    assert len(rs) == 3
    assert rs[0].label == "P000"


def test_query_sequence(tiny_protein):
    rs = tiny_protein.query("MKTVL" * 10, k=3)  # >=20 chars to be treated as a sequence
    assert len(rs) == 3


def test_path(tiny_protein):
    pr = tiny_protein.path("P000", "P005")
    assert pr.nodes[0].label == "P000"
    assert pr.nodes[-1].label == "P005"


def test_gallery_empty_by_default(tiny_protein):
    assert tiny_protein.gallery() == []


def test_alphafold_url_in_extra(tiny_protein):
    rs = tiny_protein.query("P000", k=3)
    assert "alphafold_url" in rs[0].extra
    assert "P000" in rs[0].extra["alphafold_url"]
