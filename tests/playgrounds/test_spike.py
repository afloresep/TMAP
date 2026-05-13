import pandas as pd
import pytest

from tmap import TMAP
from tmap.playgrounds.spike import SpikePlayground


@pytest.fixture
def tiny_spike(tmp_path):
    seqs = ["MFVFLVLLPLVSSQCVNL", "MFVFLVLLPLVSSQCANL", "MFVFLVLLPLVSSQCVAA", "MFVFLVLLPLVSSQCANA"]
    df = pd.DataFrame({
        "strain": [f"Strain{i}" for i in range(4)],
        "clade": ["A", "A", "B", "B"],
        "country": ["X", "Y", "X", "Y"],
        "date_numeric": [2020.0, 2020.5, 2021.0, 2021.5],
        "sequence": seqs,
    })
    # Same shingling format the playground uses internally
    kmers = [[s[i:i + 4] for i in range(len(s) - 4 + 1)] for s in seqs]
    model = TMAP(metric="jaccard", n_neighbors=3, seed=0, store_index=True).fit(kmers)
    model_path = tmp_path / "m.tmap"
    meta_path = tmp_path / "m.parquet"
    model.save(model_path)
    df.to_parquet(meta_path)
    return SpikePlayground(model_path, meta_path, k=4, n_perm=64)


def test_query_sequence(tiny_spike):
    rs = tiny_spike.query("MFVFLVLLPLVSSQCVNL", k=3)
    assert len(rs) == 3
    assert rs[0].label == "Strain0"


def test_query_strain_name_resolves(tiny_spike):
    rs = tiny_spike.query("Strain1", k=3)
    assert len(rs) == 3


def test_path(tiny_spike):
    pr = tiny_spike.path("Strain0", "Strain3")
    assert pr.nodes[0].label == "Strain0" and pr.nodes[-1].label == "Strain3"
