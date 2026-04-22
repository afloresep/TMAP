"""Unit tests for endosymbiosis example helpers.

Uses synthetic data; no network calls, no ESM-2 invocation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))

from mixed_proteome_tmap import ProteinRecord  # noqa: E402


def test_protein_record_minimal_construction():
    r = ProteinRecord(
        accession="P00001", organism="Homo sapiens", source="mitocarta",
        domain="Eukarya-mito", compartment="matrix",
    )
    assert r.accession == "P00001"
    assert r.source == "mitocarta"


def test_fetch_uniprot_proteome_uses_cache_file(tmp_path, monkeypatch):
    from mixed_proteome_tmap import fetch_uniprot_proteome

    # Pre-populate the cache — the function must NOT hit the network.
    cache = tmp_path / "UP000002480.fasta"
    cache.write_text(
        ">sp|P12345|TEST_RICPR Test protein OS=Rickettsia prowazekii\n"
        "MAAAAA\n"
        ">sp|P67890|TEST2_RICPR Another OS=Rickettsia prowazekii\n"
        "MBBBBBB\n"
    )

    def _boom(*_a, **_k):
        raise RuntimeError("network should not be called when cache exists")

    monkeypatch.setattr("urllib.request.urlopen", _boom)

    ids, seqs = fetch_uniprot_proteome("UP000002480", cache_dir=tmp_path)
    assert list(ids) == ["P12345", "P67890"]
    assert seqs[0] == "MAAAAA"
    assert seqs[1] == "MBBBBBB"


def test_parse_uniprot_accession():
    from mixed_proteome_tmap import _parse_uniprot_accession
    assert _parse_uniprot_accession("sp|P12345|TEST_HUMAN") == "P12345"
    assert _parse_uniprot_accession("tr|Q9NZC2|NAME_MOUSE") == "Q9NZC2"
    assert _parse_uniprot_accession("plain_id") == "plain_id"


def test_load_mitocarta_returns_accessions_and_compartments(tmp_path):
    import openpyxl
    from mixed_proteome_tmap import load_mitocarta

    xlsx = tmp_path / "Human.MitoCarta3.0.xls"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "A Human MitoCarta3.0"
    ws.append(["Symbol", "EnsemblGeneID", "UniProt", "MitoCarta3.0_SubMitoLocalization"])
    ws.append(["COX1", "ENSG0", "P00395", "MIM_matrix"])
    ws.append(["ATP5F1A", "ENSG1", "P25705", "MIM_matrix"])
    ws.append(["TOMM20", "ENSG2", "Q15388", "OMM"])
    wb.save(xlsx)

    records = load_mitocarta(xlsx)
    by_acc = {r.accession: r for r in records}
    assert by_acc["P00395"].compartment == "MIM_matrix"
    assert by_acc["P00395"].source == "mitocarta"
    assert by_acc["P00395"].domain == "Eukarya-mito"
    assert by_acc["P00395"].organism == "Homo sapiens"
    assert set(by_acc.keys()) == {"P00395", "P25705", "Q15388"}


def test_filter_cytosolic_drops_organellar_targeting(monkeypatch):
    from mixed_proteome_tmap import filter_cytosolic

    annotations = {
        "P00001": "SUBCELLULAR LOCATION: Cytoplasm.",
        "P00002": "SUBCELLULAR LOCATION: Mitochondrion.",
        "P00003": "SUBCELLULAR LOCATION: Plastid, chloroplast.",
        "P00004": "SUBCELLULAR LOCATION: Cytoplasm; Nucleus.",
        "P00005": "",
    }
    def fake_fetch(ids, **_):
        return {
            "accession": np.array(ids, dtype=object),
            "cc_subcellular_location": np.array(
                [annotations.get(i, "") for i in ids], dtype=object,
            ),
        }
    monkeypatch.setattr("mixed_proteome_tmap.fetch_uniprot", fake_fetch)

    kept = filter_cytosolic(list(annotations.keys()))
    assert set(kept) == {"P00001", "P00004"}


def test_build_endosymbiosis_dataset_assembles_all_sources(tmp_path, monkeypatch):
    from mixed_proteome_tmap import ProteinRecord, build_endosymbiosis_dataset

    def fake_proteome(pid, *, cache_dir, reviewed_only=False):
        return ([f"{pid}_A", f"{pid}_B"], ["MAAA", "MBBB"])
    def fake_mitocarta(_path):
        return [ProteinRecord(
            accession="P00395", organism="Homo sapiens", source="mitocarta",
            domain="Eukarya-mito", compartment="MIM_matrix",
        )]
    def fake_cytosolic_filter(ids, **_):
        return [ids[0]]  # keep the first yeast ID only

    monkeypatch.setattr("mixed_proteome_tmap.fetch_uniprot_proteome", fake_proteome)
    monkeypatch.setattr("mixed_proteome_tmap.load_mitocarta", fake_mitocarta)
    monkeypatch.setattr("mixed_proteome_tmap.filter_cytosolic", fake_cytosolic_filter)
    # Fake MitoCarta sequence fetch: return a dict with `sequence` per accession.
    def fake_fetch(ids, **_):
        return {
            "accession": np.array(ids, dtype=object),
            "sequence": np.array(["MCCC"] * len(ids), dtype=object),
        }
    monkeypatch.setattr("mixed_proteome_tmap.fetch_uniprot", fake_fetch)

    records, sequences = build_endosymbiosis_dataset(cache_dir=tmp_path)

    sources = {r.source for r in records}
    assert sources == {"rickettsia", "pelagibacter", "synechocystis",
                       "yeast-cytosol", "mitocarta"}
    assert len(records) == len(sequences)
    assert all(isinstance(s, str) and len(s) >= 3 for s in sequences)


def test_write_dataset_fasta_preserves_accession_order(tmp_path):
    from mixed_proteome_tmap import ProteinRecord, write_dataset_fasta

    records = [
        ProteinRecord(accession="A1", organism="Org1", source="rickettsia",
                      domain="Bacteria-alpha", compartment="-"),
        ProteinRecord(accession="B2", organism="Org2", source="mitocarta",
                      domain="Eukarya-mito", compartment="matrix"),
    ]
    sequences = ["MAAA", "MBBB"]
    fasta = tmp_path / "dataset.fasta"
    write_dataset_fasta(records, sequences, fasta_path=fasta)

    text = fasta.read_text()
    assert text.startswith(">A1")
    assert "\nMAAA\n" in text
    assert ">B2" in text
    assert "\nMBBB\n" in text


def test_write_dataset_metadata_tsv_columns(tmp_path):
    from mixed_proteome_tmap import ProteinRecord, write_dataset_metadata_tsv

    records = [
        ProteinRecord(accession="P00395", organism="Homo sapiens", source="mitocarta",
                      domain="Eukarya-mito", compartment="MIM_matrix"),
    ]
    tsv = tmp_path / "dataset_metadata.tsv"
    write_dataset_metadata_tsv(records, tsv_path=tsv)

    lines = tsv.read_text().strip().split("\n")
    assert lines[0].split("\t") == [
        "accession", "organism", "source", "domain", "compartment",
    ]
    assert lines[1].split("\t") == [
        "P00395", "Homo sapiens", "mitocarta", "Eukarya-mito", "MIM_matrix",
    ]


def test_load_external_embeddings_validates_shape_and_order(tmp_path):
    from mixed_proteome_tmap import load_external_embeddings

    npz = tmp_path / "emb.npz"
    np.savez(npz,
             embeddings=np.ones((3, 1280), dtype=np.float32),
             accessions=np.array(["A", "B", "C"], dtype=object))

    X, accs = load_external_embeddings(npz, expected_accessions=["A", "B", "C"])
    assert X.shape == (3, 1280)
    assert list(accs) == ["A", "B", "C"]

    # Mismatched order must raise.
    with pytest.raises(ValueError, match="accession"):
        load_external_embeddings(npz, expected_accessions=["A", "C", "B"])


def test_mito_to_alpha_path_stats_recovers_endosymbiotic_signal():
    from mixed_proteome_tmap import mito_to_alpha_path_stats

    from tmap.graph.types import Tree

    # Linear tree: mito(0) -- alpha(1) -- other(2) -- cyto(3).
    # Hops: mito→alpha=1, mito→cyto=3.
    edges = np.array([(0, 1), (1, 2), (2, 3)], dtype=np.int32)
    weights = np.ones(3, dtype=np.float32)
    tree = Tree(n_nodes=4, edges=edges, weights=weights)
    sources = np.array(["mitocarta", "rickettsia", "pelagibacter", "yeast-cytosol"])

    stats = mito_to_alpha_path_stats(tree=tree, sources=sources)
    assert stats["median_hops_to_alpha"] == 1.0
    assert stats["median_hops_to_cytosol"] == 3.0
    assert stats["n_mito"] == 1


def test_alpha_branch_mito_fraction_counts_mito_within_radius():
    from mixed_proteome_tmap import alpha_branch_mito_fraction

    from tmap.graph.types import Tree

    # Tree:    mito(0) -- alpha(1) -- alpha(2) -- cyto(3) -- mito(4)
    # mito 0 is 1 hop from alpha → counted. mito 4 is 2 hops from alpha(2) → counted.
    # Expected: 2 / 2 = 1.0 at radius 2.
    edges = np.array([(0, 1), (1, 2), (2, 3), (3, 4)], dtype=np.int32)
    weights = np.ones(4, dtype=np.float32)
    tree = Tree(n_nodes=5, edges=edges, weights=weights)
    sources = np.array(["mitocarta", "rickettsia", "pelagibacter",
                        "yeast-cytosol", "mitocarta"])
    frac = alpha_branch_mito_fraction(tree=tree, sources=sources, radius=2)
    assert frac == 1.0

    # Radius 1 → only mito 0 is within 1 hop of an alpha. → 0.5.
    frac_tight = alpha_branch_mito_fraction(tree=tree, sources=sources, radius=1)
    assert frac_tight == 0.5


def test_plot_endosymbiosis_tree_writes_png(tmp_path):
    from mixed_proteome_tmap import plot_endosymbiosis_tree

    rng = np.random.default_rng(0)
    n = 60
    layout = rng.standard_normal((n, 2)).astype(np.float32)
    domains = np.array(["Bacteria-alpha"] * 20 + ["Bacteria-cyano"] * 20 +
                       ["Eukarya-mito"] * 15 + ["Eukarya-cytosolic"] * 5)
    edges = [(i, i + 1) for i in range(n - 1)]
    out = tmp_path / "tree.png"
    plot_endosymbiosis_tree(layout=layout, domains=domains,
                            tree_edges=edges, out_path=out)
    assert out.exists() and out.stat().st_size > 5_000
