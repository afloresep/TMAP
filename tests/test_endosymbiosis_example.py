"""Unit tests for endosymbiosis example helpers.

Uses synthetic data; no network calls, no ESM-2 invocation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))

from endosymbiosis_mito_tmap import ProteinRecord  # noqa: E402


def test_protein_record_minimal_construction():
    r = ProteinRecord(
        accession="P00001", organism="Homo sapiens", source="mitocarta",
        domain="Eukarya-mito", compartment="matrix",
    )
    assert r.accession == "P00001"
    assert r.source == "mitocarta"


def test_fetch_uniprot_proteome_uses_cache_file(tmp_path, monkeypatch):
    from endosymbiosis_mito_tmap import fetch_uniprot_proteome

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
    from endosymbiosis_mito_tmap import _parse_uniprot_accession
    assert _parse_uniprot_accession("sp|P12345|TEST_HUMAN") == "P12345"
    assert _parse_uniprot_accession("tr|Q9NZC2|NAME_MOUSE") == "Q9NZC2"
    assert _parse_uniprot_accession("plain_id") == "plain_id"


def test_load_mitocarta_returns_accessions_and_compartments(tmp_path):
    import openpyxl
    from endosymbiosis_mito_tmap import load_mitocarta

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
    from endosymbiosis_mito_tmap import filter_cytosolic

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
    monkeypatch.setattr("endosymbiosis_mito_tmap.fetch_uniprot", fake_fetch)

    kept = filter_cytosolic(list(annotations.keys()))
    assert set(kept) == {"P00001", "P00004"}


def test_build_endosymbiosis_dataset_assembles_all_sources(tmp_path, monkeypatch):
    from endosymbiosis_mito_tmap import ProteinRecord, build_endosymbiosis_dataset

    def fake_proteome(pid, *, cache_dir, reviewed_only=False):
        return ([f"{pid}_A", f"{pid}_B"], ["MAAA", "MBBB"])
    def fake_mitocarta(_path):
        return [ProteinRecord(
            accession="P00395", organism="Homo sapiens", source="mitocarta",
            domain="Eukarya-mito", compartment="MIM_matrix",
        )]
    def fake_cytosolic_filter(ids, **_):
        return [ids[0]]  # keep the first yeast ID only

    monkeypatch.setattr("endosymbiosis_mito_tmap.fetch_uniprot_proteome", fake_proteome)
    monkeypatch.setattr("endosymbiosis_mito_tmap.load_mitocarta", fake_mitocarta)
    monkeypatch.setattr("endosymbiosis_mito_tmap.filter_cytosolic", fake_cytosolic_filter)
    # Fake MitoCarta sequence fetch: return a dict with `sequence` per accession.
    def fake_fetch(ids, **_):
        return {
            "accession": np.array(ids, dtype=object),
            "sequence": np.array(["MCCC"] * len(ids), dtype=object),
        }
    monkeypatch.setattr("endosymbiosis_mito_tmap.fetch_uniprot", fake_fetch)

    records, sequences = build_endosymbiosis_dataset(cache_dir=tmp_path)

    sources = {r.source for r in records}
    assert sources == {"rickettsia", "pelagibacter", "synechocystis",
                       "yeast-cytosol", "mitocarta"}
    assert len(records) == len(sequences)
    assert all(isinstance(s, str) and len(s) >= 3 for s in sequences)
