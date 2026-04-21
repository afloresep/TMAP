"""Unit tests for endosymbiosis example helpers.

Uses synthetic data; no network calls, no ESM-2 invocation.
"""
from __future__ import annotations

import sys
from pathlib import Path

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
