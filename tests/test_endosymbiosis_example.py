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
