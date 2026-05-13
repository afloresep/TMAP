"""Tests for the flu H3N2 example (parsing + feature extraction)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest  # noqa: F401

HERE = Path(__file__).parent.parent / "examples"


def _load_module():
    import sys

    spec = importlib.util.spec_from_file_location(
        "flu_h3n2_ha_tmap", HERE / "flu_h3n2_ha_tmap.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["flu_h3n2_ha_tmap"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_stub_importable():
    mod = _load_module()
    assert hasattr(mod, "Atlas")
    assert hasattr(mod, "StrainMeta")


def test_reconstruct_tip_sequence_from_muts():
    mod = _load_module()
    root_seq = "MKTI"  # 1-based positions
    # Tree: root -> A (mut "T2L") -> B (mut "L3V") = "MLVI"
    tree = {
        "name": "ROOT",
        "branch_attrs": {"mutations": {"HA1": []}},
        "children": [
            {
                "name": "A",
                "branch_attrs": {"mutations": {"HA1": ["T2L"]}},
                "children": [
                    {
                        "name": "B",
                        "branch_attrs": {"mutations": {"HA1": ["L3V"]}},
                    }
                ],
            }
        ],
    }
    seq_map = mod.reconstruct_tip_sequences(tree, root_seq, gene="HA1")
    assert seq_map["B"] == "MLVI"
    assert seq_map["A"] == "MLTI"


def test_collect_tips():
    mod = _load_module()
    tree = {
        "name": "ROOT",
        "children": [
            {"name": "A"},
            {"name": "B", "children": [{"name": "C"}, {"name": "D"}]},
        ],
    }
    names = [t["name"] for t in mod.collect_tips(tree)]
    assert sorted(names) == ["A", "C", "D"]


def test_kmers_aa_basic():
    mod = _load_module()
    out = mod.kmers_aa("ACDEACDE", k=3)
    # All 3-mers from "ACDEACDE" containing only AA alphabet
    expected = sorted({"ACD", "CDE", "DEA", "EAC"})
    assert out == expected


def test_build_atlas_shapes():
    mod = _load_module()
    metas = [
        mod.StrainMeta(strain="t1", date_str="2020.1", num_date=2020.1, clade="A"),
        mod.StrainMeta(strain="t2", date_str="2021.2", num_date=2021.2, clade="B"),
    ]
    seqs = {"t1": "ACDEACDE", "t2": "ACDEACDF"}
    atlas = mod.build_atlas(metas, seqs, k=3)
    assert atlas.names == ["t1", "t2"]
    assert len(atlas.kmers) == 2
    assert atlas.num_dates.shape == (2,)
    assert list(atlas.clades) == ["A", "B"]
