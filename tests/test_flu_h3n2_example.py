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
    tips = mod.reconstruct_tip_sequences(tree, root_seq, gene="HA1")
    assert tips["B"] == "MLVI"
    assert tips["A"] == "MLTI"


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
