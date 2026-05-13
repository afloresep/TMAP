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
