"""Tests for the Indo-European languages example."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).parent.parent / "examples"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "ie_languages_tmap", HERE / "ie_languages_tmap.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ie_languages_tmap"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_stub_importable():
    mod = _load_module()
    assert hasattr(mod, "CognateAtlas")
    assert hasattr(mod, "LangMeta")
