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


def test_build_cognate_atlas_with_synthetic_cldf(tmp_path):
    mod = _load_module()
    # Minimal synthetic CLDF using the actual IE-CoR column names:
    (tmp_path / "languages.csv").write_text(
        "ID,Name,Glottocode,Family,Clade\n"
        "L1,Lang1,abcd1234,Indo-European,Germanic\n"
        "L2,Lang2,efgh5678,Indo-European,Germanic\n"
        "L3,Lang3,ijkl9012,Indo-European,Romance\n"
    )
    (tmp_path / "forms.csv").write_text(
        "ID,Language_ID,Parameter_ID,Form\n"
        "f1,L1,c_water,wasser\n"
        "f2,L2,c_water,water\n"
        "f3,L3,c_water,aqua\n"
        "f4,L1,c_fire,feuer\n"
        "f5,L2,c_fire,fire\n"
    )
    (tmp_path / "cognates.csv").write_text(
        "ID,Form_ID,Cognateset_ID\n"
        "cg1,f1,water_PIE_1\n"
        "cg2,f2,water_PIE_1\n"
        "cg3,f3,water_PIE_2\n"
        "cg4,f4,fire_PIE_1\n"
        "cg5,f5,fire_PIE_1\n"
    )
    atlas = mod.build_cognate_atlas(tmp_path)
    # L1 and L2 share two cognate classes; L3 shares neither
    assert set(atlas.names) == {"L1", "L2", "L3"}
    i1 = atlas.names.index("L1")
    i2 = atlas.names.index("L2")
    i3 = atlas.names.index("L3")
    assert set(atlas.cognate_sets[i1]) == {"c_water::water_PIE_1", "c_fire::fire_PIE_1"}
    assert set(atlas.cognate_sets[i2]) == {"c_water::water_PIE_1", "c_fire::fire_PIE_1"}
    assert set(atlas.cognate_sets[i3]) == {"c_water::water_PIE_2"}
    assert list(atlas.subgroups) == ["Germanic", "Germanic", "Romance"]
