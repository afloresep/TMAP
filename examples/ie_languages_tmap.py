"""Indo-European languages TMAP on the IE-CoR cognate dataset.

Downloads the IE-CoR CLDF bundle (Heggarty et al. 2023, Science), parses
cognate-class membership per concept per language into Jaccard-shingled
feature sets, and fits TMAP to recover the IE language family structure.

Outputs:
    paper/images/ie_languages_tmap.png
    examples/ie_languages_report.txt
"""
from __future__ import annotations

import argparse  # noqa: F401
import urllib.request  # noqa: F401
import zipfile  # noqa: F401
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
from numpy.typing import NDArray

HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "ie_languages"
CLDF_DIR = DATA_DIR / "iecor_cldf"
ZIP_PATH = DATA_DIR / "iecor.zip"
IMG_PATH = HERE.parent / "paper" / "images" / "ie_languages_tmap.png"
REPORT_PATH = HERE / "ie_languages_report.txt"

IECOR_URLS = (
    "https://zenodo.org/records/10026029/files/lexibank-iecor-v1.0.zip",
    "https://zenodo.org/records/8089360/files/iecor.zip",
    "https://github.com/lexibank/iecor/archive/refs/heads/main.zip",
)


@dataclass(slots=True)
class LangMeta:
    glottocode: str
    name: str
    family: str         # e.g. "Indo-European"
    subgroup: str       # e.g. "Germanic", "Romance"


@dataclass(slots=True)
class CognateAtlas:
    names: list[str]
    cognate_sets: list[list[str]]  # tokens "concept::cogid" per language
    families: NDArray[np.str_]
    subgroups: NDArray[np.str_]


def main() -> None:
    raise NotImplementedError("Filled in by later tasks.")


if __name__ == "__main__":
    main()
