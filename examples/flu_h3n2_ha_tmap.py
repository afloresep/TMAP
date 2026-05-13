"""Seasonal influenza H3N2 HA TMAP on the Nextstrain 12y build.

Downloads the Nextstrain Auspice JSON for the seasonal H3N2 HA tree
(~12-year window) and the aligned HA sequence FASTA, fits TMAP on
amino-acid k-mer Jaccard, and audits the tree against collection date
and Nextstrain clade labels.

Outputs:
    paper/images/flu_h3n2_ha_tmap.png
    examples/flu_h3n2_ha_report.txt
"""
from __future__ import annotations

import argparse  # noqa: F401
import json  # noqa: F401
import urllib.request  # noqa: F401
from dataclasses import dataclass, field
from datetime import date as _date  # noqa: F401
from pathlib import Path

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr  # noqa: F401

HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "flu_h3n2"
AUSPICE_JSON = DATA_DIR / "h3n2_ha_12y.json"
FASTA_PATH = DATA_DIR / "h3n2_ha_12y_sequences.fasta"
IMG_PATH = HERE.parent / "paper" / "images" / "flu_h3n2_ha_tmap.png"
REPORT_PATH = HERE / "flu_h3n2_ha_report.txt"

AUSPICE_URL = "https://data.nextstrain.org/flu_seasonal_h3n2_ha_12y.json"
# Fallback (current canonical path; verify with `curl -sI`):
FASTA_URL = "https://data.nextstrain.org/files/workflows/seasonal-flu/h3n2/ha/12y/sequences.fasta.xz"

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(slots=True)
class StrainMeta:
    strain: str
    date_str: str
    num_date: float
    clade: str
    subclade: str = ""


@dataclass(slots=True)
class Atlas:
    names: list[str]
    kmers: list[list[str]]
    num_dates: NDArray[np.float32]
    clades: NDArray[np.str_]
    subclades: NDArray[np.str_]
    sequences: list[str] = field(default_factory=list)


def main() -> None:
    raise NotImplementedError("Filled in by later tasks.")


if __name__ == "__main__":
    main()
