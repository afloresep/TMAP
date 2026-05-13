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


def reconstruct_tip_sequences(
    tree_root: dict, root_seq: str, *, gene: str = "HA1"
) -> dict[str, str]:
    """Walk the Auspice tree from root, applying inherited AA mutations.

    Mutation format is e.g. "T2L" (parent=T, position=2 (1-based), child=L).
    Returns a {tip_name: reconstructed_aa_seq} mapping for all tips.
    """
    out: dict[str, str] = {}

    def visit(node: dict, parent_seq: str) -> None:
        muts = (node.get("branch_attrs") or {}).get("mutations") or {}
        gene_muts = muts.get(gene) or []
        seq_chars = list(parent_seq)
        for mut in gene_muts:
            if len(mut) < 3:
                continue
            child_aa = mut[-1]
            try:
                pos1 = int(mut[1:-1])
            except ValueError:
                continue
            idx = pos1 - 1
            if 0 <= idx < len(seq_chars):
                seq_chars[idx] = child_aa
        node_seq = "".join(seq_chars)
        children = node.get("children") or []
        out[node.get("name", "")] = node_seq
        for c in children:
            visit(c, node_seq)

    visit(tree_root, root_seq)
    return out


def collect_tips(node: dict, out: list[dict] | None = None) -> list[dict]:
    """Recursively collect all leaf nodes from an Auspice tree."""
    if out is None:
        out = []
    if not node.get("children"):
        out.append(node)
    else:
        for c in node["children"]:
            collect_tips(c, out)
    return out


def load_auspice_h3n2(path: Path = AUSPICE_JSON) -> tuple[dict, str, list[StrainMeta]]:
    """Load the Auspice JSON; return (tree_root, root_HA1_aa, tip metadata)."""
    with path.open() as f:
        data = json.load(f)
    tree = data["tree"]
    # Reference HA1 AA sequence:
    root_seq = data["root_sequence"]["HA1"]
    tips = collect_tips(tree)
    meta: list[StrainMeta] = []
    for t in tips:
        attrs = t.get("node_attrs", {})
        nd = (attrs.get("num_date") or {}).get("value")
        clade = (attrs.get("clade_membership") or {}).get("value", "?")
        subclade = (attrs.get("subclade") or {}).get("value", "")
        meta.append(
            StrainMeta(
                strain=t.get("name", ""),
                date_str=str(nd) if nd is not None else "",
                num_date=float(nd) if nd is not None else float("nan"),
                clade=str(clade),
                subclade=str(subclade),
            )
        )
    return tree, root_seq, meta


def main() -> None:
    raise NotImplementedError("Filled in by later tasks.")


if __name__ == "__main__":
    main()
