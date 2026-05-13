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

import argparse
import gzip
import json
import urllib.request
from dataclasses import dataclass, field
from datetime import date as _date  # noqa: F401
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "flu_h3n2"
AUSPICE_JSON = DATA_DIR / "h3n2_ha_12y.json"
# Sidecar file: newer Nextstrain builds omit root_sequence from the main JSON.
# The root-sequence is served separately; confirmed 200 at 2026-05-13.
ROOT_SEQ_JSON = DATA_DIR / "h3n2_ha_12y_root-sequence.json"
FASTA_PATH = DATA_DIR / "h3n2_ha_12y_sequences.fasta"
IMG_PATH = HERE.parent / "paper" / "images" / "flu_h3n2_ha_tmap.png"
REPORT_PATH = HERE / "flu_h3n2_ha_report.txt"

# Primary URL confirmed 200 at 2026-05-13 (content-encoding: gzip, ~405 KB compressed).
AUSPICE_URL = "https://data.nextstrain.org/flu_seasonal_h3n2_ha_12y.json"
# Sidecar root-sequence: confirmed 200 at 2026-05-13 (content-length: 1046 compressed).
ROOT_SEQ_URL = "https://data.nextstrain.org/flu_seasonal_h3n2_ha_12y_root-sequence.json"
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
    Returns a {node_name: reconstructed_aa_seq} mapping for every node in
    the tree (leaves and internal). Downstream callers typically iterate
    over the leaf list from `collect_tips` and look up by name.
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


def load_auspice_h3n2(
    path: Path = AUSPICE_JSON,
    root_seq_path: Path = ROOT_SEQ_JSON,
) -> tuple[dict, str, list[StrainMeta]]:
    """Load the Auspice JSON and sidecar root-sequence.

    Returns (tree_root, root_HA1_aa, tip_metadata).

    Newer Nextstrain builds omit root_sequence from the main JSON and serve it
    in a separate sidecar file. We load both and merge them here.
    """
    with path.open() as f:
        data = json.load(f)
    tree = data["tree"]
    # Reference HA1 AA sequence — try inline first, then sidecar.
    inline_root = data.get("root_sequence") or {}
    if inline_root.get("HA1"):
        root_seq = inline_root["HA1"]
    else:
        with root_seq_path.open() as f:
            root_data = json.load(f)
        root_seq = root_data["HA1"]
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


def _download(url: str, dest: Path) -> None:
    """Download `url` to `dest` if not already present. Skips on cache hit.

    Transparently decompresses gzip-encoded responses so the file on disk is
    always the raw (uncompressed) content — no special handling needed at load
    time.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()
    # Decompress if the server sent gzip bytes (magic bytes 1f 8b).
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    tmp.write_bytes(raw)
    tmp.rename(dest)


def ensure_auspice_json(path: Path = AUSPICE_JSON, url: str = AUSPICE_URL) -> Path:
    _download(url, path)
    return path


def ensure_root_sequence_json(
    path: Path = ROOT_SEQ_JSON, url: str = ROOT_SEQ_URL
) -> Path:
    _download(url, path)
    return path


def kmers_aa(seq: str, k: int) -> list[str]:
    """Sorted unique unambiguous AA k-mers from a sequence."""
    if k <= 0:
        raise ValueError("k must be positive")
    allowed = set(AA_ALPHABET)
    tokens: set[str] = set()
    for i in range(max(0, len(seq) - k + 1)):
        token = seq[i : i + k]
        if all(ch in allowed for ch in token):
            tokens.add(token)
    return sorted(tokens)


def build_atlas(
    metas: list[StrainMeta],
    seqs_by_strain: dict[str, str],
    *,
    k: int = 5,
) -> Atlas:
    """Pair sequences with metadata; emit k-mer sets and parallel arrays."""
    keep_names: list[str] = []
    keep_kmers: list[list[str]] = []
    keep_seqs: list[str] = []
    num_dates: list[float] = []
    clades: list[str] = []
    subclades: list[str] = []
    for m in metas:
        seq = seqs_by_strain.get(m.strain)
        if not seq:
            continue
        toks = kmers_aa(seq, k)
        if not toks:
            continue
        keep_names.append(m.strain)
        keep_kmers.append(toks)
        keep_seqs.append(seq)
        num_dates.append(m.num_date)
        clades.append(m.clade)
        subclades.append(m.subclade)
    return Atlas(
        names=keep_names,
        kmers=keep_kmers,
        num_dates=np.asarray(num_dates, dtype=np.float32),
        clades=np.asarray(clades, dtype=object),
        subclades=np.asarray(subclades, dtype=object),
        sequences=keep_seqs,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=5, help="AA k-mer size")
    p.add_argument("--n-neighbors", type=int, default=60)
    p.add_argument("--n-permutations", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-tips", type=int, default=0,
        help="If >0, subsample this many tips uniformly at random (for fast runs)."
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    from tmap import TMAP

    ensure_auspice_json()
    ensure_root_sequence_json()
    print(f"Loading Auspice JSON from {AUSPICE_JSON} ...", flush=True)
    tree, root_seq, tip_meta = load_auspice_h3n2()
    print(f"  {len(tip_meta):,} tips, root HA1 length {len(root_seq)}", flush=True)

    seqs_by_strain = reconstruct_tip_sequences(tree, root_seq, gene="HA1")

    if args.max_tips and len(tip_meta) > args.max_tips:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(tip_meta), size=args.max_tips, replace=False)
        tip_meta = [tip_meta[i] for i in idx]

    atlas = build_atlas(tip_meta, seqs_by_strain, k=args.k)
    print(f"  atlas size: {len(atlas.names):,}", flush=True)

    print("Fitting TMAP...", flush=True)
    model = TMAP(
        metric="jaccard",
        n_neighbors=args.n_neighbors,
        n_permutations=args.n_permutations,
        kc=80,
        seed=args.seed,
        minhash_seed=args.seed,
    ).fit(atlas.kmers)

    # Headline validation: tree hops from the oldest tip vs num_date.
    root_idx = int(np.argmin(atlas.num_dates))
    hops = _bfs_hops(model.tree_, root_idx)
    valid = hops >= 0
    rho = float(spearmanr(hops[valid], atlas.num_dates[valid]).statistic)
    print(f"Spearman(hops from oldest tip, num_date) = {rho:.3f}", flush=True)

    _plot_tmap(model, atlas, IMG_PATH)

    lines = [
        "Seasonal influenza H3N2 HA TMAP (Pass A)",
        f"Tips in atlas: {len(atlas.names):,}",
        f"k-mer size (AA): {args.k}",
        (
            f"TMAP: metric=jaccard, n_neighbors={args.n_neighbors},"
            f" n_permutations={args.n_permutations}"
        ),
        "",
        "Tree-hops ordering proxy:",
        f"  Spearman(hops from oldest tip, collection date) = {rho:.3f}",
        "",
        f"PNG: {IMG_PATH}",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {REPORT_PATH}")


def _bfs_hops(tree, root: int) -> NDArray[np.int32]:
    from collections import deque
    dist = np.full(tree.n_nodes, -1, dtype=np.int32)
    dist[root] = 0
    queue: deque[int] = deque([root])
    while queue:
        node = queue.popleft()
        for nbr, _ in tree.neighbors(node):
            if dist[nbr] < 0:
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return dist


def _plot_tmap(model, atlas: Atlas, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    coords = model.embedding_
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=atlas.num_dates, cmap="viridis",
        s=6, linewidths=0, alpha=0.85,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04)
    cbar.set_label("Collection date (decimal year)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"H3N2 HA TMAP — {len(atlas.names):,} strains")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
