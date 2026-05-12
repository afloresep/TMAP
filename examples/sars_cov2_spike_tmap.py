"""Pass A: SARS-CoV-2 spike TMAP on Nextstrain Open subsamples.

Loads aligned full-genome sequences and metadata from the Nextstrain Open
build, subsamples evenly across collection time and Nextstrain clade,
extracts the spike region, fits TMAP with amino-acid k-mer Jaccard, and
audits the result against the published clade labels and collection dates.

Inputs:
    examples/data/sars_cov2_spike/metadata.tsv.zst   (Nextstrain Open)
    examples/data/sars_cov2_spike/aligned.fasta.zst  (Nextstrain Open)

Outputs:
    paper/images/sars_cov2_spike_tmap_<n>.png
    paper/images/sars_cov2_spike_tmap_<n>.html
    examples/sars_cov2_spike_report_<n>.txt
    examples/sars_cov2_spike_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import io
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zstandard as zstd
from matplotlib.collections import LineCollection
from numpy.typing import NDArray
from scipy.stats import spearmanr

HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "sars_cov2_spike"
META_ZST = DATA_DIR / "metadata.tsv.zst"
FASTA_ZST = DATA_DIR / "aligned.fasta.zst"
IMG_DIR = HERE.parent / "paper" / "images"
SUMMARY_CSV = HERE / "sars_cov2_spike_summary.csv"

# NC_045512.2 spike CDS: 21563..25384 (1-based inclusive).
SPIKE_START_0 = 21562
SPIKE_END_0 = 25384
SPIKE_LEN_NT = SPIKE_END_0 - SPIKE_START_0
WUHAN_STRAIN_CANDIDATES = ("Wuhan-Hu-1/2019", "Wuhan/Hu-1/2019", "Wuhan-Hu-1", "MN908947")

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# Coarse WHO-named variant groups for SARS-CoV-2 (vs the 59 Nextstrain clades).
# Maps the leading Nextstrain clade code (e.g. "20I") to a major variant name.
VARIANT_GROUPS: dict[str, str] = {
    "19A": "Pre-VOC", "19B": "Pre-VOC",
    "20A": "Pre-VOC", "20B": "Pre-VOC", "20C": "Pre-VOC", "20D": "Pre-VOC",
    "20E": "Pre-VOC", "20F": "Pre-VOC", "20G": "Pre-VOC",
    "20I": "Alpha", "20H": "Beta", "20J": "Gamma",
    "21A": "Delta", "21I": "Delta", "21J": "Delta",
    "21B": "Kappa", "21C": "Epsilon", "21D": "Eta",
    "21E": "Theta", "21F": "Iota", "21G": "Lambda", "21H": "Mu",
    "21K": "Omicron BA.1", "21L": "Omicron BA.2",
    "22A": "Omicron BA.4/5", "22B": "Omicron BA.4/5",
    "22C": "Omicron BA.2.12", "22D": "Omicron BA.2.75",
    "22E": "Omicron BQ.1", "22F": "Omicron XBB",
}
VARIANT_FALLBACK_22 = "Omicron XBB-era"
VARIANT_FALLBACK_23 = "Omicron XBB-era"
VARIANT_FALLBACK_24_PLUS = "Omicron JN.1-era"
VARIANT_COLORS: dict[str, str] = {
    "Pre-VOC": "#7f7f7f",
    "Alpha": "#1f77b4",
    "Beta": "#ff7f0e",
    "Gamma": "#2ca02c",
    "Delta": "#d62728",
    "Kappa": "#9467bd",
    "Epsilon": "#8c564b",
    "Eta": "#e377c2",
    "Theta": "#bcbd22",
    "Iota": "#17becf",
    "Lambda": "#aec7e8",
    "Mu": "#ffbb78",
    "Omicron BA.1": "#98df8a",
    "Omicron BA.2": "#ff9896",
    "Omicron BA.4/5": "#c5b0d5",
    "Omicron BA.2.12": "#c49c94",
    "Omicron BA.2.75": "#f7b6d2",
    "Omicron BQ.1": "#dbdb8d",
    "Omicron XBB": "#9edae5",
    "Omicron XBB-era": "#393b79",
    "Omicron JN.1-era": "#843c39",
    "Other": "#000000",
}


def variant_group(clade: str) -> str:
    """Map a Nextstrain clade label (e.g. '21J (Delta)') to a major variant name."""
    if not clade or clade == "?":
        return "Other"
    code = clade.split()[0]
    if code in VARIANT_GROUPS:
        return VARIANT_GROUPS[code]
    year = code[:2] if len(code) >= 2 else ""
    if year == "22" or year == "23":
        return VARIANT_FALLBACK_23
    if year in ("24", "25", "26"):
        return VARIANT_FALLBACK_24_PLUS
    return "Other"


@dataclass(slots=True)
class StrainMeta:
    strain: str
    date_str: str
    num_date: float
    clade: str
    pango: str
    country: str = ""


@dataclass(slots=True)
class Atlas:
    names: list[str]
    kmers: list[list[str]]
    num_dates: NDArray[np.float32]
    clades: NDArray[np.str_]
    pangos: NDArray[np.str_]
    is_reference: NDArray[np.bool_]
    spike_aa: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GraphAudit:
    edge_date_delta_mean: float
    random_date_delta_mean: float
    date_delta_p_le: float
    edge_clade_boundary_fraction: float
    random_clade_boundary_fraction: float
    clade_boundary_p_le: float


def num_date_from_iso(date_str: str) -> float | None:
    """Convert ISO date (YYYY-MM-DD or YYYY-MM or YYYY) to decimal year."""
    parts = date_str.strip().split("-")
    if not parts or not parts[0].isdigit():
        return None
    try:
        year = int(parts[0])
        if year < 2019 or year > 2027:
            return None
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
            d = _date(year, int(parts[1]), int(parts[2]))
        elif len(parts) >= 2 and parts[1].isdigit():
            d = _date(year, int(parts[1]), 15)
        else:
            d = _date(year, 7, 1)
    except ValueError:
        return None
    year_start = _date(d.year, 1, 1).toordinal()
    next_year_start = _date(d.year + 1, 1, 1).toordinal()
    return year + (d.toordinal() - year_start) / (next_year_start - year_start)


def open_zst_text(path: Path) -> io.TextIOWrapper:
    """Open a .zst file as a streamed text reader."""
    fh = path.open("rb")
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(fh)
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def load_metadata(path: Path = META_ZST) -> list[StrainMeta]:
    """Stream-decode metadata.tsv.zst and keep usable rows."""
    out: list[StrainMeta] = []
    with open_zst_text(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            strain = row.get("strain", "").strip()
            if not strain or strain == "?":
                continue
            clade = (row.get("Nextstrain_clade") or row.get("clade_membership") or "").strip()
            pango = (row.get("pango_lineage") or row.get("Nextclade_pango") or "").strip()
            if not clade or clade == "?" or not pango or pango == "?":
                continue
            date_str = (row.get("date") or "").strip()
            num_date = num_date_from_iso(date_str)
            if num_date is None:
                continue
            qc = (row.get("QC_overall_status") or "").lower()
            if qc and qc not in ("good", "mediocre"):
                continue
            cov_str = (row.get("coverage") or "").strip()
            try:
                coverage = float(cov_str) if cov_str and cov_str != "?" else 1.0
            except ValueError:
                coverage = 0.0
            if coverage < 0.95:
                continue
            country = (row.get("country") or "").strip()
            out.append(
                StrainMeta(
                    strain=strain,
                    date_str=date_str,
                    num_date=float(num_date),
                    clade=clade,
                    pango=pango,
                    country=country,
                )
            )
    return out


def stratified_subsample(
    meta: list[StrainMeta],
    n_target: int,
    *,
    seed: int,
) -> list[StrainMeta]:
    """Stratified subsample across (year-month, Nextstrain clade) bins."""
    rng = np.random.default_rng(seed)
    bins: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        year = int(m.num_date)
        month = int(round((m.num_date - year) * 12)) + 1
        month = min(max(month, 1), 12)
        bins[(year, month, m.clade)].append(i)

    keys = sorted(bins)
    quota_each = max(1, n_target // max(1, len(keys)))
    selected: list[int] = []
    leftovers: list[int] = []
    for key in keys:
        idxs = bins[key]
        rng.shuffle(idxs)
        take = min(quota_each, len(idxs))
        selected.extend(idxs[:take])
        leftovers.extend(idxs[take:])
    if len(selected) < n_target:
        rng.shuffle(leftovers)
        need = n_target - len(selected)
        selected.extend(leftovers[:need])
    elif len(selected) > n_target:
        rng.shuffle(selected)
        selected = selected[:n_target]
    return [meta[i] for i in selected]


def stream_aligned_fasta(
    path: Path,
    wanted: set[str],
) -> dict[str, str]:
    """Stream-decode aligned.fasta.zst and yield only the wanted strains."""
    found: dict[str, str] = {}
    header: str | None = None
    chunks: list[str] = []
    with open_zst_text(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith(">"):
                if header is not None and header in wanted:
                    found[header] = "".join(chunks)
                header = line[1:].split()[0] if len(line) > 1 else None
                chunks = []
                if wanted and len(found) >= len(wanted):
                    break
            elif header is not None:
                chunks.append(line)
        if header is not None and header in wanted:
            found[header] = "".join(chunks)
    return found


def extract_spike_aa(aligned_seq: str) -> str:
    """Cut spike region from a genome aligned to NC_045512.2 and translate."""
    nt = aligned_seq[SPIKE_START_0:SPIKE_END_0].upper()
    if len(nt) < SPIKE_LEN_NT:
        nt = nt + "N" * (SPIKE_LEN_NT - len(nt))
    aas: list[str] = []
    for i in range(0, SPIKE_LEN_NT - 2, 3):
        codon = nt[i : i + 3].replace("-", "N")
        if "N" in codon or any(c not in "ACGT" for c in codon):
            aas.append("X")
        else:
            aas.append(CODON_TABLE.get(codon, "X"))
    return "".join(aas).rstrip("*").replace("*", "X")


def kmers_aa(seq: str, k: int) -> list[str]:
    """Unique unambiguous AA k-mers from a sequence."""
    if k <= 0:
        raise ValueError("k must be positive")
    allowed = set(AA_ALPHABET)
    tokens: set[str] = set()
    for i in range(max(0, len(seq) - k + 1)):
        token = seq[i : i + k]
        if all(ch in allowed for ch in token):
            tokens.add(token)
    return sorted(tokens)


def aa_hamming(a: str, b: str) -> int:
    """Hamming distance on aligned AA sequences, ignoring X."""
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i] and a[i] != "X" and b[i] != "X")


def build_atlas(
    selected: list[StrainMeta],
    spike_by_strain: dict[str, str],
    wuhan_spike_aa: str,
    *,
    k: int,
) -> Atlas:
    names = ["Wuhan-Hu-1"]
    spike_aa: list[str] = [wuhan_spike_aa]
    num_dates = [num_date_from_iso("2019-12-30") or 2019.99]
    clades = ["19A"]
    pangos = ["B"]
    is_ref = [True]
    for m in selected:
        seq = spike_by_strain.get(m.strain)
        if seq is None:
            continue
        names.append(m.strain)
        spike_aa.append(seq)
        num_dates.append(m.num_date)
        clades.append(m.clade)
        pangos.append(m.pango)
        is_ref.append(False)
    return Atlas(
        names=names,
        kmers=[kmers_aa(s, k) for s in spike_aa],
        num_dates=np.asarray(num_dates, dtype=np.float32),
        clades=np.asarray(clades, dtype=object),
        pangos=np.asarray(pangos, dtype=object),
        is_reference=np.asarray(is_ref, dtype=bool),
        spike_aa=spike_aa,
    )


def rooted_depth(tree, root: int) -> NDArray[np.float32]:
    depth = np.full(tree.n_nodes, np.nan, dtype=np.float32)
    for node, _, d in tree.bfs(root):
        depth[node] = float(d)
    return depth


def count_components(tree) -> tuple[int, list[int]]:
    visited = np.zeros(tree.n_nodes, dtype=bool)
    sizes: list[int] = []
    for start in range(tree.n_nodes):
        if visited[start]:
            continue
        size = 0
        for node, _, _ in tree.bfs(start):
            visited[node] = True
            size += 1
        sizes.append(size)
    sizes.sort(reverse=True)
    return len(sizes), sizes


def graph_audit(tree, atlas: Atlas, *, seed: int, n_trials: int = 200) -> GraphAudit:
    rng = np.random.default_rng(seed)
    edges = tree.edges
    nd = atlas.num_dates
    edge_dd = np.abs(nd[edges[:, 0]] - nd[edges[:, 1]])

    n_nodes = tree.n_nodes
    rand_dd = np.empty(n_trials, dtype=np.float32)
    for i in range(n_trials):
        pairs = rng.integers(0, n_nodes, size=(len(edges), 2))
        rand_dd[i] = float(np.mean(np.abs(nd[pairs[:, 0]] - nd[pairs[:, 1]])))

    cl = atlas.clades
    edge_clade_cross = np.mean(cl[edges[:, 0]] != cl[edges[:, 1]])
    rand_clade = np.empty(n_trials, dtype=np.float32)
    for i in range(n_trials):
        pairs = rng.integers(0, n_nodes, size=(len(edges), 2))
        rand_clade[i] = float(np.mean(cl[pairs[:, 0]] != cl[pairs[:, 1]]))

    return GraphAudit(
        edge_date_delta_mean=float(np.mean(edge_dd)),
        random_date_delta_mean=float(np.mean(rand_dd)),
        date_delta_p_le=float(np.mean(rand_dd <= np.mean(edge_dd))),
        edge_clade_boundary_fraction=float(edge_clade_cross),
        random_clade_boundary_fraction=float(np.mean(rand_clade)),
        clade_boundary_p_le=float(np.mean(rand_clade <= edge_clade_cross)),
    )


def descendants_of(tree, child_node: int, parent_node: int) -> NDArray[np.int32]:
    """Nodes reachable from `child_node` without crossing back through `parent_node`."""
    visited = {int(parent_node), int(child_node)}
    out = [int(child_node)]
    queue = [int(child_node)]
    while queue:
        node = queue.pop(0)
        for nbr, _ in tree.neighbors(node):
            if nbr not in visited:
                visited.add(nbr)
                out.append(int(nbr))
                queue.append(int(nbr))
    return np.asarray(out, dtype=np.int32)


def select_distinct_clade_subtrees(
    purity_rows: list[tuple[int, int, int, str, float]],
    *,
    n_clades: int,
) -> list[tuple[int, int, int, str, float]]:
    """Keep the largest pure subtree per dominant clade, up to n_clades total."""
    seen: set[str] = set()
    out: list[tuple[int, int, int, str, float]] = []
    for row in purity_rows:
        clade = row[3]
        if clade in seen:
            continue
        seen.add(clade)
        out.append(row)
        if len(out) >= n_clades:
            break
    return out


def branch_subtree_purity(
    tree,
    labels: NDArray,
    root: int,
    *,
    min_size: int = 50,
    top_k: int = 12,
) -> list[tuple[int, int, int, str, float]]:
    """For each child subtree under a branch point, report dominant-label purity."""
    parent = np.full(tree.n_nodes, -1, dtype=np.int32)
    order: list[int] = []
    for node, p, _ in tree.bfs(root):
        parent[node] = -1 if p is None else int(p)
        order.append(int(node))

    degrees = np.zeros(tree.n_nodes, dtype=np.int32)
    for src, tgt in tree.edges:
        degrees[int(src)] += 1
        degrees[int(tgt)] += 1
    branch_set = {int(i) for i in np.flatnonzero(degrees >= 3)}

    unique, codes = np.unique(labels, return_inverse=True)
    n_lab = len(unique)
    counts = np.zeros((tree.n_nodes, n_lab), dtype=np.int32)
    sizes = np.ones(tree.n_nodes, dtype=np.int32)
    counts[np.arange(tree.n_nodes), codes] = 1
    for node in order[::-1]:
        p = parent[node]
        if p >= 0:
            counts[p] += counts[node]
            sizes[p] += sizes[node]

    rows: list[tuple[int, int, int, str, float]] = []
    for child in range(tree.n_nodes):
        p = int(parent[child])
        if p in branch_set and sizes[child] >= min_size:
            dom = int(np.argmax(counts[child]))
            rows.append(
                (p, child, int(sizes[child]), str(unique[dom]),
                 float(counts[child][dom] / sizes[child]))
            )
    rows.sort(key=lambda r: (r[4], r[2]), reverse=True)
    return rows[:top_k]


def assign_clade_colors(clades: NDArray) -> dict[str, str]:
    """Stable color map across the most common Nextstrain clades."""
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    ]
    counts = Counter(str(c) for c in clades)
    ordered = [c for c, _ in counts.most_common()]
    out: dict[str, str] = {}
    for i, c in enumerate(ordered):
        out[c] = palette[i % len(palette)]
    return out


def plot_panels(
    *,
    model,
    atlas: Atlas,
    depth: NDArray,
    rho_date: float,
    out_path: Path,
) -> None:
    layout = model.embedding_
    tree = model.tree_
    edges = tree.edges
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=160)
    segments = layout[edges]

    variants = np.asarray([variant_group(str(c)) for c in atlas.clades])
    ordered_variants = [
        v for v in (
            "Pre-VOC", "Alpha", "Beta", "Gamma", "Delta",
            "Kappa", "Epsilon", "Eta", "Theta", "Iota", "Lambda", "Mu",
            "Omicron BA.1", "Omicron BA.2", "Omicron BA.4/5",
            "Omicron BA.2.12", "Omicron BA.2.75", "Omicron BQ.1",
            "Omicron XBB", "Omicron XBB-era", "Omicron JN.1-era", "Other",
        ) if v in set(variants.tolist())
    ]

    ax = axes[0, 0]
    ax.add_collection(LineCollection(segments, colors="#bbbbbb", linewidths=0.2, alpha=0.5))
    for label in ordered_variants:
        mask = variants == label
        ax.scatter(
            layout[mask, 0], layout[mask, 1],
            s=4, c=VARIANT_COLORS.get(label, "#888888"),
            linewidths=0, alpha=0.9, label=label,
        )
    ax.set_title("TMAP, colored by major variant")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)

    ax = axes[0, 1]
    ax.add_collection(LineCollection(segments, colors="#bbbbbb", linewidths=0.2, alpha=0.5))
    points = ax.scatter(
        layout[:, 0], layout[:, 1], c=atlas.num_dates,
        s=4, cmap="turbo", linewidths=0,
    )
    fig.colorbar(points, ax=ax, fraction=0.04, pad=0.01, label="num_date")
    ax.set_title("TMAP, colored by collection date")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1, 0]
    ax.add_collection(LineCollection(segments, colors="#bbbbbb", linewidths=0.2, alpha=0.5))
    finite = np.isfinite(depth)
    points = ax.scatter(
        layout[finite, 0], layout[finite, 1], c=depth[finite],
        s=4, cmap="turbo", linewidths=0,
    )
    ax.scatter(layout[~finite, 0], layout[~finite, 1], c="#d0d0d0", s=3, linewidths=0)
    ax.scatter(layout[0, 0], layout[0, 1], c="red", s=40, marker="*")
    fig.colorbar(points, ax=ax, fraction=0.04, pad=0.01, label="tree hops from Wuhan-Hu-1")
    ax.set_title("TMAP, tree depth from Wuhan-Hu-1")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1, 1]
    non_ref = ~atlas.is_reference
    valid = non_ref & np.isfinite(depth)
    ax.scatter(depth[valid], atlas.num_dates[valid], s=8, c="#333", alpha=0.6)
    ax.set_title(f"Tree hops vs collection date (Spearman={rho_date:.3f})")
    ax.set_xlabel("TMAP tree hops from Wuhan-Hu-1")
    ax.set_ylabel("Collection num_date")

    fig.suptitle(f"SARS-CoV-2 spike TMAP, n={atlas.is_reference.size - 1:,}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pure_subtrees(
    *,
    model,
    atlas: Atlas,
    purity_rows: list[tuple[int, int, int, str, float]],
    out_path: Path,
    n_clades: int = 8,
) -> list[tuple[int, int, int, str, float]]:
    """Highlight the largest pure subtree per clade on top of a grey background."""
    layout = model.embedding_
    tree = model.tree_
    edges = tree.edges
    segments = layout[edges]

    selected = select_distinct_clade_subtrees(purity_rows, n_clades=n_clades)
    color_map = assign_clade_colors(atlas.clades)

    fig, ax = plt.subplots(figsize=(11, 9), dpi=160)
    ax.add_collection(
        LineCollection(segments, colors="#e0e0e0", linewidths=0.2, alpha=0.6)
    )
    ax.scatter(
        layout[:, 0], layout[:, 1],
        s=2, c="#cccccc", linewidths=0, alpha=0.55,
    )

    legend_handles = []
    for branch, child, size, clade, purity in selected:
        nodes = descendants_of(tree, child, branch)
        color = color_map.get(clade, "#000000")
        ax.scatter(
            layout[nodes, 0], layout[nodes, 1],
            s=10, c=color, linewidths=0, alpha=0.95,
        )
        cx, cy = float(np.mean(layout[nodes, 0])), float(np.mean(layout[nodes, 1]))
        ax.annotate(
            f"{clade}\nn={size}",
            xy=(cx, cy),
            fontsize=8, color="#202020",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1.0, alpha=0.85),
        )
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                       markersize=8, label=f"{clade} (n={size}, purity={purity:.2f})")
        )

    ax.legend(
        handles=legend_handles, loc="upper left",
        bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False,
        title="Pure subtrees",
    )
    ax.set_title(
        f"SARS-CoV-2 spike TMAP, top {len(selected)} clade-pure subtrees "
        f"(n={atlas.is_reference.size - 1:,})"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return selected


def write_interactive_html(
    *,
    model,
    atlas: Atlas,
    depth: NDArray,
    out_path: Path,
    pure_subtrees: list[tuple[int, int, int, str, float]] | None = None,
) -> Path:
    viz = model.to_tmapviz(include_edges=True)
    viz.title = f"SARS-CoV-2 spike TMAP (n={atlas.is_reference.size - 1:,})"
    viz.point_size = 3
    viz.set_edge_style(color="#9a9a9a", width=0.6, opacity=0.3)
    color_map = assign_clade_colors(atlas.clades)
    variants = [variant_group(str(c)) for c in atlas.clades]
    viz.add_label("Strain", atlas.names)
    viz.add_label("Pango lineage", atlas.pangos.tolist())
    viz.add_color_layout(
        "Major variant", variants,
        categorical=True, color=VARIANT_COLORS,
    )
    viz.add_color_layout(
        "Nextstrain clade", atlas.clades.tolist(),
        categorical=True, color=color_map,
    )
    viz.add_color_layout("Collection num_date", atlas.num_dates, color="turbo")
    viz.add_color_layout("Tree hops from Wuhan-Hu-1", depth, color="turbo")

    filter_layers = [
        "Major variant",
        "Nextstrain clade",
        "Collection num_date",
        "Tree hops from Wuhan-Hu-1",
    ]

    if pure_subtrees:
        membership = ["other"] * atlas.is_reference.size
        subtree_colors = {"other": "#cccccc"}
        for branch, child, size, clade, _purity in pure_subtrees:
            label = f"{clade} (n={size})"
            subtree_colors[label] = color_map.get(clade, "#888888")
            for n in descendants_of(model.tree_, child, branch):
                membership[int(n)] = label
        viz.add_color_layout(
            "Pure subtree", membership,
            categorical=True, color=subtree_colors,
        )
        filter_layers.append("Pure subtree")

    viz.searchable = ["Strain", "Pango lineage"]
    viz.filterable = filter_layers
    return viz.write_html(out_path)


def write_report(
    *,
    path: Path,
    atlas: Atlas,
    depth: NDArray,
    rho_date: float,
    audit: GraphAudit,
    components: tuple[int, list[int]],
    purity_rows: list[tuple[int, int, int, str, float]],
    k: int,
    n_neighbors: int,
    png_path: Path,
    html_path: Path,
) -> None:
    n_comp, comp_sizes = components
    non_ref = ~atlas.is_reference
    lines = [
        f"SARS-CoV-2 spike TMAP Pass A, n={int(non_ref.sum()):,}",
        "Data source: Nextstrain ncov Open build (open metadata + aligned full genomes).",
        "Anchor: Wuhan-Hu-1 spike (extracted from NC_045512.2 reference).",
        "",
        f"k-mer size (AA): {k}",
        f"TMAP: metric=jaccard, n_neighbors={n_neighbors}",
        f"Unique Nextstrain clades: {len(set(atlas.clades.tolist()))}",
        f"Unique Pango lineages: {len(set(atlas.pangos.tolist()))}",
        f"Date range: {float(np.min(atlas.num_dates)):.3f} to {float(np.max(atlas.num_dates)):.3f}",
        "",
        "Tree shape:",
        f"  components: {n_comp}",
        f"  largest component: {comp_sizes[0] if comp_sizes else 0}",
        f"  max tree depth from Wuhan-Hu-1: {float(np.nanmax(depth)):.0f}",
        f"  median tree depth from Wuhan-Hu-1: {float(np.nanmedian(depth)):.0f}",
        "",
        "Headline:",
        f"  Spearman(tree hops from Wuhan-Hu-1, collection num_date) = {rho_date:.3f}",
        "",
        "Graph-local audit:",
        f"  Mean edge date delta: {audit.edge_date_delta_mean:.3f} yr",
        f"  Mean random-pair date delta: {audit.random_date_delta_mean:.3f} yr",
        f"  Random-pair p(random <= observed): {audit.date_delta_p_le:.3f}",
        f"  Edge cross-clade fraction: {audit.edge_clade_boundary_fraction:.3f}",
        f"  Random cross-clade fraction: {audit.random_clade_boundary_fraction:.3f}",
        f"  Random-pair p(random <= observed): {audit.clade_boundary_p_le:.3f}",
        "",
        f"Top {len(purity_rows)} branch subtrees by dominant-clade purity:",
    ]
    for branch, child, size, dom, purity in purity_rows:
        lines.append(
            f"  branch={branch} child={child} size={size:,} dom={dom} purity={purity:.3f}"
        )
    lines.extend(["", f"PNG: {png_path}", f"HTML: {html_path}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_subsample(
    data_dir: Path,
    n_target: int,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (df, sequences) ready for TMAP.

    df has columns: strain, clade, country, date_numeric.
    sequences is a list of spike AA sequences in df row order.
    The Wuhan-Hu-1 reference is prepended as the first row.
    """
    meta_path = data_dir / "metadata.tsv.zst"
    fasta_path = data_dir / "aligned.fasta.zst"

    meta = load_metadata(meta_path)
    selected = stratified_subsample(meta, n_target=n_target, seed=seed)
    np.random.default_rng(seed).shuffle(selected)

    wanted = {m.strain for m in selected}
    wanted.update(WUHAN_STRAIN_CANDIDATES)
    aligned = stream_aligned_fasta(fasta_path, wanted)

    wuhan_seq = None
    for cand in WUHAN_STRAIN_CANDIDATES:
        if cand in aligned:
            wuhan_seq = aligned[cand]
            break
    if wuhan_seq is None:
        raise RuntimeError("Could not locate Wuhan-Hu-1 in aligned FASTA.")
    wuhan_spike_aa = extract_spike_aa(wuhan_seq)

    wuhan_num_date = num_date_from_iso("2019-12-30") or 2019.99

    rows = [
        {"strain": "Wuhan-Hu-1", "clade": "19A", "country": "China", "date_numeric": wuhan_num_date}
    ]
    sequences = [wuhan_spike_aa]
    for m in selected:
        seq = aligned.get(m.strain)
        if seq is None:
            continue
        rows.append({
            "strain": m.strain,
            "clade": m.clade,
            "country": m.country,
            "date_numeric": m.num_date,
        })
        sequences.append(extract_spike_aa(seq))

    df = pd.DataFrame(rows)
    return df, sequences


def shingles_to_minhash(
    sequences: list[str],
    k: int = 6,
    n_perm: int = 128,
) -> list[list[str]]:
    """Return k-mer shingle sets for each sequence.

    The returned list of k-mer lists can be passed directly to
    TMAP(metric='jaccard').fit(), which handles MinHash internally.

    Args:
        sequences: List of amino acid spike sequences.
        k: k-mer length.
        n_perm: Kept for API symmetry; not used (MinHash is applied by TMAP).

    Returns:
        List of k-mer string lists, one per sequence.
    """
    return [kmers_aa(seq, k) for seq in sequences]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, nargs="+", default=[5000, 15000, 25000])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metadata", type=Path, default=META_ZST)
    p.add_argument("--fasta", type=Path, default=FASTA_ZST)
    return p


def main() -> None:
    args = build_parser().parse_args()
    from tmap import TMAP

    print(f"Loading metadata from {args.metadata}...", flush=True)
    meta = load_metadata(args.metadata)
    print(f"  retained {len(meta):,} usable metadata rows", flush=True)

    n_targets = sorted(set(args.n), reverse=True)
    n_max = max(n_targets)
    print(f"Stratified subsample n={n_max:,} from metadata...", flush=True)
    selected_max = stratified_subsample(meta, n_max, seed=args.seed)
    # Shuffle so smaller scales are unbiased subsets of larger ones rather than
    # the earliest bin-ordered prefix.
    np.random.default_rng(args.seed).shuffle(selected_max)
    wanted_strains = {m.strain for m in selected_max}
    wanted_strains.update(WUHAN_STRAIN_CANDIDATES)
    print(f"  wanted {len(wanted_strains):,} strains (incl. Wuhan candidates)", flush=True)

    print(f"Streaming spike regions from {args.fasta}...", flush=True)
    aligned = stream_aligned_fasta(args.fasta, wanted_strains)
    print(f"  read {len(aligned):,} aligned sequences", flush=True)

    wuhan_seq = None
    for cand in WUHAN_STRAIN_CANDIDATES:
        if cand in aligned:
            wuhan_seq = aligned[cand]
            break
    if wuhan_seq is None:
        raise RuntimeError("Could not locate Wuhan-Hu-1 in aligned FASTA.")
    wuhan_spike_aa = extract_spike_aa(wuhan_seq)
    print(f"  Wuhan spike AA length: {len(wuhan_spike_aa)}")

    spike_by_strain = {
        s: extract_spike_aa(seq) for s, seq in aligned.items()
        if s not in WUHAN_STRAIN_CANDIDATES
    }

    summary_rows: list[dict] = []
    for n_target in n_targets:
        selected = selected_max[:n_target]
        print(f"\n=== n={n_target:,} ===", flush=True)
        atlas = build_atlas(selected, spike_by_strain, wuhan_spike_aa, k=args.k)
        print(f"  atlas size: {len(atlas.names):,} (incl. reference)", flush=True)

        model = TMAP(
            metric="jaccard",
            n_neighbors=args.n_neighbors,
            n_permutations=256,
            kc=80,
            seed=args.seed,
            minhash_seed=args.seed,
        ).fit(atlas.kmers)

        depth = rooted_depth(model.tree_, root=0)
        non_ref = ~atlas.is_reference
        valid = non_ref & np.isfinite(depth)
        rho_date = float(
            spearmanr(depth[valid], atlas.num_dates[valid]).statistic
        ) if valid.sum() >= 3 else float("nan")

        audit = graph_audit(model.tree_, atlas, seed=args.seed)
        components = count_components(model.tree_)
        purity_rows = branch_subtree_purity(
            model.tree_, atlas.clades, root=0, top_k=40,
        )

        png_path = IMG_DIR / f"sars_cov2_spike_tmap_{n_target}.png"
        pure_png_path = IMG_DIR / f"sars_cov2_spike_tmap_{n_target}_pure_subtrees.png"
        html_path = IMG_DIR / f"sars_cov2_spike_tmap_{n_target}.html"
        report_path = HERE / f"sars_cov2_spike_report_{n_target}.txt"

        plot_panels(
            model=model, atlas=atlas, depth=depth, rho_date=rho_date, out_path=png_path,
        )
        pure_selected = plot_pure_subtrees(
            model=model, atlas=atlas, purity_rows=purity_rows, out_path=pure_png_path,
        )
        html_written = write_interactive_html(
            model=model, atlas=atlas, depth=depth, out_path=html_path,
            pure_subtrees=pure_selected,
        )
        write_report(
            path=report_path, atlas=atlas, depth=depth,
            rho_date=rho_date, audit=audit, components=components,
            purity_rows=purity_rows, k=args.k, n_neighbors=args.n_neighbors,
            png_path=png_path, html_path=html_written,
        )

        print(
            f"  spearman={rho_date:.3f} comp={components[0]} "
            f"max_depth={float(np.nanmax(depth)):.0f} "
            f"edge_dd={audit.edge_date_delta_mean:.3f} "
            f"rand_dd={audit.random_date_delta_mean:.3f}"
        )
        summary_rows.append({
            "n_target": n_target,
            "n_kept": int(non_ref.sum()),
            "spearman_date": rho_date,
            "n_components": components[0],
            "max_depth": float(np.nanmax(depth)),
            "edge_date_delta_yr": audit.edge_date_delta_mean,
            "random_date_delta_yr": audit.random_date_delta_mean,
            "edge_clade_cross_frac": audit.edge_clade_boundary_fraction,
            "random_clade_cross_frac": audit.random_clade_boundary_fraction,
        })

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
