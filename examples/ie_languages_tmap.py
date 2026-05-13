"""Indo-European languages TMAP on the IE-CoR cognate dataset.

Downloads the IE-CoR CLDF bundle (Heggarty et al. 2023, Science), parses
cognate-class membership per concept per language into Jaccard-shingled
feature sets, and fits TMAP to recover the IE language family structure.

Outputs:
    paper/images/ie_languages_tmap.png
    examples/ie_languages_report.txt
"""
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "ie_languages"
CLDF_DIR = DATA_DIR / "iecor_cldf"
ZIP_PATH = DATA_DIR / "iecor.zip"
IMG_PATH = HERE.parent / "paper" / "images" / "ie_languages_tmap.png"
REPORT_PATH = HERE / "ie_languages_report.txt"

IECOR_URLS = (
    "https://github.com/lexibank/iecor/archive/refs/tags/v1.2.zip",
    "https://github.com/lexibank/iecor/archive/refs/tags/v1.1.zip",
    "https://github.com/lexibank/iecor/archive/refs/heads/master.zip",
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


def _download(url: str, dest: Path) -> None:
    """Download `url` to `dest` if not already present. Skips on cache hit.

    Transparently decompresses gzip-encoded responses so the file on disk is
    always raw bytes — no special handling needed at extract time.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()
    if raw[:2] == b"\x1f\x8b":
        import gzip
        raw = gzip.decompress(raw)
    tmp.write_bytes(raw)
    tmp.rename(dest)


def ensure_iecor_cldf(
    dest_dir: Path = CLDF_DIR,
    zip_path: Path = ZIP_PATH,
    urls: tuple[str, ...] = IECOR_URLS,
) -> Path:
    """Ensure IE-CoR CLDF tables are extracted under `dest_dir`. Returns `dest_dir`.

    Tries each URL in order; raises on total failure.
    """
    forms = dest_dir / "forms.csv"
    if forms.exists() and forms.stat().st_size > 0:
        return dest_dir

    last_err: Exception | None = None
    for url in urls:
        try:
            _download(url, zip_path)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {url}: {e}", flush=True)
            last_err = e
    else:
        raise RuntimeError(f"All IE-CoR URLs failed; last error: {last_err}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir.parent)

    # IE-CoR archives extract to a single top-level dir; find forms.csv and
    # alias that directory as CLDF_DIR.
    for cand in dest_dir.parent.rglob("forms.csv"):
        if cand.parent != dest_dir:
            if dest_dir.exists():
                import shutil
                shutil.rmtree(dest_dir)
            cand.parent.rename(dest_dir)
        break
    if not (dest_dir / "forms.csv").exists():
        raise RuntimeError(f"forms.csv not found after extracting {zip_path}")
    return dest_dir


def build_cognate_atlas(cldf_dir: Path = CLDF_DIR) -> CognateAtlas:
    """Parse CLDF tables -> cognate feature sets per language.

    Each language is represented as a set of "<concept_id>::<cognate_class>"
    tokens, one per (concept, cognate-class) it participates in. Jaccard between
    two such sets measures shared cognate vocabulary.
    """
    import csv

    # Languages -- use IE-CoR's `Clade` column for the subgroup.
    lang_rows: dict[str, LangMeta] = {}
    with (cldf_dir / "languages.csv").open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lid = row.get("ID") or row.get("Language_ID")
            if not lid:
                continue
            lang_rows[lid] = LangMeta(
                glottocode=row.get("Glottocode", "") or "",
                name=row.get("Name", "") or lid,
                family=row.get("Family", "") or "Indo-European",
                subgroup=row.get("Clade") or row.get("SubGroup") or "Other",
            )

    # Forms -- form_id -> (language_id, concept_id)
    form_to_lang_concept: dict[str, tuple[str, str]] = {}
    with (cldf_dir / "forms.csv").open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fid = row.get("ID")
            lid = row.get("Language_ID")
            cid = row.get("Parameter_ID") or row.get("Concept_ID")
            if fid and lid and cid:
                form_to_lang_concept[fid] = (lid, cid)

    # Cognates -- build per-language token sets from form -> cognate-class links
    tokens_by_lang: dict[str, set[str]] = {lid: set() for lid in lang_rows}
    with (cldf_dir / "cognates.csv").open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fid = row.get("Form_ID")
            cog = row.get("Cognateset_ID") or row.get("Cognate_Class")
            if not fid or not cog:
                continue
            lc = form_to_lang_concept.get(fid)
            if not lc:
                continue
            lid, cid = lc
            if lid in tokens_by_lang:
                tokens_by_lang[lid].add(f"{cid}::{cog}")

    names: list[str] = []
    sets: list[list[str]] = []
    fams: list[str] = []
    subs: list[str] = []
    for lid, meta in lang_rows.items():
        toks = sorted(tokens_by_lang.get(lid, set()))
        if not toks:
            continue
        names.append(lid)
        sets.append(toks)
        fams.append(meta.family)
        subs.append(meta.subgroup)

    return CognateAtlas(
        names=names,
        cognate_sets=sets,
        families=np.asarray(fams, dtype=object),
        subgroups=np.asarray(subs, dtype=object),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-neighbors", type=int, default=15,
                   help="Smaller default than flu since only ~160 languages.")
    p.add_argument("--n-permutations", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p


def top_level_clade(label: str) -> str:
    """Reduce 'Germanic;North-West;West' to 'Germanic' for coloring/purity."""
    return label.split(";")[0] if label else "Other"


def main() -> None:
    args = build_parser().parse_args()
    from tmap import TMAP

    ensure_iecor_cldf()
    atlas = build_cognate_atlas()
    print(f"Languages in atlas: {len(atlas.names):,}", flush=True)

    print("Fitting TMAP...", flush=True)
    model = TMAP(
        metric="jaccard",
        n_neighbors=args.n_neighbors,
        n_permutations=args.n_permutations,
        kc=50,
        seed=args.seed,
        minhash_seed=args.seed,
    ).fit(atlas.cognate_sets)

    # Validation uses the flat top-level family labels.
    top_subs = np.asarray([top_level_clade(s) for s in atlas.subgroups], dtype=object)
    purity = _edge_subgroup_purity(model.tree_, top_subs)
    chance = _chance_subgroup_purity(top_subs)
    print(f"Edge same-family fraction: {purity:.3f} (chance {chance:.3f})", flush=True)

    _plot_tmap(model, atlas, top_subs, IMG_PATH)

    lines = [
        "Indo-European languages TMAP (Pass A)",
        f"Languages: {len(atlas.names):,}",
        f"Top-level families: {sorted(set(top_subs))}",
        f"TMAP: metric=jaccard, n_neighbors={args.n_neighbors}, "
        f"n_permutations={args.n_permutations}",
        "",
        f"Edge same-family fraction: {purity:.3f}",
        f"Chance baseline (random pairs): {chance:.3f}",
        f"Lift vs chance: {purity / chance:.2f}x" if chance > 0 else "Lift vs chance: n/a",
        "",
        f"PNG: {IMG_PATH}",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {REPORT_PATH}")


def _edge_subgroup_purity(tree, subgroups: NDArray) -> float:
    """Fraction of TMAP tree edges whose endpoints share a family label."""
    matches = 0
    total = 0
    for src, tgt in tree.edges:
        if src < len(subgroups) and tgt < len(subgroups):
            total += 1
            if subgroups[src] == subgroups[tgt]:
                matches += 1
    return matches / total if total else 0.0


def _chance_subgroup_purity(
    subgroups: NDArray, n_pairs: int = 50000, seed: int = 0
) -> float:
    """Chance baseline: probability two random tips share a family label."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(subgroups), size=n_pairs)
    b = rng.integers(0, len(subgroups), size=n_pairs)
    return float(np.mean(subgroups[a] == subgroups[b]))


def _plot_tmap(
    model,
    atlas: CognateAtlas,
    top_subs: NDArray,
    out_path: Path,
) -> None:
    import matplotlib
    coords = model.embedding_
    uniq = sorted(set(top_subs))
    cmap = matplotlib.colormaps.get_cmap("tab20")
    colors = {g: cmap(i / max(1, len(uniq) - 1)) for i, g in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    # Tree edges first, then colored leaves on top.
    for src, tgt in model.tree_.edges:
        if src < len(coords) and tgt < len(coords):
            ax.plot(
                [coords[src, 0], coords[tgt, 0]],
                [coords[src, 1], coords[tgt, 1]],
                color="lightgray", lw=0.4, alpha=0.7, zorder=1,
            )
    for g in uniq:
        mask = top_subs == g
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            color=colors[g], s=36, label=g, edgecolors="black",
            linewidths=0.4, zorder=2,
        )
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Indo-European languages TMAP — {len(atlas.names)} languages")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
