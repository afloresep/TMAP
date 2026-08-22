"""WikiArt — art history as a navigable tree.

Build a TMAP of paintings from WikiArt, colored by artistic style.
Trace paths between art movements to see how painting styles evolve
through visual similarity: from Impressionism through Post-Impressionism
to Cubism and Abstract — each step a tiny visual shift, the endpoints
radically different.

Outputs
-------
examples/wikiart_tmap.html          Interactive TMAP with painting tooltips
examples/wikiart_report.txt         Art style analysis report

Data
----
Downloads WikiArt dataset from HuggingFace (~6 GB, cached after first run).

Usage
-----
    python examples/images/wikiart_tmap.py
    python examples/images/wikiart_tmap.py --max-images 10000
    python examples/images/wikiart_tmap.py --serve

Requirements
------------
    pip install datasets torch torchvision
"""

from __future__ import annotations

import argparse
import base64
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from tmap import TMAP
from tmap.graph.analysis import (
    boundary_edges,
    confusion_matrix_from_tree,
    subtree_purity,
)

# Scripts live in a subfolder; data, caches and outputs stay at examples/.
EXAMPLES_DIR = Path(__file__).resolve().parents[1]

CACHE_DIR = EXAMPLES_DIR / "data" / "wikiart_cache"
OUTPUT_DIR = EXAMPLES_DIR

# Map styles to broad eras for supercategory analysis
STYLE_ERA = {
    "Early_Renaissance": "Renaissance",
    "High_Renaissance": "Renaissance",
    "Northern_Renaissance": "Renaissance",
    "Mannerism_Late_Renaissance": "Renaissance",
    "Baroque": "Classical",
    "Rococo": "Classical",
    "Romanticism": "19th Century",
    "Realism": "19th Century",
    "Impressionism": "19th Century",
    "Post_Impressionism": "19th Century",
    "Pointillism": "19th Century",
    "Art_Nouveau_Modern": "Early Modern",
    "Symbolism": "Early Modern",
    "Fauvism": "Early Modern",
    "Expressionism": "Early Modern",
    "Cubism": "Modern",
    "Analytical_Cubism": "Modern",
    "Synthetic_Cubism": "Modern",
    "Naive_Art_Primitivism": "Modern",
    "Abstract_Expressionism": "Contemporary",
    "Action_painting": "Contemporary",
    "Color_Field_Painting": "Contemporary",
    "Minimalism": "Contemporary",
    "Pop_Art": "Contemporary",
    "Contemporary_Realism": "Contemporary",
    "New_Realism": "Contemporary",
    "Ukiyo_e": "East Asian",
}


# 1. Data loading


def load_wikiart(max_images: int | None) -> tuple:
    """Load WikiArt from HuggingFace.

    Returns (images, styles, artists, style_names, artist_names).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library is required.\nInstall with: pip install datasets"
        )

    print("Loading WikiArt from HuggingFace...")
    ds = load_dataset("huggan/wikiart", split="train")
    style_names = ds.features["style"].names
    artist_names = ds.features["artist"].names
    print(f"  {len(ds):,} paintings, {len(style_names)} styles, {len(artist_names)} artists")

    if max_images and len(ds) > max_images:
        print(f"  Subsampling to {max_images:,} images...")
        ds = ds.shuffle(seed=42).select(range(max_images))

    # Extract metadata
    styles = np.array(ds["style"])
    artists = np.array(ds["artist"])

    return ds, styles, artists, style_names, artist_names


# 2. Embedding extraction


def extract_embeddings(
    ds,
    batch_size: int,
    device: torch.device,
    cache_tag: str,
) -> np.ndarray:
    """Extract ResNet-50 avgpool embeddings (2048-d), cached to disk."""
    cache_path = CACHE_DIR / f"embeddings_{cache_tag}.npy"
    if cache_path.exists():
        print(f"  Loading cached embeddings: {cache_path.name}")
        return np.load(cache_path)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval().to(device)

    features: list[torch.Tensor] = []

    def _hook(_mod: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        features.append(out.squeeze(-1).squeeze(-1).cpu())

    model.avgpool.register_forward_hook(_hook)

    n = len(ds)
    print(f"  Extracting embeddings ({n:,} images, batch_size={batch_size})...")
    t0 = time.time()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = ds[start:end]
        images = batch["image"]

        tensors = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            tensors.append(transform(img.convert("RGB")))

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            model(batch_tensor)

        if (start // batch_size) % 20 == 0:
            pct = start * 100 // n
            print(f"    {pct}% ({start:,}/{n:,})", flush=True)

    embeddings = torch.cat(features).numpy()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — shape: {embeddings.shape}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


# 3. Image encoding for tooltips


def encode_images(ds, size: int = 80, quality: int = 70) -> list[str]:
    """Encode images as base64 JPEG data URIs for tooltips."""
    print(f"  Encoding {len(ds):,} images for tooltips ({size}x{size})...")
    uris: list[str] = []
    for i in range(len(ds)):
        img = ds[i]["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB").resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        uris.append(f"data:image/jpeg;base64,{b64}")
    return uris


# 4. Analysis


def analyze_styles(
    model: TMAP,
    styles: np.ndarray,
    artists: np.ndarray,
    style_names: list[str],
    artist_names: list[str],
) -> str:
    """Generate art style analysis report."""
    tree = model.tree_
    lines: list[str] = []
    w = lines.append

    style_labels = np.array([style_names[s] for s in styles])
    eras = np.array([STYLE_ERA.get(s, "Other") for s in style_labels])

    w(f"WikiArt Style Analysis — {len(styles):,} paintings, {len(style_names)} styles\n")

    # 1. Era boundaries
    be_era = boundary_edges(tree, eras)
    n_edges = len(tree.edges)
    w("1. Art era boundaries:")
    w(
        f"   Same-era edges: {n_edges - len(be_era)} / {n_edges} "
        f"({(n_edges - len(be_era)) / n_edges:.1%})"
    )

    cmat_era, cls_era = confusion_matrix_from_tree(tree, eras)
    np.fill_diagonal(cmat_era, 0)
    upper = np.triu_indices_from(cmat_era, k=1)
    pair_counts_era = cmat_era[upper] + cmat_era.T[upper]
    top_era = np.argsort(pair_counts_era)[::-1][:10]
    w("   Most connected eras:")
    for i in top_era:
        if pair_counts_era[i] == 0:
            break
        r, c = upper[0][i], upper[1][i]
        w(f"     {pair_counts_era[i]:4d} edges: {cls_era[r]:>15s} <-> {cls_era[c]}")
    w("")

    # 2. Style boundaries
    be_style = boundary_edges(tree, style_labels)
    w("2. Style boundaries:")
    w(
        f"   Same-style edges: {n_edges - len(be_style)} / {n_edges} "
        f"({(n_edges - len(be_style)) / n_edges:.1%})"
    )

    cmat, classes = confusion_matrix_from_tree(tree, style_labels)
    np.fill_diagonal(cmat, 0)
    upper = np.triu_indices_from(cmat, k=1)
    pair_counts = cmat[upper] + cmat.T[upper]
    top_idx = np.argsort(pair_counts)[::-1][:15]
    w("   Most connected style pairs:")
    for i in top_idx:
        if pair_counts[i] == 0:
            break
        r, c = upper[0][i], upper[1][i]
        w(f"     {pair_counts[i]:4d} edges: {classes[r]:>30s} <-> {classes[c]}")
    w("")

    # 3. Subtree purity
    purity_style = subtree_purity(tree, style_labels, min_size=20)
    valid = purity_style[~np.isnan(purity_style)]
    purity_era = subtree_purity(tree, eras, min_size=20)
    valid_era = purity_era[~np.isnan(purity_era)]
    w("3. Subtree purity:")
    w(f"   By era:   mean={valid_era.mean():.3f}  median={np.median(valid_era):.3f}")
    w(f"   By style: mean={valid.mean():.3f}  median={np.median(valid):.3f}\n")

    # 4. Art historical paths
    w("4. Art historical paths:")
    path_pairs = [
        ("Impressionism", "Cubism"),
        ("Impressionism", "Abstract_Expressionism"),
        ("High_Renaissance", "Impressionism"),
        ("Baroque", "Pop_Art"),
        ("Realism", "Minimalism"),
        ("Romanticism", "Expressionism"),
        ("Ukiyo_e", "Impressionism"),
    ]
    for style_a, style_b in path_pairs:
        idx_a = np.where(style_labels == style_a)[0]
        idx_b = np.where(style_labels == style_b)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            w(f"   {style_a} -> {style_b}: (style not found in subsample)")
            continue

        node_a, node_b = int(idx_a[0]), int(idx_b[0])
        try:
            path_nodes = tree.path(node_a, node_b)
        except IndexError:
            w(f"   {style_a:>30s} -> {style_b:<30s}  (disconnected)")
            continue

        path_styles = style_labels[path_nodes]
        unique_styles = []
        for s in path_styles:
            if not unique_styles or unique_styles[-1] != s:
                unique_styles.append(s)

        w(
            f"   {style_a:>30s} -> {style_b:<30s}  "
            f"hops={len(path_nodes):4d}  styles crossed={len(set(path_styles))}"
        )
        w(f"     Route: {' -> '.join(unique_styles[:8])}{'...' if len(unique_styles) > 8 else ''}")
    w("")

    # 5. Per-style coherence
    w("5. Per-style tree coherence:")
    w(f"   {'Style':>30s}  {'Count':>6s}  {'Boundary %':>10s}")
    style_counts = {}
    for s in style_labels:
        style_counts[s] = style_counts.get(s, 0) + 1

    for style in sorted(style_counts, key=style_counts.get, reverse=True):
        mask = style_labels == style
        style_idx = set(np.where(mask)[0])
        boundary = 0
        internal = 0
        for s, t in tree.edges:
            s_in = s in style_idx
            t_in = t in style_idx
            if s_in and t_in:
                internal += 1
            elif s_in or t_in:
                boundary += 1
        total = boundary + internal
        bfrac = boundary / total if total > 0 else 0.0
        w(f"   {style:>30s}  {style_counts[style]:6d}  {bfrac:10.1%}")

    return "\n".join(lines)


# 5. Visualization


def create_visualization(
    model: TMAP,
    styles: np.ndarray,
    artists: np.ndarray,
    style_names: list[str],
    artist_names: list[str],
    image_uris: list[str],
):
    """Build TmapViz with style coloring and painting tooltips."""
    viz = model.to_tmapviz()
    viz.title = f"WikiArt — {len(styles):,} Paintings"

    style_labels = [style_names[s] for s in styles]
    eras = [STYLE_ERA.get(s, "Other") for s in style_labels]

    # Style and era coloring
    viz.add_color_layout("style", style_labels, categorical=True, color="tab20")
    viz.add_color_layout("era", eras, categorical=True, color="Set2")

    # Top artists only (too many to show all)
    artist_labels = [artist_names[a] for a in artists]
    artist_counts = {}
    for a in artist_labels:
        artist_counts[a] = artist_counts.get(a, 0) + 1
    top_artists = {a for a, c in sorted(artist_counts.items(), key=lambda x: -x[1])[:30]}
    artist_display = [a if a in top_artists else "Other" for a in artist_labels]
    viz.add_color_layout("artist (top 30)", artist_display, categorical=True, color="tab20")

    # Tooltips
    viz.add_images(image_uris, tooltip_size=100)
    viz.add_label(
        "painting", [f"{style_names[s]} | {artist_names[a]}" for s, a in zip(styles, artists)]
    )

    return viz


# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="WikiArt TMAP")
    parser.add_argument(
        "--max-images",
        type=int,
        default=20000,
        help="Maximum images to use (default: 20000, set 0 for all)",
    )
    parser.add_argument("--k", type=int, default=15, help="Number of neighbors")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    max_img = args.max_images if args.max_images > 0 else None
    print(f"Device: {device}")

    # Load data
    ds, styles, artists, style_names, artist_names = load_wikiart(max_img)
    n = len(ds)

    # Extract embeddings
    cache_tag = f"resnet50_{n}"
    print("Extracting ResNet-50 embeddings...")
    embeddings = extract_embeddings(ds, args.batch_size, device, cache_tag)

    # Build TMAP
    print(f"Building TMAP (metric='cosine', k={args.k})...")
    t0 = time.time()
    model = TMAP(
        metric="cosine",
        n_neighbors=args.k,
        layout_iterations=1000,
        seed=42,
    ).fit(embeddings.astype(np.float32))
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Analysis
    report = analyze_styles(model, styles, artists, style_names, artist_names)
    report_path = OUTPUT_DIR / "wikiart_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {report_path}")
    print("\n" + report)

    # Visualization
    print("Encoding images for tooltips...")
    image_uris = encode_images(ds, size=80, quality=70)

    print("Building visualization...")
    viz = create_visualization(
        model,
        styles,
        artists,
        style_names,
        artist_names,
        image_uris,
    )
    html_path = viz.write_html(OUTPUT_DIR / "wikiart_tmap")
    print(f"HTML saved to {html_path}")

    if args.serve:
        print(f"Serving on http://127.0.0.1:{args.port}")
        viz.serve(port=args.port)


if __name__ == "__main__":
    main()
