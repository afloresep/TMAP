"""Oxford Flowers 102 — morphological gradients with TMAP.

Build a TMAP of 8K flower images across 102 species. The tree reveals
visual morphological gradients: paths trace smooth transitions between
flower shapes, colors, and petal structures.

Paths like sunflower -> daisy -> dandelion (yellow, radial) or
rose -> camellia -> magnolia show clear visual gradients where each
step is a tiny change in petal shape, color, or structure.

Outputs
-------
examples/flowers_tmap.html          Interactive TMAP with flower tooltips
examples/flowers_report.txt         Morphological analysis report

Data
----
Downloads Oxford Flowers 102 via torchvision (~350 MB, cached).

Usage
-----
    python examples/images/flowers_tmap.py
    python examples/images/flowers_tmap.py --serve
    python examples/images/flowers_tmap.py --device cuda

Requirements
------------
    pip install torch torchvision
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
from torchvision import datasets, models, transforms

from tmap import TMAP
from tmap.graph.analysis import (
    boundary_edges,
    confusion_matrix_from_tree,
    subtree_purity,
)

# Scripts live in a subfolder; data, caches and outputs stay at examples/.
EXAMPLES_DIR = Path(__file__).resolve().parents[1]

CACHE_DIR = EXAMPLES_DIR / "data" / "flowers_cache"
OUTPUT_DIR = EXAMPLES_DIR

# 102 flower category names (0-indexed, matching torchvision labels)
FLOWER_NAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]

# Group flowers by visual characteristics for supercategory analysis
FLOWER_GROUP = {
    "sunflower": "daisy-like",
    "oxeye daisy": "daisy-like",
    "common dandelion": "daisy-like",
    "barbeton daisy": "daisy-like",
    "black-eyed susan": "daisy-like",
    "gazania": "daisy-like",
    "osteospermum": "daisy-like",
    "blanket flower": "daisy-like",
    "mexican aster": "daisy-like",
    "colt's foot": "daisy-like",
    "marigold": "daisy-like",
    "english marigold": "daisy-like",
    "rose": "rose-like",
    "camellia": "rose-like",
    "carnation": "rose-like",
    "lenten rose": "rose-like",
    "sweet william": "rose-like",
    "garden phlox": "rose-like",
    "azalea": "rose-like",
    "pink primrose": "rose-like",
    "pelargonium": "rose-like",
    "geranium": "rose-like",
    "mallow": "rose-like",
    "tree mallow": "rose-like",
    "hibiscus": "rose-like",
    "desert-rose": "rose-like",
    "moon orchid": "orchid-like",
    "hard-leaved pocket orchid": "orchid-like",
    "ruby-lipped cattleya": "orchid-like",
    "siam tulip": "orchid-like",
    "tiger lily": "lily-like",
    "fire lily": "lily-like",
    "giant white arum lily": "lily-like",
    "canna lily": "lily-like",
    "water lily": "lily-like",
    "sword lily": "lily-like",
    "toad lily": "lily-like",
    "peruvian lily": "lily-like",
    "blackberry lily": "lily-like",
    "bearded iris": "iris-like",
    "yellow iris": "iris-like",
    "spring crocus": "iris-like",
    "morning glory": "trumpet-shaped",
    "trumpet creeper": "trumpet-shaped",
    "petunia": "trumpet-shaped",
    "foxglove": "trumpet-shaped",
    "snapdragon": "trumpet-shaped",
    "canterbury bells": "trumpet-shaped",
    "mexican petunia": "trumpet-shaped",
    "passion flower": "exotic",
    "bird of paradise": "exotic",
    "anthurium": "exotic",
    "red ginger": "exotic",
    "frangipani": "exotic",
    "bougainvillea": "exotic",
    "king protea": "exotic",
    "bromelia": "exotic",
    "cautleya spicata": "exotic",
    "lotus": "aquatic",
    "hippeastrum": "bulb",
    "grape hyacinth": "bulb",
    "daffodil": "bulb",
    "buttercup": "simple",
    "windflower": "simple",
    "corn poppy": "simple",
    "californian poppy": "simple",
    "tree poppy": "simple",
    "wild pansy": "simple",
    "primula": "simple",
    "sweet pea": "simple",
    "love in the mist": "simple",
    "balloon flower": "simple",
    "columbine": "simple",
    "magnolia": "tree flower",
    "clematis": "vine",
    "cyclamen": "bulb",
    "poinsettia": "exotic",
    "bee balm": "exotic",
    "wallflower": "simple",
    "artichoke": "thistle-like",
    "globe thistle": "thistle-like",
    "spear thistle": "thistle-like",
    "alpine sea holly": "thistle-like",
    "purple coneflower": "thistle-like",
    "pincushion flower": "thistle-like",
    "ball moss": "other",
}


def load_flowers() -> tuple[list[Image.Image], np.ndarray]:
    """Load all splits of Flowers 102, return (images, labels)."""
    data_dir = EXAMPLES_DIR / "data" / "flowers"
    all_images: list[Image.Image] = []
    all_labels: list[int] = []

    for split in ("train", "val", "test"):
        ds = datasets.Flowers102(
            root=str(data_dir),
            split=split,
            download=True,
        )
        for img, label in ds:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            all_images.append(img.convert("RGB"))
            all_labels.append(label)

    labels = np.array(all_labels, dtype=np.int64)
    print(f"  {len(all_images)} images, {len(np.unique(labels))} species")
    return all_images, labels


def extract_embeddings(
    images: list[Image.Image],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract ResNet-50 avgpool embeddings (2048-d), cached to disk."""
    cache_path = CACHE_DIR / f"embeddings_{len(images)}.npy"
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

    print(f"  Extracting embeddings ({len(images)} images)...")
    t0 = time.time()
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch = [transform(img) for img in images[start:end]]
        tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            model(tensor)

    embeddings = torch.cat(features).numpy()
    print(f"  Done in {time.time() - t0:.1f}s — shape: {embeddings.shape}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def encode_images(images: list[Image.Image], size: int = 96) -> list[str]:
    """Encode images as base64 JPEG data URIs."""
    print(f"  Encoding {len(images)} images for tooltips ({size}x{size})...")
    uris: list[str] = []
    for img in images:
        img = img.resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        uris.append(f"data:image/jpeg;base64,{b64}")
    return uris


def analyze_flowers(
    model: TMAP,
    labels: np.ndarray,
) -> str:
    """Generate flower morphological analysis report."""
    tree = model.tree_
    lines: list[str] = []
    w = lines.append

    species = np.array([FLOWER_NAMES[l] for l in labels])
    groups = np.array([FLOWER_GROUP.get(s, "other") for s in species])

    n_species = len(np.unique(labels))
    w("Oxford Flowers 102 — Morphological Analysis")
    w(f"  {len(labels):,} images, {n_species} species\n")

    # 1. Morphological group boundaries
    be_group = boundary_edges(tree, groups)
    n_edges = len(tree.edges)
    w("1. Morphological group boundaries:")
    w(
        f"   Same-group edges: {n_edges - len(be_group)} / {n_edges} "
        f"({(n_edges - len(be_group)) / n_edges:.1%})"
    )

    cmat_g, cls_g = confusion_matrix_from_tree(tree, groups)
    np.fill_diagonal(cmat_g, 0)
    upper = np.triu_indices_from(cmat_g, k=1)
    pair_counts = cmat_g[upper] + cmat_g.T[upper]
    top_idx = np.argsort(pair_counts)[::-1][:10]
    w("   Most connected groups:")
    for i in top_idx:
        if pair_counts[i] == 0:
            break
        r, c = upper[0][i], upper[1][i]
        w(f"     {pair_counts[i]:4d} edges: {cls_g[r]:>15s} <-> {cls_g[c]}")
    w("")

    # 2. Species-level boundaries
    be_species = boundary_edges(tree, species)
    w("2. Species boundaries:")
    w(
        f"   Same-species edges: {n_edges - len(be_species)} / {n_edges} "
        f"({(n_edges - len(be_species)) / n_edges:.1%})"
    )

    cmat_s, cls_s = confusion_matrix_from_tree(tree, species)
    np.fill_diagonal(cmat_s, 0)
    upper_s = np.triu_indices_from(cmat_s, k=1)
    pair_counts_s = cmat_s[upper_s] + cmat_s.T[upper_s]
    top_s = np.argsort(pair_counts_s)[::-1][:15]
    w("   Most visually similar species pairs:")
    for i in top_s:
        if pair_counts_s[i] == 0:
            break
        r, c = upper_s[0][i], upper_s[1][i]
        w(f"     {pair_counts_s[i]:3d} edges: {cls_s[r]:>25s} <-> {cls_s[c]}")
    w("")

    # 3. Subtree purity
    purity_g = subtree_purity(tree, groups, min_size=10)
    valid_g = purity_g[~np.isnan(purity_g)]
    purity_s = subtree_purity(tree, species, min_size=10)
    valid_s = purity_s[~np.isnan(purity_s)]
    w("3. Subtree purity:")
    w(f"   By group:   mean={valid_g.mean():.3f}  median={np.median(valid_g):.3f}")
    w(f"   By species: mean={valid_s.mean():.3f}  median={np.median(valid_s):.3f}\n")

    # 4. Morphological paths
    w("4. Morphological paths (species to species along the tree):")
    path_pairs = [
        ("sunflower", "oxeye daisy"),  # both daisy-like, yellow
        ("sunflower", "rose"),  # very different morphology
        ("rose", "camellia"),  # visually similar
        ("water lily", "lotus"),  # aquatic flowers
        ("bearded iris", "moon orchid"),  # complex petals
        ("corn poppy", "californian poppy"),  # both poppies
        ("trumpet creeper", "morning glory"),  # both trumpet-shaped
        ("sunflower", "passion flower"),  # radial vs complex
        ("king protea", "artichoke"),  # both spiky/thistle-like
    ]
    for sp_a, sp_b in path_pairs:
        idx_a = np.where(species == sp_a)[0]
        idx_b = np.where(species == sp_b)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            w(f"   {sp_a} -> {sp_b}: (species not found)")
            continue

        node_a, node_b = int(idx_a[0]), int(idx_b[0])
        try:
            path_nodes = tree.path(node_a, node_b)
        except IndexError:
            w(f"   {sp_a:>25s} -> {sp_b:<25s}  (disconnected)")
            continue

        path_species = species[path_nodes]
        unique_species = []
        for s in path_species:
            if not unique_species or unique_species[-1] != s:
                unique_species.append(s)

        w(
            f"   {sp_a:>25s} -> {sp_b:<25s}  "
            f"hops={len(path_nodes):3d}  species crossed={len(set(path_species))}"
        )
        # Show route through species
        if len(unique_species) <= 8:
            w(f"     Route: {' -> '.join(unique_species)}")
        else:
            route = unique_species[:4] + ["..."] + unique_species[-3:]
            w(f"     Route: {' -> '.join(route)}")
    w("")

    # 5. Per-group coherence
    w("5. Per-group tree coherence:")
    w(f"   {'Group':>18s}  {'Count':>6s}  {'Boundary %':>10s}")
    group_counts = {}
    for g in groups:
        group_counts[g] = group_counts.get(g, 0) + 1

    for group in sorted(group_counts, key=group_counts.get, reverse=True):
        mask = groups == group
        grp_idx = set(np.where(mask)[0])
        boundary = 0
        internal = 0
        for s, t in tree.edges:
            s_in = s in grp_idx
            t_in = t in grp_idx
            if s_in and t_in:
                internal += 1
            elif s_in or t_in:
                boundary += 1
        total = boundary + internal
        bfrac = boundary / total if total > 0 else 0.0
        w(f"   {group:>18s}  {group_counts[group]:6d}  {bfrac:10.1%}")

    return "\n".join(lines)


# 5. Visualization


def create_visualization(
    model: TMAP,
    labels: np.ndarray,
    image_uris: list[str],
):
    """Build TmapViz with species coloring and flower tooltips."""
    viz = model.to_tmapviz()
    viz.title = f"Oxford Flowers 102 — {len(labels):,} Images"

    species = [FLOWER_NAMES[l] for l in labels]
    groups = [FLOWER_GROUP.get(s, "other") for s in species]

    viz.add_label("species", species)
    viz.add_color_layout("group", groups, categorical=True, color="tab20")

    viz.add_images(image_uris, tooltip_size=100)
    viz.add_label("flower", species)

    return viz


# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="Oxford Flowers 102 TMAP")
    parser.add_argument("--k", type=int, default=15, help="Number of neighbors")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Load data
    print("Loading Flowers 102...")
    images, labels = load_flowers()

    # Embeddings
    print("Extracting ResNet-50 embeddings...")
    embeddings = extract_embeddings(images, args.batch_size, device)

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
    report = analyze_flowers(model, labels)
    report_path = OUTPUT_DIR / "flowers_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {report_path}")
    print("\n" + report)

    # Visualization
    print("Encoding images for tooltips...")
    image_uris = encode_images(images)

    print("Building visualization...")
    viz = create_visualization(model, labels, image_uris)
    html_path = viz.write_html(OUTPUT_DIR / "flowers_tmap")
    print(f"HTML saved to {html_path}")

    if args.serve:
        print(f"Serving on http://127.0.0.1:{args.port}")
        viz.serve(port=args.port)


if __name__ == "__main__":
    main()
