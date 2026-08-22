"""CUB-200 bird species — morphological paths across 200 species.

Build a TMAP of 12K bird images across 200 species. The tree reveals
visual similarity: similar-looking species cluster together, and paths
trace morphological gradients across bird families — from hummingbird
to kingfisher to woodpecker (small, colorful, pointed beak).

Outputs
-------
examples/cub200_tmap.html           Interactive TMAP with bird tooltips
examples/cub200_report.txt          Species analysis report

Data
----
Downloads CUB-200-2011 from HuggingFace (~1.2 GB, cached).

Usage
-----
    python examples/images/cub200_birds_tmap.py
    python examples/images/cub200_birds_tmap.py --serve

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

CACHE_DIR = EXAMPLES_DIR / "data" / "cub200_cache"
OUTPUT_DIR = EXAMPLES_DIR

# Bird families for supercategory analysis (prefix → family)
BIRD_FAMILY = {
    "Albatross": "seabird",
    "Auklet": "seabird",
    "Cormorant": "seabird",
    "Frigate": "seabird",
    "Fulmar": "seabird",
    "Gull": "seabird",
    "Jaeger": "seabird",
    "Pelican": "seabird",
    "Puffin": "seabird",
    "Tern": "seabird",
    "Booby": "seabird",
    "Petrel": "seabird",
    "Grebe": "waterbird",
    "Heron": "waterbird",
    "Crane": "waterbird",
    "Duck": "waterbird",
    "Mallard": "waterbird",
    "Merganser": "waterbird",
    "Kingfisher": "waterbird",
    "Sparrow": "songbird",
    "Warbler": "songbird",
    "Wren": "songbird",
    "Finch": "songbird",
    "Bunting": "songbird",
    "Vireo": "songbird",
    "Tanager": "songbird",
    "Grosbeak": "songbird",
    "Oriole": "songbird",
    "Cardinal": "songbird",
    "Towhee": "songbird",
    "Junco": "songbird",
    "Goldfinch": "songbird",
    "Woodpecker": "woodpecker",
    "Flicker": "woodpecker",
    "Hummingbird": "hummingbird",
    "Jay": "corvid",
    "Crow": "corvid",
    "Raven": "corvid",
    "Hawk": "raptor",
    "Eagle": "raptor",
    "Falcon": "raptor",
    "Owl": "raptor",
    "Osprey": "raptor",
    "Kite": "raptor",
    "Swallow": "aerial",
    "Swift": "aerial",
    "Nighthawk": "aerial",
    "Mockingbird": "mimic",
    "Thrasher": "mimic",
    "Catbird": "mimic",
    "Blackbird": "icterid",
    "Meadowlark": "icterid",
    "Cowbird": "icterid",
    "Grackle": "icterid",
    "Bobolink": "icterid",
    "Cuckoo": "other",
    "Pigeon": "other",
    "Parakeet": "other",
    "Flycatcher": "flycatcher",
    "Kingbird": "flycatcher",
    "Pewee": "flycatcher",
    "Phoebe": "flycatcher",
    "Nuthatch": "tree-clinger",
    "Creeper": "tree-clinger",
    "Chickadee": "tit",
    "Titmouse": "tit",
    "Waxwing": "other",
    "Shrike": "other",
    "Pipit": "other",
    "Lark": "other",
    "Whip": "other",
}


def _guess_family(species_name: str) -> str:
    """Map a CUB-200 species name to a bird family."""
    # Species names are like "Black_footed_Albatross"
    for key, family in BIRD_FAMILY.items():
        if key.lower() in species_name.lower():
            return family
    return "other"


# 1. Data loading


def load_cub200() -> tuple:
    """Load CUB-200-2011, return (dataset, labels, species_names)."""
    try:
        from datasets import concatenate_datasets, load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print("Loading CUB-200-2011 from HuggingFace...")
    ds = load_dataset("bentrevett/caltech-ucsd-birds-200-2011")
    species_names = ds["train"].features["label"].names

    # Combine train + test
    combined = concatenate_datasets([ds["train"], ds["test"]])
    labels = np.array(combined["label"])
    print(f"  {len(combined):,} images, {len(species_names)} species")

    return combined, labels, species_names


# 2. Embeddings


def extract_embeddings(
    ds,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract ResNet-50 avgpool embeddings (2048-d), cached."""
    n = len(ds)
    cache_path = CACHE_DIR / f"embeddings_resnet50_{n}.npy"
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

    def _hook(_mod, _inp, out):
        features.append(out.squeeze(-1).squeeze(-1).cpu())

    model.avgpool.register_forward_hook(_hook)

    print(f"  Extracting embeddings ({n:,} images)...")
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = ds[start:end]
        tensors = []
        for img in batch["image"]:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            tensors.append(transform(img.convert("RGB")))
        with torch.no_grad():
            model(torch.stack(tensors).to(device))
        if (start // batch_size) % 20 == 0:
            print(f"    {start * 100 // n}%", flush=True)

    embeddings = torch.cat(features).numpy()
    print(f"  Done in {time.time() - t0:.1f}s — shape: {embeddings.shape}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


# 3. Image encoding
def encode_images(ds, size: int = 96) -> list[str]:
    """Encode bird images as base64 JPEG data URIs."""
    print(f"  Encoding {len(ds):,} images for tooltips...")
    uris: list[str] = []
    for i in range(len(ds)):
        img = ds[i]["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB").resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        uris.append(f"data:image/jpeg;base64,{b64}")
    return uris


# 4. Analysis
def _clean_name(name: str) -> str:
    return name.replace("_", " ")


def analyze_birds(
    model: TMAP,
    labels: np.ndarray,
    species_names: list[str],
) -> str:
    """Generate bird species analysis report."""
    tree = model.tree_
    lines: list[str] = []
    w = lines.append

    species = np.array([species_names[l] for l in labels])
    families = np.array([_guess_family(s) for s in species])

    w(f"CUB-200 Bird Species Analysis — {len(labels):,} images, {len(species_names)} species\n")

    # 1. Family boundaries
    be_fam = boundary_edges(tree, families)
    n_edges = len(tree.edges)
    w("1. Bird family boundaries:")
    w(
        f"   Same-family edges: {n_edges - len(be_fam)} / {n_edges} "
        f"({(n_edges - len(be_fam)) / n_edges:.1%})"
    )

    cmat_f, cls_f = confusion_matrix_from_tree(tree, families)
    np.fill_diagonal(cmat_f, 0)
    upper = np.triu_indices_from(cmat_f, k=1)
    pair_counts = cmat_f[upper] + cmat_f.T[upper]
    top_idx = np.argsort(pair_counts)[::-1][:10]
    w("   Most connected families:")
    for i in top_idx:
        if pair_counts[i] == 0:
            break
        r, c = upper[0][i], upper[1][i]
        w(f"     {pair_counts[i]:4d} edges: {cls_f[r]:>15s} <-> {cls_f[c]}")
    w("")

    # 2. Species confusion
    cmat_s, cls_s = confusion_matrix_from_tree(tree, species)
    np.fill_diagonal(cmat_s, 0)
    upper_s = np.triu_indices_from(cmat_s, k=1)
    pair_counts_s = cmat_s[upper_s] + cmat_s.T[upper_s]
    top_s = np.argsort(pair_counts_s)[::-1][:15]
    w("2. Most visually similar species pairs:")
    for i in top_s:
        if pair_counts_s[i] == 0:
            break
        r, c = upper_s[0][i], upper_s[1][i]
        w(
            f"     {pair_counts_s[i]:3d} edges: {_clean_name(cls_s[r]):>30s} <-> "
            f"{_clean_name(cls_s[c])}"
        )
    w("")

    # 3. Subtree purity
    purity_f = subtree_purity(tree, families, min_size=10)
    valid_f = purity_f[~np.isnan(purity_f)]
    purity_s = subtree_purity(tree, species, min_size=10)
    valid_s = purity_s[~np.isnan(purity_s)]
    w("3. Subtree purity:")
    w(f"   By family:  mean={valid_f.mean():.3f}  median={np.median(valid_f):.3f}")
    w(f"   By species: mean={valid_s.mean():.3f}  median={np.median(valid_s):.3f}\n")

    # 4. Morphological paths
    w("4. Morphological paths between species:")
    path_pairs = [
        ("Ruby_throated_Hummingbird", "Belted_Kingfisher"),
        ("Bald_Eagle", "Osprey"),
        ("American_Crow", "Common_Raven"),
        ("House_Sparrow", "Song_Sparrow"),
        ("Laysan_Albatross", "Herring_Gull"),
        ("Red_headed_Woodpecker", "Downy_Woodpecker"),
        ("Scarlet_Tanager", "Northern_Cardinal"),
        ("Mallard", "Pelican"),
    ]
    for sp_a, sp_b in path_pairs:
        # Fuzzy match — species names might have slight variations
        idx_a = np.where([sp_a.lower() in s.lower() for s in species])[0]
        idx_b = np.where([sp_b.lower() in s.lower() for s in species])[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            w(f"   {_clean_name(sp_a)} -> {_clean_name(sp_b)}: (not found)")
            continue

        node_a, node_b = int(idx_a[0]), int(idx_b[0])
        try:
            path_nodes = tree.path(node_a, node_b)
        except IndexError:
            w(f"   {_clean_name(sp_a):>30s} -> {_clean_name(sp_b):<30s}  (disconnected)")
            continue

        path_species = species[path_nodes]
        path_families = families[path_nodes]
        unique_sp = []
        for s in path_species:
            if not unique_sp or unique_sp[-1] != s:
                unique_sp.append(_clean_name(s))

        w(
            f"   {_clean_name(sp_a):>30s} -> {_clean_name(sp_b):<30s}  "
            f"hops={len(path_nodes):3d}  species={len(set(path_species))}  "
            f"families={len(set(path_families))}"
        )
        if len(unique_sp) <= 6:
            w(f"     Route: {' -> '.join(unique_sp)}")
        else:
            route = unique_sp[:3] + ["..."] + unique_sp[-2:]
            w(f"     Route: {' -> '.join(route)}")
    w("")

    return "\n".join(lines)


# 5. Visualization


def create_visualization(
    model: TMAP,
    labels: np.ndarray,
    species_names: list[str],
    image_uris: list[str],
):
    viz = model.to_tmapviz()
    viz.title = f"CUB-200:  {len(labels):,} Bird Images"

    species = [species_names[l] for l in labels]
    families = [_guess_family(s) for s in species]

    viz.add_color_layout("family", families, categorical=True, color="tab20")

    viz.add_images(image_uris, tooltip_size=100)
    viz.add_label("species", [_clean_name(s) for s in species])

    return viz


# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="CUB-200 Birds TMAP")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Load
    ds, labels, species_names = load_cub200()

    # Embeddings
    print("Extracting ResNet-50 embeddings...")
    embeddings = extract_embeddings(ds, args.batch_size, device)

    # Build TMAP
    print(f"Building TMAP (metric='cosine', k={args.k})...")
    t0 = time.time()
    model = TMAP(
        metric="cosine",
        n_neighbors=args.k,
        layout_iterations=1000,
        seed=42,
    ).fit(embeddings.astype(np.float32))
    print(f"  Done in {time.time() - t0:.1f}s")

    # Analysis
    report = analyze_birds(model, labels, species_names)
    report_path = OUTPUT_DIR / "cub200_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {report_path}")
    print("\n" + report)

    # Visualization
    print("Encoding images for tooltips...")
    image_uris = encode_images(ds)

    print("Building visualization...")
    viz = create_visualization(model, labels, species_names, image_uris)
    html_path = viz.write_html(OUTPUT_DIR / "cub200_tmap")
    print(f"HTML saved to {html_path}")

    if args.serve:
        print(f"Serving on http://127.0.0.1:{args.port}")
        viz.serve(port=args.port)


if __name__ == "__main__":
    main()
