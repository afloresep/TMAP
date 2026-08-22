"""EMNIST handwritten characters — why OCR confuses letters and digits.

Build a TMAP of handwritten characters (digits + letters) to reveal
visual similarity bridges: 0 ↔ O ↔ Q ↔ D, or 1 ↔ l ↔ I. Each step
along the path shows a tiny handwriting variation, and the endpoints
are different characters that happen to look alike.

The tree explains exactly why OCR systems confuse certain characters —
you can trace the path and see the gradual morphing at each step.

Outputs
-------
examples/emnist_tmap.html          Interactive TMAP with character tooltips
examples/emnist_report.txt         Character confusion analysis

Data
----
Downloads EMNIST via torchvision (~500 MB, cached).

Usage
-----
    python examples/images/emnist_characters_tmap.py
    python examples/images/emnist_characters_tmap.py --max-images 30000
    python examples/images/emnist_characters_tmap.py --serve
"""

from __future__ import annotations

import argparse
import base64
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets

from tmap import TMAP
from tmap.graph.analysis import (
    boundary_edges,
    confusion_matrix_from_tree,
    subtree_purity,
)

# Scripts live in a subfolder; data, caches and outputs stay at examples/.
EXAMPLES_DIR = Path(__file__).resolve().parents[1]

CACHE_DIR = EXAMPLES_DIR / "data" / "emnist_cache"
OUTPUT_DIR = EXAMPLES_DIR

# EMNIST balanced split: 47 classes
# 0-9 = digits, 10-35 = A-Z, 36-46 = select lowercase (a,b,d,e,f,g,h,n,q,r,t)
CHAR_MAP_BALANCED = list("0123456789") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("abdefghnqrt")

# Group characters by visual shape for analysis
SHAPE_GROUP = {
    "0": "round",
    "O": "round",
    "Q": "round",
    "D": "round",
    "C": "round",
    "G": "round",
    "1": "vertical",
    "I": "vertical",
    "l": "vertical",
    "t": "vertical",
    "7": "angular",
    "T": "angular",
    "Y": "angular",
    "V": "angular",
    "2": "curved",
    "S": "curved",
    "5": "curved",
    "Z": "curved",
    "3": "curved",
    "8": "looped",
    "B": "looped",
    "6": "looped",
    "9": "looped",
    "b": "looped",
    "d": "looped",
    "q": "looped",
    "g": "looped",
    "4": "angular",
    "A": "angular",
    "H": "angular",
    "K": "angular",
    "M": "angular",
    "N": "angular",
    "W": "angular",
    "E": "angular",
    "F": "angular",
    "L": "angular",
    "P": "mixed",
    "R": "mixed",
    "J": "mixed",
    "U": "mixed",
    "X": "angular",
    "a": "round",
    "e": "round",
    "f": "vertical",
    "h": "vertical",
    "n": "curved",
    "r": "curved",
}


# 1. Data loading


def load_emnist(max_images: int | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load EMNIST balanced split, return (images_flat, labels, char_names)."""
    data_dir = EXAMPLES_DIR / "data" / "emnist"

    print("Loading EMNIST (balanced split)...")
    ds_train = datasets.EMNIST(root=str(data_dir), split="balanced", train=True, download=True)
    ds_test = datasets.EMNIST(root=str(data_dir), split="balanced", train=False, download=True)

    # Combine train + test
    all_images = []
    all_labels = []
    for ds in [ds_train, ds_test]:
        for img, label in ds:
            # EMNIST images are transposed — fix orientation
            img = img.transpose(Image.TRANSPOSE)
            all_images.append(np.array(img, dtype=np.float32).flatten())
            all_labels.append(label)

    images = np.stack(all_images)
    labels = np.array(all_labels, dtype=np.int64)

    print(f"  {len(images):,} images, {len(np.unique(labels))} classes, {images.shape[1]}D")

    # Subsample if needed
    if max_images and len(images) > max_images:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(images), max_images, replace=False)
        idx.sort()
        images = images[idx]
        labels = labels[idx]
        print(f"  Subsampled to {len(images):,}")

    return images, labels, CHAR_MAP_BALANCED


# 2. Image encoding for tooltips


def encode_char_images(
    images_flat: np.ndarray,
    size: int = 48,
) -> list[str]:
    """Convert flat 784-d vectors back to upscaled images for tooltips."""
    print(f"  Encoding {len(images_flat):,} character images ({size}x{size})...")
    uris: list[str] = []
    side = int(np.sqrt(images_flat.shape[1]))
    for vec in images_flat:
        arr = vec.reshape(side, side).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        img = img.resize((size, size), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        uris.append(f"data:image/png;base64,{b64}")
    return uris


# 3. Analysis


def analyze_characters(
    model: TMAP,
    labels: np.ndarray,
    char_names: list[str],
) -> str:
    """Generate character confusion analysis report."""
    tree = model.tree_
    lines: list[str] = []
    w = lines.append

    chars = np.array([char_names[l] for l in labels])
    groups = np.array([SHAPE_GROUP.get(c, "other") for c in chars])

    n_classes = len(np.unique(labels))
    w("EMNIST Character Confusion Analysis")
    w(f"  {len(labels):,} images, {n_classes} character classes\n")

    # 1. Character boundaries
    be = boundary_edges(tree, chars)
    n_edges = len(tree.edges)
    w("1. Character boundaries:")
    w(
        f"   Same-character edges: {n_edges - len(be)} / {n_edges} "
        f"({(n_edges - len(be)) / n_edges:.1%})"
    )

    # Shape group boundaries
    be_g = boundary_edges(tree, groups)
    w(
        f"   Same-shape-group edges: {n_edges - len(be_g)} / {n_edges} "
        f"({(n_edges - len(be_g)) / n_edges:.1%})\n"
    )

    # 2. Most confused character pairs
    cmat, classes = confusion_matrix_from_tree(tree, chars)
    np.fill_diagonal(cmat, 0)
    upper = np.triu_indices_from(cmat, k=1)
    pair_counts = cmat[upper] + cmat.T[upper]
    top_idx = np.argsort(pair_counts)[::-1][:20]
    w("2. Most confused character pairs (shared tree edges):")
    for i in top_idx:
        if pair_counts[i] == 0:
            break
        r, c = upper[0][i], upper[1][i]
        w(f"     {pair_counts[i]:4d} edges: '{classes[r]}' <-> '{classes[c]}'")
    w("")

    # 3. Subtree purity
    purity = subtree_purity(tree, chars, min_size=20)
    valid = purity[~np.isnan(purity)]
    purity_g = subtree_purity(tree, groups, min_size=20)
    valid_g = purity_g[~np.isnan(purity_g)]
    w("3. Subtree purity:")
    w(f"   By character:   mean={valid.mean():.3f}  median={np.median(valid):.3f}")
    w(f"   By shape group: mean={valid_g.mean():.3f}  median={np.median(valid_g):.3f}\n")

    # 4. OCR confusion paths
    w("4. OCR confusion paths (why these characters get mixed up):")
    path_pairs = [
        ("0", "O"),  # digit vs letter, identical
        ("0", "D"),  # round shapes
        ("1", "I"),  # vertical strokes
        ("5", "S"),  # similar curves
        ("8", "B"),  # looped
        ("6", "b"),  # mirror
        ("9", "q"),  # mirror
        ("2", "Z"),  # similar shape
        ("Q", "9"),  # tail similarity
        ("W", "M"),  # inversions
        ("g", "9"),  # looped tail
        ("A", "4"),  # angular, pointed top
    ]
    for ch_a, ch_b in path_pairs:
        idx_a = np.where(chars == ch_a)[0]
        idx_b = np.where(chars == ch_b)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            continue

        node_a, node_b = int(idx_a[0]), int(idx_b[0])
        try:
            path_nodes = tree.path(node_a, node_b)
        except IndexError:
            w(f"   '{ch_a}' -> '{ch_b}': (disconnected)")
            continue

        path_chars = chars[path_nodes]
        unique = []
        for ch in path_chars:
            if not unique or unique[-1] != ch:
                unique.append(ch)

        w(
            f"   '{ch_a}' -> '{ch_b}': {len(path_nodes):3d} hops, "
            f"characters crossed: {len(set(path_chars))}"
        )
        if len(unique) <= 10:
            route = " -> ".join(f"'{c}'" for c in unique)
        else:
            route = " -> ".join(f"'{c}'" for c in unique[:5])
            route += " -> ... -> " + " -> ".join(f"'{c}'" for c in unique[-3:])
        w(f"     Route: {route}")
    w("")

    # 5. Digit vs letter separation
    is_digit = np.array(["digit" if l < 10 else "letter" for l in labels])
    be_dl = boundary_edges(tree, is_digit)
    w("5. Digit vs letter separation:")
    w(
        f"   Same-type edges: {n_edges - len(be_dl)} / {n_edges} "
        f"({(n_edges - len(be_dl)) / n_edges:.1%})"
    )
    w(f"   Cross-type edges: {len(be_dl)} ({len(be_dl) / n_edges:.1%})")

    return "\n".join(lines)


# 4. Visualization


def create_visualization(
    model: TMAP,
    labels: np.ndarray,
    char_names: list[str],
    image_uris: list[str],
):
    """Build TmapViz with character coloring and image tooltips."""
    viz = model.to_tmapviz()
    viz.title = f"EMNIST — {len(labels):,} Handwritten Characters"

    chars = [char_names[l] for l in labels]
    groups = [SHAPE_GROUP.get(c, "other") for c in chars]
    char_type = ["digit" if l < 10 else "letter" for l in labels]

    viz.add_color_layout("type", char_type, categorical=True, color="Set1")
    viz.add_color_layout("shape group", groups, categorical=True, color="tab10")

    viz.add_images(image_uris, tooltip_size=48)
    viz.add_label("character", [f"'{c}'" for c in chars])

    return viz


# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="EMNIST Character TMAP")
    parser.add_argument("--max-images", type=int, default=20000)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    # Load
    images, labels, char_names = load_emnist(
        args.max_images if args.max_images > 0 else None,
    )

    # Build TMAP (raw pixels + cosine, like MNIST example)
    print(f"Building TMAP (metric='cosine', k={args.k})...")
    t0 = time.time()
    model = TMAP(
        metric="cosine",
        n_neighbors=args.k,
        layout_iterations=1000,
        seed=42,
    ).fit(images)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Analysis
    report = analyze_characters(model, labels, char_names)
    report_path = OUTPUT_DIR / "emnist_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {report_path}")
    print("\n" + report)

    # Visualization
    print("Encoding character images...")
    image_uris = encode_char_images(images)

    print("Building visualization...")
    viz = create_visualization(model, labels, char_names, image_uris)
    html_path = viz.write_html(OUTPUT_DIR / "emnist_tmap")
    print(f"HTML saved to {html_path}")

    if args.serve:
        print(f"Serving on http://127.0.0.1:{args.port}")
        viz.serve(port=args.port)


if __name__ == "__main__":
    main()
