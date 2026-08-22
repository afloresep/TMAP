"""Oxford-IIIT Pets classifier audit with TMAP.

Extract ResNet-50 embeddings from the Oxford-IIIT Pets dataset (37 breeds),
train a linear probe, and build a TMAP to inspect how the classifier
organizes breeds: where boundaries lie, which breeds get confused, and
where failures cluster.

Outputs
-------
examples/pets_tmap.html          Interactive TMAP with 5 color layers + image tooltips
examples/pets_analysis_report.txt   Text report with 6 analysis sections

Usage
-----
    python examples/images/pet_breed_audit.py                    # auto device
    python examples/images/pet_breed_audit.py --device cuda      # force GPU
    python examples/images/pet_breed_audit.py --epochs 15        # longer probe training

Bring your own data
-------------------
Replace ``_extract_embeddings`` with your own feature extractor and
``_train_probe`` with your classifier.  The analysis and visualization
sections work on any (embeddings, true_labels, predicted_labels, confidences)
tuple — no architecture-specific code required.
"""

from __future__ import annotations

import argparse
import base64
import io
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
    edge_delta,
    path_properties,
    subtree_purity,
)
from tmap.visualization import TmapViz

# Scripts live in a subfolder; data, caches and outputs stay at examples/.
EXAMPLES_DIR = Path(__file__).resolve().parents[1]


CACHE_DIR = EXAMPLES_DIR / "data" / "pets_cache"
OUTPUT_DIR = EXAMPLES_DIR


def _extract_embeddings(
    split: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ResNet-50 avgpool embeddings (2048-d), cached to disk."""
    cache_emb = CACHE_DIR / f"{split}_embeddings.npy"
    cache_lbl = CACHE_DIR / f"{split}_labels.npy"
    if cache_emb.exists() and cache_lbl.exists():
        return np.load(cache_emb), np.load(cache_lbl)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.OxfordIIITPet(
        root=str(EXAMPLES_DIR / "data" / "oxford-iiit-pet"),
        split=split,
        target_types="category",
        download=True,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval().to(device)
    # Hook avgpool output (2048-d)
    features: list[torch.Tensor] = []

    def _hook(_mod: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        features.append(out.squeeze(-1).squeeze(-1).cpu())

    model.avgpool.register_forward_hook(_hook)

    all_labels: list[int] = []
    with torch.no_grad():
        for imgs, labels in loader:
            model(imgs.to(device))
            all_labels.extend(labels.tolist())

    embeddings = torch.cat(features).numpy()
    labels_arr = np.array(all_labels, dtype=np.int64)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_emb, embeddings)
    np.save(cache_lbl, labels_arr)
    return embeddings, labels_arr


def _train_probe(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    n_classes: int,
    epochs: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a linear classifier on frozen features, return (preds, confidences) on test."""
    # L2-normalize — critical for linear probe on raw avgpool features
    X_train = torch.tensor(train_emb, dtype=torch.float32, device=device)
    X_train = nn.functional.normalize(X_train, dim=1)
    y_train = torch.tensor(train_labels, dtype=torch.long, device=device)
    X_test = torch.tensor(test_emb, dtype=torch.float32, device=device)
    X_test = nn.functional.normalize(X_test, dim=1)

    probe = nn.Linear(train_emb.shape[1], n_classes).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    probe.train()
    n = X_train.shape[0]
    batch_size = 256
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = probe(X_train[idx])
            loss = nn.functional.cross_entropy(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    probe.eval()
    with torch.no_grad():
        test_logits = probe(X_test)
        probs = torch.softmax(test_logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()
        confidences = probs.max(dim=1).values.cpu().numpy()

    return preds, confidences


def _encode_images(split: str, size: int = 128) -> list[str]:
    """Load raw images, resize to *size*px, return base64 JPEG data URIs."""
    dataset = datasets.OxfordIIITPet(
        root=str(EXAMPLES_DIR / "data" / "oxford-iiit-pet"),
        split=split,
        target_types="category",
        download=False,
    )
    uris: list[str] = []
    for img, _ in dataset:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB").resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        uris.append(f"data:image/jpeg;base64,{b64}")
    return uris


def _build_tmap(embeddings: np.ndarray) -> TMAP:
    """Build a TMAP from cosine similarity on embeddings."""
    model = TMAP(
        n_neighbors=15,
        metric="cosine",
        layout_iterations=1000,
    )
    model.fit(embeddings.astype(np.float32))
    return model


def _find_best_failure_paths(
    tree,
    true_labels: np.ndarray,
    preds: np.ndarray,
    confidences: np.ndarray,
    class_names: list[str],
    top_k: int = 5,
) -> list[dict]:
    """Find the most informative failure paths in the tree.

    A failure path connects a misclassified point to its nearest correctly
    classified same-class neighbor, showing how the model's confidence
    changes along the way.
    """
    wrong = np.where(preds != true_labels)[0]
    if len(wrong) == 0:
        return []

    correct_mask = preds == true_labels
    paths: list[dict] = []

    for idx in wrong:
        true_cls = true_labels[idx]
        # Find nearest correct same-class node via BFS
        same_class_correct = np.where((true_labels == true_cls) & correct_mask)[0]
        if len(same_class_correct) == 0:
            continue

        # Use tree path length as distance proxy — pick shortest
        best_target = None
        best_len = float("inf")
        for candidate in same_class_correct[:20]:  # cap search
            p = tree.path(idx, candidate)
            if len(p) < best_len:
                best_len = len(p)
                best_target = candidate

        if best_target is None:
            continue

        conf_along = path_properties(tree, idx, best_target, confidences)
        paths.append(
            {
                "from": idx,
                "to": best_target,
                "true_class": class_names[true_cls],
                "pred_class": class_names[preds[idx]],
                "path_length": int(best_len),
                "conf_start": float(confidences[idx]),
                "conf_end": float(confidences[best_target]),
                "conf_min": float(conf_along.min()),
            }
        )

    paths.sort(key=lambda p: -p["path_length"])
    return paths[:top_k]


def _analyze_tree(
    model: TMAP,
    class_names: list[str],
    true_labels: np.ndarray,
    preds: np.ndarray,
    confidences: np.ndarray,
) -> str:
    """Generate a 6-section text report from the TMAP tree."""
    tree = model.tree_
    lines: list[str] = []
    w = lines.append

    acc = (preds == true_labels).mean()
    w(
        f"Oxford-IIIT Pets Classifier Audit  "
        f"({len(true_labels)} test images, {len(class_names)} breeds)"
    )
    w(f"Overall accuracy: {acc:.1%}\n")

    # 1. Boundary edges
    be = boundary_edges(tree, preds)
    w(f"1. Boundary edges: {len(be)} / {len(tree.edges)} edges ({len(be) / len(tree.edges):.1%})")
    w("   Edges where neighboring points have different predicted breeds.\n")

    # 2. Confusion matrix
    cmat, classes = confusion_matrix_from_tree(tree, preds)
    np.fill_diagonal(cmat, 0)
    upper = np.triu_indices_from(cmat, k=1)
    pair_counts = cmat[upper]
    top_idx = np.argsort(pair_counts)[::-1][:10]
    w("2. Most confused breed pairs (by tree boundary edges):")
    for i in top_idx:
        if pair_counts[i] == 0:
            break
        r, c = upper[0][i], upper[1][i]
        w(
            f"   {class_names[classes[r]]:>30s}  <->  "
            f"{class_names[classes[c]]:<30s}  ({pair_counts[i]} edges)"
        )
    w("")

    # 3. Confidence gradients
    deltas = edge_delta(tree, confidences)
    w("3. Confidence gradients across edges:")
    w(f"   Mean delta: {deltas.mean():.4f}   Max delta: {deltas.max():.4f}")
    steep = np.where(deltas > np.percentile(deltas, 99))[0]
    w(f"   Steep edges (>99th pct): {len(steep)}\n")

    # 4. Subtree purity
    purity = subtree_purity(tree, preds, min_size=10)
    valid = purity[~np.isnan(purity)]
    w("4. Subtree purity (predicted labels, min_size=10):")
    w(f"   Mean: {valid.mean():.3f}   Median: {np.median(valid):.3f}   Min: {valid.min():.3f}\n")

    # 5. Failure paths
    paths = _find_best_failure_paths(tree, true_labels, preds, confidences, class_names)
    w(f"5. Top failure paths ({len(paths)} shown):")
    for p in paths:
        w(
            f"   Node {p['from']} ({p['true_class']}, "
            f"predicted {p['pred_class']}, conf={p['conf_start']:.2f})"
        )
        w(
            f"     -> Node {p['to']} (correct, conf={p['conf_end']:.2f})  "
            f"path_len={p['path_length']}  min_conf={p['conf_min']:.2f}"
        )
    w("")

    # 6. Mislabel candidates
    wrong = np.where(preds != true_labels)[0]
    if len(wrong) > 0:
        # High-confidence misclassifications — model disagrees with label
        wrong_conf = confidences[wrong]
        top_wrong = wrong[np.argsort(wrong_conf)[::-1][:10]]
        w("6. Mislabel candidates (high-confidence misclassifications):")
        for idx in top_wrong:
            w(
                f"   Index {idx:5d}  true={class_names[true_labels[idx]]:>25s}  "
                f"pred={class_names[preds[idx]]:<25s}  conf={confidences[idx]:.3f}"
            )
    else:
        w("6. No misclassifications found.")
    w("")

    return "\n".join(lines)


def _create_visualization(
    model: TMAP,
    class_names: list[str],
    true_labels: np.ndarray,
    preds: np.ndarray,
    confidences: np.ndarray,
    purity: np.ndarray,
    image_uris: list[str],
) -> TmapViz:
    """Build a TmapViz with 5 color layers and image tooltips."""
    viz = model.to_tmapviz()
    viz.title = "Oxford Pets — Breed Classifier Audit"

    breed_names = [class_names[i] for i in true_labels]
    pred_names = [class_names[i] for i in preds]
    correct = ["correct" if p == t else "wrong" for p, t in zip(preds, true_labels)]

    viz.add_color_layout("breed", breed_names, categorical=True, color="tab20")
    viz.add_color_layout("predicted breed", pred_names, categorical=True, color="tab20")
    viz.add_color_layout("correct?", correct, categorical=True, color="Set1")
    viz.add_color_layout("confidence", confidences.tolist(), categorical=False, color="RdYlGn")
    viz.add_color_layout("subtree purity", purity.tolist(), categorical=False, color="RdYlBu")
    viz.add_images(image_uris, tooltip_size=128)

    return viz


def main() -> None:
    parser = argparse.ArgumentParser(description="Oxford Pets classifier audit with TMAP")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Extract embeddings
    print("Extracting train embeddings...")
    train_emb, train_labels = _extract_embeddings("trainval", args.batch_size, device)
    print("Extracting test embeddings...")
    test_emb, test_labels = _extract_embeddings("test", args.batch_size, device)

    # Class names
    ds = datasets.OxfordIIITPet(
        root=str(EXAMPLES_DIR / "data" / "oxford-iiit-pet"),
        split="test",
        target_types="category",
        download=False,
    )
    class_names = [name.replace("_", " ").title() for name in ds.classes]
    n_classes = len(class_names)
    print(f"{n_classes} breeds, {len(test_emb)} test images")

    # Train linear probe
    print(f"Training linear probe ({args.epochs} epochs)...")
    preds, confidences = _train_probe(
        train_emb,
        train_labels,
        test_emb,
        test_labels,
        n_classes,
        args.epochs,
        device,
    )
    acc = (preds == test_labels).mean()
    print(f"Probe accuracy: {acc:.1%}")

    # Build TMAP on test embeddings
    print("Building TMAP...")
    model = _build_tmap(test_emb)

    # Analysis
    purity = subtree_purity(model.tree_, preds, min_size=10)
    report = _analyze_tree(model, class_names, test_labels, preds, confidences)
    report_path = OUTPUT_DIR / "pets_analysis_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to {report_path}")

    # Encode images
    print("Encoding test images for tooltips...")
    image_uris = _encode_images("test", size=128)

    # Visualization
    print("Building visualization...")
    viz = _create_visualization(
        model,
        class_names,
        test_labels,
        preds,
        confidences,
        purity,
        image_uris,
    )
    html_path = viz.write_html(OUTPUT_DIR / "pets_tmap")
    print(f"HTML saved to {html_path}")


if __name__ == "__main__":
    main()
