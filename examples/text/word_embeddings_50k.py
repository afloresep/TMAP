"""50K word embedding TMAP — semantic map of English vocabulary.

Embed 50,000 English nouns (from WordNet) with a sentence-transformer,
build a TMAP with cosine metric, and explore the natural semantic
organization of the English language.

WordNet provides automatic semantic categories (animal, food, artifact,
plant, person, etc.) for coloring the map.

Outputs
-------
examples/word_embeddings_50k_tmap.html     Interactive TMAP
examples/word_embeddings_50k_report.txt    Analysis report

Usage
-----
    python examples/text/word_embeddings_50k.py
    python examples/text/word_embeddings_50k.py --n 50000
    python examples/text/word_embeddings_50k.py --serve

Requirements
------------
    pip install sentence-transformers nltk
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tmap import TMAP
from tmap.graph.analysis import (
    boundary_edges,
    confusion_matrix_from_tree,
    subtree_purity,
)

# Scripts live in a subfolder; data, caches and outputs stay at examples/.
EXAMPLES_DIR = Path(__file__).resolve().parents[1]

CACHE_DIR = EXAMPLES_DIR / "data" / "word50k_cache"
OUTPUT_DIR = EXAMPLES_DIR

# Readable labels for WordNet lexnames
LEXNAME_LABELS = {
    # Nouns
    "noun.Tops": "general",
    "noun.act": "action",
    "noun.animal": "animal",
    "noun.artifact": "artifact",
    "noun.attribute": "attribute",
    "noun.body": "body",
    "noun.cognition": "cognition",
    "noun.communication": "communication",
    "noun.event": "event",
    "noun.feeling": "feeling",
    "noun.food": "food",
    "noun.group": "group",
    "noun.location": "location",
    "noun.motive": "motive",
    "noun.object": "object",
    "noun.person": "person",
    "noun.phenomenon": "phenomenon",
    "noun.plant": "plant",
    "noun.possession": "possession",
    "noun.process": "process",
    "noun.quantity": "quantity",
    "noun.relation": "relation",
    "noun.shape": "shape",
    "noun.state": "state",
    "noun.substance": "substance",
    "noun.time": "time",
    # Verbs
    "verb.body": "v:body",
    "verb.change": "v:change",
    "verb.cognition": "v:thinking",
    "verb.communication": "v:speaking",
    "verb.competition": "v:competing",
    "verb.consumption": "v:consuming",
    "verb.contact": "v:touching",
    "verb.creation": "v:creating",
    "verb.emotion": "v:emotion",
    "verb.motion": "v:motion",
    "verb.perception": "v:perception",
    "verb.possession": "v:having",
    "verb.social": "v:social",
    "verb.stative": "v:being",
    "verb.weather": "v:weather",
    # Adjectives & Adverbs
    "adj.all": "adjective",
    "adj.pert": "adj:relational",
    "adj.ppl": "adj:participle",
    "adv.all": "adverb",
}


# ---------------------------------------------------------------------------
# 1. Vocabulary from WordNet
# ---------------------------------------------------------------------------


def load_wordnet_vocabulary(n: int) -> tuple[list[str], list[str]]:
    """Extract n single words from WordNet (all POS) with semantic categories."""
    import nltk

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn

    print(f"Extracting vocabulary from WordNet (target: {n:,} words)...")
    t0 = time.time()

    # Collect (word, category) pairs. For words in multiple categories,
    # use the most common synset's category (first synset = most frequent).
    word_cat: dict[str, str] = {}
    for synset in wn.all_synsets():
        cat = LEXNAME_LABELS.get(synset.lexname(), synset.lexname())
        for lemma in synset.lemmas():
            name = lemma.name().lower()
            if "_" not in name and name.isalpha() and 3 <= len(name) <= 20:
                if name not in word_cat:
                    word_cat[name] = cat

    # Sort by word length then alphabetically for determinism, take first n
    all_words = sorted(word_cat.keys(), key=lambda w: (len(w), w))
    selected = all_words[:n]

    words = selected
    categories = [word_cat[w] for w in words]

    elapsed = time.time() - t0
    print(f"  {len(words):,} words extracted in {elapsed:.1f}s")

    # Category distribution
    from collections import Counter

    counts = Counter(categories)
    print(f"  {len(counts)} categories:")
    for cat, count in counts.most_common():
        print(f"    {cat:20s} {count:6,}")

    return words, categories


# ---------------------------------------------------------------------------
# 2. Embedding
# ---------------------------------------------------------------------------


def compute_embeddings(
    words: list[str],
    model_name: str,
    batch_size: int = 512,
) -> np.ndarray:
    """Embed words with sentence-transformers, cached to disk."""
    cache_path = CACHE_DIR / f"embeddings_{len(words)}_{model_name.replace('/', '_')}.npy"
    if cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    print(f"  Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  Encoding {len(words):,} words (batch_size={batch_size})...")
    t0 = time.time()
    embeddings = model.encode(
        words,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — shape: {embeddings.shape}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


# ---------------------------------------------------------------------------
# 3. TMAP
# ---------------------------------------------------------------------------


def build_tmap(embeddings: np.ndarray, k: int) -> TMAP:
    """Build TMAP with cosine metric. Stores index for later querying."""
    print(f"Fitting TMAP (metric='cosine', k={k}, n={len(embeddings):,})...")
    t0 = time.time()
    model = TMAP(
        metric="cosine",
        n_neighbors=k,
        layout_iterations=1000,
        seed=42,
        store_index=True,
    ).fit(embeddings.astype(np.float32))
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return model


# ---------------------------------------------------------------------------
# 4. Analysis
# ---------------------------------------------------------------------------


def analyze(
    model: TMAP,
    words: list[str],
    categories: list[str],
) -> str:
    """Generate analysis report."""
    tree = model.tree_
    cat_arr = np.array(categories)
    lines: list[str] = []
    w = lines.append

    w(f"Word Embedding TMAP — {len(words):,} words, {len(set(categories))} categories\n")

    # 1. Category clustering
    be = boundary_edges(tree, cat_arr)
    w("1. Category boundaries:")
    w(
        f"   Same-category edges: {len(tree.edges) - len(be):,} / {len(tree.edges):,} "
        f"({1 - len(be) / len(tree.edges):.1%})"
    )
    w(f"   Cross-category edges: {len(be):,} ({len(be) / len(tree.edges):.1%})\n")

    # 2. Subtree purity
    purity = subtree_purity(tree, cat_arr, min_size=20)
    valid = purity[~np.isnan(purity)]
    w(f"2. Subtree purity: mean={valid.mean():.3f}  median={np.median(valid):.3f}\n")

    # 3. Most connected category pairs
    cmat, classes = confusion_matrix_from_tree(tree, cat_arr)
    np.fill_diagonal(cmat, 0)
    upper = np.triu_indices_from(cmat, k=1)
    pair_counts = cmat[upper]
    top_idx = np.argsort(pair_counts)[::-1][:15]
    w("3. Most connected category pairs:")
    for i in top_idx:
        if pair_counts[i] == 0:
            break
        r, c = upper[0][i], upper[1][i]
        w(f"   {pair_counts[i]:5d} edges: {classes[r]:>15s}  <->  {classes[c]}")
    w("")

    # 4. Sample paths
    word_to_idx = {w: i for i, w in enumerate(words)}
    paths_to_trace = [
        ("dog", "cat"),
        ("dog", "wolf"),
        ("apple", "banana"),
        ("guitar", "piano"),
        ("doctor", "nurse"),
        ("mountain", "ocean"),
        ("happy", "sad"),
        ("sword", "shield"),
        ("rain", "snow"),
        ("dog", "guitar"),
        ("brain", "computer"),
    ]
    w("4. Semantic paths:")
    for word_a, word_b in paths_to_trace:
        if word_a not in word_to_idx or word_b not in word_to_idx:
            continue
        idx_a = word_to_idx[word_a]
        idx_b = word_to_idx[word_b]
        path_nodes = tree.path(idx_a, idx_b)
        path_words = [words[n] for n in path_nodes]
        if len(path_words) <= 10:
            path_str = " → ".join(path_words)
        else:
            path_str = (
                " → ".join(path_words[:5])
                + f" → [{len(path_words) - 7} more] → "
                + " → ".join(path_words[-2:])
            )
        w(f"   {word_a} → {word_b} ({len(path_words)} hops): {path_str}")
    w("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------


def create_visualization(
    model: TMAP,
    words: list[str],
    categories: list[str],
):
    viz = model.to_tmapviz()
    viz.title = f"English Vocabulary — {len(words):,} Words"
    viz.add_color_layout("category", categories, categorical=True, color="tab20")
    viz.add_label("word", words)
    return viz


# ---------------------------------------------------------------------------
# 6. Playground — interactive word query
# ---------------------------------------------------------------------------

MODEL_SAVE_PATH = CACHE_DIR / "word_tmap.model"
WORDS_SAVE_PATH = CACHE_DIR / "word_list.npy"
CATS_SAVE_PATH = CACHE_DIR / "word_categories.npy"


def playground(model_name: str, k_show: int = 10) -> None:
    """Interactive REPL: type a word (or phrase), see where it lands."""
    from sentence_transformers import SentenceTransformer

    print("Loading saved model...")
    tmap_model = TMAP.load(MODEL_SAVE_PATH)
    words = np.load(WORDS_SAVE_PATH, allow_pickle=True).tolist()
    categories = np.load(CATS_SAVE_PATH, allow_pickle=True).tolist()
    word_to_idx = {w: i for i, w in enumerate(words)}

    print(f"Loading encoder: {model_name}")
    encoder = SentenceTransformer(model_name)

    print(f"\nPlayground ready — {len(words):,} words in map")
    print("Type a word or phrase to see where it would land.")
    print("Type two words separated by ' -> ' to trace a path.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() == "quit":
            break

        # Path mode: "word_a -> word_b"
        if " -> " in query:
            parts = [p.strip() for p in query.split(" -> ", 1)]
            idx_a = word_to_idx.get(parts[0].lower())
            idx_b = word_to_idx.get(parts[1].lower())
            if idx_a is None:
                print(f"  '{parts[0]}' not in vocabulary")
                continue
            if idx_b is None:
                print(f"  '{parts[1]}' not in vocabulary")
                continue
            path_nodes = tmap_model.tree_.path(idx_a, idx_b)
            path_words = [words[n] for n in path_nodes]
            print(f"  Path ({len(path_words)} hops):")
            print(f"  {' → '.join(path_words)}")
            print()
            continue

        # Query mode: embed and find neighbors
        emb = encoder.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        indices, distances = tmap_model.kneighbors(emb)
        idx_row = indices[0]
        dist_row = distances[0]

        # Check if the query itself is in the vocabulary
        in_vocab = query.lower() in word_to_idx

        print(f'  Query: "{query}"' + (" (in vocabulary)" if in_vocab else " (not in vocabulary)"))
        print("  Nearest neighbors on the map:")
        for rank, (idx, dist) in enumerate(zip(idx_row, dist_row)):
            if idx < 0:
                break
            word = words[idx]
            cat = categories[idx]
            coord = tmap_model.embedding_[idx]
            marker = " <--" if word == query.lower() else ""
            print(
                f"    {rank + 1:2d}. {word:25s} [{cat:>15s}]  "
                f"dist={dist:.4f}  pos=({coord[0]:.1f}, {coord[1]:.1f}){marker}"
            )

        # Show where it would be placed (centroid of top neighbors)
        valid = idx_row[idx_row >= 0][:5]
        if len(valid) > 0:
            centroid = tmap_model.embedding_[valid].mean(axis=0)
            print(f"  Approximate position: ({centroid[0]:.1f}, {centroid[1]:.1f})")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="50K word embedding TMAP")
    parser.add_argument("--n", type=int, default=50000, help="Number of words")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--k", type=int, default=20, help="Number of neighbors")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument(
        "--playground",
        action="store_true",
        help="Interactive mode: query words against the saved model",
    )
    args = parser.parse_args()

    if args.playground:
        if not MODEL_SAVE_PATH.exists():
            print(f"No saved model at {MODEL_SAVE_PATH}")
            print("Run without --playground first to build and save the model.")
            return
        playground(args.model)
        return

    words, categories = load_wordnet_vocabulary(args.n)
    embeddings = compute_embeddings(words, args.model)
    model = build_tmap(embeddings, args.k)

    # Save model + word list for playground mode
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    np.save(WORDS_SAVE_PATH, np.array(words, dtype=object))
    np.save(CATS_SAVE_PATH, np.array(categories, dtype=object))
    print(f"Model saved to {MODEL_SAVE_PATH}")

    report = analyze(model, words, categories)
    report_path = OUTPUT_DIR / "word_embeddings_50k_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {report_path}")
    print("\n" + report)

    print("Building visualization...")
    viz = create_visualization(model, words, categories)
    html_path = viz.write_html(OUTPUT_DIR / "word_embeddings_50k_tmap")
    print(f"HTML saved to {html_path}")

    if args.serve:
        print(f"Serving on http://127.0.0.1:{args.port}")
        viz.serve(port=args.port)


if __name__ == "__main__":
    main()
