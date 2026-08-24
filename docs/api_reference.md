# TMAP API Reference

This is a practical reference for the public API. For a guided workflow, start with [molecule_tutorial.md](molecule_tutorial.md).

## `TMAP`

High-level estimator for building a tree-shaped 2D map.

### Constructor

```python
TMAP(
    n_neighbors=20,
    metric="jaccard",
    n_permutations=512,
    kc=50,
    seed=42,
    layout_iterations=1000,
    layout_config=None,
    store_index=False,
)
```

### Key Methods

| Method | What it does |
|--------|---------------|
| `fit(X)` | Build the graph, tree, and 2D embedding |
| `fit_transform(X)` | Fit and return `(x, y, s, t)` |
| `kneighbors(X)` | Query nearest fitted neighbors for new points without placement or mutation |
| `transform(X_new)` | Place new points on the existing map without changing the model |
| `add_points(X_new)` | Add new points into the fitted model |
| `to_tmapviz()` | Create a `TmapViz` object for notebook or HTML output |
| `to_html(path)` | Write an HTML visualization |
| `serve(port=8050)` | Start a local HTTP server for the visualization |
| `save(path)` / `load(path)` | Save or load the fitted model |

### Key Attributes

| Attribute | Meaning |
|-----------|---------|
| `embedding_` | 2D coordinates, shape `(n, 2)` |
| `tree_` | Tree extracted from the kNN graph |
| `graph_` | k-nearest-neighbor graph |
| `lsh_forest_` | Jaccard search index for set / string inputs |
| `index_` | USearch index for cosine / euclidean (`store_index=True`) and binary Jaccard |

### Metrics

| Metric | Input | Backend |
|--------|-------|---------|
| `jaccard` | Binary matrix | USearch |
| `jaccard` | Sets / strings | MinHash + LSHForest |
| `cosine` | Dense float matrix | USearch |
| `euclidean` | Dense float matrix | USearch |
| `precomputed` | Distance matrix | Direct graph construction |

### Reproducibility

The `seed` parameter controls the OGDF tree layout, which is fully
deterministic: same kNN graph + same seed = identical coordinates.

The kNN step depends on the backend:

- **MinHash + LSHForest** (sets / strings): deterministic for a given
  `seed` and `minhash_seed`.
- **USearch HNSW** (binary Jaccard, cosine, euclidean): approximate and
  multi-threaded. Neighbor sets may vary slightly across runs or
  platforms. The resulting trees are nearly identical in practice because
  the MST is robust to small kNN variations.

If you need bit-exact kNN reproducibility for binary data, use the
MinHash + LSHForest pipeline directly:

```python
from tmap import MinHash, LSHForest
from tmap.layout import LayoutConfig, layout_from_lsh_forest

mh = MinHash(num_perm=128, seed=42)
signatures = mh.batch_from_binary_array(X)

lsh = LSHForest(d=128, l=64)
lsh.batch_add(signatures)
lsh.index()

cfg = LayoutConfig(k=20, kc=50, deterministic=True, seed=42)
x, y, s, t = layout_from_lsh_forest(lsh, cfg)
```

### Lower-Level Search Backends

`TMAP` is the main entry point for maps. For direct nearest-neighbor queries,
the lower-level search classes are also public.

#### `USearchIndex`

Use for dense cosine / euclidean search and binary Jaccard search.

| Method | What it does |
|--------|---------------|
| `build_from_vectors(X, metric=...)` | Build a dense cosine / euclidean index |
| `build_from_binary(X)` | Build a binary Jaccard index |
| `query_point(x, k)` | Query one vector |
| `query_batch(X, k)` | Query many vectors |
| `query_knn(k)` | Build the all-vs-all kNN graph |
| `add(X)` | Append new vectors to the index |
| `save(path)` / `load(path)` | Persist the index |

#### `LSHForest`

Use for lower-level MinHash workflows and Jaccard on sets / strings.

| Method | What it does |
|--------|---------------|
| `query(signature, k)` | LSH-only candidate lookup |
| `query_linear_scan(signature, k, kc)` | Query one external signature with candidate refinement |
| `query_linear_scan_by_id(id, k, kc)` | Query one indexed signature by ID |
| `query_external_batch(signatures, k, kc)` | Batch query external signatures |
| `get_knn_graph(k, kc)` | Build the all-vs-all kNN graph |
| `get_all_distances(signature)` | Exact MinHash distances to all indexed signatures |
| `save(path)` / `load(path)` | Persist the index |

## `TmapViz`

Visualization object returned by `model.to_tmapviz()`.

### Common Methods

| Method | What it does |
|--------|---------------|
| `add_color_layout(name, values, ...)` | Add a colorable column |
| `add_filter(name, values, ...)` | Add a filter-panel column without colors |
| `add_label(name, values)` | Add a tooltip column |
| `add_smiles(values)` | Add molecule structures to tooltips |
| `to_widget(...)` | Build a Jupyter widget |
| `write_html(path)` | Write one self-contained HTML file |
| `write_static(path)` | Write static assets for hosting |
| `serve(port=8050)` | Start a local HTTP server |

### Minimal Example

```python
viz = model.to_tmapviz()
viz.title = "My Map"
viz.add_color_layout("Score", scores.tolist(), color="viridis")
viz.add_label("Name", names)
viz.write_html("my_map.html")
```

## Chemistry Helpers

These live in `tmap.utils`.

| Function | What it does |
|----------|---------------|
| `fingerprints_from_smiles(smiles, fp_type="morgan", ...)` | Build fingerprints from SMILES |
| `molecular_properties(smiles, properties=None)` | Compute RDKit properties |
| `murcko_scaffolds(smiles)` | Compute Murcko scaffold strings |

### Chemistry Example

```python
from tmap.utils import fingerprints_from_smiles, molecular_properties

fps = fingerprints_from_smiles(smiles, fp_type="morgan", radius=2, n_bits=2048)
props = molecular_properties(smiles, properties=["mw", "logp", "n_rings"])
```

## Power-User Layout API

If you need lower-level control, these functions are still available:

| Function | Use it when |
|----------|-------------|
| `layout_from_lsh_forest(...)` | You want the classic MinHash + LSH workflow directly |
| `layout_from_knn_graph(...)` | You already have a `KNNGraph` |
| `tree_from_knn_graph(...)` | You want a `Tree` from a custom kNN graph |
| `layout_from_edge_list(...)` | You already have weighted edges |

For the parameter details, see [layout_guide.md](layout_guide.md).
