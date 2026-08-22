# Understanding TMAP Layout: A Visual Guide

This guide explains how TMAP creates tree visualizations from your data. Instead of diving into technical implementation details, we'll focus on **what each piece does** and **how parameters affect your visualization**.

---

## The Big Picture: What TMAP Does

TMAP takes high-dimensional data (like molecular fingerprints) and creates a 2D tree visualization where:

- **Similar items are close together**
- **Clusters form branches**
- **The overall structure reveals relationships**

Think of it like creating a family tree, but for your data points based on their similarity.

```txt
Your Data (1000 molecules with 2048-bit fingerprints)
        ↓
    [Encoding] → Compact signatures
        ↓
    [Indexing] → Find similar neighbors
        ↓
    [Graph Building] → Connect neighbors
        ↓
    [Tree Extraction] → Keep essential connections
        ↓
    [Layout] → Arrange in 2D space
        ↓
    Beautiful Tree Visualization
```

---

## Step 1: Encoding with MinHash

**What it does:** Compresses your data into compact "signatures" that preserve similarity.

```python
from tmap import MinHash

mh = MinHash(num_perm=128, seed=42)
signatures = mh.batch_from_binary_array(fingerprints)
```

### Parameters

| Parameter | What it controls | Trade-off |
|-----------|-----------------|-----------|
| `num_perm` | Signature size (default: 128) | Higher = more accurate similarity, but slower |
| `seed` | Randomization seed | Set for reproducible results |

### How to think about it

Imagine each molecule as a long barcode. MinHash creates a much shorter "summary barcode" that still captures what makes each molecule unique. When two molecules have similar original barcodes, their summary barcodes will also be similar.

**Rule of thumb:** Start with `num_perm=128`. Increase to 256 or 512 if you need more precision for very similar items.

---

## Step 2: Building the LSH Forest Index

**What it does:** Creates a searchable index so we can quickly find similar items.

```python
from tmap import LSHForest

lsh = LSHForest(d=128, l=64)
lsh.batch_add(signatures)
lsh.index()
```

### Parameters

| Parameter | What it controls | Trade-off |
|-----------|-----------------|-----------|
| `d` | Signature dimension (must match MinHash) | Must equal `num_perm` |
| `l` | Number of "trees" in the forest | Higher = better recall, but more memory |

### How to think about it

The LSH Forest is like a library catalog system. Instead of one index, it maintains multiple overlapping indexes (`l` of them). When searching for similar items, it checks all indexes and combines results. More indexes means fewer items slip through the cracks.

**Rule of thumb:** Use `l=32` for small datasets (<10k), `l=64` for medium (10k-100k), `l=128` for large (>100k).

---

## Step 3: Finding Neighbors (k-NN Graph)

**What it does:** For each item, finds its k most similar neighbors.

This happens internally when you call `layout_from_lsh_forest`, controlled by two parameters:

```python
cfg = LayoutConfig()
cfg.k = 20   # Number of neighbors to keep
cfg.kc = 50  # Search multiplier
```

### Parameters

| Parameter | What it controls | Trade-off |
|-----------|-----------------|-----------|
| `k` | Neighbors per item (default: 10) | Higher = denser graph, better connectivity |
| `kc` | Search depth multiplier (default: 10) | Higher = better quality neighbors, slower |

### How to think about it

Imagine you're at a party trying to find your 20 closest friends (`k=20`). You could ask everyone in the room, but that's slow. Instead, you ask around in your general social circle, checking maybe 1000 people (`k * kc = 20 * 50 = 1000`), and pick the 20 you're closest to.

**Why kc matters:** LSH is approximate. By searching more candidates (`k * kc`), we're more likely to find the true nearest neighbors, not just "pretty close" ones.

**Rule of thumb:**

- For exploration: `k=10, kc=10` (fast, might miss some connections)
- For publication quality: `k=20-50, kc=50-100` (slower, better connectivity)

---

## Step 4: The Minimum Spanning Tree (MST)

**What it does:** From the neighbor graph, extracts the essential tree structure.

The k-NN graph has many connections (each node connects to k neighbors). The MST keeps only the connections needed to:

1. Connect all points
2. Minimize total "distance" (maximize similarity)

### The Key Insight

The MST is computed by OGDF (the layout engine) on the **full k-NN graph**. This is important:

```
k-NN Graph (many edges)  →  MST (n-1 edges)  →  Tree Structure
     ↓                           ↓
   Dense                    Sparse but
  connections              connected
```

If your k-NN graph has disconnected clusters (some molecules have no path to others), the MST will also be disconnected. This shows up as a grid of isolated points in your visualization.

**To fix disconnected trees:**

- Increase `k` (more neighbors)
- Increase `kc` (better neighbor quality)
- Increase `l` in LSHForest (better recall)

---

## Step 5: Force-Directed Layout

**What it does:** Arranges the tree in 2D space so it looks nice.

This is where OGDF's magic happens. It uses physics simulation:

- **Edges act like springs** (pull connected nodes together)
- **Nodes repel each other** (prevent overlap)
- **The system finds equilibrium** (balanced forces)

```python
cfg = LayoutConfig()
cfg.fme_iterations = 1000  # Simulation steps
cfg.node_size = 1/30       # Repulsion strength
cfg.mmm_repeats = 2        # Refinement passes
```

### Layout Parameters Explained

#### `fme_iterations` (default: 1000)

**Number of simulation steps**

| Value | Effect |
|-------|--------|
| 100-500 | Fast but rough. Good for exploration. |
| 1000 | Default. Good balance. |
| 2000-5000 | Smoother, more refined. Takes longer. |

Think of it like shaking a box of magnets. More iterations = more time to settle into a stable arrangement.

#### `node_size` (default: 1/65 ≈ 0.015)

**How much nodes repel each other**

| Value | Effect |
|-------|--------|
| 1/100 (0.01) | Compact, tight clusters |
| 1/65 (0.015) | Default, balanced |
| 1/30 (0.033) | Spread out, more space between branches |
| 1/10 (0.1) | Very spread out |

Larger values = nodes push each other away more = more spread out tree.

#### `mmm_repeats` (default: 1)

**Multilevel refinement passes**

The layout algorithm works in levels:

1. Coarsen (simplify) the graph
2. Layout the simplified version
3. Expand back and refine

More repeats = more refinement at each level = smoother result.

| Value | Effect |
|-------|--------|
| 1 | Fast, good enough for most cases |
| 2 | Noticeably smoother |
| 3+ | Diminishing returns, mostly for very large graphs |

---

## Step 6: Scaling and Final Touches

### `sl_scaling_type`

**How the layout scales during refinement**

| Type | When to use |
|------|-------------|
| `RelativeToDrawing` (default) | Most cases. Scales based on current drawing. |
| `RelativeToAvgLength` | When edge weights vary a lot. Respects average edge length. |
| `Absolute` | When you need exact control over scale. |

#### `sl_extra_scaling_steps` (default: 2)

Additional refinement steps for scaling. Higher = smoother scaling transitions.

---

## Advanced: Placer and Merger

These control the multilevel algorithm's internal behavior.

### Placer (where to put nodes when expanding)

| Type | Description | Best for |
|------|-------------|----------|
| `Barycenter` (default) | Center of neighbors | Most cases |
| `Zero` | Same as parent | Stable, predictable |
| `Median` | Median of neighbors | Outlier-resistant |
| `Circle` | Circle around neighbors | Spreading out |
| `Solar` | Solar system arrangement | Hierarchical data |
| `Random` | Random placement | Exploration (non-deterministic!) |

### Merger (how to simplify the graph)

| Type | Description | Best for |
|------|-------------|----------|
| `LocalBiconnected` (default) | Preserves local structure | Most cases |
| `EdgeCover` | Based on edge coverage | Dense graphs |
| `Solar` | Sun/planet hierarchy | Hierarchical data |
| `IndependentSet` | GRIP-style | Very large graphs |

**Rule of thumb:** Stick with defaults (`Barycenter` + `LocalBiconnected`) unless you have a specific reason to change.

---

## Putting It All Together

Here's the recommended setup for different scenarios:

### Quick Exploration

```python
cfg = LayoutConfig()
cfg.k = 10
cfg.kc = 20
cfg.fme_iterations = 500
```

### Publication Quality

```python
cfg = LayoutConfig()
cfg.k = 30
cfg.kc = 100
cfg.node_size = 1/30
cfg.mmm_repeats = 2
cfg.sl_extra_scaling_steps = 10
cfg.sl_scaling_type = ScalingType.RelativeToAvgLength
cfg.fme_iterations = 1000
cfg.deterministic = True
cfg.seed = 42
```

### Very Large Datasets (>100k points)

```python
cfg = LayoutConfig()
cfg.k = 20
cfg.kc = 50
cfg.fme_iterations = 500  # Lower for speed
cfg.mmm_repeats = 1
cfg.merger = Merger.IndependentSet  # Better for large graphs
```

---

## Common Issues and Fixes

### Problem: Disconnected points (grid of dots)

**Cause:** The k-NN graph has disconnected components.

**Fixes:**

1. Increase `k` (more neighbors per point)
2. Increase `kc` (better quality neighbors)
3. Increase `l` in LSHForest (better recall)
4. Check if your data naturally has isolated clusters

### Problem: Tree looks too compact/cluttered

**Fixes:**

1. Increase `node_size` (e.g., from 1/65 to 1/30)
2. Increase `sl_extra_scaling_steps`
3. Increase `fme_iterations`

### Problem: Tree looks too spread out

**Fixes:**

1. Decrease `node_size` (e.g., from 1/65 to 1/100)
2. Use `ScalingType.RelativeToDrawing`

### Problem: Layout changes each run

**Fixes:**

```python
cfg.deterministic = True
cfg.seed = 42  # Any fixed number
```

### Problem: Layout takes too long

**Fixes:**

1. Decrease `fme_iterations` (e.g., 500 instead of 1000)
2. Decrease `mmm_repeats` to 1
3. Decrease `k` and `kc`
4. Use `l=32` in LSHForest

---

## Complete Example

```python
import numpy as np
from tmap import MinHash, LSHForest
from tmap.layout import layout_from_lsh_forest, LayoutConfig, ScalingType

# Your data: binary fingerprints (n_samples x n_bits)
fingerprints = load_your_data()  # shape: (1000, 2048)

# Step 1: Encode
mh = MinHash(num_perm=128, seed=42)
signatures = mh.batch_from_binary_array(fingerprints)

# Step 2: Index
lsh = LSHForest(d=128, l=64)
lsh.batch_add(signatures)
lsh.index()

# Step 3-6: Layout (all in one call!)
cfg = LayoutConfig()
cfg.k = 20
cfg.kc = 50
cfg.node_size = 1/30
cfg.mmm_repeats = 2
cfg.deterministic = True
cfg.seed = 42

x, y, s, t = layout_from_lsh_forest(lsh, cfg)

# x, y: coordinates for each point
# s, t: edges (s[i] connects to t[i])
print(f"Laid out {len(x)} points with {len(s)} edges")
```

---

## Parameter Reference Card

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| **k** | 10 | 5-100 | Neighbors per point. Higher = denser graph. |
| **kc** | 10 | 10-100 | Search depth. Higher = better neighbors. |
| **fme_iterations** | 1000 | 100-5000 | Simulation steps. Higher = smoother. |
| **node_size** | 1/65 | 0.001-0.1 | Repulsion. Higher = more spread. |
| **mmm_repeats** | 1 | 1-3 | Refinement passes. Higher = smoother. |
| **sl_extra_scaling_steps** | 2 | 0-20 | Scaling refinement. Higher = smoother. |
| **merger_factor** | 2.0 | 1.5-4.0 | Coarsening aggressiveness. |

---

## Next Steps

- Try the example script: `examples/chemistry/smiles_tmap.py`
- Experiment with parameters on your data
- Check `TmapViz` for interactive visualization options

---

## Related Documentation

- [MinHash Guide](minhash_guide.md) - Data encoding with MinHash signatures
- [LSHForest Guide](lshforest_guide.md) - Building the LSH index and k-NN graphs
- [Graph Guide](graph_guide.md) - MST extraction and Tree traversal
- [API Reference](api_reference.md) - Quick reference for all functions

Happy mapping!
