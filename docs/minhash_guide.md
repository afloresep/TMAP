# Understanding MinHash: A Visual Guide

This guide explains how MinHash encoding works in TMAP and helps you choose the right method for your data.

---

## What MinHash Does

MinHash is a **locality-sensitive hashing** technique that compresses your data into compact "signatures" while preserving similarity information.

**The key insight:** If two items are similar (high Jaccard similarity), their MinHash signatures will also be similar.

```txt
Your Data (2048-bit fingerprint)
        ↓
    [MinHash Encoding]
        ↓
Compact Signature (128 values)
```

This compression is essential for TMAP because:

1. **Speed**: Comparing 128 values is faster than comparing 2048 bits
2. **Indexing**: LSH Forest can efficiently index these signatures
3. **Scalability**: Smaller signatures = less memory for millions of points

---

## Quick Start

```python
from tmap import MinHash

# Create encoder
mh = MinHash(num_perm=128, seed=42)

# Encode fingerprints (most common case)
fingerprints = load_your_data()  # shape: (n_samples, n_bits)
signatures = mh.batch_from_binary_array(fingerprints)

# Now use with LSHForest
from tmap import LSHForest
lsh = LSHForest(d=128, l=64)
lsh.batch_add(signatures)
lsh.index()
```

---

## Choosing the Right Method

TMAP's MinHash provides multiple encoding methods. Here's when to use each:

### Decision Tree

```txt
What type of data do you have?
│
├─ Binary fingerprints (0/1 arrays)?
│  │
│  ├─ One fingerprint at a time? → from_binary_array()
│  │
│  └─ Many fingerprints? → batch_from_binary_array() ⭐ RECOMMENDED
│
├─ Sparse indices (list of "on" positions)?
│  │
│  ├─ One sample? → from_sparse_binary_array()
│  │
│  └─ Many samples? → batch_from_sparse_binary_array()
│
├─ Text/string data (words, tokens)?
│  │
│  ├─ One document? → from_string_array()
│  │
│  └─ Many documents? → batch_from_string_array()
│
└─ Auto-detect input type? → encode()
```

### Method Comparison

| Method | Input | Backend | Speed | When to Use |
|--------|-------|---------|-------|-------------|
| `batch_from_binary_array()` | 2D array (n, bits) | Numba | ⚡⚡⚡ | **Most cases** - many fingerprints |
| `from_binary_array()` | 1D array (bits,) | Numba | ⚡⚡⚡ | Single fingerprint |
| `batch_from_sparse_binary_array()` | List of index lists | Numba | ⚡⚡⚡ | Sparse data (< 30% fill rate) |
| `from_sparse_binary_array()` | List of indices | Numba | ⚡⚡⚡ | Single sparse fingerprint |
| `batch_from_string_array()` | List of string lists | Numba + xxhash | ⚡⚡ | Text data with xxhash64 token hashing |
| `from_string_array()` | List of strings | Numba + xxhash | ⚡⚡ | Single text document |
| `encode()` | Auto-detect | Auto | Varies | General purpose, flexible |

---

## Method Details

### `batch_from_binary_array()` - The Primary Method

**Use this for most molecular fingerprint work.**

```python
import numpy as np
from tmap import MinHash

mh = MinHash(num_perm=128, seed=42)

# Your fingerprints: n_samples × n_bits
fingerprints = np.random.randint(0, 2, size=(10000, 2048), dtype=np.uint8)

# Encode all at once (fastest)
signatures = mh.batch_from_binary_array(fingerprints)
print(signatures.shape)  # (10000, 128)
```

**Performance:** ~150,000 fingerprints/second (2048-bit, 128 permutations)

**When to use:**

- Molecular fingerprints (ECFP, MACCS, etc.)
- Any dense binary data
- Processing datasets of any size

---

### `from_binary_array()` - Single Fingerprint

**For encoding one fingerprint at a time.**

```python
# Single fingerprint
fp = np.array([1, 0, 1, 1, 0, 0, 1, 0, ...], dtype=np.uint8)
sig = mh.from_binary_array(fp)
print(sig.shape)  # (128,)
```

**When to use:**

- Processing one item at a time (streaming)
- Querying a new point against existing index
- Testing and debugging

**Note:** For multiple fingerprints, use `batch_from_binary_array()` instead - it's faster due to Numba parallelization.

---

### `batch_from_sparse_binary_array()` - Sparse Data

**For fingerprints stored as lists of "on" indices.**

```python
# Sparse format: each item is a list of indices where bit = 1
sparse_fingerprints = [
    [0, 5, 42, 100, 256],      # First fingerprint has bits 0, 5, 42, 100, 256 set
    [1, 5, 43, 101, 257],      # Second fingerprint
    [0, 6, 44, 102, 258],      # Third fingerprint
]

signatures = mh.batch_from_sparse_binary_array(sparse_fingerprints)
```

**When to use:**

- Your data is already in sparse format
- Fill rate is low (< 30% of bits are 1)
- Memory efficiency is important

**Important:** Sparse and dense representations produce **identical** signatures:

```python
# These produce the same signature
indices = [0, 5, 10]
sig_sparse = mh.from_sparse_binary_array(indices)

dense = np.zeros(100, dtype=np.uint8)
dense[indices] = 1
sig_dense = mh.from_binary_array(dense)

np.testing.assert_array_equal(sig_sparse, sig_dense)  # ✓ Same!
```

---

### `batch_from_string_array()` - Text Data

**For text data like words, tokens, or n-grams.**

```python
# Each document is a list of strings (tokens, words, etc.)
documents = [
    ["apple", "banana", "cherry"],
    ["banana", "date", "elderberry"],
    ["apple", "fig", "grape"],
]

signatures = mh.batch_from_string_array(documents)
```

**When to use:**

- Text documents (bag-of-words)
- Chemical SMILES treated as character n-grams
- Any string-based set data

**Important:** String signatures use `xxhash64` token hashing before the Numba MinHash path. This is different from the binary hash function, so **binary and string signatures are NOT comparable**.

---

### `encode()` - Auto-Detection

**Flexible method that auto-detects input type.**

```python
# Works with numpy arrays
data = np.random.randint(0, 2, size=(100, 50), dtype=np.uint8)
sigs = mh.encode(data)

# Also works with sets
data = [{1, 2, 3}, {2, 3, 4}, {3, 4, 5}]
sigs = mh.encode(data)

# And lists
data = [[1, 2, 3], [2, 3, 4]]
sigs = mh.encode(data)
```

**When to use:**

- Writing generic code that handles multiple input types
- Quick prototyping
- When input type varies at runtime

**Behavior:**

- NumPy array → Uses Numba (fast)
- List of integer sets/lists → Uses the sparse Numba path
- List of string sets/lists → Uses `xxhash64` token hashing + Numba

---

## WeightedMinHash - For Float Vectors

When your data has weights (not just 0/1), use `WeightedMinHash`:

```python
from tmap import WeightedMinHash

# Must specify dimension upfront
wmh = WeightedMinHash(dim=100, num_perm=128, seed=42)

# Weight vectors (e.g., TF-IDF, count vectors)
weights = np.random.uniform(0.1, 10, size=(1000, 100))  # Must be > 0

# Encode
signatures = wmh.batch_from_weight_array(weights)
print(signatures.shape)  # (1000, 128, 2) - note: 2 values per hash

# Distance calculation is different
sig1 = wmh.from_weight_array(weights[0])
sig2 = wmh.from_weight_array(weights[1])
distance = wmh.get_weighted_distance(sig1, sig2)
```

**Key differences from MinHash:**

1. Requires knowing `dim` at construction time
2. Input must be strictly positive (> 0, not just >= 0)
3. Output has shape `(n, num_perm, 2)` - two values per hash
4. Use `get_weighted_distance()` for comparisons

---

## Parameters

### `num_perm` - Signature Size

Controls the tradeoff between accuracy and size.

| Value | Accuracy | Memory | Speed | Use Case |
|-------|----------|--------|-------|----------|
| 32 | Low | 256 bytes | Fastest | Quick exploration |
| 64 | Medium | 512 bytes | Fast | Development |
| 128 | Good | 1 KB | Good | **Default, most cases** |
| 256 | High | 2 KB | Slower | High precision needed |
| 512 | Very High | 4 KB | Slowest | Publication quality |

**Rule of thumb:** Start with 128. Increase only if you need better Jaccard estimation accuracy.

### `seed` - Reproducibility

```python
# Same seed = same signatures for same input
mh1 = MinHash(num_perm=128, seed=42)
mh2 = MinHash(num_perm=128, seed=42)

sig1 = mh1.encode(data)
sig2 = mh2.encode(data)
np.testing.assert_array_equal(sig1, sig2)  # ✓ Identical
```

Always set a seed for reproducible results.

---

## Computing Distances

### Jaccard Distance (MinHash)

```python
mh = MinHash(num_perm=128, seed=42)
sig1 = mh.from_binary_array(fp1)
sig2 = mh.from_binary_array(fp2)

# Method 1: Static method
distance = MinHash.get_distance(sig1, sig2)

# Method 2: Alias
distance = MinHash.jaccard_distance(sig1, sig2)

# Distance is in [0, 1]
# 0 = identical, 1 = completely different
```

### Weighted Jaccard Distance (WeightedMinHash)

```python
wmh = WeightedMinHash(dim=100, num_perm=128, seed=42)
sig1 = wmh.from_weight_array(w1)
sig2 = wmh.from_weight_array(w2)

distance = WeightedMinHash.get_weighted_distance(sig1, sig2)
```

### Accuracy vs True Jaccard

MinHash **estimates** Jaccard distance. The estimate improves with more permutations:

```python
# True Jaccard for sets: 1 - |A ∩ B| / |A ∪ B|
true_distance = 1 - len(set1 & set2) / len(set1 | set2)

# MinHash estimate (more accurate with higher num_perm)
mh = MinHash(num_perm=512, seed=42)  # High for accuracy
sig1 = mh.encode([set1])[0]
sig2 = mh.encode([set2])[0]
estimated_distance = mh.get_distance(sig1, sig2)

# Typically within 0.05 of true value with 512 permutations
```

---

## Common Pitfalls

### 1. Mixing Binary and String Signatures

**Problem:** Binary and string data use different hash functions.

```python
# ❌ WRONG - these signatures are not comparable!
sig_binary = mh.from_binary_array(fingerprint)
sig_string = mh.from_string_array(["1", "2", "3"])
distance = mh.get_distance(sig_binary, sig_string)  # Meaningless!
```

**Solution:** Use the same encoding method for all data you want to compare.

### 2. Using Batch Methods for Single Items

**Problem:** Overhead from method dispatch.

```python
# ❌ Inefficient - calling batch method in a loop
for fp in fingerprints:
    sig = mh.batch_from_binary_array(fp.reshape(1, -1))[0]

# ✓ Efficient - one batch call
sigs = mh.batch_from_binary_array(fingerprints)
```

### 3. Wrong Shape for from_binary_array

**Problem:** Passing 2D array to single-item method.

```python
# ❌ WRONG - will raise error
fp = np.array([[1, 0, 1], [0, 1, 1]])  # 2D!
sig = mh.from_binary_array(fp)  # Error!

# ✓ Correct
sig = mh.batch_from_binary_array(fp)  # For multiple
sig = mh.from_binary_array(fp[0])     # For single
```

### 4. Weighted MinHash with Zero Values

**Problem:** WeightedMinHash requires strictly positive values.

```python
# ❌ WRONG - will raise error
weights = np.array([1.0, 0.0, 2.0])  # Contains 0!
sig = wmh.from_weight_array(weights)

# ✓ Correct - add small epsilon
weights = np.maximum(weights, 1e-10)
sig = wmh.from_weight_array(weights)
```

### 5. Mismatched num_perm

**Problem:** Comparing signatures with different `num_perm`.

```python
mh1 = MinHash(num_perm=64)
mh2 = MinHash(num_perm=128)

sig1 = mh1.from_binary_array(fp)  # Shape: (64,)
sig2 = mh2.from_binary_array(fp)  # Shape: (128,)

# ❌ Will raise error
distance = mh1.get_distance(sig1, sig2)  # Mismatched lengths!
```

---

## Performance Tips

### 1. Use Batch Methods

```python
# ⚡ Fast
sigs = mh.batch_from_binary_array(all_fingerprints)

# 🐌 Slow
sigs = [mh.from_binary_array(fp) for fp in all_fingerprints]
```

### 2. Ensure Correct dtype

```python
# ⚡ Fast - uint8 is optimal
fps = fingerprints.astype(np.uint8)
sigs = mh.batch_from_binary_array(fps)

# 🐌 Slower - requires conversion
fps_float = fingerprints.astype(np.float64)  # Wastes memory
sigs = mh.batch_from_binary_array(fps_float)  # Internal conversion
```

### 3. Use Sparse Format When Appropriate

```python
# If your fingerprints have < 30% fill rate:
sparse_fps = [np.nonzero(fp)[0].tolist() for fp in fingerprints]
sigs = mh.batch_from_sparse_binary_array(sparse_fps)
```

### 4. Numba JIT Compilation

The first call compiles Numba functions (takes ~1-2 seconds). Subsequent calls are fast:

```python
# First call: ~2 seconds (compilation)
sigs = mh.batch_from_binary_array(batch1)

# Second call: ~0.01 seconds (already compiled)
sigs = mh.batch_from_binary_array(batch2)
```

---

## Complete Example

```python
import numpy as np
from tmap import MinHash, LSHForest
from tmap.layout import layout_from_lsh_forest, LayoutConfig

# 1. Load or generate binary fingerprints
n_molecules = 10000
n_bits = 2048
fingerprints = (np.random.rand(n_molecules, n_bits) < 0.1).astype(np.uint8)

# 2. Create MinHash encoder
mh = MinHash(num_perm=128, seed=42)

# 3. Encode all fingerprints (batch is fastest)
signatures = mh.batch_from_binary_array(fingerprints)
print(f"Encoded {len(signatures)} molecules to shape {signatures.shape}")

# 4. Build LSH Forest index
lsh = LSHForest(d=128, l=64)  # d must match num_perm
lsh.batch_add(signatures)
lsh.index()

# 5. Create layout
cfg = LayoutConfig()
cfg.k = 20
cfg.kc = 50
cfg.deterministic = True
cfg.seed = 42

x, y, s, t = layout_from_lsh_forest(lsh, cfg)
print(f"Layout: {len(x)} points, {len(s)} edges")
```

---

## Next Steps

- See [LSHForest Guide](lshforest_guide.md) for indexing and querying
- See [Layout Guide](layout_guide.md) for visualization parameters
- Try the example: `examples/chemistry/smiles_tmap.py`
