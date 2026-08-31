"""
Numba JIT-accelerated functions for LSH Forest operations.

This module provides high-performance distance computation and linear scan
operations used by LSHForest for k-NN graph construction.

Key optimizations:
1. Vectorized distance computation across candidates
2. Parallel linear scan over all queries for k-NN graph
3. Efficient top-k selection using partial sorting
"""

import numba
import numpy as np
from numba import prange
from numpy.typing import NDArray


@numba.njit(cache=True)
def jaccard_distance(sig_a: NDArray[np.uint64], sig_b: NDArray[np.uint64]) -> float:
    """
    Compute Jaccard distance between two MinHash signatures.

    Distance = 1 - (number of matching hash values) / num_perm

    Args:
        sig_a: First MinHash signature of shape (d,)
        sig_b: Second MinHash signature of shape (d,)

    Returns:
        Jaccard distance (0.0 to 1.0)
    """
    d = len(sig_a)
    matches = 0
    for i in range(d):
        if sig_a[i] == sig_b[i]:
            matches += 1
    return 1.0 - matches / d


@numba.njit(cache=True)
def weighted_jaccard_distance(sig_a: NDArray[np.uint64], sig_b: NDArray[np.uint64]) -> float:
    """
    Compute weighted Jaccard distance between two weighted MinHash signatures.

    For weighted MinHash, both columns (k, y_k) must match for a row to count.
    Distance = 1 - (number of matching rows) / num_perm

    Args:
        sig_a: First weighted MinHash signature of shape (d, 2)
        sig_b: Second weighted MinHash signature of shape (d, 2)

    Returns:
        Weighted Jaccard distance (0.0 to 1.0)
    """
    d = sig_a.shape[0]
    matches = 0
    for i in range(d):
        if sig_a[i, 0] == sig_b[i, 0] and sig_a[i, 1] == sig_b[i, 1]:
            matches += 1
    return 1.0 - matches / d


@numba.njit(parallel=True, cache=True)
def compute_distances_to_candidates(
    query: NDArray[np.uint64],
    candidates: NDArray[np.uint64],
) -> NDArray[np.float32]:
    """
    Compute Jaccard distances from query to all candidates (vectorized).

    Args:
        query: Query signature of shape (d,)
        candidates: Candidate signatures of shape (n_candidates, d)

    Returns:
        Distances array of shape (n_candidates,)
    """
    n_candidates = candidates.shape[0]
    d = query.shape[0]
    distances = np.empty(n_candidates, dtype=np.float32)

    for i in prange(n_candidates):
        matches = 0
        for j in range(d):
            if query[j] == candidates[i, j]:
                matches += 1
        distances[i] = 1.0 - matches / d

    return distances


@numba.njit(parallel=True, cache=True)
def compute_weighted_distances_to_candidates(
    query: NDArray[np.uint64],
    candidates: NDArray[np.uint64],
) -> NDArray[np.float32]:
    """
    Compute weighted Jaccard distances from query to all candidates.

    Args:
        query: Query signature of shape (d, 2)
        candidates: Candidate signatures of shape (n_candidates, d, 2)

    Returns:
        Distances array of shape (n_candidates,)
    """
    n_candidates = candidates.shape[0]
    d = query.shape[0]
    distances = np.empty(n_candidates, dtype=np.float32)

    for i in prange(n_candidates):
        matches = 0
        for j in range(d):
            if query[j, 0] == candidates[i, j, 0] and query[j, 1] == candidates[i, j, 1]:
                matches += 1
        distances[i] = 1.0 - matches / d

    return distances


@numba.njit(cache=True)
def _argsort_topk(arr: NDArray[np.float32], k: int) -> NDArray[np.int32]:
    """Get indices of k smallest elements (partial sort)."""
    n = len(arr)
    if k >= n:
        return np.argsort(arr).astype(np.int32)

    # Use argpartition-like approach
    indices = np.arange(n, dtype=np.int32)

    # Simple selection sort for top-k (efficient for small k)
    for i in range(k):
        min_idx = i
        for j in range(i + 1, n):
            if arr[indices[j]] < arr[indices[min_idx]]:
                min_idx = j
        # Swap
        indices[i], indices[min_idx] = indices[min_idx], indices[i]

    return indices[:k]


@numba.njit(parallel=True, cache=True)
def linear_scan_batch(
    queries: NDArray[np.uint64],
    signatures: NDArray[np.uint64],
    candidate_indices: NDArray[np.int32],
    candidate_counts: NDArray[np.int32],
    k: int,
    exclude_self: bool = True,
    query_offset: int = 0,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """
    Perform linear scan for multiple queries in parallel.

    For each query, computes distances to its candidates and returns top-k.

    Args:
        queries: Query signatures of shape (n_queries, d)
        signatures: All stored signatures of shape (n_total, d)
        candidate_indices: Candidate indices for each query, shape (n_queries, max_candidates)
                          Padded with -1 for queries with fewer candidates
        candidate_counts: Number of valid candidates per query, shape (n_queries,)
        k: Number of nearest neighbors to return per query
        exclude_self: If True, candidate with index == query index is
            skipped (for self-kNN).  Set False for external queries.
        query_offset: Global row offset of ``queries[0]`` for blocked
            self-kNN construction.

    Returns:
        Tuple of:
        - indices: (n_queries, k) array of neighbor indices
        - distances: (n_queries, k) array of distances
    """
    n_queries = queries.shape[0]
    d = queries.shape[1]

    result_indices = np.full((n_queries, k), -1, dtype=np.int32)
    result_distances = np.full((n_queries, k), np.float32(2.0), dtype=np.float32)

    for q in prange(n_queries):
        n_cand = candidate_counts[q]
        if n_cand == 0:
            continue

        # Compute distances to candidates
        cand_distances = np.empty(n_cand, dtype=np.float32)
        for i in range(n_cand):
            cand_idx = candidate_indices[q, i]
            if cand_idx < 0:
                cand_distances[i] = 2.0  # Invalid candidate
            elif exclude_self and cand_idx == q + query_offset:
                cand_distances[i] = 2.0  # Exclude self
            else:
                # Compute Jaccard distance
                matches = 0
                for j in range(d):
                    if queries[q, j] == signatures[cand_idx, j]:
                        matches += 1
                cand_distances[i] = 1.0 - matches / d

        # Get top-k indices
        actual_k = min(k, n_cand)
        top_k = _argsort_topk(cand_distances, actual_k)

        # Only assign valid results (distance < 2.0 means valid neighbor)
        result_idx = 0
        for i in range(actual_k):
            idx = top_k[i]
            if cand_distances[idx] < 2.0:
                result_indices[q, result_idx] = candidate_indices[q, idx]
                result_distances[q, result_idx] = cand_distances[idx]
                result_idx += 1

    return result_indices, result_distances


@numba.njit(parallel=True, cache=True)
def linear_scan_batch_weighted(
    queries: NDArray[np.uint64],
    signatures: NDArray[np.uint64],
    candidate_indices: NDArray[np.int32],
    candidate_counts: NDArray[np.int32],
    k: int,
    exclude_self: bool = True,
    query_offset: int = 0,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """
    Perform linear scan for multiple weighted queries in parallel.

    Args:
        queries: Query signatures of shape (n_queries, d, 2)
        signatures: All stored signatures of shape (n_total, d, 2)
        candidate_indices: Candidate indices, shape (n_queries, max_candidates)
        candidate_counts: Number of valid candidates per query
        k: Number of nearest neighbors to return
        exclude_self: If True, candidate with index == query index is
            skipped (for self-kNN).  Set False for external queries.
        query_offset: Global row offset of ``queries[0]`` for blocked
            self-kNN construction.

    Returns:
        Tuple of (indices, distances) arrays
    """
    n_queries = queries.shape[0]
    d = queries.shape[1]

    result_indices = np.full((n_queries, k), -1, dtype=np.int32)
    result_distances = np.full((n_queries, k), np.float32(2.0), dtype=np.float32)

    for q in prange(n_queries):
        n_cand = candidate_counts[q]
        if n_cand == 0:
            continue

        cand_distances = np.empty(n_cand, dtype=np.float32)
        for i in range(n_cand):
            cand_idx = candidate_indices[q, i]
            if cand_idx < 0 or (exclude_self and cand_idx == q + query_offset):
                cand_distances[i] = 2.0
            else:
                matches = 0
                for j in range(d):
                    if (
                        queries[q, j, 0] == signatures[cand_idx, j, 0]
                        and queries[q, j, 1] == signatures[cand_idx, j, 1]
                    ):
                        matches += 1
                cand_distances[i] = 1.0 - matches / d

        actual_k = min(k, n_cand)
        top_k = _argsort_topk(cand_distances, actual_k)

        # Only assign valid results (distance < 2.0 means valid neighbor)
        result_idx = 0
        for i in range(actual_k):
            idx = top_k[i]
            if cand_distances[idx] < 2.0:
                result_indices[q, result_idx] = candidate_indices[q, idx]
                result_distances[q, result_idx] = cand_distances[idx]
                result_idx += 1

    return result_indices, result_distances


@numba.njit(cache=True, inline="always")
def _compare_prefix(
    signatures: NDArray[np.uint64],
    row: int,
    query: NDArray[np.uint64],
    start: int,
    prefix_length: int,
) -> int:
    """Compare one stored band prefix with a query prefix lexicographically."""
    for offset in range(prefix_length):
        stored = signatures[row, start + offset]
        wanted = query[start + offset]
        if stored < wanted:
            return -1
        if stored > wanted:
            return 1
    return 0


@numba.njit(cache=True, inline="always")
def _compare_prefix_weighted(
    signatures: NDArray[np.uint64],
    row: int,
    query: NDArray[np.uint64],
    start: int,
    prefix_length: int,
) -> int:
    """Compare weighted MinHash prefixes, including both values per permutation."""
    for offset in range(prefix_length):
        for component in range(2):
            stored = signatures[row, start + offset, component]
            wanted = query[start + offset, component]
            if stored < wanted:
                return -1
            if stored > wanted:
                return 1
    return 0


@numba.njit(cache=True, inline="always")
def _prefix_range(
    signatures: NDArray[np.uint64],
    sorted_indices: NDArray[np.int32],
    query: NDArray[np.uint64],
    start: int,
    prefix_length: int,
) -> tuple[int, int]:
    """Return the half-open range of rows equal to an unweighted prefix."""
    lo = 0
    hi = len(sorted_indices)
    while lo < hi:
        mid = (lo + hi) // 2
        comparison = _compare_prefix(signatures, sorted_indices[mid], query, start, prefix_length)
        if comparison < 0:
            lo = mid + 1
        else:
            hi = mid
    left = lo

    hi = len(sorted_indices)
    while lo < hi:
        mid = (lo + hi) // 2
        comparison = _compare_prefix(signatures, sorted_indices[mid], query, start, prefix_length)
        if comparison <= 0:
            lo = mid + 1
        else:
            hi = mid
    return left, lo


@numba.njit(cache=True, inline="always")
def _prefix_range_weighted(
    signatures: NDArray[np.uint64],
    sorted_indices: NDArray[np.int32],
    query: NDArray[np.uint64],
    start: int,
    prefix_length: int,
) -> tuple[int, int]:
    """Return the half-open range of rows equal to a weighted prefix."""
    lo = 0
    hi = len(sorted_indices)
    while lo < hi:
        mid = (lo + hi) // 2
        comparison = _compare_prefix_weighted(
            signatures, sorted_indices[mid], query, start, prefix_length
        )
        if comparison < 0:
            lo = mid + 1
        else:
            hi = mid
    left = lo

    hi = len(sorted_indices)
    while lo < hi:
        mid = (lo + hi) // 2
        comparison = _compare_prefix_weighted(
            signatures, sorted_indices[mid], query, start, prefix_length
        )
        if comparison <= 0:
            lo = mid + 1
        else:
            hi = mid
    return left, lo


@numba.njit(cache=True, inline="always")
def _seen_or_insert(seen: NDArray[np.int32], value: int) -> bool:
    """Insert into a small open-addressed set; return whether it was present."""
    mask = len(seen) - 1
    slot = (value * 2654435761) & mask
    while True:
        current = seen[slot]
        if current == value:
            return True
        if current == -1:
            seen[slot] = value
            return False
        slot = (slot + 1) & mask


@numba.njit(cache=True, inline="always")
def _seen_table_size(max_results: int) -> int:
    """Choose a power-of-two hash table with a load factor at most 0.5."""
    size = 2
    target = max_results * 2
    while size < target:
        size *= 2
    return size


@numba.njit(parallel=True, cache=True)
def query_lsh_forest_batch(
    queries: NDArray[np.uint64],
    signatures: NDArray[np.uint64],
    sorted_indices_flat: NDArray[np.int32],
    band_offsets: NDArray[np.int64],
    band_width: int,
    max_results: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Retrieve candidates using adaptive LSH-Forest prefix backoff.

    Every tree owns a disjoint band of ``band_width`` MinHash values. Rows are
    sorted lexicographically by their complete band. Queries first retrieve
    exact full-band matches, then shorten the prefix one value at a time until
    ``max_results`` unique candidates have been collected or the one-value
    prefixes are exhausted. This mirrors the original TMAP C++ LSH Forest.
    """
    n_queries = queries.shape[0]
    n_trees = len(band_offsets) - 1
    n_indexed = signatures.shape[0]
    result_limit = min(max_results, n_indexed)

    candidates = np.full((n_queries, result_limit), -1, dtype=np.int32)
    counts = np.zeros(n_queries, dtype=np.int32)
    table_size = _seen_table_size(result_limit)

    for q in prange(n_queries):
        seen = np.full(table_size, -1, dtype=np.int32)
        n_candidates = 0

        for prefix_length in range(band_width, 0, -1):
            for tree in range(n_trees):
                offset = band_offsets[tree]
                end = band_offsets[tree + 1]
                sorted_indices = sorted_indices_flat[offset:end]
                start = tree * band_width
                left, right = _prefix_range(
                    signatures,
                    sorted_indices,
                    queries[q],
                    start,
                    prefix_length,
                )

                for position in range(left, right):
                    candidate = int(sorted_indices[position])
                    if not _seen_or_insert(seen, candidate):
                        candidates[q, n_candidates] = candidate
                        n_candidates += 1
                        if n_candidates == result_limit:
                            break
                if n_candidates == result_limit:
                    break
            if n_candidates == result_limit:
                break

        counts[q] = n_candidates

    return candidates, counts


@numba.njit(parallel=True, cache=True)
def query_lsh_forest_batch_weighted(
    queries: NDArray[np.uint64],
    signatures: NDArray[np.uint64],
    sorted_indices_flat: NDArray[np.int32],
    band_offsets: NDArray[np.int64],
    band_width: int,
    max_results: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Weighted adaptive-prefix retrieval using both values per permutation."""
    n_queries = queries.shape[0]
    n_trees = len(band_offsets) - 1
    n_indexed = signatures.shape[0]
    result_limit = min(max_results, n_indexed)

    candidates = np.full((n_queries, result_limit), -1, dtype=np.int32)
    counts = np.zeros(n_queries, dtype=np.int32)
    table_size = _seen_table_size(result_limit)

    for q in prange(n_queries):
        seen = np.full(table_size, -1, dtype=np.int32)
        n_candidates = 0

        for prefix_length in range(band_width, 0, -1):
            for tree in range(n_trees):
                offset = band_offsets[tree]
                end = band_offsets[tree + 1]
                sorted_indices = sorted_indices_flat[offset:end]
                start = tree * band_width
                left, right = _prefix_range_weighted(
                    signatures,
                    sorted_indices,
                    queries[q],
                    start,
                    prefix_length,
                )

                for position in range(left, right):
                    candidate = int(sorted_indices[position])
                    if not _seen_or_insert(seen, candidate):
                        candidates[q, n_candidates] = candidate
                        n_candidates += 1
                        if n_candidates == result_limit:
                            break
                if n_candidates == result_limit:
                    break
            if n_candidates == result_limit:
                break

        counts[q] = n_candidates

    return candidates, counts
