"""
LSH Forest implementation for approximate nearest neighbor search.

This module provides a custom Numba-accelerated implementation:
- Lexicographically sorted prefix trees with adaptive prefix backoff
- Numba JIT candidate retrieval and distance computation
- Numba JIT for distance computation and linear scan

The LSH Forest is optimized for Jaccard similarity on MinHash signatures,
"""

from __future__ import annotations

import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._lsh_numba import (
    compute_distances_to_candidates,
    compute_weighted_distances_to_candidates,
    jaccard_distance,
    linear_scan_batch,
    linear_scan_batch_weighted,
    query_lsh_forest_batch,
    query_lsh_forest_batch_weighted,
    weighted_jaccard_distance,
)
from .types import KNNGraph

__all__ = ["LSHForest"]

_CANDIDATE_BUFFER_BYTES = 256 * 1024 * 1024


class LSHForest:
    """
    LSH Forest data structure for approximate nearest neighbor search.

    Args:
        d: Dimensionality of MinHash vectors (number of permutations). Default: 128
        l: Number of prefix trees. Each tree owns ``d // l`` consecutive
            signature values. Default: 8, matching the original TMAP API.
        store: Expose signatures for linear scan and distance methods. The
            prefix index itself always retains the values needed for queries.
            Default: True.
        weighted: Whether using weighted MinHash signatures. Default: False

    Example:
        >>> from tmap.index.encoders import MinHash
        >>> from tmap.index import LSHForest
        >>>
        >>> # Create MinHash signatures (Numba-accelerated)
        >>> mh = MinHash(num_perm=128)
        >>> sigs = mh.batch_from_binary_array(fingerprints)
        >>>
        >>> # Build LSH Forest
        >>> lsh = LSHForest(d=128)
        >>> lsh.batch_add(sigs)
        >>> lsh.index()
        >>>
        >>> # Build k-NN graph (Numba-accelerated linear scan)
        >>> knn_graph = lsh.get_knn_graph(k=20, kc=10)
    """

    def __init__(
        self,
        d: int = 128,
        l: int | None = None,
        store: bool = True,
        weighted: bool = False,
    ) -> None:
        if d <= 0:
            raise ValueError("d must be positive")

        if l is None:
            l = min(8, d)

        if l <= 0:
            raise ValueError("l must be positive")
        if l > d:
            raise ValueError("l cannot be greater than d")

        self._d = d
        self._l = l
        self._k = d // l
        self._store = store
        self._weighted = weighted

        # Pending signatures are collected in batches and made contiguous by index().
        self._signatures_list: list[NDArray[np.uint64]] = []
        self._index_signatures: NDArray[np.uint64] | None = None
        self._signatures: NDArray[np.uint64] | None = None

        # Each tree stores row IDs in lexicographic order of its full band.
        self._sorted_indices_flat: NDArray[np.int32] | None = None
        self._band_offsets: NDArray[np.int64] | None = None

        # State tracking
        self._n_indexed: int = 0
        self._is_indexed: bool = False
        self._needs_reindex: bool = False

    @property
    def size(self) -> int:
        """Number of indexed MinHash signatures."""
        return self._n_indexed

    @property
    def is_clean(self) -> bool:
        """Whether the index is up-to-date (index() called after last add)."""
        return self._is_indexed and not self._needs_reindex

    @property
    def d(self) -> int:
        """Number of permutations (signature dimensionality)."""
        return self._d

    @property
    def l(self) -> int:
        """Number of prefix trees."""
        return self._l

    @property
    def is_indexed(self) -> bool:
        """Whether the index has been built (index() called after adding)."""
        return self._is_indexed

    # Internal helpers

    def _validate_signature_shape(self, signature: NDArray[np.uint64], batch: bool = False) -> None:
        """Validate signature shape matches configuration."""
        if self._weighted:
            if batch:
                if signature.ndim != 3 or signature.shape[1:] != (self._d, 2):
                    raise ValueError(
                        f"Expected shape (n, {self._d}, 2) for weighted, got {signature.shape}"
                    )
            else:
                if signature.shape != (self._d, 2):
                    raise ValueError(
                        f"Expected shape ({self._d}, 2) for weighted, got {signature.shape}"
                    )
        else:
            if batch:
                if signature.ndim != 2 or signature.shape[1] != self._d:
                    raise ValueError(f"Expected shape (n, {self._d}), got {signature.shape}")
            else:
                if signature.shape != (self._d,):
                    raise ValueError(f"Expected shape ({self._d},), got {signature.shape}")

    def _compute_distance(self, sig_a: NDArray[np.uint64], sig_b: NDArray[np.uint64]) -> float:
        """Compute distance using appropriate method based on weighted flag."""
        if self._weighted:
            return cast(float, weighted_jaccard_distance(sig_a, sig_b))
        else:
            return cast(float, jaccard_distance(sig_a, sig_b))

    def _get_index_signatures(self) -> NDArray[np.uint64] | None:
        """Return index data, including compatibility with older object pickles."""
        signatures = getattr(self, "_index_signatures", None)
        if signatures is None:
            signatures = self._signatures
            self._index_signatures = signatures
        return signatures

    @staticmethod
    def _validate_query_parameters(k: int, kc: int | None = None) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if kc is not None and kc <= 0:
            raise ValueError(f"kc must be positive, got {kc}")

    # Add methods

    def add(self, signature: NDArray[np.uint64]) -> None:
        """
        Add a MinHash signature to the LSH forest.

        Args:
            signature: MinHash vector of shape (d,) or (d, 2) for weighted

        Note:
            Call index() after adding signatures to build/update the index.
        """
        self._validate_signature_shape(signature)

        # Prefix queries require the raw band values even when store=False.
        self._signatures_list.append(signature[np.newaxis].copy())

        self._needs_reindex = True

    def batch_add(self, signatures: NDArray[np.uint64]) -> None:
        """
        Add multiple MinHash signatures to the LSH forest (optimized).

        Args:
            signatures: MinHash vectors of shape (n, d) or (n, d, 2) for weighted

        Note:
            Call index() after adding signatures to build/update the index.
        """
        self._validate_signature_shape(signatures, batch=True)

        # Store a whole batch rather than N individual Python objects.
        self._signatures_list.append(signatures.copy())

        self._needs_reindex = True

    # Index method

    def index(self) -> None:
        """
        Build/rebuild the LSH forest index.

        Must be called after adding signatures with add() or batch_add().
        """
        previous = self._get_index_signatures()
        if not self._signatures_list:
            if previous is None:
                self._n_indexed = 0
                self._is_indexed = True
                self._needs_reindex = False
            return

        # Convert list to contiguous array for efficient Numba access.
        # Include previously indexed signatures if this is a re-index.
        all_parts = self._signatures_list
        if previous is not None and len(previous) > 0:
            all_parts = [previous] + all_parts

        if len(all_parts) == 1:
            index_signatures = np.ascontiguousarray(all_parts[0])
        else:
            index_signatures = np.concatenate(all_parts)
        self._signatures_list = []  # free intermediate copies
        n = index_signatures.shape[0]

        def sort_tree(tree: int) -> tuple[int, NDArray[np.int32]]:
            start = tree * self._k
            end = start + self._k
            if self._weighted:
                band = index_signatures[:, start:end, :].reshape(n, self._k * 2)
            else:
                band = index_signatures[:, start:end]
            # np.lexsort uses its final key as the primary key, hence reverse
            # the columns so the first value in a band is compared first.
            keys = tuple(band[:, column] for column in range(band.shape[1] - 1, -1, -1))
            order = np.lexsort(keys).astype(np.int32, copy=False)
            return tree, order

        sorted_indices = np.empty((self._l, n), dtype=np.int32)
        max_workers = min(self._l, os.cpu_count() or 1) if n >= 4096 else 1
        if max_workers == 1:
            sorted_trees = map(sort_tree, range(self._l))
            for tree, order in sorted_trees:
                sorted_indices[tree] = order
        else:
            # NumPy releases the GIL while lexsorting. Trees are independent,
            # so building them concurrently closes the gap to the OpenMP C++
            # reference without changing their deterministic order.
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for tree, order in executor.map(sort_tree, range(self._l)):
                    sorted_indices[tree] = order

        self._index_signatures = index_signatures
        self._signatures = index_signatures if self._store else None
        self._sorted_indices_flat = sorted_indices.reshape(-1)
        self._band_offsets = np.arange(self._l + 1, dtype=np.int64) * n

        self._n_indexed = n
        self._is_indexed = True
        self._needs_reindex = False

    # Query methods

    def query(self, signature: NDArray[np.uint64], k: int) -> NDArray[np.int32]:
        """
        Query the LSH forest for k-nearest neighbors.

        Uses LSH tree traversal only (no linear scan). For better accuracy,
        use query_linear_scan().

        Args:
            signature: Query MinHash vector of shape (d,) or (d, 2)
            k: Number of nearest neighbors to retrieve

        Returns:
            Array of neighbor indices, shape (k,) or fewer if not enough neighbors
        """
        if not self._is_indexed:
            raise RuntimeError("Must call index() before querying")
        self._validate_query_parameters(k)
        if self._n_indexed == 0:
            return np.array([], dtype=np.int32)

        self._validate_signature_shape(signature)
        index_signatures = self._get_index_signatures()
        if (
            index_signatures is None
            or self._sorted_indices_flat is None
            or self._band_offsets is None
        ):
            raise RuntimeError("Index structures not initialized")

        query_batch = signature[np.newaxis]
        query_fn = query_lsh_forest_batch_weighted if self._weighted else query_lsh_forest_batch
        candidates, counts = query_fn(
            query_batch,
            index_signatures,
            self._sorted_indices_flat,
            self._band_offsets,
            self._k,
            k,
        )

        # Extract valid candidates
        n_valid = counts[0]
        return cast(NDArray[np.int32], candidates[0, :n_valid].copy())

    def query_by_id(self, id: int, k: int) -> NDArray[np.int32]:
        """
        Query k-nearest neighbors for an indexed signature by its ID.

        Args:
            id: Index of the query signature (0-based, order of insertion)
            k: Number of nearest neighbors to retrieve

        Returns:
            Array of neighbor indices

        Raises:
            ValueError: If store=False (signatures not retained)
        """
        if not self._store:
            raise ValueError("query_by_id requires store=True")
        if self._signatures is None:
            raise RuntimeError("Must call index() before querying")
        if id < 0 or id >= self._n_indexed:
            raise IndexError(f"ID {id} out of range [0, {self._n_indexed})")

        return self.query(self._signatures[id], k)

    # Linear scan methods

    def linear_scan(
        self,
        signature: NDArray[np.uint64],
        indices: NDArray[np.int32] | list[int],
        k: int = 10,
    ) -> list[tuple[float, int]]:
        """
        Query a subset of indexed signatures using linear scan.

        Computes exact distances to all specified candidates and returns top-k.

        Args:
            signature: Query MinHash vector
            indices: Subset of indices to search
            k: Number of nearest neighbors to retrieve

        Returns:
            List of (distance, index) tuples, sorted by distance
        """
        if not self._store:
            raise ValueError("linear_scan requires store=True")
        if self._signatures is None:
            raise RuntimeError("Must call index() before linear scan")
        self._validate_query_parameters(k)

        self._validate_signature_shape(signature)

        indices_arr = np.asarray(indices, dtype=np.int32)
        if len(indices_arr) == 0:
            return []

        # Get candidate signatures
        candidates = self._signatures[indices_arr]

        # Compute distances
        if self._weighted:
            distances = compute_weighted_distances_to_candidates(signature, candidates)
        else:
            distances = compute_distances_to_candidates(signature, candidates)

        # Get top-k
        actual_k = min(k, len(indices_arr))
        top_k_idx = np.argpartition(distances, actual_k - 1)[:actual_k]
        top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]

        return [(float(distances[i]), int(indices_arr[i])) for i in top_k_idx]

    def query_linear_scan(
        self,
        signature: NDArray[np.uint64],
        k: int,
        kc: int = 10,
    ) -> list[tuple[float, int]]:
        """
        Query k-nearest neighbors with LSH forest + linear scan combination.

        First retrieves k*kc candidates using LSH forest, then performs
        linear scan on candidates to find exact k nearest neighbors.

        Args:
            signature: Query MinHash vector
            k: Number of nearest neighbors to retrieve
            kc: Multiplier for LSH forest retrieval (retrieves k*kc candidates)

        Returns:
            List of (distance, index) tuples, sorted by distance
        """
        # Get candidates from LSH
        candidates = self.query(signature, k * kc)

        if len(candidates) == 0:
            return []

        # Linear scan on candidates
        return self.linear_scan(signature, candidates, k)

    def query_linear_scan_by_id(
        self,
        id: int,
        k: int,
        kc: int = 10,
    ) -> list[tuple[float, int]]:
        """
        Query k-nearest neighbors by ID with LSH forest + linear scan.

        Args:
            id: Index of the query signature
            k: Number of nearest neighbors to retrieve
            kc: Multiplier for LSH forest retrieval

        Returns:
            List of (distance, index) tuples, sorted by distance
        """
        if not self._store:
            raise ValueError("query_linear_scan_by_id requires store=True")
        if self._signatures is None:
            raise RuntimeError("Must call index() before querying")
        if id < 0 or id >= self._n_indexed:
            raise IndexError(f"ID {id} out of range [0, {self._n_indexed})")

        results = self.query_linear_scan(self._signatures[id], k + 1, kc)

        # Exclude self from results
        return [(d, i) for d, i in results if i != id][:k]

    # k-NN Graph methods (main output for TMAP pipeline)

    def get_all_nearest_neighbors(
        self,
        k: int,
        kc: int = 10,
    ) -> NDArray[np.int32]:
        """
        Get k-nearest neighbors of all indexed signatures.

        Args:
            k: Number of nearest neighbors per point
            kc: Multiplier for LSH forest retrieval

        Returns:
            Flattened array of neighbor indices, shape (n * k,).
            Use reshape(n, k) to get per-point neighbors.
        """
        knn = self.get_knn_graph(k, kc)
        return knn.indices.flatten()

    def get_knn_graph(
        self,
        k: int,
        kc: int = 10,
    ) -> KNNGraph:
        """
        Construct the k-nearest neighbor graph of all indexed signatures.

        This is the primary output method because it produces input for OGDF layout
        and MST construction APIs.

        Args:
            k: Number of nearest neighbors per point
            kc: Multiplier for LSH forest retrieval

        Returns:
            KNNGraph with indices and distances arrays
        """
        if not self._store:
            raise ValueError("get_knn_graph requires store=True")
        if self._signatures is None or self._n_indexed == 0:
            raise RuntimeError("Must add signatures and call index() first")
        self._validate_query_parameters(k, kc)

        max_candidates = min(k * kc, self._n_indexed)
        index_signatures = self._get_index_signatures()
        if (
            index_signatures is None
            or self._sorted_indices_flat is None
            or self._band_offsets is None
        ):
            raise RuntimeError("Index structures not initialized")

        indices = np.full((self._n_indexed, k), -1, dtype=np.int32)
        distances = np.full((self._n_indexed, k), np.float32(2.0), dtype=np.float32)
        bytes_per_query = max(max_candidates * np.dtype(np.int32).itemsize, 1)
        batch_size = max(1, min(self._n_indexed, _CANDIDATE_BUFFER_BYTES // bytes_per_query))
        query_fn = query_lsh_forest_batch_weighted if self._weighted else query_lsh_forest_batch
        scan_fn = linear_scan_batch_weighted if self._weighted else linear_scan_batch

        for start in range(0, self._n_indexed, batch_size):
            end = min(start + batch_size, self._n_indexed)
            queries = self._signatures[start:end]
            candidates, counts = query_fn(
                queries,
                index_signatures,
                self._sorted_indices_flat,
                self._band_offsets,
                self._k,
                max_candidates,
            )
            batch_indices, batch_distances = scan_fn(
                queries,
                self._signatures,
                candidates,
                counts,
                k,
                True,
                start,
            )
            indices[start:end] = batch_indices
            distances[start:end] = batch_distances

        return KNNGraph(indices=indices, distances=distances)

    def query_external_batch(
        self,
        signatures: NDArray[np.uint64],
        k: int,
        kc: int = 10,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Query k-nearest stored neighbors for a batch of *external* signatures.

        Uses the same Numba-parallel pipeline as :meth:`get_knn_graph`
        (adaptive-prefix retrieval -> linear scan) but does **not** exclude
        self-matches, since the query signatures are not part of the index.

        Parameters
        ----------
        signatures : NDArray[np.uint64]
            ``(m, d)`` array of MinHash signatures to query.
        k : int
            Number of nearest neighbors per query.
        kc : int
            Candidate multiplier for LSH retrieval (retrieves ``k * kc``
            candidates, then refines with exact distances).

        Returns
        -------
        indices : NDArray[np.int32]
            ``(m, k)`` neighbor indices (padded with ``-1``).
        distances : NDArray[np.float32]
            ``(m, k)`` Jaccard distances (padded with ``inf``).
        """
        if not self._store:
            raise ValueError("query_external_batch requires store=True")
        if self._signatures is None or self._n_indexed == 0:
            raise RuntimeError("Must add signatures and call index() first")
        self._validate_query_parameters(k, kc)
        self._validate_signature_shape(signatures, batch=True)
        index_signatures = self._get_index_signatures()
        if (
            index_signatures is None
            or self._sorted_indices_flat is None
            or self._band_offsets is None
        ):
            raise RuntimeError("Index structures not initialized")

        max_candidates = min(k * kc, self._n_indexed)
        query_fn = query_lsh_forest_batch_weighted if self._weighted else query_lsh_forest_batch
        all_candidates, candidate_counts = query_fn(
            signatures,
            index_signatures,
            self._sorted_indices_flat,
            self._band_offsets,
            self._k,
            max_candidates,
        )

        # Batch linear scan with exclude_self=False
        if self._weighted:
            indices, distances = linear_scan_batch_weighted(
                signatures,
                self._signatures,
                all_candidates,
                candidate_counts,
                k,
                False,
                0,
            )
        else:
            indices, distances = linear_scan_batch(
                signatures,
                self._signatures,
                all_candidates,
                candidate_counts,
                k,
                False,
                0,
            )

        # Replace sentinel 2.0 distances with inf for consistency
        distances[indices < 0] = np.inf

        return indices, distances

    # Distance methods

    @staticmethod
    def get_distance(
        sig_a: NDArray[np.uint64],
        sig_b: NDArray[np.uint64],
    ) -> float:
        """
        Calculate Jaccard distance between two MinHash signatures.

        Args:
            sig_a: First MinHash vector
            sig_b: Second MinHash vector

        Returns:
            Jaccard distance (0.0 to 1.0)
        """
        return cast(float, jaccard_distance(sig_a, sig_b))

    @staticmethod
    def get_weighted_distance(
        sig_a: NDArray[np.uint64],
        sig_b: NDArray[np.uint64],
    ) -> float:
        """
        Calculate weighted Jaccard distance between two weighted MinHash signatures.

        Args:
            sig_a: First weighted MinHash vector, shape (d, 2)
            sig_b: Second weighted MinHash vector, shape (d, 2)

        Returns:
            Weighted Jaccard distance (0.0 to 1.0)
        """
        return cast(float, weighted_jaccard_distance(sig_a, sig_b))

    def get_distance_by_id(self, a: int, b: int) -> float:
        """
        Calculate Jaccard distance between two indexed signatures.

        Args:
            a: Index of first signature
            b: Index of second signature

        Returns:
            Jaccard distance

        Raises:
            ValueError: If store=False
        """
        if not self._store:
            raise ValueError("get_distance_by_id requires store=True")
        if self._signatures is None:
            raise RuntimeError("Must call index() first")

        return self._compute_distance(self._signatures[a], self._signatures[b])

    def get_all_distances(
        self,
        signature: NDArray[np.uint64],
    ) -> NDArray[np.float32]:
        """
        Calculate distances from a signature to all indexed signatures.

        Args:
            signature: Query MinHash vector

        Returns:
            Array of distances, shape (n_indexed,)

        Raises:
            ValueError: If store=False
        """
        if not self._store:
            raise ValueError("get_all_distances requires store=True")
        if self._signatures is None:
            raise RuntimeError("Must call index() first")

        self._validate_signature_shape(signature)

        if self._weighted:
            return cast(
                NDArray[np.float32],
                compute_weighted_distances_to_candidates(signature, self._signatures),
            )
        else:
            return cast(
                NDArray[np.float32], compute_distances_to_candidates(signature, self._signatures)
            )

    def get_hash(self, id: int) -> NDArray[np.uint64]:
        """
        Retrieve the MinHash signature of an indexed entry.

        Args:
            id: Index of the signature (0-based, order of insertion)

        Returns:
            MinHash vector, shape (d,) or (d, 2) for weighted

        Raises:
            ValueError: If store=False
            IndexError: If id out of range
        """
        if not self._store:
            raise ValueError("get_hash requires store=True")
        if self._signatures is None:
            raise RuntimeError("Must call index() first")
        if id < 0 or id >= self._n_indexed:
            raise IndexError(f"ID {id} out of range [0, {self._n_indexed})")

        return cast(NDArray[np.uint64], self._signatures[id].copy())

    def save(self, path: str) -> None:
        """
        Serialize the LSH forest to disk.

        Args:
            path: File path for serialization
        """
        state = {
            "format_version": 2,
            "d": self._d,
            "l": self._l,
            "k": self._k,
            "store": self._store,
            "weighted": self._weighted,
            "index_signatures": self._get_index_signatures(),
            "signatures": self._signatures,
            "signatures_list": self._signatures_list,
            "sorted_indices_flat": self._sorted_indices_flat,
            "band_offsets": self._band_offsets,
            "n_indexed": self._n_indexed,
            "is_indexed": self._is_indexed,
            "needs_reindex": self._needs_reindex,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> LSHForest:
        """
        Load a serialized LSH forest from disk.

        Args:
            path: File path to load from

        Returns:
            Restored LSHForest instance, ready for queries
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls(
            d=state["d"],
            l=state["l"],
            store=state["store"],
            weighted=state["weighted"],
        )
        instance._k = state["k"]
        if state.get("format_version", 1) >= 2:
            instance._index_signatures = state["index_signatures"]
            instance._signatures = state["signatures"]
            instance._signatures_list = state["signatures_list"]
            instance._sorted_indices_flat = state["sorted_indices_flat"]
            instance._band_offsets = state["band_offsets"]
            instance._n_indexed = state["n_indexed"]
            instance._is_indexed = state["is_indexed"]
            instance._needs_reindex = state.get("needs_reindex", False)
        else:
            # Version 1 stored exact-band hash tables. Rebuild them as adaptive
            # prefix trees rather than silently retaining the defective index.
            parts: list[NDArray[np.uint64]] = []
            if state.get("signatures") is not None:
                parts.append(state["signatures"])
            parts.extend(state.get("signatures_list", []))
            instance._signatures_list = parts
            instance.index()

        return instance

    def clear(self) -> None:
        """Clear all added data and computed indices."""
        self._signatures_list = []
        self._index_signatures = None
        self._signatures = None
        self._sorted_indices_flat = None
        self._band_offsets = None
        self._n_indexed = 0
        self._is_indexed = False
        self._needs_reindex = False
