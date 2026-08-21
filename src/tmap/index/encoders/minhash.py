"""
MinHash encoders for binary/set data and weighted vectors.

This module provides two MinHash implementations:
- MinHash: For binary fingerprints and set data (e.g., molecular fingerprints)
- WeightedMinHash: For weighted/float vectors (e.g., count vectors)

Performance Notes:
    Binary data uses the Numba backend (50-100x faster than datasketch).
    String data uses xxhash64 for token hashing, then the same Numba backend.

API Compatibility:
    All public methods maintain backward compatibility with the original TMAP API.
"""

import os
from collections.abc import Collection, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ._minhash_numba import (
    binary_to_sparse,
    init_permutations,
    minhash_batch_from_dense,
    minhash_batch_from_sparse,
)

__all__ = [
    "MinHash",
    "WeightedMinHash",
]


def _get_weighted_minhash_generator() -> Any:
    """Load datasketch's weighted MinHash generator on demand."""
    try:
        from datasketch.weighted_minhash import WeightedMinHashGenerator
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "WeightedMinHash requires optional dependency 'datasketch'. "
            "Install it with `pip install datasketch`."
        ) from exc
    return WeightedMinHashGenerator


def _get_xxhash() -> Any:
    """Load xxhash only when string token hashing is requested."""
    try:
        import xxhash
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "String MinHash encoding requires optional dependency 'xxhash'. "
            "Install it with `pip install xxhash`."
        ) from exc
    return xxhash


def _encode_weighted_chunk(
    args: tuple[NDArray[np.float32], int, int, int],
) -> NDArray[np.uint64]:
    """Encode a chunk of weighted vectors (creates its own generator)."""
    data, dim, num_perm, seed = args
    generator_cls = _get_weighted_minhash_generator()
    generator = generator_cls(dim=dim, sample_size=num_perm, seed=seed)
    n_samples = data.shape[0]
    signatures = np.zeros((n_samples, num_perm, 2), dtype=np.uint64)
    for i in range(n_samples):
        signatures[i] = generator.minhash(data[i]).hashvalues
    return signatures


class MinHash:
    """MinHash encoder for binary/set data (e.g., molecular fingerprints).

    Automatically selects the optimal backend:
    - Binary arrays -> Numba JIT
    - String data -> xxhash64 token IDs -> Numba JIT

    Args:
        num_perm: Number of permutation functions (hash functions). Higher values
            give more accurate Jaccard similarity estimates but use more memory.
        seed: Random seed for reproducibility.

    Example:
        >>> mh = MinHash(num_perm=128, seed=42)
        >>> # From binary fingerprint (uses Numba - fast!)
        >>> fp = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        >>> sig = mh.from_binary_array(fp)
        >>> # From sparse indices (uses Numba)
        >>> sig = mh.from_sparse_binary_array([0, 2, 3, 6])
        >>> # From strings
        >>> sig = mh.from_string_array(["hello", "world"])
        >>> # Batch processing (very fast with Numba!)
        >>> fps = np.random.randint(0, 2, size=(10000, 2048), dtype=np.uint8)
        >>> sigs = mh.batch_from_binary_array(fps)

    Notes:
        - Binary data uses universal hash function (a*x + b) mod prime
        - String data is hashed to int64 IDs via xxhash64, then uses the same
          universal hash
        - Signatures from binary and string inputs are NOT comparable
          (binary uses column indices as element IDs, strings use xxhash outputs)
    """

    def __init__(self, num_perm: int = 128, seed: int = 1):
        self._num_perm = num_perm
        self._seed = seed

        # Pre-compute permutation parameters (same formula as datasketch)
        self._a, self._b = init_permutations(num_perm, seed)

    def encode(
        self, data: NDArray[np.uint8] | Sequence[Collection[int | str]]
    ) -> NDArray[np.uint64]:
        """
        Encode data into MinHash signatures.

        Automatically selects the optimal backend based on input type:
        - NumPy array -> Numba JIT
        - Sequence of sets/lists -> xxhash64 + Numba JIT

        Args:
            data: EITHER
                - (n_samples, n_features) binary NumPy array (0/1 values)
                - sequence of collections (e.g., sets/lists of ints or strings)

        Returns:
            signatures: (n_samples, num_perm) uint64 array
        """
        # Fast path: binary numpy array with Numba
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                raise ValueError(
                    "encode() received a 1D array. If this is a single binary "
                    "vector, reshape to (1, n_features). If these are element "
                    "IDs, use from_sparse_binary_array() instead."
                )
            return self._encode_binary_numba(data)

        if isinstance(data, list) and all(isinstance(s, set) for s in data):
            sets = data
        else:
            sets = [set(s) for s in data]

        # Peek at first non-empty set to determine element type
        first_elem = None
        for s in sets:
            if s:
                first_elem = next(iter(s))
                break

        if first_elem is not None and isinstance(first_elem, str):
            # Validate all elements are strings
            for s in sets:
                bad = [e for e in s if not isinstance(e, str)]
                if bad:
                    raise TypeError(
                        f"Sets contain mixed types. First non-string element: "
                        f"{bad[0]!r} ({type(bad[0]).__name__}). "
                        f"All elements must be the same type (all str or all int)."
                    )
            return self._encode_strings(sets)

        # Validate all elements are numeric (not strings mixed in)
        for s in sets:
            for e in s:
                if isinstance(e, str):
                    raise TypeError(
                        f"Sets contain mixed types: found str {e!r} in an "
                        f"integer set. All elements must be the same type."
                    )

        # Integer sets -> sparse numba path (integers are already valid IDs)
        return self.batch_from_sparse_binary_array([list(s) for s in sets])

    def _encode_binary_numba(self, data: NDArray[np.uint8]) -> NDArray[np.uint64]:
        """Fast binary array encoding using Numba JIT."""
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_samples = data.shape[0]

        # Choose between dense and sparse based on fill rate
        # Sparse is better for fill rates < ~30%
        fill_rate = np.mean(data)

        if fill_rate < 0.3:
            # Sparse path: convert to CSR-like format
            indices_flat, offsets = binary_to_sparse(data)
            return cast(
                NDArray[np.uint64],
                minhash_batch_from_sparse(
                    indices_flat,
                    offsets,
                    self._a,
                    self._b,
                    self._num_perm,
                    n_samples,
                ),
            )
        else:
            # Dense path: process directly
            return cast(
                NDArray[np.uint64],
                minhash_batch_from_dense(
                    data,
                    self._a,
                    self._b,
                    self._num_perm,
                ),
            )

    @staticmethod
    def get_distance(vec_a: NDArray[np.uint64], vec_b: NDArray[np.uint64]) -> float:
        """
        Calculate the Jaccard distance between two MinHash vectors.

        Jaccard distance = 1 - (number of matching hash values) / num_perm

        Args:
            vec_a: A MinHash vector
            vec_b: A MinHash vector

        Returns:
            The estimated Jaccard distance (0.0 to 1.0)
        """
        if len(vec_a) != len(vec_b):
            raise ValueError(
                "Cannot compute Jaccard distance: MinHash vectors have different "
                f"numbers of permutations ({len(vec_a)} vs {len(vec_b)})"
            )
        return float(1 - np.mean(vec_a == vec_b))

    # Alias for backward compatibility
    jaccard_distance = get_distance

    def from_binary_array(self, arr: NDArray[np.uint8] | list[int]) -> NDArray[np.uint64]:
        """
        Create a MinHash signature from a single binary vector.

        Args:
            arr: 1D binary array (0/1 values) where 1s indicate set membership

        Returns:
            1D array of shape (num_perm,) containing the MinHash signature
        """
        if isinstance(arr, list):
            arr = np.array(arr, dtype=np.uint8)
        if len(arr.shape) > 1:
            raise ValueError("vector must be 1D")
        return cast(NDArray[np.uint64], self.encode(arr.reshape(1, -1))[0])

    def from_sparse_binary_array(self, indices: Sequence[int]) -> NDArray[np.uint64]:
        """
        Create a MinHash signature from sparse representation (list of indices).

        Args:
            indices: 1D sequence of integers representing positions of 1s

        Returns:
            1D array of shape (num_perm,) containing the MinHash signature
        """
        # Reject nested sequences/arrays like [[1, 2], [3]]
        if isinstance(indices, np.ndarray):
            if indices.ndim != 1:
                raise ValueError(f"indices must be 1D, got array with shape {indices.shape}")
        else:
            if any(isinstance(x, (list, tuple, set, dict, np.ndarray)) for x in indices):
                raise ValueError("indices must be a 1D sequence of ints, not a nested sequence")

        indices_arr = np.array(indices, dtype=np.int64)
        if indices_arr.size > 0 and indices_arr.min() < 0:
            raise ValueError(
                f"Indices must be non-negative. Got minimum index {indices_arr.min()}."
            )
        offsets = np.array([0, len(indices)], dtype=np.int64)
        return cast(
            NDArray[np.uint64],
            minhash_batch_from_sparse(
                indices_arr,
                offsets,
                self._a,
                self._b,
                self._num_perm,
                1,
            )[0],
        )

    def _encode_strings(self, sets: list[set[str]]) -> NDArray[np.uint64]:
        """Encode batches of string tokens through the sparse MinHash path."""
        xxhash = _get_xxhash()
        cache: dict[str, int] = {}
        # CSR
        indices_flat: list[int] = []
        offsets = [0]
        n_tokens = 0

        for token_set in sets:
            for token in token_set:
                n_tokens += 1
                if token not in cache:
                    # xxhas64 could be up to 2^64 but numpy int64 holds up to  2^63 -1
                    # so mask to 63 bits ANDing the number

                    cache[token] = xxhash.xxh64_intdigest(token.encode("utf-8")) & 0x7FFFFFFFFFFFFFFF
                indices_flat.append(cache[token])
            offsets.append(n_tokens)

        return cast(
            NDArray[np.uint64],
            minhash_batch_from_sparse(
                np.array(indices_flat, dtype=np.int64),
                np.array(offsets, dtype=np.int64),
                self._a,
                self._b,
                self._num_perm,
                len(sets),
            ),
        )

    def from_string_array(self, strings: Sequence[str]) -> NDArray[np.uint64]:
        """
        Create a MinHash signature from a list of strings.

        Args:
            strings: 1D sequence of strings to be treated as set elements

        Returns:
            1D array of shape (num_perm,) containing the MinHash signature
        """
        # Reject nested sequences/arrays like [["a"], ["b"]]
        if isinstance(strings, np.ndarray):
            if strings.ndim != 1:
                raise ValueError(f"strings must be 1D, got array with shape {strings.shape}")
        else:
            if any(isinstance(x, (list, tuple, set, dict, np.ndarray)) for x in strings):
                raise ValueError("strings must be a 1D sequence of str, not a nested sequence")

        if not all(isinstance(x, str) for x in strings):
            raise ValueError("All elements must be strings")

        return cast(NDArray[np.uint64], self._encode_strings([set(strings)])[0])

    # Batch methods
    def batch_from_binary_array(
        self, arrays: Sequence[NDArray[np.uint8]] | NDArray[np.uint8]
    ) -> NDArray[np.uint64]:
        """
        Create MinHash signatures from multiple binary vectors.

        This is the fastest method for encoding many fingerprints.

        Args:
            arrays: Either a 2D array of shape (n_samples, n_features) or
                    a sequence of 1D binary arrays

        Returns:
            2D array of shape (n_samples, num_perm) containing MinHash signatures
        """
        # Ensure we have a 2D array
        if isinstance(arrays, np.ndarray) and arrays.ndim == 2:
            data = arrays.astype(np.uint8, copy=False)
        else:
            data = np.stack([np.asarray(arr, dtype=np.uint8) for arr in arrays])

        return self._encode_binary_numba(data)

    def batch_from_sparse_binary_array(
        self, indices_list: Sequence[Sequence[int]]
    ) -> NDArray[np.uint64]:
        """
        Create MinHash signatures from multiple sparse representations.

        Args:
            indices_list: Sequence of sequences, where each inner sequence contains
                          the indices of 1s in a sparse binary vector

        Returns:
            2D array of shape (n_samples, num_perm) containing MinHash signatures
        """
        n_samples = len(indices_list)

        # Build CSR-like structure for Numba kernel
        lengths = [len(indices) for indices in indices_list]
        offsets = np.zeros(n_samples + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        total_nnz = offsets[-1]
        indices_flat = np.empty(total_nnz, dtype=np.int64)
        for i, indices in enumerate(indices_list):
            indices_flat[offsets[i] : offsets[i + 1]] = indices

        # Validate: reject non-integer floats and negative indices
        if total_nnz > 0:
            if indices_flat.min() < 0:
                raise ValueError(
                    f"Indices must be non-negative. Got minimum index {indices_flat.min()}."
                )
            # Check original values for float truncation
            for seq in indices_list:
                for val in seq:
                    if isinstance(val, float) and val != int(val):
                        raise TypeError(
                            f"Sparse indices must be integers, got float {val}. "
                            f"Non-integer floats would be silently truncated."
                        )

        return cast(
            NDArray[np.uint64],
            minhash_batch_from_sparse(
                indices_flat,
                offsets,
                self._a,
                self._b,
                self._num_perm,
                n_samples,
            ),
        )

    def batch_from_string_array(
        self,
        string_lists: Sequence[Sequence[str]],
    ) -> NDArray[np.uint64]:
        """
        Create MinHash signatures from multiple string lists.

        Args:
            string_lists: Sequence of string sequences, where each inner sequence
                          contains strings to be treated as set elements
        Returns:
            2D array of shape (n_samples, num_perm) containing MinHash signatures
        """
        return self._encode_strings([set(s) for s in string_lists])


# for integer/float data
class WeightedMinHash:
    """Weighted MinHash for float/integer vectors.

    Uses consistent weighted sampling to create MinHash signatures that
    estimate weighted Jaccard similarity.

    Args:
        dim: Number of features (must know upfront for generator)
        num_perm: Number of hash permutations
        seed: Random seed for reproducibility
    """

    def __init__(self, dim: int, num_perm: int = 128, seed: int = 1):
        self._num_perm = num_perm
        self._seed = seed
        self._dim = dim

        # Generator must be created with known dimension
        generator_cls = _get_weighted_minhash_generator()
        self._generator = generator_cls(dim=dim, sample_size=num_perm, seed=seed)

    def encode(self, data: NDArray[np.float32]) -> NDArray[np.uint64]:
        """
        Encode weighted vectors into MinHash signatures.

        Args:
            data: (n_samples, n_features) array of positive weights (> 0)

        Returns:
            signatures: (n_samples, num_perm, 2) uint64 array
                Each signature has 2 values per hash (k, y_k from the algorithm)
        """
        n_samples, n_features = data.shape
        if not np.all(data > 0):
            raise ValueError("Vector must contain only positive values")
        if n_features != self._dim:
            raise ValueError(f"Expected {self._dim} features, got {n_features}")

        signatures = np.zeros(shape=(n_samples, self._num_perm, 2), dtype=np.uint64)

        for i in range(n_samples):
            signatures[i] = self._generator.minhash(data[i]).hashvalues

        return signatures

    def from_weight_array(self, vec: NDArray[np.floating]) -> NDArray[np.uint64]:
        """
        Create a weighted MinHash vector from a float array.

        Args:
            vec: 1D array of positive float values (> 0)

        Returns:
            2D array of shape (num_perm, 2) containing the weighted MinHash signature

        Note:
            The original TMAP supported 'method' parameter for ICWS or I2CWS.
            This implementation uses datasketch's consistent weighted sampling.
        """
        return cast(NDArray[np.uint64], self.encode(vec.reshape(1, -1))[0])

    def batch_from_weight_array(
        self,
        vectors: Sequence[NDArray[np.floating]] | NDArray[np.floating],
        n_jobs: int | None = None,
    ) -> NDArray[np.uint64]:
        """
        Create weighted MinHash signatures from multiple weight vectors (parallelized).

        Args:
            vectors: Either a 2D array of shape (n_samples, dim) or
                     a sequence of 1D weight arrays
            n_jobs: Number of parallel workers. None = number of CPUs.
                    Note: Currently parallelization is limited because the
                    WeightedMinHashGenerator is not easily serializable.

        Returns:
            3D array of shape (n_samples, num_perm, 2) containing weighted MinHash signatures
        """
        if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
            data = vectors.astype(np.float32, copy=False)
        else:
            data = np.stack([np.asarray(vec, dtype=np.float32) for vec in vectors])

        n_samples = data.shape[0]

        # For large datasets with n_jobs > 1, use chunk-based parallelism
        # Each worker gets its own generator to avoid serialization issues
        if n_jobs is None:
            n_jobs = os.cpu_count() or 1

        if n_samples < 100 or n_jobs == 1:
            return self.encode(data.astype(np.float32))

        # Chunk the data for parallel processing
        chunk_size = max(1, n_samples // n_jobs)
        chunks = [
            (data[i : i + chunk_size], self._dim, self._num_perm, self._seed)
            for i in range(0, n_samples, chunk_size)
        ]

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(_encode_weighted_chunk, chunks))

        return np.concatenate(results, axis=0)

    @staticmethod
    def get_weighted_distance(vec_a: NDArray[np.uint64], vec_b: NDArray[np.uint64]) -> float:
        """
        Calculate the weighted Jaccard distance between two weighted MinHash vectors.

        For weighted MinHash, two hashes match only if both components (k, y_k) match.
        Distance = 1 - (number of matching rows) / num_perm

        Args:
            vec_a: A weighted MinHash vector of shape (num_perm, 2)
            vec_b: A weighted MinHash vector of shape (num_perm, 2)

        Returns:
            The estimated weighted Jaccard distance (0.0 to 1.0)
        """
        if vec_a.shape != vec_b.shape:
            raise ValueError(f"Shape mismatch: vec_a {vec_a.shape} vs vec_b {vec_b.shape}")
        if vec_a.ndim != 2 or vec_a.shape[1] != 2:
            raise ValueError(f"Expected shape (num_perm, 2), got {vec_a.shape}")
        # Both columns must match for a row to count as matching
        matches = np.all(vec_a == vec_b, axis=1)
        return float(1 - np.mean(matches))
