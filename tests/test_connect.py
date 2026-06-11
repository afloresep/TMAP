"""Tests for connecting a fragmented k-NN graph into a single tree."""

import numpy as np
import pytest

from tmap.graph.connect import connect_knn_components
from tmap.index.types import KNNGraph
from tmap.layout import OGDF_AVAILABLE


def _n_components(knn: KNNGraph) -> int:
    """Count connected components of an (undirected) KNNGraph."""
    from tmap.graph.connect import _DSU, _valid_edges

    src, dst, _ = _valid_edges(knn)
    dsu = _DSU(knn.n_nodes)
    for a, b in zip(src.tolist(), dst.tolist()):
        dsu.union(a, b)
    return dsu.n_sets


def _separated_blobs(n_per: int = 150, n_blobs: int = 4, dim: int = 8, seed: int = 0):
    """Far-apart Gaussian blobs: a low-k k-NN graph cannot bridge them."""
    rng = np.random.default_rng(seed)
    parts = []
    for c in range(n_blobs):
        center = np.zeros(dim)
        center[c] = 100.0
        parts.append(rng.normal(center, 0.5, size=(n_per, dim)))
    return np.vstack(parts).astype(np.float32)


def test_disconnected_knn_has_multiple_components():
    # Build a graph that is two disjoint triangles (no cross edges).
    indices = np.array([[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]], dtype=np.int32)
    distances = np.ones((6, 2), dtype=np.float32)
    knn = KNNGraph.from_arrays(indices, distances)
    assert _n_components(knn) == 2


def test_connect_without_index_uses_representative_fallback():
    # Two disjoint triangles, no requery callable -> fallback chaining.
    indices = np.array([[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]], dtype=np.int32)
    distances = np.ones((6, 2), dtype=np.float32)
    knn = KNNGraph.from_arrays(indices, distances)

    connected, n_comp, n_bridges = connect_knn_components(knn, requery=None)

    assert n_comp == 2
    assert n_bridges == 1  # components - 1
    assert _n_components(connected) == 1
    assert connected.n_nodes == 6


def test_connect_with_requery_finds_real_bridges():
    # A 1-D chain split into two halves; requery with larger k exposes the gap edge.
    n = 10
    coords = np.arange(n, dtype=np.float64)

    def requery(k: int) -> KNNGraph:
        d = np.abs(coords[:, None] - coords[None, :])
        return KNNGraph.from_distance_matrix(d, k=min(k, n - 1))

    # k=1 chain that skips the 4->5 link -> two components {0..4}, {5..9}.
    idx = np.array([[1], [0], [1], [2], [3], [6], [5], [6], [7], [8]], dtype=np.int32)
    dist = np.ones((n, 1), dtype=np.float32)
    knn = KNNGraph.from_arrays(idx, dist)
    assert _n_components(knn) == 2

    connected, n_comp, n_bridges = connect_knn_components(knn, requery=requery)
    assert n_comp == 2
    assert n_bridges == 1
    assert _n_components(connected) == 1


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not available")
def test_estimator_connects_forest_into_single_tree(recwarn):
    from tmap import TMAP

    X = _separated_blobs()
    n = X.shape[0]

    off = TMAP(metric="euclidean", n_neighbors=3, seed=0, connect_components=False).fit(X)
    assert _n_components(off.graph_) > 1  # genuinely a forest
    assert off.n_components_ is None

    on = TMAP(metric="euclidean", n_neighbors=3, seed=0).fit(X)
    assert on.n_components_ > 1
    assert on.n_bridges_ == on.n_components_ - 1
    # graph_ stays the raw (n, k) graph; the connected graph drives the tree.
    assert on.graph_.indices.shape == (n, 3)
    # Tree spans every node and a path exists across former components.
    assert on.tree_.edges.shape[0] == n - 1
    assert len(on.tree_.path(0, n - 1)) >= 2
    assert any("bridge" in str(w.message) for w in recwarn.list)
