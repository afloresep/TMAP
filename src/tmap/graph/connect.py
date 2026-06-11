"""Connect a disconnected k-NN graph into a single component."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from tmap.index.types import KNNGraph

RequeryFn = Callable[[int], KNNGraph]


class _DSU:
    """Disjoint-set union with path halving."""

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.n_sets = n

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return int(x)

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        self.n_sets -= 1
        return True

    @classmethod
    def from_labels(cls, labels: NDArray[np.int64], n_sets: int) -> _DSU:
        """Build a DSU whose sets are precomputed component labels (0..n_sets-1).

        Each node points one hop to its component's representative (the first
        node carrying that label), so no per-edge union pass is needed.
        """
        dsu = cls(labels.shape[0])
        _, first = np.unique(labels, return_index=True)
        dsu.parent = first[labels].astype(np.int64)
        dsu.n_sets = n_sets
        return dsu


def _valid_edges(
    knn: KNNGraph,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Flatten a KNNGraph to (src, dst, dist) arrays, dropping -1/self/out-of-range."""
    idx = knn.indices
    n, k = idx.shape
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = idx.ravel().astype(np.int64)
    dist = knn.distances.ravel().astype(np.float64)
    mask = (dst >= 0) & (dst < n) & (dst != src) & np.isfinite(dist)
    return src[mask], dst[mask], dist[mask]


def _components(
    n: int, src: NDArray[np.int64], dst: NDArray[np.int64]
) -> tuple[int, NDArray[np.int64]]:
    """(n_components, component-label-per-node) for the undirected edge set."""
    adj = coo_matrix((np.ones(src.size, dtype=np.int8), (src, dst)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    return n_comp, labels.astype(np.int64)


def _labels(dsu: _DSU, n: int) -> NDArray[np.int64]:
    """Current root (component id) of every node (vectorized full find)."""
    parent = dsu.parent
    root = np.arange(n)
    while True:
        up = parent[root]
        if np.array_equal(up, root):
            return root.astype(np.int64)
        root = up


def _augment(knn: KNNGraph, bridges: list[tuple[int, int, float]]) -> KNNGraph:
    """Return a KNNGraph with bridge edges appended as extra neighbor columns."""
    if not bridges:
        return knn
    n, k = knn.indices.shape
    inc = np.zeros(n, dtype=np.int64)
    for a, b, _ in bridges:
        inc[a] += 1
        inc[b] += 1
    pad = int(inc.max())

    idx = np.full((n, k + pad), -1, dtype=np.int32)
    dist = np.full((n, k + pad), np.inf, dtype=np.float32)
    idx[:, :k] = knn.indices
    dist[:, :k] = knn.distances

    nxt = np.full(n, k, dtype=np.int64)
    for a, b, w in bridges:
        idx[a, nxt[a]] = b
        dist[a, nxt[a]] = w
        nxt[a] += 1
        idx[b, nxt[b]] = a
        dist[b, nxt[b]] = w
        nxt[b] += 1
    return KNNGraph.from_arrays(idx, dist)


def connect_knn_components(
    knn: KNNGraph,
    requery: RequeryFn | None = None,
    *,
    max_neighbors: int = 512,
) -> tuple[KNNGraph, int, int]:
    """Join a disconnected k-NN graph into a single connected component.

    Parameters
    ----------
    knn : KNNGraph
        The graph to connect (may be disconnected).
    requery : callable, optional
        ``requery(k) -> KNNGraph`` producing a graph with ``k`` neighbors per
        node, used to discover cross-component bridges.  If ``None``, only the
        representative-chaining fallback is available.
    max_neighbors : int, default 512
        Cap on the neighbor count used while searching for bridges.

    Returns
    -------
    (knn, n_components, n_bridges)
        The (possibly augmented) graph, the number of components found in the
        input, and the number of bridge edges added.
    """
    n = knn.n_nodes
    if n < 2:
        return knn, 1, 0

    # 1. Components of the current graph. scipy's connected_components runs in C,
    #    avoiding a Python union pass over all n*k edges on every fit.
    src, dst, _ = _valid_edges(knn)
    n_components, labels = _components(n, src, dst)
    if n_components == 1:
        return knn, 1, 0

    # Seed the union-find from those labels so the bridging sweep starts from the
    # real components without rescanning edges.
    dsu = _DSU.from_labels(labels, n_components)
    bridges: list[tuple[int, int, float]] = []

    # 2. Boruvka/Kruskal sweep. Re-query with a growing neighbor count and add
    #    the shortest cross-component edges until everything is joined.
    if requery is not None:
        k_try = max(knn.k * 4, 16)
        while dsu.n_sets > 1:
            k_eff = min(k_try, n - 1, max_neighbors)
            cand = requery(k_eff)
            csrc, cdst, cdist = _valid_edges(cand)
            comp = _labels(dsu, n)
            cross = comp[csrc] != comp[cdst]
            csrc, cdst, cdist = csrc[cross], cdst[cross], cdist[cross]
            if csrc.size:
                order = np.argsort(cdist, kind="stable")
                for e in order:
                    a, b, w = int(csrc[e]), int(cdst[e]), float(cdist[e])
                    if dsu.union(a, b):
                        bridges.append((a, b, w))
                        if dsu.n_sets == 1:
                            break
            if k_eff >= min(n - 1, max_neighbors):
                # neighbor count maxed out
                break
            k_try *= 2

    # 3. Last resort: chain any still-separate components through representatives.
    if dsu.n_sets > 1:
        reps: dict[int, int] = {}
        for i in range(n):
            r = dsu.find(i)
            if r not in reps:
                reps[r] = i
        rep_nodes = list(reps.values())

        # Weight bridges as the most expensive edges so the MST adds them last.
        fallback_w = max((w for _, _, w in bridges), default=0.0)
        if fallback_w == 0.0:
            _, _, d = _valid_edges(knn)
            fallback_w = float(np.max(d)) if d.size else 1.0
        for a, b in zip(rep_nodes[:-1], rep_nodes[1:]):
            if dsu.union(a, b):
                bridges.append((a, b, fallback_w))

    return _augment(knn, bridges), n_components, len(bridges)
