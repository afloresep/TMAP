"""Tree extraction from k-NN graphs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from tmap.graph.types import Tree
from tmap.index.types import KNNGraph


def _edge_weights_from_knn(knn: KNNGraph) -> dict[tuple[int, int], float]:
    """Return the minimum observed weight for each undirected k-NN edge."""
    edge_weights: dict[tuple[int, int], float] = {}

    for i in range(knn.n_nodes):
        for j_idx in range(knn.k):
            j = int(knn.indices[i, j_idx])
            if j < 0 or j == i:
                continue

            weight = float(knn.distances[i, j_idx])
            if not np.isfinite(weight):
                continue

            key = (min(i, j), max(i, j))
            prev = edge_weights.get(key)
            if prev is None or weight < prev:
                edge_weights[key] = weight

    return edge_weights


def _tree_from_ogdf_edges(
    knn: KNNGraph,
    s: NDArray[np.uint32],
    t: NDArray[np.uint32],
) -> Tree:
    """Build a Tree from OGDF MST edge topology and original k-NN weights."""
    if len(s) == 0:
        return Tree(
            n_nodes=knn.n_nodes,
            edges=np.empty((0, 2), dtype=np.int32),
            weights=np.empty(0, dtype=np.float32),
            root=0,
        )

    edges = np.column_stack(
        [
            s.astype(np.int32, copy=False),
            t.astype(np.int32, copy=False),
        ]
    )

    edge_weights = _edge_weights_from_knn(knn)
    weights = np.ones(len(edges), dtype=np.float32)
    for idx, (src, tgt) in enumerate(edges):
        key = (min(int(src), int(tgt)), max(int(src), int(tgt)))
        weight = edge_weights.get(key)
        if weight is not None:
            weights[idx] = np.float32(weight)

    degree = np.zeros(knn.n_nodes, dtype=np.int32)
    np.add.at(degree, edges[:, 0], 1)
    np.add.at(degree, edges[:, 1], 1)

    return Tree(
        n_nodes=knn.n_nodes,
        edges=edges,
        weights=weights,
        root=int(np.argmax(degree)),
    )


def tree_from_knn_graph(knn: KNNGraph, config: Any | None = None) -> Tree:
    """Compute a Tree from a KNNGraph using OGDF's MST path."""
    from tmap.layout._ogdf import LayoutConfig, layout_from_knn_graph

    if config is None and LayoutConfig is not None:
        config = LayoutConfig()
        config.fme_iterations = 1
        config.deterministic = True
        config.seed = 0
        # This path only needs the MST topology (s, t), not coordinates, so skip
        # the (now default-on) adaptive layout work and crossing-reduction pass.
        config.adaptive = False
        config.untangle = False

    _, _, s, t = layout_from_knn_graph(knn, config=config, create_mst=True)
    return _tree_from_ogdf_edges(knn, s, t)
