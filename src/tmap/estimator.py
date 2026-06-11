from __future__ import annotations

import math
import pickle
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from numpy.typing import NDArray

from tmap.graph.connect import RequeryFn, connect_knn_components
from tmap.graph.mst import _tree_from_ogdf_edges, tree_from_knn_graph
from tmap.graph.types import Tree
from tmap.index.lsh_forest import LSHForest
from tmap.index.types import KNNGraph
from tmap.index.usearch_index import USearchIndex
from tmap.layout._ogdf import (
    LayoutConfig,
    layout_from_knn_graph,
    require_ogdf,
)

if TYPE_CHECKING:
    from tmap.visualization import TmapViz


def _resolve_ann_backend(
    seed: int | None = None,
    threads: int = 0,
) -> USearchIndex:
    """Return an ANN index for cosine/euclidean kNN search.

    Uses USearch (included as a core dependency).
    """
    return USearchIndex(seed=seed, threads=threads)


def _select_lsh_l(d: int, n_samples: int) -> int:
    """Auto-select number of LSH prefix trees based on d and dataset size.

    The LSH band width (k_band = d // l) controls discrimination:
    - Short bands (small k_band): high recall but many false positives.
      At large N, the candidate budget (k*kc) overflows with random matches.
    - Long bands (large k_band): precise candidates but low recall.
      At small N, too few collisions to find any neighbors.

    We use k_band=4 up to ~1M points (tested optimal for 100k molecules),
    then gradually increase for larger datasets where the collision pool
    would otherwise overwhelm the candidate budget.
    """
    if n_samples <= 1_000_000:
        k_band = 4
    else:
        # Scale up gently: k_band=5 at ~2M, 6 at ~4M, 7 at ~8M
        k_band = max(4, 4 + round(math.log2(n_samples / 1_000_000)))
    n_trees = max(8, d // k_band)
    return min(n_trees, d)


def _make_minhash_encoder(num_perm: int, seed: int) -> Any:
    """Create a MinHash encoder with a clear dependency error if missing."""
    try:
        from tmap.index.encoders.minhash import MinHash
    except ModuleNotFoundError as exc:
        if exc.name in {"datasketch", "xxhash"}:
            raise ModuleNotFoundError(
                "metric='jaccard' requires optional dependencies "
                "'datasketch' and 'xxhash'. Install them with:\n"
                "  pip install datasketch xxhash"
            ) from exc
        raise
    return MinHash(num_perm=num_perm, seed=seed)


class TMAP:
    """Build a tree-shaped 2D map from high-dimensional data.

    TMAP builds a k-nearest-neighbor graph, extracts a tree, and lays that
    tree out in 2D. Use it for binary fingerprints, dense embeddings, or
    precomputed distances.

    Args:
        n_neighbors: Number of neighbors per point.
        metric: Distance metric. Use ``"jaccard"`` for binary data,
            ``"cosine"`` or ``"euclidean"`` for dense vectors, or
            ``"precomputed"`` if you already have distances.
        n_permutations: Number of MinHash permutations for jaccard.
        kc: Search multiplier for the LSH forest. Larger values usually give
            better recall but cost more time.
        seed: Random seed for the OGDF layout. The MinHash + LSH kNN path
            is fully deterministic for a given seed. The USearch HNSW path
            (binary jaccard, cosine, euclidean) is multi-threaded by default
            and may return slightly different equidistant neighbors across
            runs, which can change the tree topology and layout. Pass
            ``reproducible=True`` to force a deterministic build.
        minhash_seed: Random seed for MinHash when metric is ``"jaccard"``.
        layout_iterations: Number of OGDF layout iterations.
        layout_config: Optional advanced OGDF layout config. Its
            ``deterministic`` and ``seed`` fields are managed by the estimator
            (always a seeded, deterministic layout) and are set on the object
            you pass, in place; set other knobs (e.g. ``ns_cap``, ``untangle``)
            freely. For a nondeterministic / multi-threaded layout, use
            ``tmap.layout.layout_from_knn_graph``.
        store_index: If True, keep the dense ANN index after ``fit()`` so you
            can later call ``transform()`` or ``add_points()``.
        reproducible: If True, build the USearch HNSW index single-threaded
            so the kNN graph is bit-identical across runs. Slower (roughly
            5-7x for HNSW build at 10k+ points), but guarantees the same
            coordinates from the same data and seed. Default ``False``.
        connect_components: If True (default), bridge a disconnected kNN graph
            (which a too-low ``n_neighbors`` can produce) into a single tree by
            adding minimum-weight cross-component edges, so ``path``/``distance``
            stay defined and the layout is connected. Set ``False`` to keep the
            forest. See ``n_components_`` for how many components were found.

    Example:
        >>> model = TMAP(metric="jaccard", n_neighbors=20).fit(binary_matrix)
        >>> coords = model.embedding_
        >>> model = TMAP(metric="cosine", n_neighbors=20).fit(embeddings)
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        metric: str = "jaccard",
        n_permutations: int = 512,
        kc: int = 50,
        seed: int = 42,
        minhash_seed: int = 42,
        layout_iterations: int = 1000,
        layout_config: Any | None = None,
        store_index: bool = False,
        reproducible: bool = False,
        connect_components: bool = True,
    ) -> None:
        if n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be > 0, got {n_neighbors}")
        if n_permutations <= 0:
            raise ValueError(f"n_permutations must be > 0, got {n_permutations}")
        if kc <= 0:
            raise ValueError(f"kc must be > 0, got {kc}")

        metric = metric.lower()
        valid_metrics = {"jaccard", "precomputed", "cosine", "euclidean"}
        if metric not in valid_metrics:
            valid_list = ", ".join(sorted(valid_metrics))
            raise ValueError(f"Unsupported metric {metric!r}. Supported metrics: {valid_list}")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_permutations = n_permutations
        self.kc = kc
        self.seed = seed
        self.minhash_seed = minhash_seed
        self.layout_iterations = layout_iterations
        self.layout_config = layout_config
        self.store_index = store_index
        self.reproducible = reproducible
        self.connect_components = connect_components

        self._embedding: NDArray[np.float32] | None = None
        self._n_components: int | None = None
        self._n_bridges: int = 0
        self._connected_graph: KNNGraph | None = None
        self._index: Any | None = None
        self._tree: Tree | None = None
        self._graph: KNNGraph | None = None
        self._lsh_forest: LSHForest | None = None
        self._n_features: int | None = None
        self._jaccard_mode: str | None = None  # "binary", "sets", or "strings"

    def fit(
        self,
        X: Any | None = None,
        *,
        knn_graph: KNNGraph | None = None,
    ) -> Self:
        """Fit the model and build the map.

        Pass either raw data in ``X`` or a precomputed ``knn_graph``.

        Args:
            X: Input data for the selected metric.
            knn_graph: Precomputed neighbor graph. If given, TMAP skips the
                neighbor search step.

        Returns:
            The fitted model.
        """
        if knn_graph is not None and X is not None:
            raise ValueError("Pass either X or knn_graph, not both.")
        if knn_graph is None and X is None:
            raise ValueError("Either X or knn_graph must be provided.")

        # Closure used to look for cross-component bridges if the k-NN graph is
        # disconnected (see connect_knn_components). Set per backend below; left
        # None for a user-supplied knn_graph (no index to re-query).
        requery: RequeryFn | None = None

        if knn_graph is not None:
            knn = knn_graph
            self._lsh_forest = None
            self._n_features = None
        elif self.metric == "jaccard":
            # Try the fast USearch path for binary data first. If coercion
            # fails (non-0/1 values, ragged lists, etc.), fall through to
            # the MinHash + LSH path which handles sets and strings.
            _use_usearch = False
            if self._is_binary_input(X):
                try:
                    binary = self._coerce_binary_matrix(X)
                    _use_usearch = True
                except (ValueError, TypeError):
                    pass

            if _use_usearch:
                n_samples, n_features = binary.shape
                self._n_features = n_features
                self._jaccard_mode = "binary"
                if self.n_neighbors >= n_samples:
                    raise ValueError(
                        f"n_neighbors={self.n_neighbors} must be < n_samples={n_samples}"
                    )
                index = USearchIndex(
                    seed=self.seed,
                    expansion_search=512,
                    threads=1 if self.reproducible else 0,
                )
                index.build_from_binary(binary)
                knn = index.query_knn(k=self.n_neighbors)
                self._index = index
                self._lsh_forest = None
                requery = lambda kk: index.query_knn(k=kk)  # noqa: E731
            else:
                # Sets/strings → MinHash + LSH Forest.
                signatures, n_samples, n_features = self._encode_jaccard(X)
                self._n_features = n_features

                lsh_l = _select_lsh_l(self.n_permutations, n_samples)
                forest = LSHForest(d=self.n_permutations, l=lsh_l)
                forest.batch_add(signatures)
                del signatures  # forest has its own copy
                forest.index()
                knn = forest.get_knn_graph(k=self.n_neighbors, kc=self.kc)

                # Check that LSH returned a usable graph.
                n_missing = int(np.sum(knn.indices == -1))
                n_total = knn.indices.size
                if n_total > 0 and n_missing == n_total:
                    raise ValueError(
                        "kNN graph is completely empty (all neighbors are -1). "
                        "The data may be too sparse for LSH to find any "
                        "neighbors. Try increasing kc, n_permutations, or "
                        "check that input rows have sufficient overlap."
                    )
                if n_total > 0 and n_missing / n_total > 0.9:
                    warnings.warn(
                        f"kNN graph is very sparse: {n_missing}/{n_total} "
                        f"neighbor slots are empty (-1). Embedding quality "
                        f"may be poor. Consider increasing kc (currently "
                        f"{self.kc}) or n_permutations (currently "
                        f"{self.n_permutations}).",
                        UserWarning,
                        stacklevel=2,
                    )

                self._lsh_forest = forest
                self._index = None
                requery = lambda kk: forest.get_knn_graph(k=kk, kc=self.kc)  # noqa: E731
        elif self.metric == "precomputed":
            distance_matrix = self._coerce_distance_matrix(X)
            knn = KNNGraph.from_distance_matrix(distance_matrix, k=self.n_neighbors)
            self._lsh_forest = None
            self._n_features = None
            requery = lambda kk: KNNGraph.from_distance_matrix(distance_matrix, k=kk)  # noqa: E731
        elif self.metric in {"cosine", "euclidean"}:
            X_dense = self._coerce_dense_matrix(X)
            if self.n_neighbors >= X_dense.shape[0]:
                raise ValueError(
                    f"n_neighbors={self.n_neighbors} must be < n_samples={X_dense.shape[0]}"
                )
            index = _resolve_ann_backend(
                seed=self.seed,
                threads=1 if self.reproducible else 0,
            )
            index.build_from_vectors(X_dense, metric=self.metric)
            knn = index.query_knn(k=self.n_neighbors)
            self._lsh_forest = None
            self._n_features = X_dense.shape[1]
            self._index = index if self.store_index else None
            requery = lambda kk: index.query_knn(k=kk)  # noqa: E731
        else:
            # Defensive fallback; __init__ already validates metrics.
            raise ValueError(f"Unsupported metric {self.metric!r}")

        self._graph = knn  # the raw k-NN graph as the backend returned it

        # A k-NN graph with too-low k can fragment into several components, whose
        # MST is a forest. Bridge them into one connected graph (default on) so
        # path/distance stay well-defined and the layout is connected. graph_ keeps
        # the raw graph; this connected graph drives the tree and the embedding.
        if self.connect_components:
            connected, n_comp, n_bridges = connect_knn_components(knn, requery)
            self._n_components = n_comp
            self._n_bridges = n_bridges
            if n_bridges > 0:
                warnings.warn(
                    f"k-NN graph had {n_comp} disconnected components "
                    f"(n_neighbors={self.n_neighbors} may be too low); added "
                    f"{n_bridges} bridge edge(s) to form a single connected tree. "
                    f"Increase n_neighbors/kc for a more faithful graph, or pass "
                    f"connect_components=False to keep the forest.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            connected = knn
        self._connected_graph = connected

        require_ogdf()
        config = self._make_layout_config()
        x, y, s, t = layout_from_knn_graph(connected, config=config, create_mst=True)
        self._tree = _tree_from_ogdf_edges(connected, s, t)

        self._embedding = np.column_stack([x, y]).astype(np.float32, copy=False)
        return self

    def fit_transform(
        self,
        X: Any | None = None,
        *,
        knn_graph: KNNGraph | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]:
        """Fit the model and return coordinates and tree edges.

        Args:
            X: Input data for the selected metric.
            knn_graph: Precomputed neighbor graph.

        Returns:
            Tuple ``(x, y, s, t)`` where ``x`` and ``y`` are the coordinates
            and ``s`` and ``t`` are the tree edges.
        """
        self.fit(X, knn_graph=knn_graph)
        # Return x,y coordinates  + s,t edges
        return (
            self.embedding_[:, 0],
            self.embedding_[:, 1],
            self.tree_.edges[:, 0],
            self.tree_.edges[:, 1],
        )

    def kneighbors(
        self,
        X: Any,
        *,
        return_distance: bool = True,
    ) -> NDArray[np.int32] | tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Query nearest fitted neighbors for new points without mutating the model.

        Args:
            X: Query data in the same format accepted by ``transform()``.
            return_distance: If True, return ``(indices, distances)``.
                If False, return only ``indices``.

        Returns:
            Either ``indices`` with shape ``(m, k)`` or
            ``(indices, distances)`` with shapes ``(m, k)``.
        """
        if self._embedding is None or self._graph is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")

        indices, distances, _ = self._query_new_points(X, update_state=False)
        if return_distance:
            return indices, distances
        return indices

    def transform(self, X: Any) -> NDArray[np.float32]:
        """Place new points on the existing map without changing the model.

        The new points are matched to the fitted data and then placed near
        their nearest neighbors. The original map stays unchanged.

        Args:
            X: New data. For jaccard, pass the same kind of input you used in
                ``fit()``. For cosine and euclidean, pass a float matrix with
                the same number of features. For precomputed, pass distances
                from the new points to the fitted points.

        Returns:
            Array of shape ``(m, 2)`` with the new point coordinates.

        Raises:
            RuntimeError: If the model is not fitted.
            RuntimeError: If cosine or euclidean was fitted without
                ``store_index=True``.
            ValueError: If the input shape does not match the fitted data.
        """
        if self._embedding is None or self._graph is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")

        new_indices, _, m = self._query_new_points(X, update_state=False)
        if m == 0:
            return np.empty((0, 2), dtype=np.float32)
        return self._position_new_points(new_indices)

    @property
    def embedding_(self) -> NDArray[np.float32]:
        """Return the fitted 2D coordinates."""
        if self._embedding is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")
        return self._embedding

    @property
    def n_components_(self) -> int | None:
        """Number of connected components in the raw kNN graph at fit time.

        ``1`` means the graph was already connected. ``> 1`` means a too-low
        ``n_neighbors`` fragmented it and ``n_bridges_`` edges were added to
        join it (when ``connect_components=True``). ``None`` if components were
        not checked (``connect_components=False``).
        """
        return self._n_components

    @property
    def n_bridges_(self) -> int:
        """Number of bridge edges added to connect a fragmented kNN graph."""
        return self._n_bridges

    @property
    def tree_(self) -> Tree:
        """Return the fitted tree."""
        if self._tree is None:
            # Prefer the connected graph so the tree spans all nodes; fall back
            # to the raw graph (e.g. a model restored from an older save).
            source = getattr(self, "_connected_graph", None) or self._graph
            if source is None:
                raise RuntimeError("Estimator is not fitted. Call fit() first.")
            self._tree = tree_from_knn_graph(source)
        return self._tree

    @property
    def graph_(self) -> KNNGraph:
        """Return the fitted k-nearest-neighbor graph."""
        if self._graph is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")
        return self._graph

    @property
    def lsh_forest_(self) -> LSHForest:
        """Return the fitted LSH forest (sets/strings Jaccard only).

        Only available when ``metric='jaccard'`` and the input was
        variable-length (sets or strings). Binary matrix inputs use
        USearch instead — access via ``index_``.
        """
        if self._lsh_forest is None:
            raise RuntimeError(
                "No fitted LSHForest available. "
                "Binary Jaccard uses USearch (see index_). "
                "LSHForest is only used for set/string Jaccard inputs."
            )
        return self._lsh_forest

    @property
    def index_(self) -> Any:
        """Return the stored USearch index.

        Available for ``metric='cosine'``/``'euclidean'`` (requires
        ``store_index=True``) and ``metric='jaccard'`` with binary
        matrix input (always stored).
        """
        if self._index is None:
            raise RuntimeError(
                "No index stored. For cosine/euclidean, use "
                "store_index=True. For Jaccard with sets/strings, "
                "use lsh_forest_ instead."
            )
        return self._index

    def to_tmapviz(self, include_edges: bool = True) -> TmapViz:
        """Create a ``TmapViz`` object from the fitted model.

        Args:
            include_edges: If True, include the tree edges.

        Returns:
            A ``TmapViz`` object ready for HTML or notebook rendering.
        """
        from tmap.visualization import TmapViz

        embedding = self.embedding_
        tree = self.tree_

        viz = TmapViz()
        viz.set_points(embedding[:, 0], embedding[:, 1])

        if include_edges and len(tree.edges) > 0:
            viz.set_edges(
                tree.edges[:, 0].astype(np.uint32, copy=False),
                tree.edges[:, 1].astype(np.uint32, copy=False),
            )

        return viz

    def serve(self, port: int = 8050, include_edges: bool = True, **kwargs: Any) -> None:
        """Serve the fitted map on a local HTTP server.

        This is useful for larger datasets where a single self-contained HTML
        file would be too heavy.

        Args:
            port: TCP port for the local server.
            include_edges: If True, include the tree edges.
            **kwargs: Extra arguments passed to ``TmapViz.serve()``.
        """
        viz = self.to_tmapviz(include_edges=include_edges)
        viz.serve(port=port, **kwargs)

    def to_html(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        include_edges: bool = True,
    ) -> Path:
        """Write the fitted map to an HTML file.

        Args:
            path: Output file path.
            title: Optional page title.
            include_edges: If True, include the tree edges.

        Returns:
            The written file path.
        """
        viz = self.to_tmapviz(include_edges=include_edges)
        if title is not None:
            viz.title = title
        return viz.write_html(path)

    def save(self, path: str | Path) -> Path:
        """Save the fitted model to disk.

        The saved file includes the map, tree, graph, and any stored index.

        Args:
            path: Output file path, for example ``"model.tmap"``.

        Returns:
            The written file path.

        Example:
            >>> model = TMAP(metric="cosine", store_index=True).fit(X)
            >>> model.save("my_model.tmap")
        """
        if self._embedding is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: str | Path) -> TMAP:
        """Load a model saved with ``save()``.

        Args:
            path: File path written by ``save()``.

        Returns:
            The restored model.
        """
        path = Path(path)
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Expected a TMAP instance, got {type(model).__name__}")
        return model

    def plot(
        self,
        *,
        color_by: Any | None = None,
        color_map: str | list[str] | dict[str, str] | None = None,
        data: Any | None = None,
        tooltip_properties: list[str] | None = None,
        point_size: float = 3,
        opacity: float = 0.8,
        width: int | str = 800,
        height: int = 420,
        show: bool = True,
        controls: bool = False,
    ) -> Any:
        """Show the fitted map in a Jupyter notebook.

        This method needs ``jupyter-scatter``.

        Args:
            color_by: Column name or array used for coloring.
            color_map: Optional colormap override.
            data: Optional DataFrame with metadata.
            tooltip_properties: Column names to show on hover.
            point_size: Point size.
            opacity: Point opacity.
            width: Widget width. Use ``"auto"`` to follow the notebook cell.
            height: Widget height in pixels.
            show: If True, display the widget.
            controls: If True, show notebook controls when available.

        Returns:
            The configured notebook widget.
        """
        from tmap.visualization.jupyter import _display_scatter, to_jscatter

        scatter = to_jscatter(
            self.embedding_,
            color_by=color_by,
            color_map=color_map,
            data=data,
            tooltip_properties=tooltip_properties,
            point_size=point_size,
            opacity=opacity,
            width=width,
            height=height,
        )

        if show:
            _display_scatter(scatter, controls=controls)

        return scatter

    def plot_static(
        self,
        *,
        color_by: Any | None = None,
        color_map: str | None = None,
        data: Any | None = None,
        edges: bool = True,
        edge_color: str = "#cccccc",
        edge_alpha: float = 0.3,
        edge_linewidth: float = 0.3,
        point_size: float = 1.0,
        alpha: float = 0.8,
        ax: Any | None = None,
        figsize: tuple[float, float] = (8, 8),
    ) -> Any:
        """Render the embedding as a static matplotlib scatter plot.

        Parameters
        ----------
        color_by : str, array-like, or None
            Column name in *data*, or a raw array of values.
        color_map : str or None
            Matplotlib colormap name.
        data : DataFrame or None
            Metadata DataFrame.
        edges : bool, default True
            If True, draw tree edges behind the points.
        edge_color : str, default '#cccccc'
            Color for edge lines.
        edge_alpha : float, default 0.3
            Opacity for edge lines.
        edge_linewidth : float, default 0.3
            Line width for edges.
        point_size : float, default 1.0
            Marker size.
        alpha : float, default 0.8
            Point opacity.
        ax : matplotlib Axes or None
            Draw into an existing axes.
        figsize : tuple, default (8, 8)
            Figure size when *ax* is None.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from tmap.visualization.static import plot_static

        edge_arr = self.tree_.edges if edges else None
        return plot_static(
            self.embedding_,
            color_by=color_by,
            color_map=color_map,
            data=data,
            edges=edge_arr,
            edge_color=edge_color,
            edge_alpha=edge_alpha,
            edge_linewidth=edge_linewidth,
            point_size=point_size,
            alpha=alpha,
            ax=ax,
            figsize=figsize,
        )

    def add_points(self, X: Any) -> NDArray[np.float32]:
        """Add new points to the existing map without a full refit.

        New points are matched to the fitted data, placed near their nearest
        neighbors, and then appended to the tree, graph, and embedding.
        Existing coordinates stay unchanged.

        Args:
            X: New data. For jaccard, pass a binary matrix. For cosine and
                euclidean, pass a float matrix with the same number of
                features. For precomputed, pass distances from the new points
                to the fitted points.

        Returns:
            Array of shape ``(m, 2)`` with the new point coordinates.

        Raises:
            RuntimeError: If the model is not fitted.
            RuntimeError: If cosine or euclidean was fitted without
                ``store_index=True``.
            ValueError: If the input shape does not match the fitted data.

        Note:
            Later dense batches can see previously added dense points, but
            points inside the same batch are still queried against the
            pre-existing index only.
        """
        if self._embedding is None or self._graph is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")

        new_indices, new_distances, m = self._query_new_points(X, update_state=True)

        if m == 0:
            return np.empty((0, 2), dtype=np.float32)

        new_coords = self._position_new_points(new_indices)
        self._extend_tree(new_indices, new_distances, m)

        # Extend KNN graph
        self._graph = KNNGraph(
            indices=np.concatenate([self._graph.indices, new_indices]),
            distances=np.concatenate([self._graph.distances, new_distances]),
        )

        # Update embedding
        self._embedding = np.concatenate([self._embedding, new_coords])

        return new_coords

    def _query_new_points(
        self,
        X: Any,
        *,
        update_state: bool,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32], int]:
        """Dispatch neighbor queries based on metric.

        Returns ``(indices, distances, m)`` where shapes are ``(m, k)``.

        When ``update_state`` is ``True`` this method performs any side effects
        needed for incremental insertion (currently only Jaccard LSH updates).
        """
        k = self.n_neighbors
        _empty = (
            np.empty((0, k), dtype=np.int32),
            np.empty((0, k), dtype=np.float32),
            0,
        )

        if self.metric == "jaccard" and self._index is not None:
            # USearch binary Jaccard path (batch query).
            binary = self._coerce_binary_matrix(X, min_samples=0)
            if self._n_features is not None and binary.shape[1] != self._n_features:
                raise ValueError(
                    f"Feature dimension mismatch: fit() saw {self._n_features} "
                    f"features, but received {binary.shape[1]}."
                )
            m = binary.shape[0]
            if m == 0:
                return _empty

            all_indices, all_distances = self._index.query_batch(binary, k)

            if update_state:
                self._index.add(binary)

            return all_indices, all_distances, m

        elif self.metric == "jaccard":
            # MinHash + LSH path (sets/strings).
            signatures, m = self._encode_jaccard_queries(
                X,
                allow_original_mode=not update_state,
            )
            if m == 0:
                return _empty

            forest = self._lsh_forest
            if forest is None:
                raise RuntimeError(
                    "No LSH Forest available. Cannot query new points "
                    "without a jaccard-fitted estimator."
                )

            all_indices, all_distances = forest.query_external_batch(
                signatures,
                k,
                self.kc,
            )

            if update_state:
                forest.batch_add(signatures)
                forest.index()

            return all_indices, all_distances, m

        elif self.metric in {"cosine", "euclidean"}:
            if self._index is None:
                raise RuntimeError(
                    "No ANN index stored. Reconstruct with store_index=True "
                    f"to use transform() or add_points() with "
                    f"metric={self.metric!r}."
                )
            X_dense = self._coerce_dense_matrix(X, min_samples=0)
            if self._n_features is not None and X_dense.shape[1] != self._n_features:
                raise ValueError(
                    f"Feature dimension mismatch: fit() saw {self._n_features} "
                    f"features, but received {X_dense.shape[1]}."
                )
            m = X_dense.shape[0]
            if m == 0:
                return _empty

            all_indices, all_distances = self._index.query_batch(X_dense, k)

            if update_state:
                self._index.add(X_dense)

            return all_indices, all_distances, m

        elif self.metric == "precomputed":
            dist_matrix = np.asarray(X, dtype=np.float32)
            if not np.all(np.isfinite(dist_matrix)):
                raise ValueError("Distance matrix contains NaN or Inf values.")
            if dist_matrix.ndim != 2:
                raise ValueError(
                    "metric='precomputed' expects a 2D distance matrix (m_new, n_existing)."
                )
            n_existing = self._embedding.shape[0]
            if dist_matrix.shape[1] != n_existing:
                raise ValueError(
                    f"X.shape[1]={dist_matrix.shape[1]} must equal "
                    f"n_existing={n_existing} for metric='precomputed'."
                )
            m = dist_matrix.shape[0]
            if m == 0:
                return (
                    np.empty((0, k), dtype=np.int32),
                    np.empty((0, k), dtype=np.float32),
                    0,
                )

            actual_k = min(k, n_existing)
            sorted_idx = np.argsort(dist_matrix, axis=1)[:, :actual_k].astype(np.int32)
            sorted_dist = np.take_along_axis(dist_matrix, sorted_idx.astype(np.intp), axis=1)

            # Pad to k columns if n_existing < k
            if actual_k < k:
                pad_idx = np.full((m, k - actual_k), -1, dtype=np.int32)
                pad_dist = np.full((m, k - actual_k), np.inf, dtype=np.float32)
                sorted_idx = np.concatenate([sorted_idx, pad_idx], axis=1)
                sorted_dist = np.concatenate([sorted_dist, pad_dist], axis=1)

            return sorted_idx, sorted_dist, m

        else:
            raise RuntimeError(f"Unsupported metric {self.metric!r}")

    def _encode_jaccard_queries(
        self,
        X: Any,
        *,
        allow_original_mode: bool,
    ) -> tuple[NDArray[np.uint64], int]:
        """Encode new Jaccard queries according to the fitted input mode."""
        encoder = _make_minhash_encoder(self.n_permutations, self.minhash_seed)

        if self._jaccard_mode == "binary":
            binary = self._coerce_binary_matrix(X, min_samples=0)
            if self._n_features is not None and binary.shape[1] != self._n_features:
                raise ValueError(
                    f"Feature dimension mismatch: fit() saw {self._n_features} "
                    f"features, but received {binary.shape[1]}."
                )
            m = binary.shape[0]
            if m == 0:
                return np.empty((0, self.n_permutations), dtype=np.uint64), 0
            return encoder.batch_from_binary_array(binary), m

        if not allow_original_mode:
            raise TypeError(
                f"add_points() requires a binary matrix, but fit() used "
                f"{self._jaccard_mode} input. add_points() is only supported "
                f"when fit() was called with a binary array, DataFrame, or "
                f"sparse matrix."
            )

        if not isinstance(X, (list, tuple)):
            raise TypeError(
                f"transform() requires {self._jaccard_mode} input because fit() used "
                f"{self._jaccard_mode} input."
            )

        m = len(X)
        if m == 0:
            return np.empty((0, self.n_permutations), dtype=np.uint64), 0

        if self._jaccard_mode == "sets":
            if not isinstance(X[0], (list, tuple, set, np.ndarray)):
                raise TypeError(
                    "transform() for set-based Jaccard expects a sequence of "
                    "integer-index sequences, e.g. [[1, 5, 10]]."
                )
            return encoder.batch_from_sparse_binary_array(X), m

        if self._jaccard_mode == "strings":
            return encoder.batch_from_string_array(X), m

        raise RuntimeError("Unknown fitted Jaccard input mode.")

    def _position_new_points(
        self,
        new_indices: NDArray[np.int32],
    ) -> NDArray[np.float32]:
        """Position new points near their nearest existing neighbor (tree parent).

        We place each new point at its parent plus a small offset pointing
        toward the local neighborhood centroid. We can't run a full
        force-directed layout here because that would push the new point
        far from its branch and create crossing edges that break the MST
        visual. A simple offset keeps edges short and looks clean.
        """
        m = new_indices.shape[0]
        existing = self._embedding  # (n, 2)
        new_coords = np.empty((m, 2), dtype=np.float32)

        centroid = existing.mean(axis=0)
        coord_range = existing.max(axis=0) - existing.min(axis=0)
        jitter_scale = coord_range * 0.001  # tiny jitter to avoid exact overlaps

        # We need a sense of how far apart connected nodes are in the
        # embedding so offset distances look proportional.
        # Measure the typical embedding distance between each node and its
        # nearest KNN neighbor, then take the median.
        local_scale = 1.0
        if self._graph is not None and self._graph.indices.shape[0] > 0:
            nn_idx = self._graph.indices[:, 0]
            valid = nn_idx >= 0
            if valid.any():
                diffs = existing[valid] - existing[nn_idx[valid]]
                nn_dists = np.linalg.norm(diffs, axis=1)
                local_scale = float(np.median(nn_dists)) or 1.0

        rng = np.random.default_rng(self.seed)

        for i in range(m):
            idxs = new_indices[i][new_indices[i] >= 0]

            if len(idxs) == 0:
                # No valid neighbors found, drop at the global centroid
                new_coords[i] = centroid
            else:
                # nearest neighbor becomes the tree parent
                parent_coord = existing[idxs[0]]

                if len(idxs) >= 2:
                    # Point the offset toward the centroid of the next
                    # few neighbors so the new point sits on the correct
                    # "side" of its parent branch.
                    nb_coords = existing[idxs[1 : min(5, len(idxs))]]
                    direction = nb_coords.mean(axis=0) - parent_coord
                    norm = np.linalg.norm(direction)
                    if norm > 1e-8:
                        direction /= norm
                    else:
                        # Neighbors are on top of each other then pick a random direction
                        direction = rng.normal(0, 1, size=2).astype(np.float32)
                        direction /= np.linalg.norm(direction)
                    new_coords[i] = parent_coord + direction * (local_scale * 0.3)
                else:
                    new_coords[i] = parent_coord

            new_coords[i] += rng.normal(0, 1, size=2).astype(np.float32) * jitter_scale

        return new_coords

    def _extend_tree(
        self,
        new_indices: NDArray[np.int32],
        new_distances: NDArray[np.float32],
        m: int,
    ) -> None:
        """Append each new point to the tree via its nearest existing neighbor."""
        old_tree = self.tree_  # force lazy extraction if needed
        n_existing = old_tree.n_nodes

        new_edges = np.empty((m, 2), dtype=np.int32)
        new_weights = np.empty(m, dtype=np.float32)

        for i in range(m):
            new_node = n_existing + i
            # Connect to nearest valid existing neighbor
            nn_idx = int(new_indices[i, 0]) if new_indices[i, 0] >= 0 else 0
            nn_dist = float(new_distances[i, 0]) if new_indices[i, 0] >= 0 else 1.0
            new_edges[i] = [nn_idx, new_node]
            new_weights[i] = nn_dist

        all_edges = np.concatenate([old_tree.edges, new_edges])
        all_weights = np.concatenate([old_tree.weights, new_weights])

        self._tree = Tree(
            n_nodes=n_existing + m,
            edges=all_edges,
            weights=all_weights,
            root=old_tree.root,
        )

    # Tree exploration convenience methods

    def path(self, from_idx: int, to_idx: int) -> list[int]:
        """Shortest path in the tree between two points.

        Delegates to :meth:`Tree.path`.

        Parameters
        ----------
        from_idx : int
            Source point index.
        to_idx : int
            Target point index.

        Returns
        -------
        list[int]
            Ordered node indices from source to target (inclusive).
        """
        return self.tree_.path(from_idx, to_idx)

    def distance(self, from_idx: int, to_idx: int) -> float:
        """Sum of edge weights along the tree path between two points.

        Delegates to :meth:`Tree.distance`.
        """
        return self.tree_.distance(from_idx, to_idx)

    def distances_from(self, source: int) -> NDArray[np.float32]:
        """Tree distance from *source* to every other point (pseudotime).

        Delegates to :meth:`Tree.distances_from`.

        Parameters
        ----------
        source : int
            Source point index.

        Returns
        -------
        NDArray[np.float32]
            Array of shape ``(n_samples,)`` with tree distances.
        """
        return self.tree_.distances_from(source)

    def hops(self, from_idx: int, to_idx: int) -> int:
        """Number of tree edges between two points (weight-agnostic).

        Delegates to :meth:`Tree.hops`.
        """
        return self.tree_.hops(from_idx, to_idx)

    def hops_from(self, source: int) -> NDArray[np.int32]:
        """Hop count from *source* to every other point.

        Unweighted analogue of :meth:`distances_from`; unreachable points
        receive ``-1``. Delegates to :meth:`Tree.hops_from`.

        Parameters
        ----------
        source : int
            Source point index.

        Returns
        -------
        NDArray[np.int32]
            Array of shape ``(n_samples,)`` with hop counts.
        """
        return self.tree_.hops_from(source)

    def _make_layout_config(self) -> Any | None:
        config = self.layout_config
        if config is None:
            if LayoutConfig is None:
                return None
            config = LayoutConfig()
            if hasattr(config, "fme_iterations"):
                config.fme_iterations = self.layout_iterations

        # The estimator owns layout reproducibility: always run a deterministic,
        # seeded layout, whether or not a custom layout_config was passed. The
        # HNSW index build is the separate, multi-threaded-by-default stage gated
        # by `reproducible`; the layout's own multi-threading gives no measurable
        # speedup, so determinism here is free. For a nondeterministic/
        # multi-threaded layout, call tmap.layout.layout_from_knn_graph directly.
        if hasattr(config, "deterministic"):
            config.deterministic = True
        if hasattr(config, "seed"):
            config.seed = self.seed
        return config

    def _encode_jaccard(self, X: Any) -> tuple[NDArray[np.uint64], int, int | None]:
        """Detect input type for metric='jaccard' and return MinHash signatures.

        Supports three input formats:
        - 2D binary array (n_samples, n_features) ->  batch_from_binary_array
        - scipy sparse matrix -> efficient row-wise sparse encoding
        - pandas DataFrame -> converted to ndarray
        - list of string sequences ->  batch_from_string_array
        - list of integer sequences ->  batch_from_sparse_binary_array

        Returns:
            (signatures, n_samples, n_features)
            n_features is None for list-of-sets/strings (variable width).
        """
        if X is None:
            raise ValueError("X cannot be None for metric='jaccard'.")

        encoder = _make_minhash_encoder(self.n_permutations, self.minhash_seed)

        # scipy sparse -> encode directly from sparse indices (avoids full densification)
        if hasattr(X, "tocsr"):
            import scipy.sparse as sp

            csr = sp.csr_matrix(X)
            n_samples, n_features = csr.shape
            if self.n_neighbors >= n_samples:
                raise ValueError(f"n_neighbors={self.n_neighbors} must be < n_samples={n_samples}")
            # Enforce binary values (same as dense path)
            if csr.nnz > 0 and not np.all(csr.data == 1):
                raise ValueError(
                    "Sparse matrix must contain only binary (0/1) values. "
                    "Non-zero entries other than 1 were found."
                )
            # Extract per-row nonzero column indices
            indices_list = [
                csr.indices[csr.indptr[i] : csr.indptr[i + 1]].tolist() for i in range(n_samples)
            ]
            signatures = encoder.batch_from_sparse_binary_array(indices_list)
            self._jaccard_mode = "binary"
            return signatures, n_samples, n_features

        # pandas DataFrame -> ndarray
        if hasattr(X, "values") and not isinstance(X, np.ndarray):
            X = X.values

        # 2D numpy array ->  binary path
        if isinstance(X, np.ndarray):
            binary_matrix = self._coerce_binary_matrix(X)
            n_samples = binary_matrix.shape[0]
            n_features = binary_matrix.shape[1]
            if self.n_neighbors >= n_samples:
                raise ValueError(f"n_neighbors={self.n_neighbors} must be < n_samples={n_samples}")
            signatures = encoder.batch_from_binary_array(binary_matrix)
            del binary_matrix
            self._jaccard_mode = "binary"
            return signatures, n_samples, n_features

        if not isinstance(X, (list, tuple)) or len(X) < 2:
            raise ValueError(
                "metric='jaccard' expects a 2D binary array or a list of sequences "
                "(at least 2 samples)."
            )

        # List of uniform-length numeric lists (e.g. data.tolist()) -> try binary path
        try:
            arr = np.asarray(X)
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                return self._encode_jaccard(arr)  # recurse into the ndarray branch
        except (ValueError, TypeError):
            pass  # ragged lists, mixed types, etc. --> falls through

        n_samples = len(X)
        if self.n_neighbors >= n_samples:
            raise ValueError(f"n_neighbors={self.n_neighbors} must be < n_samples={n_samples}")

        # first non-empty element to decide string vs integer
        first_elem = None
        for seq in X:
            if seq:
                first_elem = next(iter(seq))
                break

        if first_elem is not None and isinstance(first_elem, str):
            signatures = encoder.batch_from_string_array(X)
            self._jaccard_mode = "strings"
        else:
            signatures = encoder.batch_from_sparse_binary_array(X)
            self._jaccard_mode = "sets"

        return signatures, n_samples, None

    def _coerce_binary_matrix(self, X: Any | None, min_samples: int = 2) -> NDArray[np.uint8]:
        if X is None:
            raise ValueError("X cannot be None for metric='jaccard'.")
        # Convert array-like inputs (DataFrames, sparse matrices)
        if hasattr(X, "toarray"):
            X = X.toarray()
        elif hasattr(X, "values") and not isinstance(X, np.ndarray):
            X = X.values
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(
                "metric='jaccard' expects a 2D binary matrix of shape (n_samples, n_features)."
            )
        if arr.shape[0] < min_samples:
            raise ValueError(f"Need at least {min_samples} samples.")
        if arr.dtype != np.bool_ and not np.issubdtype(arr.dtype, np.number):
            raise ValueError("Binary matrix must contain numeric/boolean values.")
        if arr.shape[0] > 0 and not np.all((arr == 0) | (arr == 1)):
            raise ValueError("Binary matrix must contain only 0/1 values.")
        return arr.astype(np.uint8, copy=False)

    def _coerce_distance_matrix(self, X: Any | None) -> NDArray[np.float32]:
        if X is None:
            raise ValueError("X cannot be None for metric='precomputed'.")
        distances = np.asarray(X, dtype=np.float32)
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError(
                "metric='precomputed' expects a square distance matrix "
                "with shape (n_samples, n_samples)."
            )
        if distances.shape[0] < 2:
            raise ValueError("Distance matrix must contain at least 2 samples.")
        if not np.all(np.isfinite(distances)):
            raise ValueError("Distance matrix must contain only finite values.")
        return distances

    def _coerce_dense_matrix(self, X: Any | None, min_samples: int = 2) -> NDArray[np.float32]:
        if X is None:
            raise ValueError(f"X cannot be None for metric={self.metric!r}.")
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"metric={self.metric!r} expects a 2D matrix of shape (n_samples, n_features)."
            )
        if arr.shape[0] < min_samples:
            raise ValueError(f"Need at least {min_samples} samples.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input matrix must contain only finite values.")
        return arr

    def _is_binary_input(self, X: Any) -> bool:
        """Check whether X looks like a dense binary matrix (not sets/strings).

        Sparse matrices return False — they go through the MinHash + LSH path
        which handles sparse data without densifying.
        """
        if X is None:
            return False
        # Sparse matrices should NOT be densified; keep them on the LSH path.
        if hasattr(X, "tocsr") or hasattr(X, "toarray"):
            return False
        if hasattr(X, "values") and not isinstance(X, np.ndarray):
            return True  # DataFrame
        if isinstance(X, np.ndarray):
            return True
        if not isinstance(X, (list, tuple)) or len(X) == 0:
            return False
        # List input: try to detect if it's a rectangular numeric array
        # of 0/1 values (binary matrix as list-of-lists) vs sets/strings.
        first = X[0]
        if isinstance(first, (str, set, frozenset)):
            return False
        try:
            arr = np.asarray(X)
            if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.number):
                return False
            return bool(np.all((arr == 0) | (arr == 1)))
        except (ValueError, TypeError):
            return False
