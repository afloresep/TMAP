"""
OGDF integration layer.
"""

from __future__ import annotations

import importlib.util
import sys
import sysconfig
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from tmap.index.lsh_forest import LSHForest
    from tmap.index.types import KNNGraph

# =============================================================================
# Load C++ extension
# =============================================================================

_AVAILABLE = False
_IMPORT_ERROR: ImportError | None = None

# These will be set by _load_extension()
LayoutConfig: Any = None
LayoutResult: Any = None
Merger: Any = None
Placer: Any = None
ScalingType: Any = None
UntangleMode: Any = None
_cpp_layout_from_edge_list: Any = None


def _load_extension() -> bool:
    """
    Load the _tmap_ogdf C++ extension.

    In editable installs, Python code comes from src/ but the compiled
    extension lives in site-packages. We handle this by:
    1. Trying normal import first
    2. Falling back to direct file loading from site-packages
    """
    global _AVAILABLE, _IMPORT_ERROR
    global LayoutConfig, LayoutResult, Merger, Placer, ScalingType, UntangleMode
    global _cpp_layout_from_edge_list

    # Try normal import first
    try:
        from tmap.layout._tmap_ogdf import (  # type: ignore
            LayoutConfig as _LC,
        )
        from tmap.layout._tmap_ogdf import (
            LayoutResult as _LR,
        )
        from tmap.layout._tmap_ogdf import (
            Merger as _M,
        )
        from tmap.layout._tmap_ogdf import (
            Placer as _P,
        )
        from tmap.layout._tmap_ogdf import (
            ScalingType as _ST,
        )
        from tmap.layout._tmap_ogdf import (
            UntangleMode as _UM,
        )
        from tmap.layout._tmap_ogdf import (
            layout_from_edge_list as _lfel,
        )

        # Assign to module-level globals (global declaration above makes this work)
        LayoutConfig = _LC
        LayoutResult = _LR
        Merger = _M
        Placer = _P
        ScalingType = _ST
        UntangleMode = _UM
        _cpp_layout_from_edge_list = _lfel
        _AVAILABLE = True
        return True
    except ImportError as e:
        _IMPORT_ERROR = e

    # Fallback: find and load the .so file directly from site-packages
    platlib = Path(sysconfig.get_paths()["platlib"])
    ext_dir = platlib / "tmap" / "layout"

    if not ext_dir.exists():
        return False

    # Find the extension file (name varies by platform)
    ext_files = (
        list(ext_dir.glob("_tmap_ogdf.cpython-*.so"))  # Linux/macOS
        + list(ext_dir.glob("_tmap_ogdf.*.pyd"))  # Windows
        + list(ext_dir.glob("_tmap_ogdf*.dylib"))  # macOS alternative
    )

    if not ext_files:
        return False

    try:
        # Load the extension directly by file path
        spec = importlib.util.spec_from_file_location("_tmap_ogdf", ext_files[0])
        if spec is None or spec.loader is None:
            return False

        module = importlib.util.module_from_spec(spec)
        sys.modules["_tmap_ogdf"] = module  # Cache it
        spec.loader.exec_module(module)

        # Extract the symbols we need
        LayoutConfig = module.LayoutConfig
        LayoutResult = module.LayoutResult
        Merger = module.Merger
        Placer = module.Placer
        ScalingType = module.ScalingType
        UntangleMode = module.UntangleMode
        _cpp_layout_from_edge_list = module.layout_from_edge_list

        _AVAILABLE = True
        _IMPORT_ERROR = None
        return True

    except Exception as e:
        _IMPORT_ERROR = ImportError(f"Failed to load extension: {e}")
        return False


# Load on module import
_load_extension()


def require_ogdf() -> None:
    """Raise ImportError if OGDF extension is not available."""
    if not _AVAILABLE:
        raise ImportError(
            "OGDF layout extension not available. "
            "Reinstall with bundled OGDF: pip install -e . "
            "If a core-only install was intended, use "
            "pip install -e . --config-settings=cmake.define.TMAP_BUILD_LAYOUT=OFF"
        ) from _IMPORT_ERROR


def layout_from_edge_list(
    vertex_count: int,
    edges: list[tuple[int, int, float]],
    config: Any | None = None,
    create_mst: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32], NDArray[np.uint32]]:
    """
    Compute 2D layout from edge list.

    Parameters
    ----------
    vertex_count : int
        Number of vertices
    edges : list of (source, target, weight)
        Edge list. Weights should be positive.
    config : LayoutConfig, optional
        Layout configuration. If None, uses defaults.
    create_mst : bool, default True
        If True, compute MST first.

    Returns
    -------
    x, y, s, t : ndarrays
        Coordinates and edge topology
    """
    require_ogdf()

    if config is None:
        if LayoutConfig is None:
            raise RuntimeError("LayoutConfig is unavailable")
        config = LayoutConfig()
    if _cpp_layout_from_edge_list is None:
        raise RuntimeError("OGDF layout function is unavailable")

    result = _cpp_layout_from_edge_list(vertex_count, edges, config, create_mst)

    return (
        np.array(result.x, dtype=np.float32),
        np.array(result.y, dtype=np.float32),
        np.array(result.s, dtype=np.uint32),
        np.array(result.t, dtype=np.uint32),
    )


def layout_from_lsh_forest(
    lsh_forest: LSHForest,
    config: Any | None = None,
    create_mst: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32], NDArray[np.uint32]]:
    """
    Compute 2D layout directly from LSHForest.

    Parameters
    ----------
    lsh_forest : LSHForest
        Indexed LSHForest with signatures
    config : LayoutConfig, optional
        Layout configuration. Uses config.k and config.kc for k-NN construction.
    create_mst : bool, default True
        If True, compute MST before layout (recommended for tree visualization)

    Returns
    -------
    x, y, s, t : ndarrays
        x, y: Node coordinates (float32)
        s, t: Edge source/target indices (uint32) - edges in the final layout

    Example
    -------
    >>> from tmap import MinHash, LSHForest
    >>> from tmap.layout import layout_from_lsh_forest, LayoutConfig
    >>>
    >>> # Build LSHForest
    >>> mh = MinHash(num_perm=128)
    >>> sigs = mh.batch_from_binary_array(fingerprints)
    >>> lsh = LSHForest(d=128)
    >>> lsh.batch_add(sigs)
    >>> lsh.index()
    >>>
    >>> # Compute layout with custom config
    >>> cfg = LayoutConfig()
    >>> cfg.k = 20
    >>> cfg.kc = 50
    >>> cfg.node_size = 1/30
    >>> cfg.mmm_repeats = 2
    >>> x, y, s, t = layout_from_lsh_forest(lsh, cfg)
    """
    require_ogdf()

    if config is None:
        if LayoutConfig is None:
            raise RuntimeError("LayoutConfig is unavailable")
        config = LayoutConfig()
    if _cpp_layout_from_edge_list is None:
        raise RuntimeError("OGDF layout function is unavailable")

    # Build k-NN graph using config parameters
    knn = lsh_forest.get_knn_graph(k=config.k, kc=config.kc)

    # Convert k-NN to edge list
    edges = _knn_to_edge_list(knn)

    # Call layout with full k-NN graph - OGDF will compute MST
    result = _cpp_layout_from_edge_list(knn.n_nodes, edges, config, create_mst=create_mst)

    return (
        np.array(result.x, dtype=np.float32),
        np.array(result.y, dtype=np.float32),
        np.array(result.s, dtype=np.uint32),
        np.array(result.t, dtype=np.uint32),
    )


def layout_from_knn_graph(
    knn: KNNGraph,
    config: Any | None = None,
    create_mst: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32], NDArray[np.uint32]]:
    """
    Compute 2D layout from a k-NN graph.

    This is useful when you've already computed the k-NN graph separately
    but want OGDF to compute the MST (recommended for better connectivity).

    Parameters
    ----------
    knn : KNNGraph
        k-NN graph from LSHForest, USearch, or another source.
    config : LayoutConfig, optional
        Layout configuration
    create_mst : bool, default True
        If True, compute MST before layout

    Returns
    -------
    x, y, s, t : ndarrays
        Coordinates and edge topology
    """
    require_ogdf()

    if config is None:
        if LayoutConfig is None:
            raise RuntimeError("LayoutConfig is unavailable")
        config = LayoutConfig()
    if _cpp_layout_from_edge_list is None:
        raise RuntimeError("OGDF layout function is unavailable")

    edges = _knn_to_edge_list(knn)

    result = _cpp_layout_from_edge_list(knn.n_nodes, edges, config, create_mst=create_mst)

    return (
        np.array(result.x, dtype=np.float32),
        np.array(result.y, dtype=np.float32),
        np.array(result.s, dtype=np.uint32),
        np.array(result.t, dtype=np.uint32),
    )


def _knn_to_edge_list(knn: KNNGraph) -> list[tuple[int, int, float]]:
    """
    Convert KNNGraph to edge list for OGDF.

    Creates undirected edges from k-NN (which is directed: i -> neighbors[i]).
    Filters out self-loops and invalid entries.
    """
    edge_weights: dict[tuple[int, int], float] = {}

    n = knn.n_nodes
    k = knn.k

    for i in range(n):
        for j_idx in range(k):
            j = int(knn.indices[i, j_idx])
            dist = float(knn.distances[i, j_idx])

            if j < 0 or j == i or not np.isfinite(dist):
                continue

            edge_key = (min(i, j), max(i, j))
            prev = edge_weights.get(edge_key)
            if prev is None or dist < prev:
                edge_weights[edge_key] = dist

    return [(src, tgt, weight) for (src, tgt), weight in edge_weights.items()]


__all__ = [
    "_AVAILABLE",
    "require_ogdf",
    "LayoutConfig",
    "Placer",
    "Merger",
    "ScalingType",
    "layout_from_edge_list",
    "layout_from_lsh_forest",
    "layout_from_knn_graph",
]
