"""OGDF-backed layout helpers for TMAP visualizations."""

from tmap.layout._ogdf import _AVAILABLE as OGDF_AVAILABLE

__all__ = [
    "OGDF_AVAILABLE",
]

# Conditionally export OGDF-dependent items
if OGDF_AVAILABLE:
    from tmap.layout._ogdf import (
        LayoutConfig,
        Merger,
        Placer,
        ScalingType,
        UntangleMode,
        layout_from_edge_list,
        layout_from_knn_graph,
        layout_from_lsh_forest,
    )

    __all__ += [
        "LayoutConfig",
        "Placer",
        "Merger",
        "ScalingType",
        "UntangleMode",
        "layout_from_edge_list",
        "layout_from_lsh_forest",
        "layout_from_knn_graph",
    ]
