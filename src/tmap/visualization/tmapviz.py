from __future__ import annotations

import base64
import gzip
import json
import math
import shutil
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from matplotlib import colormaps
from numpy.typing import NDArray

try:
    from jinja2 import Environment, PackageLoader, select_autoescape

    _JINJA_AVAILABLE = True
except ImportError:
    _JINJA_AVAILABLE = False

COLORMAPS = list(colormaps)
VENDOR_DIR = Path(__file__).parent / "vendor"


def _project_root() -> Path:
    """Return repository root (assumes src/tmap/visualization/...)."""
    return Path(__file__).resolve().parents[3]


def _load_js_sources() -> dict[str, str]:
    """Load raw JS sources for inline embedding.

    First tries vendored files (included in package), then falls back
    to node_modules (for development).
    """
    # Vendored files (preferred - included in package)
    vendor_deps = {
        "regl": VENDOR_DIR / "regl.min.js",
        "scatterplot": VENDOR_DIR / "regl-scatterplot.esm.js",
        "pubsub": VENDOR_DIR / "pub-sub-es.js",
    }

    # Check if vendored files exist
    if all(path.exists() for path in vendor_deps.values()):
        return {name: path.read_text(encoding="utf-8") for name, path in vendor_deps.items()}

    # Fallback to node_modules (for development)
    # TODO(ISS-006): Remove node_modules fallback once vendored files are stable
    root = _project_root()
    node_deps = {
        "regl": root / "node_modules" / "regl" / "dist" / "regl.min.js",
        "scatterplot": root
        / "node_modules"
        / "regl-scatterplot"
        / "dist"
        / "regl-scatterplot.esm.js",
        "pubsub": root / "node_modules" / "pub-sub-es" / "dist" / "index.js",
    }

    missing = [name for name, path in node_deps.items() if not path.exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"Missing JS dependencies: {missing_list}. "
            "Vendored files not found and node_modules unavailable. "
            "This is likely a packaging issue - please reinstall the package."
        )

    return {name: path.read_text(encoding="utf-8") for name, path in node_deps.items()}


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _runtime_base64() -> dict[str, str]:
    """Cacheable helper to fetch + encode JS sources."""
    sources = _load_js_sources()
    return {
        "regl": _b64(sources["regl"]),
        "scatterplot": _b64(sources["scatterplot"]),
        "pubsub": _b64(sources["pubsub"]),
    }


@lru_cache(maxsize=1)
def _get_jinja_env() -> Environment:
    """Get or create a cached Jinja2 environment for templates."""
    if not _JINJA_AVAILABLE:
        raise ImportError(
            "Jinja2 is required for template rendering. "
            "Install full dependencies with: pip install -e ."
        )
    return Environment(
        loader=PackageLoader("tmap.visualization", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _sanitize_filename(name: str, seen: set[str] | None = None) -> str:
    """Sanitize a column name for use as a filename (URL-safe).

    When *seen* is provided, appends a numeric suffix to avoid collisions.
    The sanitized name is added to *seen* in place.
    """
    import re

    # Replace spaces, parens, and other non-alphanumeric chars with underscores
    safe = re.sub(r"[^a-zA-Z0-9_\-.]", "_", name)
    # Collapse multiple underscores
    safe = re.sub(r"_+", "_", safe).strip("_")
    if seen is not None:
        base = safe
        counter = 2
        while safe in seen:
            safe = f"{base}_{counter}"
            counter += 1
        seen.add(safe)
    return safe


def _unique_structure_filename(path: str | Path, seen: set[str]) -> str:
    """Return a URL-safe unique filename for a copied structure sidecar."""
    raw = Path(path).name or "structure"
    safe = _sanitize_filename(raw) or "structure"
    parsed = Path(safe)
    stem = parsed.stem or "structure"
    suffix = parsed.suffix

    candidate = f"{stem}{suffix}"
    counter = 2
    while candidate in seen:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    seen.add(candidate)
    return candidate


def _normalize_structure_sidecar_dir(directory: str | Path) -> str:
    """Validate and normalize a relative output directory for structure files."""
    path = Path(directory)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("directory must be a relative path inside the output bundle")
    normalized = path.as_posix().strip("/")
    if not normalized or normalized == ".":
        raise ValueError("directory must not be empty")
    return normalized


def _normalize_coords(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Normalize coordinates to [-1, 1] preserving aspect ratio.
    This is actually required by regl-scatterplot.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    x_center = (x_max + x_min) / 2.0
    y_center = (y_max + y_min) / 2.0

    x_range = x_max - x_min
    y_range = y_max - y_min
    scale = max(x_range, y_range) / 2.0
    if scale == 0:
        scale = 1.0

    x_norm = (x - x_center) / scale
    y_norm = (y - y_center) / scale
    return cast(NDArray[np.float64], np.stack([x_norm, y_norm], axis=1).astype(np.float64))


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> list[float]:
    """Convert #RRGGBB to [r, g, b, a] floats in [0, 1]."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    return [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)] + [alpha]


def _normalize_hex_color(color: str) -> str:
    """Normalize color to #rrggbb format."""
    if not isinstance(color, str):
        raise ValueError("Color must be a hex string like '#rrggbb'.")

    normalized = color.strip().lstrip("#")
    if len(normalized) == 3:
        normalized = "".join(ch * 2 for ch in normalized)

    if len(normalized) != 6:
        raise ValueError(f"Invalid hex color: {color!r}")

    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError(f"Invalid hex color: {color!r}") from exc

    return f"#{normalized.lower()}"


def _hex_to_css_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert #RRGGBB + alpha to a CSS rgba(...) color string."""
    rgb = _hex_to_rgba(hex_color, alpha)
    r, g, b = (int(round(channel * 255)) for channel in rgb[:3])
    alpha_str = f"{alpha:.6f}".rstrip("0").rstrip(".")
    if alpha_str == "":
        alpha_str = "0"
    return f"rgba({r}, {g}, {b}, {alpha_str})"


# TODO(ISS-014): Implement categorical=True preserves listed colors when available
def _colormap_to_hex(name: str) -> list[str]:
    """
    Convert a matplotlib colormap to a list of hex strings.
    """
    import matplotlib as mpl
    from matplotlib.colors import to_hex

    cmap = mpl.colormaps[name]
    hex_colors = [to_hex(cmap(i)) for i in range(cmap.N)]
    return hex_colors


def _cycle_colormaps(
    colormaps_payload: dict[str, list[str]],
    columns: dict[str, Any],
) -> None:
    """Extend colormap hex lists by cycling to cover all categorical values."""
    max_cats: dict[str, int] = {}
    for col in columns.values():
        cmap_name = col.color if col.role in ("layout", "layout+label") else None
        if cmap_name and col.dtype == "categorical":
            n = len(set(col.values))
            max_cats[cmap_name] = max(max_cats.get(cmap_name, 0), n)
    for cmap_name, needed in max_cats.items():
        if cmap_name in colormaps_payload and needed > len(colormaps_payload[cmap_name]):
            base = colormaps_payload[cmap_name]
            colormaps_payload[cmap_name] = [base[i % len(base)] for i in range(needed)]


def _contains_nan(values: Sequence[Any]) -> bool:
    """Return True when values contain at least one NaN."""
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.isnan(arr).any())


def _safe_float(v: Any) -> float:
    """Convert a value to float, treating None/empty/NA-like as NaN."""
    if v is None or (isinstance(v, str) and v == ""):
        return float("nan")
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def _coerce_json_safe(v: Any) -> Any:
    """Coerce a single value to a JSON-serializable type.

    Handles numpy scalars, pandas NA, non-finite floats, etc.
    """
    # numpy scalar -> Python scalar
    if isinstance(v, np.generic):
        v = v.item()
    # pandas NA-like sentinels
    try:
        import pandas as pd

        if isinstance(v, type(pd.NA)) or (isinstance(v, float) and pd.isna(v)):
            return None
    except ImportError:
        pass
    # Non-finite Python float -> None (renders as null in JSON)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _encode_string_column(values: list[Any], name: str) -> bytes:
    """Encode label/string column values as JSON bytes.

    Handles numpy scalars, pandas NA, non-finite floats, and other
    non-JSON-native types by coercing them to JSON-safe equivalents.
    """
    safe = [_coerce_json_safe(v) for v in values]
    return json.dumps(safe, separators=(",", ":"), allow_nan=False).encode()


def _to_json_safe(value: Any) -> Any:
    """Convert values to JSON-safe types, mapping non-finite numbers to null.

    Optimized to avoid deep-copying large lists of strings or plain numbers.
    """
    if isinstance(value, np.ndarray):
        # For float arrays that may contain NaN/Inf, replace with None
        if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
            return [None if not np.isfinite(v) else float(v) for v in value.flat]
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if not value:
            return value if isinstance(value, list) else list(value)
        # Check first element to decide strategy — avoid per-element recursion
        # for homogeneous lists of strings or plain ints (common for columns).
        first = value[0]
        if isinstance(first, str):
            return value  # strings are already JSON-safe, no copy needed
        if isinstance(first, int) and not isinstance(first, (bool, np.integer)):
            return value  # plain Python ints are JSON-safe
        return [_to_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if np.isfinite(f) else None
    return value


@dataclass
class Column:
    name: str
    values: Sequence[int | np.floating | str]
    role: Literal["layout", "label", "layout+label", "smiles", "images"]
    dtype: Literal["continuous", "categorical", "label", "smiles", "images"]
    color: str | None = None


def _pack_coords_binary(points: np.ndarray, bits: int = 16) -> bytes:
    """Pack normalized [-1,1] coordinates as gzip-compressed quantized integers."""
    if bits == 16:
        max_val = 65535
        quantized = ((points.astype(np.float64) + 1.0) * (max_val / 2.0)).astype(np.uint16)
    else:
        max_val = 4294967295
        quantized = ((points.astype(np.float64) + 1.0) * (max_val / 2.0)).astype(np.uint32)
    raw = quantized.flatten().tobytes()
    return gzip.compress(raw, compresslevel=6)


def _pack_numeric_binary(values: np.ndarray, dtype: str = "float32") -> bytes:
    """Pack numeric column as gzip-compressed typed array."""
    arr: NDArray[np.float32] | NDArray[np.int32]
    if dtype == "float32":
        arr = values.astype(np.float32)
    elif dtype == "int32":
        arr = values.astype(np.int32)
    else:
        arr = values.astype(np.float32)
    return gzip.compress(arr.tobytes(), compresslevel=6)


def _pack_categorical_binary(values: Sequence[Any]) -> tuple[bytes, list[str]]:
    """Pack categorical column using dictionary encoding."""
    unique_values: list[str] = []
    value_to_idx: dict[str, int] = {}
    indices: NDArray[np.uint32] = np.empty(len(values), dtype=np.uint32)

    for i, v in enumerate(values):
        s = str(v)
        if s not in value_to_idx:
            value_to_idx[s] = len(unique_values)
            unique_values.append(s)
        indices[i] = value_to_idx[s]

    compressed = gzip.compress(indices.tobytes(), compresslevel=6)
    return compressed, unique_values


class TmapViz:
    """Interactive scatter-plot visualization backed by regl-scatterplot.

    Supports continuous and categorical color layouts, label tooltips,
    and optional SMILES molecule rendering. Export via ``to_html()`` /
    ``write_html()`` or notebook widgets via ``to_widget()`` / ``show()``.

    Attributes:
        title: Page title and default filename stem.
        background_color: Hex background color (default ``"#7A7A7A"``).
        point_color: Default hex point color (default ``"#4a9eff"``).
        point_size: Base point radius in pixels (default ``4.0``).
        opacity: Point opacity in ``[0, 1]`` (default ``0.85``).
        edge_color: Edge hex color (default ``"#000000"``).
        edge_opacity: Edge opacity in ``[0, 1]`` (default ``0.5``).
        edge_width: Edge width in CSS pixels (default ``2.0``).
    """

    def __init__(self) -> None:
        self.title: str = "MyTMAP"
        self.background_color: str = "#FFFFFF"
        self.point_color: str = "#4a9eff"
        self._point_size: float = 4.0
        self.opacity: float = 0.85

        self.edge_color: str = "#000000"
        self.edge_opacity: float = 0.5
        self.edge_width: float = 2.0

        self._points_array: np.ndarray | None = None  # Shape: (n, 2)
        self._edges_s: np.ndarray | None = None
        self._edges_t: np.ndarray | None = None
        self._edge_weights: np.ndarray | None = None
        self._layout_keys: list[str] = []
        self._labels_keys: list[str] = []
        self._smiles_column: str | None = None
        self._images_column: str | None = None
        self._images_tooltip_size: int = 128
        self._protein_column: str | None = None
        self._structures_3d_column: str | None = None
        self._structures_3d_source: str = "url"
        self._structures_3d_format: str = "pdb"
        self._structures_3d_file_paths: list[Path | None] | None = None
        self._structures_3d_copy_sidecars: bool = False
        self._columns: dict[str, Column] = {}

        # Custom hex colormaps registered via add_color_layout(color=[...] or color={...})
        self._custom_colormaps: dict[str, list[str]] = {}

        # UI configuration (new declarative API)
        self._filterable: list[str] = []
        self._searchable: list[str] = []
        self._card_config: dict[str, Any] | None = None
        self._column_ui: dict[str, dict[str, Any]] = {}

    @property
    def point_size(self) -> float:
        return self._point_size

    @point_size.setter
    def point_size(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"point_size must be > 0, got {value}")
        self._point_size = float(value)

    def add_color_layout(
        self,
        name: str,
        values: list[Any] | NDArray,
        categorical: bool = False,
        add_as_label: bool = True,
        color: str | list[str] | dict[str, str] | None = None,
    ) -> None:
        """Add a color layout column (continuous or categorical).

        Parameters
        ----------
        name : str
            Column name shown in the layout selector and tooltip.
        values : list or ndarray
            One value per point. For categorical layouts these become the
            legend labels, so you can pass human-readable strings directly
            (e.g. ``["Bacteria", "Fungi", "Virus", ...]``).
        categorical : bool, default False
            If True, treat values as discrete categories.
        add_as_label : bool, default True
            Also show values in the hover tooltip.
        color : str, list of str, dict, or None
            How to color the points. Accepts three formats:

            - **str** (default): a matplotlib colormap name like ``"tab10"``
              or ``"viridis"``.
            - **list of hex strings**: one color per unique category, assigned
              in the order the categories first appear in *values*.
              Example: ``["#e41a1c", "#377eb8", "#4daf4a"]``.
            - **dict mapping value to hex color**: explicit color per category.
              Example: ``{"Bacteria": "#e41a1c", "Fungi": "#377eb8"}``.
              Categories not in the dict fall back to gray (``#888888``).

            Custom hex lists and dicts are only supported for categorical
            layouts. If None, defaults to ``"tab10"`` for categorical and
            ``"viridis"`` for continuous.
        """
        if isinstance(values, np.ndarray) and not categorical:
            values = np.asarray(values, dtype=np.float32)
        elif isinstance(values, np.ndarray):
            values = values.tolist()
        else:
            values = list(values)

        # Default to continuous because it will give less issues and having to pass
        # always the type can be annoying...
        # Validate continuous values are actually numeric
        if not categorical:
            for v in values:
                if v is None or (isinstance(v, str) and v == ""):
                    continue
                # pandas NA-like sentinels
                try:
                    import pandas as _pd

                    if isinstance(v, type(_pd.NA)):
                        continue
                except ImportError:
                    pass
                try:
                    float(v)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Continuous layout '{name}' contains non-numeric value "
                        f"{v!r}. Use categorical=True for string values."
                    ) from None

        _column_dtype: Literal["categorical", "continuous"] = (
            "categorical" if categorical else "continuous"
        )

        # Resolve the color parameter into a colormap name (str) that
        # we can look up later in colormaps_payload.
        color_key = self._resolve_color(name, color, values, categorical)

        if not categorical and _contains_nan(values):
            warnings.warn(
                f"Continuous layout '{name}' contains NaN values. "
                "NaN points will be rendered in black (#000000).",
                UserWarning,
                stacklevel=2,
            )

        if name not in self._layout_keys:
            self._layout_keys.append(name)

        if add_as_label:
            if name not in self._labels_keys:
                self._labels_keys.append(name)
            role: Literal["layout", "layout+label"] = "layout+label"
        else:
            if name in self._labels_keys:
                self._labels_keys.remove(name)
            role = "layout"

        self._columns[name] = Column(name, values, role, _column_dtype, color=color_key)

    def _resolve_color(
        self,
        column_name: str,
        color: str | list[str] | dict[str, str] | None,
        values: list[Any] | NDArray,
        categorical: bool,
    ) -> str:
        """Turn the user-supplied color argument into a colormap key.

        For matplotlib colormap names the key is the name itself.
        For custom hex lists or dicts we store the colors under a
        synthetic key in self._custom_colormaps and return that key.
        """
        import matplotlib

        # Default
        if color is None:
            color = "tab10" if categorical else "viridis"

        # dict {value: hex} -> convert to ordered hex list
        if isinstance(color, dict):
            if not categorical:
                raise ValueError(
                    "A color dict mapping values to hex colors is only "
                    "supported for categorical layouts (categorical=True)."
                )
            # Collect unique categories as str (matches _pack_categorical_binary)
            categories = list(dict.fromkeys(str(v) for v in values))

            # Sort numerically if ALL categories parse as finite numbers.
            # This must mirror the JS sort: unique.every(v => !isNaN(Number(v)))
            try:
                parsed = [float(c) for c in categories]
                if all(p == p for p in parsed):  # exclude NaN
                    categories.sort(key=float)
            except (ValueError, TypeError):
                pass  # keep insertion order for non-numeric categories

            # Normalize dict keys to strings for lookup
            str_color: dict[str, str] = {str(k): v for k, v in color.items()}

            # Build hex list, trying numeric-equivalent key forms
            hex_list: list[str] = []
            unmatched: list[str] = []
            for cat in categories:
                hex_val = str_color.get(cat)
                if hex_val is None:
                    # Try numeric normalization: "2.0" <-> "2", "3" <->"3.0"
                    try:
                        num = float(cat)
                        if num == int(num):
                            hex_val = str_color.get(str(int(num)))
                        if hex_val is None:
                            hex_val = str_color.get(str(num))
                    except (ValueError, TypeError, OverflowError):
                        pass
                if hex_val is None:
                    unmatched.append(cat)
                    hex_val = "#888888"
                hex_list.append(_normalize_hex_color(hex_val))

            if unmatched:
                warnings.warn(
                    f"Categorical layout '{column_name}': {len(unmatched)} of "
                    f"{len(categories)} categories had no matching key in the "
                    f"color dict and will use fallback gray (#888888): "
                    f"{unmatched[:5]}{'...' if len(unmatched) > 5 else ''}",
                    UserWarning,
                    stacklevel=3,
                )
            color = hex_list  # fall through to list handling

        # list of hex strings -> store as custom colormap
        if isinstance(color, list):
            if not categorical:
                raise ValueError(
                    "A list of hex colors is only supported for "
                    "categorical layouts (categorical=True)."
                )
            hex_colors = [_normalize_hex_color(c) for c in color]
            if not hex_colors:
                raise ValueError("color list must not be empty. Provide at least one hex color.")

            unique_count = len(set(str(v) for v in values))
            if len(hex_colors) < unique_count:
                warnings.warn(
                    f"Categorical layout '{column_name}' has {unique_count} "
                    f"unique values but only {len(hex_colors)} custom colors "
                    f"were provided. Colors will cycle.",
                    UserWarning,
                    stacklevel=2,
                )
                hex_colors = [hex_colors[i % len(hex_colors)] for i in range(unique_count)]

            key = f"_custom_{column_name}"
            self._custom_colormaps[key] = hex_colors
            return key

        # string: matplotlib colormap name
        if color not in COLORMAPS:
            raise ValueError(f"Color option not found. Choose from {list(matplotlib.colormaps)}")

        if (
            color not in set(matplotlib.colormaps).difference(set(matplotlib.color_sequences))
            and not categorical
        ):
            raise ValueError(
                f"Continuous layout requires a color scheme from "
                f"{set(matplotlib.colormaps).difference(set(matplotlib.color_sequences))}"
            )

        if categorical and color not in list(matplotlib.color_sequences):
            raise ValueError(
                f"Categorical layout requires a color scheme from "
                f"{list(matplotlib.color_sequences)}"
            )

        if categorical:
            try:
                unique_count = len(set(values))
            except TypeError as exc:
                raise ValueError(
                    "Categorical layout values must be hashable to compute unique categories."
                ) from exc

            cmap_size = matplotlib.colormaps[color].N
            if unique_count > cmap_size:
                warnings.warn(
                    f"Categorical layout '{column_name}' has {unique_count} unique values but "
                    f"colormap '{color}' only provides {cmap_size} colors. "
                    f"Colors will cycle.",
                    UserWarning,
                    stacklevel=2,
                )

        return color

    def add_label(
        self,
        name: str,
        values: list[Any],
    ) -> None:
        """Add a text-only label column (shown in tooltip, not used for coloring).

        Args:
            name: Column name displayed in the tooltip header.

        Raises:
            ValueError: If *name* already exists as a color-layout column.
                Overwriting a layout column with a label would corrupt the
                binary data (JS expects float32/uint32 but gets JSON string).
        """
        if name in self._layout_keys:
            raise ValueError(
                f"Cannot add label '{name}': it already exists as a color layout. "
                f"Layout columns with add_as_label=True already appear in tooltips. "
                f"Use a different name if you need a separate label."
            )

        if isinstance(values, np.ndarray):
            values = values.tolist()
        else:
            values = list(values)

        if name not in self._labels_keys:
            self._labels_keys.append(name)
        self._columns[name] = Column(name, values, "label", "label")

    def add_smiles(
        self,
        values: list[str],
        name: str = "SMILES",
    ) -> None:
        """Add a SMILES column for molecular structure visualization.

        Molecules are rendered in the HTML tooltip when hovering over points.

        Args:
            name: Column name (displayed in tooltip)
            values: List of SMILES strings, one per point
        """
        if isinstance(values, np.ndarray):
            values = values.tolist()
        else:
            values = list(values)

        if self._smiles_column is not None:
            raise ValueError(
                f"Only one SMILES column is supported. "
                f"Already have '{self._smiles_column}', cannot add '{name}'."
            )

        self._smiles_column = name
        if name not in self._labels_keys:
            self._labels_keys.append(name)
        self._columns[name] = Column(name, values, "smiles", "smiles")

    def add_images(
        self,
        values: list[str],
        name: str = "image",
        tooltip_size: int = 128,
    ) -> None:
        """Add an image column for thumbnail rendering in tooltips.

        Each value should be a data URI (e.g. ``"data:image/jpeg;base64,..."``).
        When hovering over a point the image is rendered in the tooltip.

        Args:
            values: List of data-URI strings, one per point.
            name: Column name (displayed in tooltip header).
            tooltip_size: Display size in pixels for the tooltip thumbnail.
                Images are scaled to this size with nearest-neighbor
                interpolation (crisp pixel art).  Default 128.
        """
        if isinstance(values, np.ndarray):
            values = values.tolist()
        else:
            values = list(values)

        if self._images_column is not None:
            raise ValueError(
                f"Only one images column is supported. "
                f"Already have '{self._images_column}', cannot add '{name}'."
            )

        self._images_column = name
        self._images_tooltip_size = int(tooltip_size)
        # Don't add to _labels_keys we don't want the raw data URI shown as
        # text in the tooltip rows; the template renders it visually instead.
        self._columns[name] = Column(name, values, "images", "images")

    def add_protein_ids(
        self,
        values: list[str],
        name: str = "UniProt ID",
    ) -> None:
        """Add a protein ID column for structure visualization.

        Pinned cards show a Mol* 3D structure viewer (lazy-loaded from
        AlphaFold DB) and on-demand UniProt metadata. IDs also appear as
        clickable links.

        Args:
            values: List of UniProt accessions (e.g. ``"E4ZVF8"``), one per point.
            name: Column name (displayed in tooltip).
        """
        if isinstance(values, np.ndarray):
            values = values.tolist()
        else:
            values = list(values)

        if self._protein_column is not None:
            raise ValueError(
                f"Only one protein column is supported. "
                f"Already have '{self._protein_column}', cannot add '{name}'."
            )

        self._protein_column = name
        if name not in self._labels_keys:
            self._labels_keys.append(name)
        self._columns[name] = Column(name, values, "label", "label")

    def add_3d_structures(
        self,
        values: list[str],
        *,
        source: str = "url",
        fmt: str = "pdb",
        name: str = "structure",
    ) -> None:
        """Attach per-point 3D molecular structures for the protein template.

        Each pinned card renders the structure with 3Dmol.js. If structures
        are attached and the default template is used, the protein template is
        selected automatically.

        Two transports:

        - ``source="url"`` — each value is an absolute URL or a path relative
          to the rendered HTML (e.g. ``"pdbs/1CRN.pdb"``). The browser fetches
          the file lazily on pin. This is the scalable default.
        - ``source="text"`` — each value is the full text of a PDB or mmCIF
          file. The structure text is gzip+base64-embedded in the HTML, so the
          data payload is self-contained but the viewer still loads 3Dmol.js
          from its CDN.

        For local files, prefer :meth:`add_3d_structure_files`; it stores
        relative URLs and can copy the sidecar files into static output
        bundles automatically.

        Args:
            values: Per-point URL/path (url) or structure content (text).
                Use ``""`` or ``None`` for points without structures.
            source: ``"url"`` (default) or ``"text"``.
            fmt: Structure format — ``"pdb"`` (default) or ``"mmcif"``/``"cif"``.
            name: Column name.
        """
        if isinstance(values, np.ndarray):
            values = values.tolist()
        else:
            values = list(values)

        if source not in ("url", "text"):
            raise ValueError(f"source must be 'url' or 'text', got {source!r}.")

        fmt = fmt.lower()
        if fmt in ("cif", "mmcif"):
            fmt = "cif"
        elif fmt != "pdb":
            raise ValueError(f"Unsupported structure format '{fmt}'. Use 'pdb' or 'cif'.")

        if self._structures_3d_column is not None:
            raise ValueError(
                f"Only one 3D structures column is supported. "
                f"Already have '{self._structures_3d_column}', cannot add '{name}'."
            )

        self._structures_3d_column = name
        self._structures_3d_source = source
        self._structures_3d_format = fmt
        self._structures_3d_file_paths = None
        self._structures_3d_copy_sidecars = False
        self._columns[name] = Column(name, values, "label", "label")

    def add_3d_structure_files(
        self,
        paths: Sequence[str | Path | None],
        *,
        copy: bool = True,
        directory: str | Path = "structures",
        fmt: str = "pdb",
        name: str = "structure",
    ) -> None:
        """Attach local PDB/mmCIF files and optionally copy them into outputs.

        This is the most convenient API when you have a folder of local
        structures. The visualization stores browser-facing relative URLs. When
        ``copy=True`` (default), :meth:`write_static` and :meth:`write_html`
        copy the referenced files into ``directory`` next to the generated
        HTML.

        Args:
            paths: Local file paths, one per point. ``None`` or ``""`` means
                no structure for that point.
            copy: If True, copy files into the rendered output bundle and store
                relative URLs like ``"structures/1CRN.pdb"``. If False, store
                the provided path strings unchanged.
            directory: Relative sidecar directory used when ``copy=True``.
            fmt: Structure format — ``"pdb"`` (default) or ``"mmcif"``/``"cif"``.
            name: Column name.
        """
        path_values = list(paths)

        if not copy:
            self.add_3d_structures(
                ["" if p is None else str(p) for p in path_values],
                source="url",
                fmt=fmt,
                name=name,
            )
            return

        sidecar_dir = _normalize_structure_sidecar_dir(directory)
        seen_files: set[str] = set()
        path_to_url: dict[str, str] = {}
        file_paths: list[Path | None] = []
        urls: list[str] = []

        for p in path_values:
            if p is None or str(p) == "":
                file_paths.append(None)
                urls.append("")
                continue

            src = Path(p)
            key = str(src)
            if key not in path_to_url:
                filename = _unique_structure_filename(src, seen_files)
                path_to_url[key] = f"{sidecar_dir}/{filename}"

            file_paths.append(src)
            urls.append(path_to_url[key])

        self.add_3d_structures(urls, source="url", fmt=fmt, name=name)
        self._structures_3d_file_paths = file_paths
        self._structures_3d_copy_sidecars = True

    @property
    def filterable(self) -> list[str]:
        """Column names shown in the filter panel."""
        return list(self._filterable)

    @filterable.setter
    def filterable(self, names: list[str]) -> None:
        if not isinstance(names, (list, tuple)):
            raise TypeError("filterable must be a list of column names")
        self._filterable = list(names)

    @property
    def searchable(self) -> list[str]:
        """Column names available for text search."""
        return list(self._searchable)

    @searchable.setter
    def searchable(self, names: list[str]) -> None:
        if not isinstance(names, (list, tuple)):
            raise TypeError("searchable must be a list of column names")
        self._searchable = list(names)

    def configure_column(
        self,
        name: str,
        *,
        display_name: str | None = None,
        link_template: str | None = None,
        copyable: bool | None = None,
        format: str | None = None,
    ) -> None:
        """Set per-column UI hints for the HTML visualization.

        Args:
            name: Column name (must match a previously added column).
            display_name: Override the label shown in cards/tooltips.
            link_template: URL template with ``{column_name}`` placeholders.
            copyable: If True, add a copy button for this value.
            format: Display format hint. Supported values are ``"fixed:N"``,
                ``"percent:N"``, ``"scientific:N"``, ``"integer"``, and
                ``"stars:N"``, where ``N`` controls precision or star count.

        Notes:
            Repeated calls merge with the column's existing hints, so one UI
            setting can be changed without restating the others.
        """
        ui: dict[str, Any] = dict(self._column_ui.get(name, {}))
        if display_name is not None:
            ui["displayName"] = display_name
        if link_template is not None:
            ui["linkTemplate"] = link_template
        if copyable is not None:
            ui["copyable"] = copyable
        if format is not None:
            ui["format"] = format
        self._column_ui[name] = ui

    def configure_card(
        self,
        *,
        title_column: str | None = None,
        subtitle_column: str | None = None,
        fields: list[str] | None = None,
        links: list[dict[str, str]] | None = None,
    ) -> None:
        """Configure the pinned-card panel shown when clicking a point.

        By default the card shows every label column as key-value rows.
        Use this method to pick a title, add a subtitle, restrict which
        fields appear, or add clickable links with per-point URLs.

        Parameters
        ----------
        title_column : str, optional
            Name of an existing column whose value becomes the card
            heading for each point.  Example: ``"Gene Name"``.
        subtitle_column : str, optional
            Name of an existing column shown as an italic line below
            the title.  Example: ``"Organism"``.
        fields : list of str, optional
            Column names to display as key-value rows in the card body.
            If omitted, all label columns are shown.
        links : list of dict, optional
            Clickable buttons rendered below the subtitle.  Each dict
            needs ``"label"`` (button text) and ``"url"`` (URL template).
            Column placeholders like ``{col_name}`` are replaced with the
            point's value for that column.

            Example::

                [{"label": "UniProt",
                  "url": "https://uniprot.org/uniprot/{UniProt ID}"}]

        Examples
        --------
        >>> viz.configure_card(
        ...     title_column="Name",
        ...     subtitle_column="Category",
        ...     fields=["Name", "Score", "Source"],
        ...     links=[{"label": "PubChem",
        ...             "url": "https://pubchem.ncbi.nlm.nih.gov/#query={SMILES}"}],
        ... )
        """
        config: dict[str, Any] = {}
        # Validate referenced columns exist
        known = set(self._columns.keys())
        refs: list[str] = []
        if title_column is not None:
            config["titleColumn"] = title_column
            refs.append(title_column)
        if subtitle_column is not None:
            config["subtitleColumn"] = subtitle_column
            refs.append(subtitle_column)
        if fields is not None:
            config["fields"] = list(fields)
            refs.extend(fields)
        missing = [r for r in refs if r not in known]
        if missing:
            warnings.warn(
                f"configure_card references columns not yet added: {missing}. "
                f"Add them before calling to_html() or the card will be incomplete.",
                UserWarning,
                stacklevel=2,
            )
        if links is not None:
            config["links"] = [dict(lnk) for lnk in links]
        self._card_config = config

    @property
    def n_points(self) -> int:
        """Return the number of points set."""
        return len(self._points_array) if self._points_array is not None else 0

    @property
    def layouts(self) -> list[Column]:
        """Return layouts added."""
        return [self._columns[layout] for layout in self._layout_keys]

    @property
    def labels(self) -> list[Column]:
        """Return labels added."""
        return [self._columns[labels] for labels in self._labels_keys]

    def to_dataframe(
        self,
        *,
        include_coords: bool = True,
        x_col: str = "_tmap_x",
        y_col: str = "_tmap_y",
    ) -> Any:
        """Return visualization metadata as a pandas DataFrame.

        This is useful for advanced notebook workflows where jscatter controls
        (search, filtering, callbacks) need direct access to metadata.

        Args:
            include_coords: If True, include normalized coordinates.
            x_col: Column name for x coordinate when ``include_coords=True``.
            y_col: Column name for y coordinate when ``include_coords=True``.

        Returns:
            ``pandas.DataFrame`` with one row per point.
        """
        import pandas as pd

        n_rows = len(self._points_array) if self._points_array is not None else 0
        if n_rows == 0 and self._columns:
            first_column = next(iter(self._columns.values()))
            n_rows = len(first_column.values)

        for col in self._columns.values():
            if len(col.values) != n_rows:
                raise ValueError(
                    f"Column '{col.name}' has {len(col.values)} values but there are "
                    f"{n_rows} points"
                )

        if self._columns:
            df = pd.DataFrame({name: col.values for name, col in self._columns.items()})
        else:
            df = pd.DataFrame(index=pd.RangeIndex(n_rows))

        if include_coords:
            if self._points_array is None:
                raise ValueError("Call set_points() before include_coords=True.")
            if x_col == y_col:
                raise ValueError("x_col and y_col must be different names.")
            if x_col in df.columns or y_col in df.columns:
                raise ValueError(
                    f"Coordinate columns '{x_col}'/'{y_col}' "
                    "conflict with existing metadata columns."
                )
            coords = self._points_array.astype(np.float32, copy=False)
            df[x_col] = coords[:, 0]
            df[y_col] = coords[:, 1]

        return df

    def set_edges(
        self,
        s: list[int] | NDArray[np.unsignedinteger],
        t: list[int] | NDArray[np.unsignedinteger],
        weights: list[float] | NDArray[np.floating] | None = None,
    ) -> None:
        """Set MST edge source/target arrays and optional edge distances.

        Args:
            s: Source vertex indices for each edge.
            t: Target vertex indices for each edge.
            weights: Optional non-negative distance for each edge. The HTML
                Neighborhood Explorer converts these distances to similarity
                scores. When omitted, it estimates similarity from the plotted
                coordinate distance.

        Raises:
            ValueError: If arrays differ in length, are not 1-D, or contain
                indices outside ``[0, n_points)``.
        """
        s_arr = np.asarray(s, dtype=np.uint32)
        t_arr = np.asarray(t, dtype=np.uint32)

        if s_arr.ndim != 1 or t_arr.ndim != 1:
            raise ValueError(
                f"Edge arrays must be 1-dimensional. Got s: {s_arr.ndim}D and t: {t_arr.ndim}D"
            )

        if s_arr.shape != t_arr.shape:
            raise ValueError(
                f"Edge arrays must have the same length. Got s: {len(s_arr)} and t: {len(t_arr)}"
            )

        weights_arr: np.ndarray | None = None
        if weights is not None:
            weights_arr = np.asarray(weights, dtype=np.float32)
            if weights_arr.ndim != 1:
                raise ValueError(f"Edge weights must be 1-dimensional. Got {weights_arr.ndim}D")
            if weights_arr.shape != s_arr.shape:
                raise ValueError(
                    "Edge weights must have the same length as the edge arrays. "
                    f"Got weights: {len(weights_arr)} and edges: {len(s_arr)}"
                )
            if not np.all(np.isfinite(weights_arr)) or np.any(weights_arr < 0):
                raise ValueError("Edge weights must contain finite, non-negative distances")

        if self.n_points > 0:
            max_idx = self.n_points
            if s_arr.size > 0 and (s_arr.max() >= max_idx or t_arr.max() >= max_idx):
                raise ValueError(
                    f"Edge indices must be < n_points ({max_idx}). "
                    f"Got max(s)={s_arr.max()}, max(t)={t_arr.max()}"
                )

        self._edges_s = s_arr
        self._edges_t = t_arr
        self._edge_weights = weights_arr

    def set_edge_style(
        self,
        color: str | None = None,
        width: float | None = None,
        opacity: float | None = None,
    ) -> None:
        """Set edge rendering style for visualization templates.

        Args:
            color: Hex color string for edges (``#rgb`` or ``#rrggbb``).
            width: Edge line width in CSS pixels. Must be > 0.
            opacity: Edge opacity in ``[0, 1]``.
        """
        if color is not None:
            self.edge_color = _normalize_hex_color(color)

        if width is not None:
            width_value = float(width)
            if not np.isfinite(width_value) or width_value <= 0:
                raise ValueError(f"Edge width must be > 0. Got {width!r}")
            self.edge_width = width_value

        if opacity is not None:
            opacity_value = float(opacity)
            if not np.isfinite(opacity_value) or not 0.0 <= opacity_value <= 1.0:
                raise ValueError(f"Edge opacity must be in [0, 1]. Got {opacity!r}")
            self.edge_opacity = opacity_value

    def set_points(
        self,
        x: list[np.floating] | NDArray[np.floating],
        y: list[np.floating] | NDArray[np.floating],
    ) -> None:
        """
        Store and normalize point coordinates.

        x: X coordinates
        y: Y coordinates
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        if x_arr.size == 0:
            raise ValueError("Cannot set empty coordinates. Provide at least one point.")

        if x_arr.shape != y_arr.shape:
            raise ValueError("x and y must have the same shape")
        if x_arr.ndim != 1 or y_arr.ndim != 1:
            raise ValueError(
                f"Both x and y should be 1 dimensional arrays,\
                Got x: {x_arr.ndim}D and y: {y_arr.ndim}D"
            )
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError(
                "Coordinates contain NaN or Inf values. All x and y values must be finite."
            )

        normalized_coords = _normalize_coords(x_arr, y_arr)
        self._points_array = normalized_coords
        n = len(normalized_coords)

        for col in self._columns.values():
            if len(col.values) != n:
                raise ValueError(
                    f"Column '{col.name}' has {len(col.values)} values but there are {n} points"
                )

        # Re-validate edges against the new point count
        if self._edges_s is not None and self._edges_s.size > 0:
            max_idx = max(int(self._edges_s.max()), int(self._edges_t.max()))
            if max_idx >= n:
                raise ValueError(
                    f"Edge indices must be < n_points ({n}). Got max edge index {max_idx}."
                )

    def _resolve_widget_color(self, color_key: str | None) -> str | list[str] | None:
        """Translate a stored color key into something jscatter accepts.

        Custom hex lists/dicts are saved under a ``_custom_<column>`` key in
        ``self._custom_colormaps``; jscatter only understands matplotlib
        colormap names or raw color lists, so resolve the key back to the
        hex list before handing it over.
        """
        if isinstance(color_key, str) and color_key in self._custom_colormaps:
            return list(self._custom_colormaps[color_key])
        return color_key

    def to_widget(
        self,
        *,
        width: int | str = 800,
        height: int = 420,
        controls: bool = True,
    ) -> Any:
        """Create a jupyter-scatter widget from the current ``TmapViz`` state.

        Args:
            width: Widget width in pixels or ``"auto"``.
            height: Widget height in pixels.
            controls: If True, expose color and categorical filter controls
                when rendering via ``scatter.show()``.

        Returns:
            Configured ``jscatter.Scatter`` instance.
        """
        if self._points_array is None:
            raise ValueError("Call set_points() before converting to a notebook widget.")

        n_points = len(self._points_array)
        for col in self._columns.values():
            if len(col.values) != n_points:
                raise ValueError(
                    f"Column '{col.name}' has {len(col.values)} values but there are "
                    f"{n_points} points"
                )

        import pandas as pd

        from tmap.visualization.jupyter import to_jscatter

        selected_layout = self._layout_keys[0] if self._layout_keys else None

        data_df: pd.DataFrame | None = None
        if self._columns:
            data_df = self.to_dataframe(include_coords=False)
            # Preserve categorical intent for all declared categorical layouts so
            # interactive layout switching can keep categorical color scaling.
            for name, col in self._columns.items():
                if col.dtype == "categorical" and name in data_df.columns:
                    data_df[name] = pd.Series(data_df[name], index=data_df.index).astype("category")

        color_map: str | list[str] | dict[str, str] | None = None
        if selected_layout is not None:
            color_map = self._resolve_widget_color(self._columns[selected_layout].color)

        tooltip_properties = [name for name in self._labels_keys if name in self._columns]

        scatter = to_jscatter(
            self._points_array.astype(np.float32, copy=False),
            color_by=selected_layout,
            color_map=color_map,
            data=data_df,
            tooltip_properties=tooltip_properties or None,
            point_size=self.point_size,
            opacity=self.opacity,
            width=width,
            height=height,
        )

        scatter.background(self.background_color)
        if selected_layout is None:
            scatter.color(default=self.point_color)

        if self._edges_s is not None and self._edges_t is not None and len(self._edges_s) > 0:
            warnings.warn(
                "Edges are not supported in notebook mode yet and will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        if controls:
            self._attach_controls_to_show(
                scatter,
                data_df=data_df,
                selected_layout=selected_layout,
            )
            # Hint notebook display helper to use scatter.show() when controls
            # are requested.
            scatter._tmap_prefers_show = True

        return scatter

    def _attach_controls_to_show(
        self,
        scatter: Any,
        *,
        data_df: Any | None,
        selected_layout: str | None,
    ) -> None:
        """Patch ``scatter.show()`` to include color and filter controls."""
        original_show = scatter.show
        layout_options = tuple(self._layout_keys)
        if data_df is None:
            # No metadata columns available for color/filter controls.
            return
        categorical_filter_values: dict[str, list[Any]] = {}
        for name, col in self._columns.items():
            if col.dtype != "categorical" or name not in data_df.columns:
                continue
            series = data_df[name]
            if hasattr(series, "cat"):
                values = [v for v in series.cat.categories.tolist() if v == v]  # drop NaN
            else:
                values = [v for v in series.dropna().unique().tolist()]
            if values:
                categorical_filter_values[name] = values

        def _show_with_layout_selector(*args: Any, **kwargs: Any) -> Any:
            base_widget = original_show(*args, **kwargs)
            try:
                import ipywidgets as widgets  # type: ignore[import-untyped]
            except ImportError:
                return base_widget

            controls_row: list[Any] = []

            current_layout = scatter.color().get("by")
            initial_layout = current_layout if current_layout in layout_options else selected_layout
            if initial_layout is None and layout_options:
                initial_layout = layout_options[0]

            if layout_options:
                color_dd = widgets.Dropdown(
                    options=list(layout_options),
                    value=initial_layout,
                    description="Color:",
                )

                def _on_layout_change(change: dict[str, Any]) -> None:
                    layout_name = change["new"]
                    col = self._columns[layout_name]
                    cmap = self._resolve_widget_color(col.color)
                    # Reset norm so jscatter re-derives min/max from the new column
                    scatter.color(by=layout_name, map=cmap, norm=None)
                    scatter.legend(True)

                color_dd.observe(_on_layout_change, names="value")
                controls_row.append(color_dd)

            if categorical_filter_values:
                filter_col_dd = widgets.Dropdown(
                    options=["None"] + list(categorical_filter_values.keys()),
                    value="None",
                    description="Filter:",
                )
                filter_val_dd = widgets.Dropdown(
                    options=["All"],
                    value="All",
                    description="Value:",
                    disabled=True,
                )

                def _apply_filter() -> None:
                    col = filter_col_dd.value
                    val = filter_val_dd.value
                    if col == "None" or val == "All":
                        scatter.filter(None)
                        return
                    idxs = np.flatnonzero(data_df[col].to_numpy() == val).tolist()
                    scatter.filter(idxs)

                def _on_filter_col_change(change: dict[str, Any]) -> None:
                    col = change["new"]
                    if col == "None":
                        filter_val_dd.options = ["All"]
                        filter_val_dd.value = "All"
                        filter_val_dd.disabled = True
                        scatter.filter(None)
                        return

                    values = categorical_filter_values[col]
                    filter_val_dd.options = values
                    filter_val_dd.disabled = False
                    filter_val_dd.value = values[0]
                    _apply_filter()

                def _on_filter_value_change(change: dict[str, Any]) -> None:
                    if filter_col_dd.value == "None":
                        return
                    if change["new"] is None:
                        return
                    _apply_filter()

                filter_col_dd.observe(_on_filter_col_change, names="value")
                filter_val_dd.observe(_on_filter_value_change, names="value")
                controls_row.extend([filter_col_dd, filter_val_dd])

            if not controls_row:
                return base_widget
            return widgets.VBox([widgets.HBox(controls_row), base_widget])

        scatter.show = _show_with_layout_selector

    def show(
        self,
        *,
        width: int | str = 800,
        height: int = 420,
        controls: bool = True,
    ) -> Any:
        """Display the notebook widget and return the configured scatter object.

        Args:
            width: Widget width in pixels or ``"auto"``.
            height: Widget height in pixels.
            controls: If True, include color/filter controls and display
                jscatter controls.

        Returns:
            Configured ``jscatter.Scatter`` instance.
        """
        from tmap.visualization.jupyter import _display_scatter

        scatter = self.to_widget(width=width, height=height, controls=controls)
        _display_scatter(scatter, controls=controls)
        return scatter

    def _validate(self) -> None:
        """Pre-flight checks before rendering HTML or writing static files.

        Catches Python-side mistakes that would otherwise surface as
        cryptic JS console errors (ArrayBuffer mismatches, missing columns).
        """
        if self._points_array is None:
            raise ValueError("Call set_points() before rendering.")

        n_points = len(self._points_array)
        for col in self._columns.values():
            if len(col.values) != n_points:
                raise ValueError(
                    f"Column '{col.name}' has {len(col.values)} values but there are "
                    f"{n_points} points"
                )

        # Every layout key must have a numeric column (float32 or uint32)
        for name in self._layout_keys:
            if name not in self._columns:
                raise ValueError(
                    f"Layout '{name}' is registered but has no column data. "
                    f"This would cause a JS fetch error."
                )
            col = self._columns[name]
            if col.dtype == "label":
                raise ValueError(
                    f"Layout '{name}' has dtype 'label' (string) but layouts "
                    f"require numeric data (float32/uint32). "
                    f"This happens when add_label() overwrites a color layout. "
                    f"Use a different name for the label."
                )

        # Every label key must have a column
        for name in self._labels_keys:
            if name not in self._columns:
                raise ValueError(f"Label '{name}' is registered but has no column data.")

    def _resolve_template_name(self, template_name: str) -> str:
        """Select the protein template automatically for local 3D structures."""
        if template_name == "base.html.j2" and self._structures_3d_column is not None:
            return "protein.html.j2"
        return template_name

    def _copy_structure_sidecars(self, output_dir: Path) -> None:
        """Copy local structure files into the generated output bundle."""
        if not self._structures_3d_copy_sidecars or not self._structures_3d_file_paths:
            return
        if self._structures_3d_column is None:
            return

        column = self._columns.get(self._structures_3d_column)
        if column is None:
            return

        copied: set[Path] = set()
        for src, url in zip(self._structures_3d_file_paths, column.values, strict=True):
            if src is None or not url:
                continue
            if not src.exists():
                raise FileNotFoundError(f"3D structure file not found: {src}")
            if not src.is_file():
                raise ValueError(f"3D structure path is not a file: {src}")

            dest = output_dir / str(url)
            if dest in copied:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.add(dest)

    def to_html(self, template_name: str = "base.html.j2") -> str:
        """Return a self-contained HTML document as a string.

        Uses the same template as ``write_static`` (toolbar, side panels,
        theme toggle, etc.) but with all data inlined as base64 so the
        resulting HTML file works without a server.

        Args:
            template_name: Jinja2 template to use.

        Returns:
            Full HTML document string.
        """
        self._validate()
        template_name = self._resolve_template_name(template_name)

        n_points = len(self._points_array)

        # Pack coordinates as binary
        coords_compressed = _pack_coords_binary(self._points_array, bits=16)
        coords_b64 = base64.b64encode(coords_compressed).decode("ascii")

        # Pack columns
        columns_b64: dict[str, str] = {}
        columns_meta: dict[str, dict[str, Any]] = {}
        # Seed with custom hex colormaps so _colormap_to_hex is not called for them
        colormaps_payload: dict[str, list[str]] = dict(self._custom_colormaps)

        for name, col in self._columns.items():
            colormap_name = col.color if col.role in ("layout", "layout+label") else None

            if col.dtype == "categorical":
                compressed, dictionary = _pack_categorical_binary(col.values)
                columns_b64[name] = base64.b64encode(compressed).decode("ascii")
                columns_meta[name] = {
                    "dtype": "uint32",
                    "role": col.role,
                    "colormap": colormap_name,
                    "dictionary": dictionary,
                }
            elif col.dtype == "continuous":
                arr = np.array(
                    [_safe_float(v) for v in col.values],
                    dtype=np.float32,
                )
                compressed = _pack_numeric_binary(arr, "float32")
                columns_b64[name] = base64.b64encode(compressed).decode("ascii")
                columns_meta[name] = {
                    "dtype": "float32",
                    "role": col.role,
                    "colormap": colormap_name,
                }
            else:
                # String columns (labels, SMILES, images) — gzip-compress as JSON
                json_bytes = _encode_string_column(col.values, name)
                compressed = gzip.compress(json_bytes, compresslevel=6)
                columns_b64[name] = base64.b64encode(compressed).decode("ascii")
                columns_meta[name] = {
                    "dtype": "string",
                    "role": col.role,
                    "colormap": None,
                }

            if colormap_name and colormap_name not in colormaps_payload:
                colormaps_payload[colormap_name] = _colormap_to_hex(colormap_name)

        _cycle_colormaps(colormaps_payload, self._columns)

        # Pack edges if present
        edges_b64 = ""
        edge_weights_b64 = ""
        n_edges = 0
        if self._edges_s is not None and self._edges_t is not None:
            n_edges = len(self._edges_s)
            edges_combined = np.concatenate([self._edges_s, self._edges_t]).astype(np.uint32)
            edges_compressed = gzip.compress(edges_combined.tobytes(), compresslevel=6)
            edges_b64 = base64.b64encode(edges_compressed).decode("ascii")
            if self._edge_weights is not None:
                weights_compressed = gzip.compress(
                    self._edge_weights.astype(np.float32, copy=False).tobytes(),
                    compresslevel=6,
                )
                edge_weights_b64 = base64.b64encode(weights_compressed).decode("ascii")

        # Build metadata (same flat structure as write_static)
        layout_options = list(self._layout_keys)
        label_options = [name for name in self._labels_keys if name in self._columns]

        effective_point_size = self.point_size
        if self.point_size == 4.0:
            if n_points > 2_000_000:
                effective_point_size = 0.5
            elif n_points > 500_000:
                effective_point_size = 1.0
            elif n_points > 100_000:
                effective_point_size = 2.0

        # Attach per-column UI hints
        for col_name, ui in self._column_ui.items():
            if col_name in columns_meta:
                columns_meta[col_name]["ui"] = ui

        inline_metadata = {
            "title": self.title,
            "nPoints": n_points,
            "coordDtype": "uint16",
            "pointColor": self.point_color,
            "pointSize": effective_point_size,
            "opacity": self.opacity,
            "edgeStrokeStyle": _hex_to_css_rgba(self.edge_color, self.edge_opacity),
            "edgeWidth": self.edge_width,
            "backgroundColor": _hex_to_rgba(self.background_color),
            "layoutOptions": layout_options,
            "labelOptions": label_options,
            "colormaps": colormaps_payload,
            "smilesColumn": self._smiles_column,
            "imagesColumn": self._images_column,
            "imagesTooltipSize": self._images_tooltip_size,
            "proteinColumn": self._protein_column,
            "structures3dColumn": self._structures_3d_column,
            "structures3dSource": self._structures_3d_source,
            "structures3dFormat": self._structures_3d_format,
            "nEdges": n_edges,
            "hasEdgeWeights": bool(edge_weights_b64),
            "columns": columns_meta,
            "card": self._card_config,
            "filters": self._filterable if self._filterable else (layout_options or None),
            "search": self._searchable if self._searchable else (label_options or None),
        }

        metadata_json_str = json.dumps(
            _to_json_safe(inline_metadata),
            separators=(",", ":"),
            allow_nan=False,
        )

        runtime = _runtime_base64()
        env = _get_jinja_env()
        template = env.get_template(template_name)

        return template.render(
            title=self.title,
            background_color=self.background_color,
            n_points=n_points,
            runtime_regl=runtime["regl"],
            runtime_pubsub=runtime["pubsub"],
            runtime_scatterplot=runtime["scatterplot"],
            # Inline data flags
            inline_data=True,
            inline_metadata=metadata_json_str,
            inline_coords=coords_b64,
            inline_columns=columns_b64,
            inline_edges=edges_b64,
            inline_edge_weights=edge_weights_b64,
        )

    def write_html(
        self,
        path: str | Path,
        template_name: str = "base.html.j2",
    ) -> Path:
        """Write HTML to disk and return the path.

        Args:
            path: Either a full file path (ending in .html) or a directory path.
                  - If a file path: saves to that exact location
                  - If a directory: uses self.title as the filename
            template_name: Jinja2 template to use for rendering.

        Returns:
            Path to the saved file

        Examples:
            >>> viz.write_html("output.html")  # Saves to output.html
            >>> viz.write_html("results/")     # Saves to results/{title}.html
            >>> viz.write_html("results/viz.html")  # Saves to results/viz.html
        """
        path = Path(path)

        # Determine if path is a file or directory
        if str(path).endswith(".html"):
            # Full file path provided
            output_path = path
        elif path.is_dir() or (not path.exists() and not str(path).endswith(".html")):
            # Directory provided (existing or will be created) - use title as filename
            if not self.title.endswith(".html"):
                filename = self.title + ".html"
            else:
                filename = self.title
            output_path = path / filename
        else:
            # Assume it's a file path without .html extension
            output_path = Path(str(path) + ".html")

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = self.to_html(template_name=template_name)

        output_path.write_text(html, encoding="utf-8")
        self._copy_structure_sidecars(output_path.parent)
        return output_path

    def write_static(
        self,
        output_dir: str | Path,
        template_name: str = "base.html.j2",
    ) -> Path:
        """Write all visualization files to a directory for static hosting.

        Writes binary data files (coords, edges, columns), metadata JSON,
        and a rendered HTML shell. The output directory can be served by
        any HTTP server (nginx, python -m http.server, S3, etc.).

        Args:
            output_dir: Directory to write files into (created if needed).
            template_name: Jinja2 template to render as index.html.

        Returns:
            Path to the output directory.
        """
        self._validate()
        template_name = self._resolve_template_name(template_name)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_points = len(self._points_array)

        # --- Write binary data files ---
        # Coordinates
        coords_compressed = _pack_coords_binary(self._points_array, bits=16)
        (output_dir / "coords.bin").write_bytes(coords_compressed)

        # Edges
        n_edges = 0
        if self._edges_s is not None and self._edges_t is not None:
            n_edges = len(self._edges_s)
            edges_combined = np.concatenate([self._edges_s, self._edges_t]).astype(np.uint32)
            edges_compressed = gzip.compress(edges_combined.tobytes(), compresslevel=6)
            (output_dir / "edges.bin").write_bytes(edges_compressed)
            if self._edge_weights is not None:
                weights_compressed = gzip.compress(
                    self._edge_weights.astype(np.float32, copy=False).tobytes(),
                    compresslevel=6,
                )
                (output_dir / "edge_weights.bin").write_bytes(weights_compressed)

        # Columns
        columns_meta: dict[str, dict[str, Any]] = {}
        colormaps_payload: dict[str, list[str]] = dict(self._custom_colormaps)

        _seen_filenames: set[str] = set()
        for name, col in self._columns.items():
            colormap_name = col.color if col.role in ("layout", "layout+label") else None
            safe_name = _sanitize_filename(name, _seen_filenames)

            if col.dtype == "categorical":
                compressed, dictionary = _pack_categorical_binary(col.values)
                (output_dir / f"col_{safe_name}.bin").write_bytes(compressed)
                columns_meta[name] = {
                    "dtype": "uint32",
                    "role": col.role,
                    "colormap": colormap_name,
                    "dictionary": dictionary,
                    "file": safe_name,
                }
            elif col.dtype == "continuous":
                arr = np.array(
                    [_safe_float(v) for v in col.values],
                    dtype=np.float32,
                )
                compressed = _pack_numeric_binary(arr, "float32")
                (output_dir / f"col_{safe_name}.bin").write_bytes(compressed)
                columns_meta[name] = {
                    "dtype": "float32",
                    "role": col.role,
                    "colormap": colormap_name,
                    "file": safe_name,
                }
            else:
                # String columns (labels, SMILES, images) — gzip-compress as JSON
                json_bytes = _encode_string_column(col.values, name)
                compressed = gzip.compress(json_bytes, compresslevel=6)
                (output_dir / f"col_{safe_name}.bin").write_bytes(compressed)
                columns_meta[name] = {
                    "dtype": "string",
                    "role": col.role,
                    "colormap": None,
                    "file": safe_name,
                }

            if colormap_name and colormap_name not in colormaps_payload:
                colormaps_payload[colormap_name] = _colormap_to_hex(colormap_name)

        _cycle_colormaps(colormaps_payload, self._columns)

        # Attach per-column UI hints
        for col_name, ui in self._column_ui.items():
            if col_name in columns_meta:
                columns_meta[col_name]["ui"] = ui

        # Auto-scale point size
        effective_point_size = self.point_size
        if self.point_size == 4.0:
            if n_points > 2_000_000:
                effective_point_size = 0.5
            elif n_points > 500_000:
                effective_point_size = 1.0
            elif n_points > 100_000:
                effective_point_size = 2.0

        layout_options = list(self._layout_keys)
        label_options = [name for name in self._labels_keys if name in self._columns]

        metadata = {
            "title": self.title,
            "nPoints": n_points,
            "coordDtype": "uint16",
            "pointColor": self.point_color,
            "pointSize": effective_point_size,
            "opacity": self.opacity,
            "edgeStrokeStyle": _hex_to_css_rgba(self.edge_color, self.edge_opacity),
            "edgeWidth": self.edge_width,
            "backgroundColor": _hex_to_rgba(self.background_color),
            "layoutOptions": layout_options,
            "labelOptions": label_options,
            "colormaps": colormaps_payload,
            "smilesColumn": self._smiles_column,
            "imagesColumn": self._images_column,
            "imagesTooltipSize": self._images_tooltip_size,
            "proteinColumn": self._protein_column,
            "structures3dColumn": self._structures_3d_column,
            "structures3dSource": self._structures_3d_source,
            "structures3dFormat": self._structures_3d_format,
            "nEdges": n_edges,
            "hasEdgeWeights": self._edge_weights is not None and n_edges > 0,
            "columns": columns_meta,
            "card": self._card_config,
            "filters": self._filterable if self._filterable else (layout_options or None),
            "search": self._searchable if self._searchable else (label_options or None),
        }

        metadata_json = json.dumps(
            _to_json_safe(metadata),
            separators=(",", ":"),
            allow_nan=False,
        )
        (output_dir / "metadata.json").write_text(metadata_json, encoding="utf-8")
        self._copy_structure_sidecars(output_dir)

        # --- Render the HTML shell in fetch mode ---
        # Data is served from files written above (metadata.json, coords.bin, etc.).
        runtime = _runtime_base64()
        env = _get_jinja_env()
        template = env.get_template(template_name)
        html = template.render(
            title=self.title,
            background_color=self.background_color,
            n_points=n_points,
            runtime_regl=runtime["regl"],
            runtime_pubsub=runtime["pubsub"],
            runtime_scatterplot=runtime["scatterplot"],
            inline_data=False,
            inline_metadata="{}",
            inline_coords="",
            inline_columns={},
            inline_edges="",
            inline_edge_weights="",
        )
        (output_dir / "index.html").write_text(html, encoding="utf-8")

        return output_dir

    def serve(self, port: int = 8050, open_browser: bool = True) -> None:
        """Serve the visualization on a local HTTP server.

        For datasets beyond what a single HTML file handles well (>1M points),
        this avoids embedding all data as base64 in one file.  Binary data
        files are served separately and color columns are loaded lazily.

        Args:
            port: TCP port for the local HTTP server.
            open_browser: If True, open the default browser automatically.
        """
        import http.server
        import tempfile
        import threading
        import webbrowser

        tmpdir = self.write_static(
            Path(tempfile.mkdtemp(prefix="tmap_serve_")),
        )

        n_points = len(self._points_array)  # type: ignore[arg-type]
        n_edges = len(self._edges_s) if self._edges_s is not None else 0

        # --- Start HTTP server ---
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, directory=str(tmpdir), **kwargs)

            def log_message(self, format: str, *args: Any) -> None:
                pass  # Suppress request logs

        server = http.server.HTTPServer(("127.0.0.1", port), Handler)
        url = f"http://127.0.0.1:{port}"

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        print(f"Serving TMAP visualization at {url}")
        print(f"  {n_points:,} points, {n_edges:,} edges")
        print("  Press Ctrl+C to stop.")

        if open_browser:
            webbrowser.open(url)

        try:
            thread.join()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server.shutdown()
