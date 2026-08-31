"""
Tests for the visualization module.

Tests cover:
- TmapViz class creation and configuration
- Point coordinate setting and validation
- Color layout (continuous and categorical)
- Label and SMILES columns
- HTML rendering (binary mode)
- Binary container format utilities
"""

import gzip
import json
import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tmap.visualization.static import plot_static

matplotlib.use("Agg")

# Check if visualization dependencies are available
try:
    from tmap.visualization import TmapViz
    from tmap.visualization.binary import (
        BinaryContainerWriter,
        dequantize_coords,
        pack_categorical_column,
        pack_coords,
        pack_numeric_column,
        quantize_coords,
    )

    _VIZ_AVAILABLE = True
except ImportError as e:
    _VIZ_AVAILABLE = False
    _IMPORT_ERROR = e

pytestmark = pytest.mark.skipif(
    not _VIZ_AVAILABLE, reason="Visualization dependencies not available (jinja2 required)"
)

try:
    from jscatter.jscatter import Scatter as _Scatter

    _JSCATTER_AVAILABLE = True
except ImportError:
    _JSCATTER_AVAILABLE = False

try:
    import ipywidgets as _widgets

    _IPYWIDGETS_AVAILABLE = True
except ImportError:
    _IPYWIDGETS_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_coords():
    """Simple coordinate arrays for testing."""
    x = np.array([0.0, 1.0, 0.5, -0.5], dtype=np.float32)
    y = np.array([0.0, 0.5, 1.0, -0.5], dtype=np.float32)
    return x, y


@pytest.fixture
def sample_data():
    """Sample data for visualization testing."""
    n = 100
    np.random.seed(42)
    return {
        "x": np.random.uniform(-1, 1, n).astype(np.float32),
        "y": np.random.uniform(-1, 1, n).astype(np.float32),
        "continuous": np.random.uniform(0, 100, n),
        "categorical": np.random.choice(["A", "B", "C"], n).tolist(),
        "labels": [f"Point_{i}" for i in range(n)],
        "smiles": [f"C{'C' * (i % 5)}" for i in range(n)],  # Simple SMILES
    }


@pytest.fixture
def viz_with_data(sample_data):
    """TmapViz instance with data already set."""
    viz = TmapViz()
    viz.set_points(sample_data["x"], sample_data["y"])
    return viz, sample_data


# =============================================================================
# TmapViz Basic Tests
# =============================================================================


class TestTmapVizCreation:
    """Tests for TmapViz creation and basic properties."""

    def test_default_values(self):
        """TmapViz should have sensible defaults."""
        viz = TmapViz()

        assert viz.title == "MyTMAP"
        assert viz.background_color == "#FFFFFF"
        assert viz.point_color == "#4a9eff"
        assert viz.point_size == 4.0
        assert viz.opacity == 0.85
        assert viz.edge_color == "#000000"
        assert viz.edge_opacity == 0.5
        assert viz.edge_width == 2.0
        assert viz.n_points == 0

    def test_properties_settable(self):
        """Properties should be settable."""
        viz = TmapViz()

        viz.title = "Test Title"
        viz.background_color = "#7A7A7A"
        viz.point_color = "#FF0000"
        viz.point_size = 10.0
        viz.opacity = 0.5

        assert viz.title == "Test Title"
        assert viz.background_color == "#7A7A7A"
        assert viz.point_color == "#FF0000"
        assert viz.point_size == 10.0
        assert viz.opacity == 0.5


# =============================================================================
# set_points Tests
# =============================================================================


class TestSetPoints:
    """Tests for set_points method."""

    def test_basic_set_points(self, simple_coords):
        """set_points should accept coordinate arrays."""
        x, y = simple_coords
        viz = TmapViz()

        viz.set_points(x, y)

        assert viz.n_points == 4

    def test_accepts_lists(self):
        """set_points should accept Python lists."""
        viz = TmapViz()

        viz.set_points([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

        assert viz.n_points == 3

    def test_mismatched_shapes_raises(self):
        """Mismatched x and y should raise ValueError."""
        viz = TmapViz()

        with pytest.raises(ValueError, match="same shape"):
            viz.set_points([0.0, 1.0], [0.0, 1.0, 2.0])

    def test_2d_arrays_raise(self):
        """2D arrays should raise ValueError."""
        viz = TmapViz()

        with pytest.raises(ValueError, match="1 dimensional"):
            viz.set_points(np.zeros((5, 2)), np.zeros((5, 2)))

    def test_coordinates_normalized(self, simple_coords):
        """Coordinates should be normalized to [-1, 1]."""
        x, y = simple_coords
        viz = TmapViz()

        viz.set_points(x, y)

        # Internal points should be normalized
        points = viz._points_array
        for p in points:
            assert -1.0 <= p[0] <= 1.0
            assert -1.0 <= p[1] <= 1.0


class TestToDataFrame:
    """Tests for TmapViz.to_dataframe helper."""

    def test_to_dataframe_with_coords(self, viz_with_data):
        """Metadata export should include coordinate columns by default."""
        viz, data = viz_with_data
        viz.add_color_layout("value", data["continuous"], categorical=False)
        viz.add_label("name", data["labels"])

        df = viz.to_dataframe()

        assert len(df) == viz.n_points
        assert "value" in df.columns
        assert "name" in df.columns
        assert "_tmap_x" in df.columns
        assert "_tmap_y" in df.columns
        np.testing.assert_allclose(df["_tmap_x"].to_numpy(), viz._points_array[:, 0])
        np.testing.assert_allclose(df["_tmap_y"].to_numpy(), viz._points_array[:, 1])

    def test_to_dataframe_without_coords(self, viz_with_data):
        """include_coords=False should omit coordinate columns."""
        viz, data = viz_with_data
        viz.add_color_layout("group", data["categorical"], categorical=True)

        df = viz.to_dataframe(include_coords=False)

        assert len(df) == viz.n_points
        assert "group" in df.columns
        assert "_tmap_x" not in df.columns
        assert "_tmap_y" not in df.columns

    def test_to_dataframe_requires_points_when_including_coords(self):
        """Requesting coords without points should raise."""
        viz = TmapViz()
        viz.add_label("name", ["a", "b"])

        with pytest.raises(ValueError, match="set_points"):
            viz.to_dataframe(include_coords=True)


# =============================================================================
# add_color_layout Tests
# =============================================================================


class TestAddColorLayout:
    """Tests for add_color_layout method."""

    def test_continuous_layout(self, viz_with_data):
        """Should add continuous color layout."""
        viz, data = viz_with_data

        viz.add_color_layout("value", data["continuous"], categorical=False)

        assert len(viz.layouts) == 1
        assert viz.layouts[0].name == "value"
        assert viz.layouts[0].dtype == "continuous"

    def test_categorical_layout(self, viz_with_data):
        """Should add categorical color layout."""
        viz, data = viz_with_data

        viz.add_color_layout("group", data["categorical"], categorical=True, color="tab10")

        assert len(viz.layouts) == 1
        assert viz.layouts[0].name == "group"
        assert viz.layouts[0].dtype == "categorical"

    def test_default_colormap_continuous(self, viz_with_data):
        """Continuous should default to viridis."""
        viz, data = viz_with_data

        viz.add_color_layout("value", data["continuous"], categorical=False)

        assert viz.layouts[0].color == "viridis"

    def test_default_colormap_categorical(self, viz_with_data):
        """Categorical should default to tab10."""
        viz, data = viz_with_data

        viz.add_color_layout("group", data["categorical"], categorical=True)

        assert viz.layouts[0].color == "tab10"

    def test_custom_colormap(self, viz_with_data):
        """Should accept custom colormap."""
        viz, data = viz_with_data

        viz.add_color_layout("value", data["continuous"], categorical=False, color="plasma")

        assert viz.layouts[0].color == "plasma"

    def test_invalid_colormap_raises(self, viz_with_data):
        """Invalid colormap should raise ValueError."""
        viz, data = viz_with_data

        with pytest.raises(ValueError, match="Color option not found"):
            viz.add_color_layout("value", data["continuous"], color="not_a_colormap")

    def test_add_as_label_true(self, viz_with_data):
        """add_as_label=True should add to labels."""
        viz, data = viz_with_data

        viz.add_color_layout("value", data["continuous"], add_as_label=True)

        assert len(viz.labels) == 1
        assert viz.labels[0].name == "value"

    def test_add_as_label_false(self, viz_with_data):
        """add_as_label=False should not add to labels."""
        viz, data = viz_with_data

        viz.add_color_layout("value", data["continuous"], add_as_label=False)

        # Should be in layouts but not labels
        assert len(viz.layouts) == 1
        assert len(viz.labels) == 0

    def test_multiple_layouts(self, viz_with_data):
        """Should support multiple color layouts."""
        viz, data = viz_with_data

        viz.add_color_layout("continuous_col", data["continuous"], categorical=False)
        viz.add_color_layout("categorical_col", data["categorical"], categorical=True)

        assert len(viz.layouts) == 2


# =============================================================================
# add_label Tests
# =============================================================================


class TestAddLabel:
    """Tests for add_label method."""

    def test_basic_label(self, viz_with_data):
        """Should add label column."""
        viz, data = viz_with_data

        viz.add_label("name", data["labels"])

        assert len(viz.labels) == 1
        assert viz.labels[0].name == "name"
        assert viz.labels[0].dtype == "label"

    def test_multiple_labels(self, viz_with_data):
        """Should support multiple labels."""
        viz, data = viz_with_data

        viz.add_label("name", data["labels"])
        viz.add_label("id", [str(i) for i in range(100)])

        assert len(viz.labels) == 2


# =============================================================================
# add_smiles Tests
# =============================================================================


class TestAddSmiles:
    """Tests for add_smiles method."""

    def test_basic_smiles(self, viz_with_data):
        """Should add SMILES column."""
        viz, data = viz_with_data

        viz.add_smiles(data["smiles"], "structure")

        assert viz._smiles_column == "structure"
        assert "structure" in [label.name for label in viz.labels]

    def test_only_one_smiles_allowed(self, viz_with_data):
        """Only one SMILES column should be allowed."""
        viz, data = viz_with_data

        viz.add_smiles(data["smiles"], "structure1")

        with pytest.raises(ValueError, match="Only one SMILES column"):
            viz.add_smiles("structure2", data["smiles"])


# =============================================================================
# render Tests
# =============================================================================


class TestToHtmlRendering:
    """Tests for to_html rendering."""

    def test_render_basic(self, viz_with_data):
        """Should render to HTML string."""
        viz, data = viz_with_data
        viz.add_color_layout("value", data["continuous"])

        html = viz.to_html()

        assert isinstance(html, str)
        assert "<html" in html.lower()
        assert "</html>" in html.lower()

    def test_render_without_points_raises(self):
        """render without set_points should raise."""
        viz = TmapViz()

        with pytest.raises(ValueError, match="set_points"):
            viz.to_html()

    def test_render_contains_title(self, viz_with_data):
        """HTML should contain title."""
        viz, data = viz_with_data
        viz.title = "Test Visualization"

        html = viz.to_html()

        assert "Test Visualization" in html

    def test_render_with_smiles_uses_smiles_template(self, viz_with_data):
        """Adding SMILES should auto-switch to smiles template."""
        viz, data = viz_with_data
        viz.add_smiles(data["smiles"], "structure")

        html = viz.to_html()

        # Should render without error (smiles template used)
        assert isinstance(html, str)
        assert len(html) > 0


# =============================================================================
# Notebook API Tests
# =============================================================================


@pytest.mark.skipif(not _JSCATTER_AVAILABLE, reason="jupyter-scatter is not installed")
class TestNotebookAPI:
    """Tests for TmapViz notebook helpers."""

    def test_to_widget_basic_style(self, viz_with_data):
        """Should map point/background style onto jscatter."""
        viz, _ = viz_with_data
        viz.background_color = "#ffffff"
        viz.point_color = "#ff0000"
        viz.point_size = 6.0
        viz.opacity = 0.7

        scatter = viz.to_widget()

        assert isinstance(scatter, _Scatter)
        assert scatter.background()["color"][:3] == pytest.approx((1.0, 1.0, 1.0))
        assert scatter.color()["by"] is None
        assert scatter.color()["default"][:3] == pytest.approx((1.0, 0.0, 0.0))
        assert scatter.size()["default"] == pytest.approx(6.0)
        assert scatter.opacity()["default"] == pytest.approx(0.7)

    def test_to_widget_uses_layout_and_labels(self, viz_with_data):
        """Should use first layout as color and labels as tooltip fields."""
        viz, data = viz_with_data
        viz.add_label("name", data["labels"])
        viz.add_color_layout("value", data["continuous"], categorical=False, color="plasma")

        scatter = viz.to_widget()

        assert isinstance(scatter, _Scatter)
        assert scatter.color()["by"] == "value"
        assert scatter.tooltip()["enable"] is True
        assert "name" in scatter.tooltip()["properties"]

    def test_to_widget_warns_when_edges_set(self, viz_with_data):
        """Edges are not rendered in notebook mode and should emit a warning."""
        viz, _ = viz_with_data
        viz.set_edges([0, 1, 2], [1, 2, 3])

        with pytest.warns(UserWarning, match="Edges are not supported"):
            scatter = viz.to_widget()

        assert isinstance(scatter, _Scatter)

    def test_show_calls_display_helper(
        self, viz_with_data, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """show() should delegate widget display to jupyter helper."""
        viz, _ = viz_with_data
        calls: list[bool] = []

        def _fake_display(scatter: _Scatter, *, controls: bool = False) -> None:
            calls.append(controls)

        monkeypatch.setattr("tmap.visualization.jupyter._display_scatter", _fake_display)

        scatter = viz.show(controls=True)
        assert isinstance(scatter, _Scatter)
        assert calls == [True]

    @pytest.mark.skipif(not _IPYWIDGETS_AVAILABLE, reason="ipywidgets is not installed")
    def test_to_widget_controls_show_includes_color_and_filter(self, viz_with_data) -> None:
        """controls=True should add color and categorical filter controls to show()."""
        viz, data = viz_with_data
        viz.add_color_layout("rings", [i % 4 for i in range(viz.n_points)], categorical=True)
        viz.add_color_layout("mw", data["continuous"], categorical=False, color="plasma")
        viz.add_label("name", data["labels"])

        scatter = viz.to_widget(controls=True)
        shown = scatter.show()

        assert isinstance(scatter, _Scatter)
        assert isinstance(shown, _widgets.VBox)

        controls = shown.children[0]
        assert isinstance(controls, _widgets.HBox)

        dropdowns = [c for c in controls.children if isinstance(c, _widgets.Dropdown)]
        assert len(dropdowns) >= 3
        color_dd = next(dd for dd in dropdowns if dd.description == "Color:")
        filter_dd = next(dd for dd in dropdowns if dd.description == "Filter:")
        value_dd = next(dd for dd in dropdowns if dd.description == "Value:")

        assert color_dd.value == "rings"
        color_dd.value = "mw"
        assert scatter.color()["by"] == "mw"

        filter_dd.value = "rings"
        assert value_dd.disabled is False
        value_dd.value = 2
        filtered = scatter.widget.filter
        assert filtered is not None
        assert len(filtered) == 25

        filter_dd.value = "None"
        assert scatter.widget.filter is None

    @pytest.mark.skipif(not _IPYWIDGETS_AVAILABLE, reason="ipywidgets is not installed")
    def test_to_widget_controls_false_does_not_add_custom_dropdowns(self, viz_with_data) -> None:
        """controls=False should not add custom color/filter dropdowns."""
        viz, data = viz_with_data
        viz.add_color_layout("rings", [i % 4 for i in range(viz.n_points)], categorical=True)
        viz.add_color_layout("mw", data["continuous"], categorical=False, color="plasma")

        scatter = viz.to_widget(controls=False)
        shown = scatter.show()

        if not isinstance(shown, _widgets.VBox):
            return
        controls = shown.children[0]
        if not isinstance(controls, _widgets.HBox):
            return
        dropdowns = [
            c
            for c in controls.children
            if isinstance(c, _widgets.Dropdown) and c.description in {"Color:", "Filter:", "Value:"}
        ]
        assert dropdowns == []

    @pytest.mark.skipif(not _IPYWIDGETS_AVAILABLE, reason="ipywidgets is not installed")
    def test_show_controls_true_uses_same_controls_ui(self, viz_with_data) -> None:
        """show(controls=True) and to_widget(controls=True).show() should match behavior."""
        viz, data = viz_with_data
        viz.add_color_layout("rings", [i % 4 for i in range(viz.n_points)], categorical=True)
        viz.add_color_layout("mw", data["continuous"], categorical=False, color="plasma")

        scatter = viz.show(controls=True)
        shown = scatter.show()
        assert isinstance(shown, _widgets.VBox)
        controls = shown.children[0]
        assert isinstance(controls, _widgets.HBox)
        dropdowns = [c for c in controls.children if isinstance(c, _widgets.Dropdown)]
        color_dd = next(dd for dd in dropdowns if dd.description == "Color:")
        color_dd.value = "mw"
        assert scatter.color()["by"] == "mw"
        assert color_dd.value == "mw"


# =============================================================================
# to_html Tests
# =============================================================================


class TestToHtml:
    """Tests for to_html."""

    def test_to_html_returns_valid_html(self, viz_with_data):
        """to_html should return a valid HTML string."""
        viz, data = viz_with_data
        viz.add_color_layout("value", data["continuous"])

        html = viz.to_html()

        assert isinstance(html, str)
        assert "<html" in html.lower()
        # Inline metadata rendered as JS object literal
        assert "const metadata =" in html


# =============================================================================
# write_html Tests
# =============================================================================


class TestWriteHtml:
    """Tests for write_html method."""

    def test_write_html_creates_file(self, viz_with_data, tmp_path):
        """write_html should create HTML file."""
        viz, data = viz_with_data
        viz.title = "test_output"

        output_path = viz.write_html(tmp_path)

        assert output_path.exists()
        assert output_path.name == "test_output.html"

    def test_write_html_adds_html_extension(self, viz_with_data, tmp_path):
        """write_html should add .html extension if missing."""
        viz, data = viz_with_data
        viz.title = "my_viz"

        output_path = viz.write_html(tmp_path)

        assert output_path.suffix == ".html"

    def test_write_html_preserves_existing_extension(self, viz_with_data, tmp_path):
        """write_html should preserve .html extension if present."""
        viz, data = viz_with_data
        viz.title = "my_viz.html"

        output_path = viz.write_html(tmp_path)

        assert output_path.name == "my_viz.html"
        assert not output_path.name.endswith(".html.html")

    def test_write_html_binary_format(self, viz_with_data, tmp_path):
        """write_html should produce binary format output."""
        viz, data = viz_with_data
        viz.title = "binary_output"

        output_path = viz.write_html(tmp_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "const metadata =" in content


# =============================================================================
# Binary Module Tests
# =============================================================================


class TestBinaryModule:
    """Tests for binary container format utilities."""

    def test_quantize_coords_16bit(self):
        """16-bit quantization should work."""
        coords = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])

        quantized = quantize_coords(coords, bits=16)

        assert quantized.dtype == np.uint16
        assert quantized[0, 0] == 0  # -1 -> 0
        assert quantized[2, 0] == 65535  # 1 -> max
        assert 32000 < quantized[1, 0] < 33000  # 0 -> ~middle

    def test_quantize_dequantize_roundtrip(self):
        """Quantize then dequantize should approximate original."""
        coords = np.array([[-0.5, 0.25], [0.75, -0.9]])

        quantized = quantize_coords(coords, bits=16)
        restored = dequantize_coords(quantized, bits=16)

        # Should be close (within quantization error)
        np.testing.assert_allclose(coords, restored, atol=1e-4)

    def test_pack_coords(self):
        """pack_coords should return compressed bytes."""
        x = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        y = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        compressed, uncompressed_size = pack_coords(x, y)

        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
        assert uncompressed_size == 12  # 3 points * 2 coords * 2 bytes

    def test_pack_numeric_column_float32(self):
        """pack_numeric_column should handle float32."""
        values = np.array([1.0, 2.0, 3.0, 4.0])

        compressed, uncompressed_size = pack_numeric_column(values, "float32")

        assert isinstance(compressed, bytes)
        assert uncompressed_size == 16  # 4 floats * 4 bytes

    def test_pack_numeric_column_int32(self):
        """pack_numeric_column should handle int32."""
        values = np.array([1, 2, 3, 4])

        compressed, uncompressed_size = pack_numeric_column(values, "int32")

        assert isinstance(compressed, bytes)
        assert uncompressed_size == 16  # 4 ints * 4 bytes

    def test_pack_categorical_column(self):
        """pack_categorical_column should create dictionary encoding."""
        values = ["A", "B", "A", "C", "B", "A"]

        compressed, uncompressed_size, dictionary = pack_categorical_column(values)

        assert isinstance(compressed, bytes)
        assert dictionary == ["A", "B", "C"]  # Order of first occurrence
        assert uncompressed_size == 24  # 6 indices * 4 bytes


class TestBinaryContainerWriter:
    """Tests for BinaryContainerWriter class."""

    def test_basic_write(self):
        """Should write basic container."""
        writer = BinaryContainerWriter()
        writer.add_coords(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
        )
        writer.set_metadata({"title": "Test"})

        result = writer.write()

        assert isinstance(result, bytes)
        assert result[:4] == b"TMAP"  # Magic bytes

    def test_write_chunked(self):
        """write_chunked should return dict of chunks."""
        writer = BinaryContainerWriter()
        writer.add_coords(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
        )
        writer.add_numeric_column("values", np.array([1.0, 2.0]), "float32")
        writer.set_metadata({"title": "Test"})

        chunks = writer.write_chunked()

        assert "header" in chunks
        assert "metadata" in chunks
        assert "coords" in chunks
        assert "col_values" in chunks


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for visualization module."""

    def test_single_point(self):
        """Should handle single point."""
        viz = TmapViz()
        viz.set_points([0.0], [0.0])
        viz.add_label("name", ["Only Point"])

        html = viz.to_html()

        assert isinstance(html, str)

    def test_empty_categorical_values(self, viz_with_data):
        """Should handle categorical with empty strings."""
        viz, data = viz_with_data
        values = ["", "A", "", "B"] * 25

        viz.add_color_layout("sparse", values, categorical=True)

        assert len(viz.layouts) == 1

    def test_unicode_labels(self, viz_with_data):
        """Should handle unicode in labels."""
        viz, data = viz_with_data
        unicode_labels = [f"Point_{i}_\u03b1\u03b2\u03b3" for i in range(100)]

        viz.add_label("name", unicode_labels)

        html = viz.to_html()
        # Binary mode: labels are gzip-compressed and base64-encoded,
        # so unicode won't appear directly. Just verify render succeeds.
        assert isinstance(html, str)
        assert "<html" in html.lower()

    def test_large_numeric_values(self, viz_with_data):
        """Should handle large numeric values."""
        viz, data = viz_with_data
        large_values = np.array([1e10, 1e-10, 0, -1e10] * 25)

        viz.add_color_layout("large", large_values)

        html = viz.to_html()
        assert isinstance(html, str)

    def test_nan_values_in_continuous(self, viz_with_data):
        """Should warn and handle NaN in continuous values."""
        viz, data = viz_with_data
        values = np.array([1.0, np.nan, 3.0, np.nan] * 25)

        with pytest.warns(UserWarning, match="contains NaN values"):
            viz.add_color_layout("with_nan", values)

        # Should not raise, NaN values are rendered in black client-side
        html = viz.to_html()
        assert isinstance(html, str)
        assert "<html" in html.lower()


# =============================================================================
# Column Length Validation
# =============================================================================


class TestColumnValidation:
    """Tests for column length validation."""

    def test_column_length_mismatch_after_set_points(self):
        """Adding column with wrong length after set_points should raise."""
        viz = TmapViz()
        viz.set_points([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

        # This should work - setting points validates existing columns
        # But the column is added to the dict without validation
        # Validation happens at render time
        viz.add_label("name", ["A", "B"])  # Wrong length

        with pytest.raises(ValueError, match="values but there are"):
            viz.to_html()

    def test_set_points_validates_existing_columns(self):
        """set_points should validate against existing columns."""
        viz = TmapViz()
        viz.add_label("name", ["A", "B", "C"])

        with pytest.raises(ValueError, match="values but there are"):
            viz.set_points([0.0, 1.0], [0.0, 1.0])  # Only 2 points, 3 labels


# =============================================================================
# set_edges Tests
# =============================================================================


class TestSetEdges:
    """Tests for set_edges method."""

    def test_basic_set_edges(self, viz_with_data):
        """set_edges should accept valid s, t arrays."""
        viz, data = viz_with_data
        s = np.array([0, 1, 2], dtype=np.uint32)
        t = np.array([1, 2, 3], dtype=np.uint32)

        viz.set_edges(s, t)

        assert viz._edges_s is not None
        assert viz._edges_t is not None
        assert len(viz._edges_s) == 3
        assert len(viz._edges_t) == 3

    def test_set_edges_accepts_distances_for_neighborhood_similarity(self, viz_with_data):
        viz, _ = viz_with_data

        viz.set_edges([0, 1, 2], [1, 2, 3], [0.1, 0.25, 0.4])

        assert viz._edge_weights is not None
        np.testing.assert_allclose(viz._edge_weights, [0.1, 0.25, 0.4])

    def test_set_edges_rejects_invalid_distances(self, viz_with_data):
        viz, _ = viz_with_data

        with pytest.raises(ValueError, match="same length"):
            viz.set_edges([0, 1], [1, 2], [0.1])

        with pytest.raises(ValueError, match="finite, non-negative"):
            viz.set_edges([0, 1], [1, 2], [0.1, -0.2])

    def test_set_edges_mismatched_length(self, viz_with_data):
        """Mismatched s and t should raise ValueError."""
        viz, data = viz_with_data

        with pytest.raises(ValueError, match="same length"):
            viz.set_edges([0, 1], [1, 2, 3])

    def test_set_edges_2d_raises(self, viz_with_data):
        """2D arrays should raise ValueError."""
        viz, data = viz_with_data

        with pytest.raises(ValueError, match="1-dimensional"):
            viz.set_edges(np.zeros((3, 2), dtype=np.uint32), np.zeros((3, 2), dtype=np.uint32))

    def test_set_edges_out_of_bounds(self, viz_with_data):
        """Edge indices >= n_points should raise ValueError."""
        viz, data = viz_with_data
        n = len(data["x"])

        with pytest.raises(ValueError, match="must be < n_points"):
            viz.set_edges([0, n], [1, 0])

    def test_render_with_edges(self, viz_with_data):
        """Rendered HTML should include edge data."""
        viz, data = viz_with_data
        viz.add_color_layout("value", data["continuous"])
        viz.set_edges([0, 1, 2], [1, 2, 3])

        html = viz.to_html()

        # Inline edges should be present
        assert "inline_edges" in html or "const metadata =" in html
        # Extract inline metadata JSON from the JS
        match = re.search(
            r"const metadata = ({.*?});",
            html,
            re.DOTALL,
        )
        assert match is not None
        meta = json.loads(match.group(1))
        assert meta["nEdges"] == 3
        assert meta["edgeStrokeStyle"] == "rgba(0, 0, 0, 0.5)"
        assert meta["edgeWidth"] == 2.0

    def test_edge_distances_are_serialized_for_neighborhoods(self, viz_with_data, tmp_path):
        viz, data = viz_with_data
        viz.add_color_layout("value", data["continuous"])
        viz.set_edges([0, 1, 2], [1, 2, 3], [0.1, 0.25, 0.4])

        out = viz.write_static(tmp_path / "weighted")
        meta = json.loads((out / "metadata.json").read_text())

        assert meta["hasEdgeWeights"] is True
        assert (out / "edge_weights.bin").exists()

    def test_custom_edge_style_in_header(self, viz_with_data):
        """Custom edge style should be serialized in header metadata."""
        viz, data = viz_with_data
        viz.set_edges([0, 1], [1, 2])
        viz.set_edge_style(color="#f03", width=4.5, opacity=0.35)

        html = viz.to_html()

        match = re.search(
            r"const metadata = ({.*?});",
            html,
            re.DOTALL,
        )
        assert match is not None
        meta = json.loads(match.group(1))
        assert meta["edgeStrokeStyle"] == "rgba(255, 0, 51, 0.35)"
        assert meta["edgeWidth"] == 4.5


class TestEdgeStyle:
    """Tests for edge style configuration."""

    def test_set_edge_style_updates_values(self):
        """set_edge_style should update style attributes."""
        viz = TmapViz()
        viz.set_edge_style(color="#abc", width=3.25, opacity=0.2)

        assert viz.edge_color == "#aabbcc"
        assert viz.edge_width == 3.25
        assert viz.edge_opacity == 0.2

    def test_set_edge_style_invalid_color_raises(self):
        """Invalid edge color should raise ValueError."""
        viz = TmapViz()
        with pytest.raises(ValueError, match="Invalid hex color"):
            viz.set_edge_style(color="not-a-color")

    def test_set_edge_style_invalid_width_raises(self):
        """Non-positive edge width should raise ValueError."""
        viz = TmapViz()
        with pytest.raises(ValueError, match="must be > 0"):
            viz.set_edge_style(width=0)

    def test_set_edge_style_invalid_opacity_raises(self):
        """Opacity outside [0, 1] should raise ValueError."""
        viz = TmapViz()
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            viz.set_edge_style(opacity=1.2)


# =============================================================================
# Static Plot Tests
# =============================================================================


class TestPlotStatic:
    """Tests for the matplotlib static plot function."""

    @pytest.fixture(autouse=True)
    def _close_figs(self):
        yield
        plt.close("all")

    @pytest.fixture
    def embedding(self):
        return np.column_stack([np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)]).astype(np.float32)

    def test_returns_axes(self, embedding):
        ax = plot_static(embedding)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_continuous_color(self, embedding):
        values = np.linspace(0, 1, 50)
        ax = plot_static(embedding, color_by=values)
        # Should have a colorbar (figure has more than 1 axes)
        assert len(ax.figure.axes) > 1

    def test_categorical_color_with_legend(self, embedding):
        labels = np.array(["A", "B"] * 25)
        ax = plot_static(embedding, color_by=labels)
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 2

    def test_edges_drawn(self, embedding):
        edges = np.array([[i, i + 1] for i in range(49)], dtype=np.int32)
        ax = plot_static(embedding, edges=edges)
        # Should have a LineCollection
        from matplotlib.collections import LineCollection

        lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(lcs) == 1

    def test_dataframe_color_by(self, embedding):
        import pandas as pd

        df = pd.DataFrame({"group": ["X", "Y", "Z", "W", "V"] * 10})
        ax = plot_static(embedding, color_by="group", data=df)
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 5

    def test_existing_axes_passthrough(self, embedding):
        fig, existing_ax = plt.subplots()
        returned_ax = plot_static(embedding, ax=existing_ax)
        assert returned_ax is existing_ax

    def test_no_ticks_or_labels(self, embedding):
        ax = plot_static(embedding)
        assert ax.get_xticks().tolist() == []
        assert ax.get_yticks().tolist() == []
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
        for spine in ax.spines.values():
            assert not spine.get_visible()

    def test_invalid_shape_raises(self):
        bad = np.ones((10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            plot_static(bad)


# =============================================================================
# Declarative UI Configuration Tests
# =============================================================================


class TestAddFilter:
    """Tests for add_filter: filter panel columns without a color layout."""

    def test_registers_column_without_layout(self, viz_with_data):
        viz, data = viz_with_data
        viz.add_filter("score", data["continuous"])

        assert "score" in viz._columns
        assert viz._columns["score"].dtype == "continuous"
        assert viz._columns["score"].color is None
        assert viz._layout_keys == []
        assert [c.name for c in viz.layouts] == []

    def test_added_as_label_by_default(self, viz_with_data):
        viz, data = viz_with_data
        viz.add_filter("score", data["continuous"])

        assert viz._columns["score"].role == "filter+label"
        assert "score" in viz._labels_keys

    def test_add_as_label_false(self, viz_with_data):
        viz, data = viz_with_data
        viz.add_filter("score", data["continuous"], add_as_label=False)

        assert viz._columns["score"].role == "filter"
        assert "score" not in viz._labels_keys

    def test_categorical_filter(self, viz_with_data):
        viz, data = viz_with_data
        viz.add_filter("group", data["categorical"], categorical=True)

        assert viz._columns["group"].dtype == "categorical"
        assert viz._columns["group"].color is None

    def test_continuous_rejects_non_numeric(self, viz_with_data):
        viz, data = viz_with_data
        with pytest.raises(ValueError, match="categorical=True"):
            viz.add_filter("group", data["categorical"])

    def test_rejects_existing_color_layout(self, viz_with_data):
        viz, data = viz_with_data
        viz.add_color_layout("score", data["continuous"])
        with pytest.raises(ValueError, match="already exists as a color layout"):
            viz.add_filter("score", data["continuous"])

    def test_repeated_call_does_not_duplicate(self, viz_with_data):
        viz, data = viz_with_data
        viz.add_filter("score", data["continuous"])
        viz.add_filter("score", data["continuous"])

        assert viz._filter_keys == ["score"]

    def test_metadata_lists_filter_without_colormap(self, viz_with_data, tmp_path):
        viz, data = viz_with_data
        viz.add_filter("score", data["continuous"])
        viz.add_filter("group", data["categorical"], categorical=True)

        out = viz.write_static(tmp_path / "out")
        meta = json.loads((out / "metadata.json").read_text())

        assert meta["filters"] == ["score", "group"]
        assert meta["layoutOptions"] == []
        # No colors were worked out for these columns.
        assert meta["colormaps"] == {}
        assert meta["columns"]["score"]["colormap"] is None
        assert meta["columns"]["score"]["dtype"] == "float32"
        assert meta["columns"]["group"]["dtype"] == "uint32"
        assert sorted(meta["columns"]["group"]["dictionary"]) == ["A", "B", "C"]

    def test_filters_come_before_color_layouts(self, viz_with_data, tmp_path):
        viz, data = viz_with_data
        viz.add_color_layout("value", data["continuous"])
        viz.add_filter("score", data["continuous"])

        out = viz.write_static(tmp_path / "out")
        meta = json.loads((out / "metadata.json").read_text())

        assert meta["filters"] == ["score", "value"]

    def test_filterable_property_overrides(self, viz_with_data, tmp_path):
        viz, data = viz_with_data
        viz.add_color_layout("value", data["continuous"])
        viz.add_filter("score", data["continuous"])
        viz.filterable = ["value"]

        out = viz.write_static(tmp_path / "out")
        meta = json.loads((out / "metadata.json").read_text())

        assert meta["filters"] == ["value"]

    def test_comparison_falls_back_to_filter_columns(self, viz_with_data):
        # With no color layouts, the neighbour comparison has to take its
        # numbers from the filter panel instead.
        viz, data = viz_with_data
        viz.add_filter("score", data["continuous"])

        html = viz.to_html()

        assert "[...new Set([...filterColumnNames, ...layoutOptions])]" in html

    def test_inline_html_carries_filter_column(self, viz_with_data):
        viz, data = viz_with_data
        viz.add_filter("score", data["continuous"])

        html = viz.to_html()
        match = re.search(r"const metadata = ({.*?});", html, re.DOTALL)
        assert match is not None
        meta = json.loads(match.group(1))

        assert meta["filters"] == ["score"]
        assert "score" in meta["columns"]


class TestFilterableProperty:
    """Tests for the filterable property."""

    def test_default_empty(self):
        viz = TmapViz()
        assert viz.filterable == []

    def test_set_and_get(self):
        viz = TmapViz()
        viz.filterable = ["col_a", "col_b"]
        assert viz.filterable == ["col_a", "col_b"]

    def test_returns_copy(self):
        viz = TmapViz()
        viz.filterable = ["col_a"]
        result = viz.filterable
        result.append("col_b")
        assert viz.filterable == ["col_a"]

    def test_invalid_type_raises(self):
        viz = TmapViz()
        with pytest.raises(TypeError, match="list"):
            viz.filterable = "not_a_list"

    def test_accepts_tuple(self):
        viz = TmapViz()
        viz.filterable = ("col_a", "col_b")
        assert viz.filterable == ["col_a", "col_b"]


class TestSearchableProperty:
    """Tests for the searchable property."""

    def test_default_empty(self):
        viz = TmapViz()
        assert viz.searchable == []

    def test_set_and_get(self):
        viz = TmapViz()
        viz.searchable = ["name", "id"]
        assert viz.searchable == ["name", "id"]

    def test_invalid_type_raises(self):
        viz = TmapViz()
        with pytest.raises(TypeError, match="list"):
            viz.searchable = 42


class TestConfigureColumn:
    """Tests for per-column HTML presentation hints."""

    def test_stores_all_ui_hints(self):
        viz = TmapViz()
        viz.configure_column(
            "score",
            display_name="Confidence",
            link_template="https://example.test/item/{name}",
            copyable=True,
            format="percent:1",
        )

        assert viz._column_ui["score"] == {
            "displayName": "Confidence",
            "linkTemplate": "https://example.test/item/{name}",
            "copyable": True,
            "format": "percent:1",
        }

    def test_repeated_calls_merge_hints(self):
        viz = TmapViz()
        viz.configure_column("score", display_name="Confidence")
        viz.configure_column("score", format="fixed:2")

        assert viz._column_ui["score"] == {
            "displayName": "Confidence",
            "format": "fixed:2",
        }

    def test_serialized_in_static_metadata(self, viz_with_data, tmp_path):
        viz, data = viz_with_data
        viz.add_label("name", data["labels"])
        viz.add_color_layout("score", data["continuous"])
        viz.configure_column(
            "score",
            display_name="Confidence",
            link_template="https://example.test/item/{name}",
            format="fixed:2",
        )

        out = viz.write_static(tmp_path / "out")
        meta = json.loads((out / "metadata.json").read_text())

        assert meta["columns"]["score"]["ui"] == {
            "displayName": "Confidence",
            "linkTemplate": "https://example.test/item/{name}",
            "format": "fixed:2",
        }

    def test_html_runtime_consumes_presentation_hints(self, viz_with_data):
        viz, data = viz_with_data
        viz.add_label("name", data["labels"])
        viz.add_color_layout("score", data["continuous"])
        viz.configure_column("score", display_name="Confidence", format="stars:5")

        html = viz.to_html()

        assert "function getColumnDisplayName" in html
        assert "function formatValueText" in html
        assert "function resolveColumnLink" in html
        assert "renderColumnValue(name, val, idx)" in html


class TestHtmlInteractionShell:
    """Tests for the responsive, accessible visualization controls."""

    def test_contains_responsive_and_keyboard_controls(self, viz_with_data):
        viz, data = viz_with_data
        viz.add_color_layout("value", data["continuous"])

        html = viz.to_html()

        assert "@media (max-width: 640px)" in html
        assert 'id="canvas" tabindex="0" role="region"' in html
        assert 'id="tb-reset"' not in html
        assert 'id="tb-help"' in html
        assert 'id="interaction-help"' in html
        assert "scatterplot.zoomToOrigin" in html

    def test_theme_honors_configured_background(self, viz_with_data):
        viz, data = viz_with_data
        viz.background_color = "#FFFFFF"
        viz.add_color_layout("value", data["continuous"])

        html = viz.to_html()

        assert "const configuredBackground = normalizeBackground(metadata.backgroundColor)" in html
        assert "Force dark background on load" not in html
        assert "edgeColorForBackground" in html

    def test_light_theme_styles_copyable_card_rows(self, viz_with_data):
        viz, data = viz_with_data
        viz.add_label("name", data["labels"])
        viz.add_color_layout("value", data["continuous"])
        viz.configure_column("name", copyable=True)

        html = viz.to_html()

        assert '[data-theme="light"] .card-copy-row {' in html
        assert '[data-theme="light"] .card-copy-row .value { color: #1d1d1f; }' in html

    def test_search_navigation_and_docked_inspector(self, viz_with_data):
        viz, data = viz_with_data
        viz.add_label("name", data["labels"])
        viz.add_color_layout("value", data["continuous"])
        viz.searchable = ["name"]

        html = viz.to_html()

        assert 'id="search-result-list"' in html
        assert 'id="search-prev"' in html
        assert 'id="search-next"' in html
        assert 'id="partial-match" checked' in html
        assert '<aside id="card-stack"' in html
        assert 'aria-label="Point inspector"' in html
        assert "function activateSearchMatch" in html
        assert "scatterplot.zoomToLocation" in html
        assert "openPointInspector(idx, { replace: true })" in html

    def test_tool_panels_do_not_collide_with_selection_details(self, viz_with_data):
        viz, data = viz_with_data
        viz.add_label("name", data["labels"])
        viz.add_color_layout("value", data["continuous"])

        html = viz.to_html()

        assert "#panel-explore," in html
        assert ".tool-panel-open #export-panel" in html
        assert ".tool-panel-open.inspector-open #export-panel" in html
        assert "max-width: none !important;" in html
        assert "function updateWorkspaceLayout" in html
        assert "&& (Boolean(openToolPanel) || exportVisible)" in html
        assert "cardStack.classList.toggle('panel-suppressed', suppressInspector)" in html
        assert "exportEl?.classList.toggle('panel-suppressed', suppressExport)" in html

    def test_neighborhood_explorer_is_embedded_in_inspector(self, viz_with_data):
        viz, data = viz_with_data
        viz.add_label("name", data["labels"])
        viz.add_color_layout("value", data["continuous"])
        viz.set_edges([0, 1, 2], [1, 2, 3], [0.1, 0.2, 0.3])

        html = viz.to_html()

        assert "function buildNeighborIndex" in html
        assert "function renderTreeExplorer" in html
        assert 'class="neighborhood-section"' in html
        assert "connected neighbors" in html
        assert "Side-by-side comparison" in html
        assert "IntersectionObserver" in html

    def test_tree_focus_modes_are_embedded_in_inspector(self, viz_with_data):
        """The inspector ships the Neighbors / Path chips and their walks."""
        viz, data = viz_with_data
        viz.add_label("name", data["labels"])
        viz.add_color_layout("value", data["continuous"])
        viz.set_edges([0, 1, 2], [1, 2, 3], [0.1, 0.2, 0.3])

        html = viz.to_html()

        # Both chips share one section and one mode variable.
        assert "function renderTreeModes" in html
        assert "data-tree-mode=" in html
        for mode, label in (("neighbors", "Neighbors"), ("path", "Path")):
            assert f"mode: '{mode}', label: '{label}'" in html
        assert "function setTreeFocus" in html
        assert "let treeFocusMode = 'off';" in html

        # Tree traversal primitives.
        assert "function treeBfs" in html
        assert "function treePath" in html

        # Per-mode bodies.
        assert "function renderNeighborBody" in html
        assert "function renderPathBody" in html

    def test_subtree_isolation_is_not_offered(self, viz_with_data):
        """Lasso selection already covers branch isolation, so no Subtree mode."""
        viz, data = viz_with_data
        viz.add_label("name", data["labels"])
        viz.add_color_layout("value", data["continuous"])
        viz.set_edges([0, 1, 2], [1, 2, 3], [0.1, 0.2, 0.3])

        html = viz.to_html()

        assert "Subtree" not in html
        assert "subtreeIndices" not in html
        assert "branchSizes" not in html
        assert "data-branch-select" not in html

    def test_tree_distance_colouring_registers_a_synthetic_column(self, viz_with_data):
        """Distance-from-here is a colour layout, routed through the Color by menu."""
        viz, data = viz_with_data
        viz.add_label("name", data["labels"])
        viz.add_color_layout("value", data["continuous"])
        viz.set_edges([0, 1, 2], [1, 2, 3], [0.1, 0.2, 0.3])

        html = viz.to_html()

        assert "function applyTreeDistanceColor" in html
        assert "function retireTreeDistanceColumn" in html
        assert "__tree_distance__" in html
        assert "Colour map by distance from here" in html

    def test_tree_explorer_bounds_its_work_on_large_maps(self, viz_with_data):
        """Long paths are capped rather than rendered wholesale."""
        viz, data = viz_with_data
        viz.add_label("name", data["labels"])
        viz.add_color_layout("value", data["continuous"])
        viz.set_edges([0, 1, 2], [1, 2, 3], [0.1, 0.2, 0.3])

        html = viz.to_html()

        assert "const MAX_VISIBLE_PATH_NODES" in html
        assert "const MAX_VISIBLE_NEIGHBORS" in html


class TestConfigureCard:
    """Tests for configure_card method."""

    def test_stores_config(self):
        viz = TmapViz()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            viz.configure_card(
                title_column="UniProt ID",
                subtitle_column="protein_name",
                fields=["cluster_size", "pLDDT"],
                links=[{"label": "UniProt", "url": "https://uniprot.org/{UniProt ID}"}],
            )
        assert viz._card_config is not None
        assert viz._card_config["titleColumn"] == "UniProt ID"
        assert viz._card_config["subtitleColumn"] == "protein_name"
        assert viz._card_config["fields"] == ["cluster_size", "pLDDT"]
        assert len(viz._card_config["links"]) == 1

    def test_partial_config(self):
        viz = TmapViz()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            viz.configure_card(title_column="Name")
        assert viz._card_config == {"titleColumn": "Name"}

    def test_serialized_in_write_static(self, viz_with_data, tmp_path):
        viz, data = viz_with_data
        viz.add_label("name", data["labels"])
        viz.add_color_layout("value", data["continuous"])
        viz.configure_card(title_column="name", fields=["value"])

        out = viz.write_static(tmp_path / "out")
        meta = json.loads((out / "metadata.json").read_text())

        assert meta["card"] is not None
        assert meta["card"]["titleColumn"] == "name"
        assert meta["card"]["fields"] == ["value"]

    def test_serialized_in_to_html(self, viz_with_data):
        viz, data = viz_with_data
        viz.add_label("name", data["labels"])
        viz.add_color_layout("value", data["continuous"])
        viz.configure_card(title_column="name", fields=["value"])

        html = viz.to_html()
        match = re.search(
            r"const metadata = ({.*?});",
            html,
            re.DOTALL,
        )
        assert match is not None
        meta = json.loads(match.group(1))
        assert meta["card"]["titleColumn"] == "name"


class Test3DStructures:
    """Tests for local 3D structure metadata and sidecar handling."""

    def test_add_3d_structures_defaults_to_url(self):
        viz = TmapViz()
        viz.add_3d_structures(["pdbs/1CRN.pdb"], fmt="mmcif")

        assert viz._structures_3d_column == "structure"
        assert viz._structures_3d_source == "url"
        assert viz._structures_3d_format == "cif"

    def test_add_3d_structures_accepts_text_source(self):
        viz = TmapViz()
        viz.add_3d_structures(["HEADER test\nEND\n"], source="text", fmt="pdb")

        assert viz._structures_3d_source == "text"

    def test_add_3d_structures_rejects_invalid_source(self):
        viz = TmapViz()
        with pytest.raises(ValueError, match="source must be 'url' or 'text'"):
            viz.add_3d_structures(["HEADER test\nEND\n"], source="inline")

    def test_3d_structures_auto_select_protein_template(self, tmp_path):
        viz = TmapViz()
        viz.set_points([0.0], [0.0])
        viz.add_3d_structures(["pdbs/1CRN.pdb"])

        html = viz.to_html()
        assert "Protein template" in html
        assert '"structures3dSource":"url"' in html

        out = viz.write_static(tmp_path / "out")
        index_html = (out / "index.html").read_text()
        assert "Protein template" in index_html

    def test_add_3d_structure_files_copies_sidecars(self, tmp_path):
        src_dir = tmp_path / "input"
        src_dir.mkdir()
        pdb = src_dir / "1CRN.pdb"
        pdb.write_text("HEADER test\nEND\n")

        viz = TmapViz()
        viz.set_points([0.0, 1.0, 2.0], [0.0, 0.5, 1.0])
        viz.add_3d_structure_files([pdb, None, pdb], fmt="pdb")

        out = viz.write_static(tmp_path / "out")

        copied = out / "structures" / "1CRN.pdb"
        assert copied.read_text() == "HEADER test\nEND\n"

        meta = json.loads((out / "metadata.json").read_text())
        assert meta["structures3dColumn"] == "structure"
        assert meta["structures3dSource"] == "url"
        assert meta["structures3dFormat"] == "pdb"

        col_file = out / f"col_{meta['columns']['structure']['file']}.bin"
        urls = json.loads(gzip.decompress(col_file.read_bytes()).decode())
        assert urls == ["structures/1CRN.pdb", "", "structures/1CRN.pdb"]

    def test_add_3d_structure_files_write_html_copies_sidecars(self, tmp_path):
        pdb = tmp_path / "1CRN.pdb"
        pdb.write_text("HEADER test\nEND\n")

        viz = TmapViz()
        viz.set_points([0.0], [0.0])
        viz.add_3d_structure_files([pdb])

        html_path = viz.write_html(tmp_path / "viz.html")

        assert html_path.exists()
        assert (tmp_path / "structures" / "1CRN.pdb").read_text() == "HEADER test\nEND\n"

    def test_add_3d_structure_files_rejects_unsafe_directory(self):
        viz = TmapViz()
        with pytest.raises(ValueError, match="relative path"):
            viz.add_3d_structure_files(["1CRN.pdb"], directory="../outside")


class TestBackwardCompat:
    """No new config calls should produce same metadata shape as before."""

    def test_no_config_no_extra_keys(self, viz_with_data, tmp_path):
        viz, data = viz_with_data
        viz.add_color_layout("value", data["continuous"])
        viz.add_label("name", data["labels"])

        out = viz.write_static(tmp_path / "out")
        meta = json.loads((out / "metadata.json").read_text())

        # card should be null when not configured
        assert meta.get("card") is None
        # filters/search auto-populated from layouts/labels
        assert meta.get("filters") == ["value"]
        assert meta.get("search") == ["value", "name"]

        # No column should have a "ui" key
        for col_meta in meta["columns"].values():
            assert "ui" not in col_meta


class TestStaticShellMode:
    """Ensure write_static emits fetch-based HTML shell."""

    def test_write_static_uses_fetch_shell(self, viz_with_data, tmp_path):
        viz, data = viz_with_data
        viz.add_color_layout("value", data["continuous"])
        viz.add_label("name", data["labels"])

        out = viz.write_static(tmp_path / "out")
        index_html = (out / "index.html").read_text()

        assert "const metadata = await fetch('./metadata.json').then(r => r.json());" in index_html
        assert "const metadata = {" not in index_html
