#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "layout.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_tmap_ogdf, m) {
    m.doc() = "TMAP OGDF layout extension";

    // Enums
    py::enum_<tmap::Placer>(m, "Placer", "Initial placement strategy during uncoarsening")
        .value("Barycenter", tmap::Placer::Barycenter, "Place at barycenter of neighbors (default)")
        .value("Solar", tmap::Placer::Solar, "Use solar merger info")
        .value("Circle", tmap::Placer::Circle, "Place in circle around barycenter")
        .value("Median", tmap::Placer::Median, "Median position of neighbors")
        .value("Random", tmap::Placer::Random, "Random position (non-deterministic)")
        .value("Zero", tmap::Placer::Zero, "Same position as representative");

    py::enum_<tmap::Merger>(m, "Merger", "Graph coarsening strategy")
        .value("EdgeCover", tmap::Merger::EdgeCover, "Edge cover based")
        .value("LocalBiconnected", tmap::Merger::LocalBiconnected, "Avoids distortions (default)")
        .value("Solar", tmap::Merger::Solar, "Solar system partitioning")
        .value("IndependentSet", tmap::Merger::IndependentSet, "GRIP-style independent set");

    py::enum_<tmap::ScalingType>(m, "ScalingType", "Scaling strategy for layout")
        .value("Absolute", tmap::ScalingType::Absolute)
        .value("RelativeToAvgLength", tmap::ScalingType::RelativeToAvgLength)
        .value("RelativeToDesiredLength", tmap::ScalingType::RelativeToDesiredLength)
        .value("RelativeToDrawing", tmap::ScalingType::RelativeToDrawing);

    py::enum_<tmap::UntangleMode>(m, "UntangleMode", "Crossing-reduction post-pass mode")
        .value("Rotate", tmap::UntangleMode::Rotate,
            "Best of several rotations about the parent + reflection (default)")
        .value("Reflect", tmap::UntangleMode::Reflect,
            "Only flip each subtree across its parent->child axis");

    // LayoutConfig
    py::class_<tmap::LayoutConfig>(m, "LayoutConfig", "Configuration for OGDF layout")
        .def(py::init<>())
        .def_readwrite("k", &tmap::LayoutConfig::k,
            "Number of nearest neighbors for k-NN graph (default: 10)")
        .def_readwrite("kc", &tmap::LayoutConfig::kc,
            "Query multiplier for LSH (queries k*kc, keeps k) (default: 10)")
        .def_readwrite("fme_iterations", &tmap::LayoutConfig::fme_iterations,
            "FastMultipoleEmbedder iterations (default: 1000)")
        .def_readwrite("fme_precision", &tmap::LayoutConfig::fme_precision,
            "Multipole expansion precision (default: 4)")
        .def_readwrite("sl_repeats", &tmap::LayoutConfig::sl_repeats,
            "ScalingLayout repeats (default: 1)")
        .def_readwrite("sl_extra_scaling_steps", &tmap::LayoutConfig::sl_extra_scaling_steps,
            "Extra scaling steps (default: 2)")
        .def_readwrite("sl_scaling_min", &tmap::LayoutConfig::sl_scaling_min,
            "Minimum scaling (default: 1.0)")
        .def_readwrite("sl_scaling_max", &tmap::LayoutConfig::sl_scaling_max,
            "Maximum scaling (default: 1.0)")
        .def_readwrite("sl_scaling_type", &tmap::LayoutConfig::sl_scaling_type,
            "Scaling type (default: RelativeToDrawing)")
        .def_readwrite("mmm_repeats", &tmap::LayoutConfig::mmm_repeats,
            "ModularMultilevelMixer repeats (default: 1)")
        .def_readwrite("placer", &tmap::LayoutConfig::placer,
            "Placer algorithm (default: Barycenter)")
        .def_readwrite("merger", &tmap::LayoutConfig::merger,
            "Merger algorithm (default: LocalBiconnected)")
        .def_readwrite("merger_factor", &tmap::LayoutConfig::merger_factor,
            "Merger factor (default: 2.0)")
        .def_readwrite("merger_adjustment", &tmap::LayoutConfig::merger_adjustment,
            "Edge length adjustment (default: 0)")
        .def_readwrite("node_size", &tmap::LayoutConfig::node_size,
            "Node size for repulsion; used by the legacy (adaptive=False) layout (default: 1/65)")

        // ---- Per-level adaptive layout (on by default) ----
        .def_readwrite("adaptive", &tmap::LayoutConfig::adaptive,
            "Use the per-level adaptive layout (default: True). Set False for the "
            "legacy single-shot ModularMultilevelMixer.")
        .def_readwrite("ns_cap", &tmap::LayoutConfig::ns_cap,
            "Adaptive per-level node-size cap; the master spread/grid knob (default: 0.03)")
        .def_readwrite("ns_coef", &tmap::LayoutConfig::ns_coef,
            "Adaptive per-level node-size coefficient (default: 2.5)")
        .def_readwrite("ns_exp", &tmap::LayoutConfig::ns_exp,
            "Adaptive per-level node-size exponent: size shrinks as ns_coef/n^ns_exp (default: 0.40)")
        .def_readwrite("quad_rotate", &tmap::LayoutConfig::quad_rotate,
            "Rotate the FME frame per level to remove the central cross (default: True)")

        // ---- Crossing-reduction post-pass (on by default) ----
        .def_readwrite("untangle", &tmap::LayoutConfig::untangle,
            "Run the crossing-reduction post-pass (default: True)")
        .def_readwrite("untangle_mode", &tmap::LayoutConfig::untangle_mode,
            "Untangle mode: Rotate (default) or Reflect")
        .def_readwrite("untangle_max_sub", &tmap::LayoutConfig::untangle_max_sub,
            "Cap on subtree size considered; 0 = no cap (default: 2000)")
        .def_readwrite("untangle_passes", &tmap::LayoutConfig::untangle_passes,
            "Greedy untangle sweeps (default: 4)")
        .def_readwrite("untangle_rot_steps", &tmap::LayoutConfig::untangle_rot_steps,
            "Rotation angles tried per subtree (default: 64)")
        .def_readwrite("untangle_max_angle", &tmap::LayoutConfig::untangle_max_angle,
            "Aesthetic cap in DEGREES on subtree turn; 0 = no cap (default: 90)")
        .def_readwrite("untangle_slide_eps", &tmap::LayoutConfig::untangle_slide_eps,
            "Bounded stem-slide fraction; 0 = off / exact edge lengths (default: 0)")
        .def_readwrite("untangle_slide_steps", &tmap::LayoutConfig::untangle_slide_steps,
            "Stem-scale samples when untangle_slide_eps > 0 (default: 5)")
        .def_readwrite("deterministic", &tmap::LayoutConfig::deterministic,
            "Enable deterministic mode (single thread, seeded RNG)")
        .def_property("seed",
            [](const tmap::LayoutConfig& c) -> py::object {
                if (c.seed.has_value()) return py::int_(c.seed.value());
                return py::none();
            },
            [](tmap::LayoutConfig& c, py::object v) {
                if (v.is_none()) c.seed = std::nullopt;
                else c.seed = v.cast<uint32_t>();
            },
            "Random seed (None for unseeded)");

    // LayoutResult
    py::class_<tmap::LayoutResult>(m, "LayoutResult", "Layout computation result")
        .def_readonly("x", &tmap::LayoutResult::x, "X coordinates")
        .def_readonly("y", &tmap::LayoutResult::y, "Y coordinates")
        .def_readonly("s", &tmap::LayoutResult::s, "Edge source indices")
        .def_readonly("t", &tmap::LayoutResult::t, "Edge target indices");

    // Layout function
    m.def("layout_from_edge_list", &tmap::layout_from_edge_list,
        py::arg("vertex_count"),
        py::arg("edges"),
        py::arg("config") = tmap::LayoutConfig{},
        py::arg("create_mst") = true,
        R"doc(
        Compute 2D layout from edge list using OGDF.

        Parameters
        ----------
        vertex_count : int
            Number of vertices in the graph
        edges : list of (int, int, float)
            Edge list as (source, target, weight) tuples
        config : LayoutConfig, optional
            Layout configuration
        create_mst : bool, optional
            If True, compute MST first (default: True)

        Returns
        -------
        LayoutResult
            Result with x, y coordinates and edge topology (s, t)
        )doc");
}
