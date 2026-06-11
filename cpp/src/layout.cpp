#include "layout.hpp"

#include "untangle.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <thread>

// OGDF includes
#include <ogdf/basic/Graph.h>
#include <ogdf/basic/GraphAttributes.h>
#include <ogdf/basic/PreprocessorLayout.h>
#include <ogdf/basic/basic.h>
#include <ogdf/basic/extended_graph_alg.h>
#include <ogdf/basic/simple_graph_alg.h>
#include <ogdf/energybased/FastMultipoleEmbedder.h>
#include <ogdf/basic/LayoutModule.h>
#include <ogdf/energybased/multilevel_mixer/BarycenterPlacer.h>
#include <ogdf/energybased/multilevel_mixer/CirclePlacer.h>
#include <ogdf/energybased/multilevel_mixer/EdgeCoverMerger.h>
#include <ogdf/energybased/multilevel_mixer/IndependentSetMerger.h>
#include <ogdf/energybased/multilevel_mixer/LocalBiconnectedMerger.h>
#include <ogdf/energybased/multilevel_mixer/MedianPlacer.h>
#include <ogdf/energybased/multilevel_mixer/ModularMultilevelMixer.h>
#include <ogdf/energybased/multilevel_mixer/MultilevelGraph.h>
#include <ogdf/energybased/multilevel_mixer/RandomPlacer.h>
#include <ogdf/energybased/multilevel_mixer/ScalingLayout.h>
#include <ogdf/energybased/multilevel_mixer/SolarMerger.h>
#include <ogdf/energybased/multilevel_mixer/SolarPlacer.h>
#include <ogdf/energybased/multilevel_mixer/ZeroPlacer.h>
#include <ogdf/graphalg/steiner_tree/EdgeWeightedGraph.h>
#include <ogdf/packing/ComponentSplitterLayout.h>
#include <ogdf/packing/TileToRowsCCPacker.h>

namespace tmap {

namespace {

// Convert our Placer enum to OGDF InitialPlacer
ogdf::InitialPlacer* create_placer(Placer p) {
    switch (p) {
        case Placer::Barycenter: return new ogdf::BarycenterPlacer();
        case Placer::Solar:      return new ogdf::SolarPlacer();
        case Placer::Circle:     return new ogdf::CirclePlacer();
        case Placer::Median:     return new ogdf::MedianPlacer();
        case Placer::Random:     return new ogdf::RandomPlacer();
        case Placer::Zero:       return new ogdf::ZeroPlacer();
        default:                 return new ogdf::BarycenterPlacer();
    }
}

// Convert our Merger enum to OGDF MultilevelBuilder
ogdf::MultilevelBuilder* create_merger(Merger m, double factor, int adjustment) {
    ogdf::MultilevelBuilder* merger = nullptr;
    switch (m) {
        case Merger::EdgeCover: {
            auto* ecm = new ogdf::EdgeCoverMerger();
            ecm->setFactor(factor);
            ecm->setEdgeLengthAdjustment(adjustment);
            merger = ecm;
            break;
        }
        case Merger::LocalBiconnected: {
            auto* lbm = new ogdf::LocalBiconnectedMerger();
            lbm->setFactor(factor);
            lbm->setEdgeLengthAdjustment(adjustment);
            merger = lbm;
            break;
        }
        case Merger::Solar: {
            merger = new ogdf::SolarMerger();
            break;
        }
        case Merger::IndependentSet: {
            auto* ism = new ogdf::IndependentSetMerger();
            ism->setSearchDepthBase(factor);
            merger = ism;
            break;
        }
        default: {
            auto* lbm = new ogdf::LocalBiconnectedMerger();
            lbm->setFactor(factor);
            lbm->setEdgeLengthAdjustment(adjustment);
            merger = lbm;
            break;
        }
    }
    return merger;
}

// Convert our ScalingType to OGDF
ogdf::ScalingLayout::ScalingType to_ogdf_scaling(ScalingType st) {
    switch (st) {
        case ScalingType::Absolute:
            return ogdf::ScalingLayout::ScalingType::Absolute;
        case ScalingType::RelativeToAvgLength:
            return ogdf::ScalingLayout::ScalingType::RelativeToAvgLength;
        case ScalingType::RelativeToDesiredLength:
            return ogdf::ScalingLayout::ScalingType::RelativeToDesiredLength;
        case ScalingType::RelativeToDrawing:
        default:
            return ogdf::ScalingLayout::ScalingType::RelativeToDrawing;
    }
}

// Per-level adaptive force layout. This is the major new addition to the layout
// in TMAP version 2.
//
// Drives the multilevel loop manually (mirroring ModularMultilevelMixer) and
// sets the FME node size on EVERY level from that level's node count, via the law
// nodeSize(n) = min(ns_cap, ns_coef / n^ns_exp). Coarse levels (few nodes) get
// larger repulsion so the global skeleton spreads and fills; the finest level
// stays small so no lattice/grid artifact forms. The coarsest level is seeded on
// a disc and each level's frame is randomly rotated (quad_rotate) to keep the
// multipole's 4-fold anisotropy from locking to fixed axes (no central cross, except in
// extreme cases).
class PerLevelAdaptiveLayout : public ogdf::LayoutModule {
public:
    PerLevelAdaptiveLayout(const LayoutConfig& cfg, unsigned threads, std::mt19937& rng)
        : cfg_(cfg), threads_(threads), rng_(rng) {}

    void call(ogdf::GraphAttributes& GA) override {
        auto* fme = new ogdf::FastMultipoleEmbedder();
        fme->setNumIterations(cfg_.fme_iterations);
        fme->setMultipolePrec(cfg_.fme_precision);
        fme->setDefaultEdgeLength(1);
        fme->setDefaultNodeSize(1);
        fme->setRandomize(false);
        fme->setNumberOfThreads(threads_);

        auto* sl = new ogdf::ScalingLayout();
        sl->setLayoutRepeats(cfg_.sl_repeats);

        // ScalingLayout takes ownership of fme, will later be deleted with it
        sl->setSecondaryLayout(fme);
        sl->setExtraScalingSteps(cfg_.sl_extra_scaling_steps);
        sl->setScalingType(to_ogdf_scaling(cfg_.sl_scaling_type));
        sl->setScaling(cfg_.sl_scaling_min, cfg_.sl_scaling_max);

        ogdf::MultilevelBuilder* merger =
            create_merger(cfg_.merger, cfg_.merger_factor, cfg_.merger_adjustment);
        ogdf::InitialPlacer* placer = create_placer(cfg_.placer);

        ogdf::MultilevelGraph mlg(GA);
        ogdf::Graph& lg = mlg.getGraph();
        merger->buildAllLevels(mlg);
        seedCoarsestOnDisc(mlg.getGraphAttributes(), lg);

        while (mlg.getLevel() > 0) {
            applyLevelSize(mlg.getGraphAttributes(), lg);
            runScalingLevel(sl, mlg.getGraphAttributes(), lg);
            mlg.moveToZero();
            placer->placeOneLevel(mlg);
        }

        // finest level
        applyLevelSize(mlg.getGraphAttributes(), lg);
        runScalingLevel(sl, mlg.getGraphAttributes(), lg);
        mlg.exportAttributes(GA);

        // also deletes fme (its secondary layout)
        delete sl;
        delete merger;
        delete placer;
    }

private:
    double nodeSizeForLevel(int nLevel) const {
        return std::min(cfg_.ns_cap,
            cfg_.ns_coef / std::pow(static_cast<double>(std::max(1, nLevel)), cfg_.ns_exp));
    }
    void applyLevelSize(ogdf::GraphAttributes& lga, const ogdf::Graph& lg) const {
        const double ns = nodeSizeForLevel(lg.numberOfNodes());
        for (ogdf::node v : lg.nodes) {
            lga.width(v) = lga.height(v) = ns;
        }
    }
    void seedCoarsestOnDisc(ogdf::GraphAttributes& cga, const ogdf::Graph& lg) {
        std::uniform_real_distribution<double> ang(0.0, 6.2831853);
        std::uniform_real_distribution<double> rad(0.0, 1.0);
        for (ogdf::node v : lg.nodes) {
            const double r = std::sqrt(rad(rng_));
            const double a = ang(rng_);
            cga.x(v) = r * std::cos(a);
            cga.y(v) = r * std::sin(a);
        }
    }
    void runScalingLevel(ogdf::ScalingLayout* sl, ogdf::GraphAttributes& lga,
            const ogdf::Graph& lg) {
        if (!cfg_.quad_rotate) {
            sl->call(lga);
            return;
        }
        std::uniform_real_distribution<double> rot(0.0, 6.2831853);
        const double th = rot(rng_), c = std::cos(th), s = std::sin(th);
        for (ogdf::node v : lg.nodes) {
            const double xx = lga.x(v), yy = lga.y(v);
            lga.x(v) = c * xx - s * yy;
            lga.y(v) = s * xx + c * yy;
        }
        sl->call(lga);

        // rotate back so the frame stays consistent
        for (ogdf::node v : lg.nodes) {
            const double xx = lga.x(v), yy = lga.y(v);
            lga.x(v) = c * xx + s * yy;
            lga.y(v) = -s * xx + c * yy;
        }
    }

    const LayoutConfig& cfg_;
    unsigned threads_;
    std::mt19937& rng_;
};

}

LayoutResult layout_from_edge_list(
    uint32_t vertex_count,
    const std::vector<std::tuple<uint32_t, uint32_t, float>>& edges,
    const LayoutConfig& config,
    bool create_mst
) {
    LayoutResult result;

    // Handle empty/trivial cases
    if (vertex_count == 0) {
        return result;
    }
    if (vertex_count == 1) {
        result.x = {0.0f};
        result.y = {0.0f};
        return result;
    }

    // Set seed for determinism if requested (default to 0 when deterministic with no seed)
    if (config.deterministic) {
        const int seed = static_cast<int>(config.seed.value_or(0u));
        ogdf::setSeed(seed);
    }

    // Build OGDF graph
    ogdf::EdgeWeightedGraph<float> g;
    std::vector<ogdf::node> nodes(vertex_count);

    for (uint32_t i = 0; i < vertex_count; i++) {
        nodes[i] = g.newNode();
    }

    // Find max weight for normalization
    float max_weight = 0.0f;
    for (const auto& [src, tgt, w] : edges) {
        if (w > max_weight) max_weight = w;
    }
    if (max_weight == 0.0f) max_weight = 1.0f;

    // Add edges (normalized, skip negatives)
    for (const auto& [src, tgt, w] : edges) {
        if (w >= 0.0f && src < vertex_count && tgt < vertex_count) {
            g.newEdge(nodes[src], nodes[tgt], w / max_weight);
        }
    }

    // Clean graph
    ogdf::makeLoopFree(g);
    ogdf::makeParallelFreeUndirected(g);

    // Count connected components
    ogdf::NodeArray<int> component(g);
    int n_components = ogdf::connectedComponents(g, component);

    // Compute MST if requested
    if (create_mst && g.numberOfEdges() > 0) {
        ogdf::EdgeArray<float> weights = g.edgeWeights();
        ogdf::makeMinimumSpanningTree(g, weights);
    }

    // Create GraphAttributes
    ogdf::GraphAttributes ga(g);
    ga.setAllHeight(config.node_size);
    ga.setAllWidth(config.node_size);

    // RNG for the adaptive per-level seeding/frame-rotation and the untangle
    // post-pass. Seeded from config.seed when given (0 in deterministic mode),
    // otherwise from the system entropy source.
    const unsigned rng_seed = config.seed.has_value()
        ? config.seed.value()
        : (config.deterministic ? 0u : std::random_device{}());
    std::mt19937 rng(rng_seed);
    const unsigned threads = config.deterministic
        ? 1u
        : std::max(1u, std::thread::hardware_concurrency());

    if (config.adaptive) {
        // The per-level adaptive layout (tuned default): per-level node sizing via
        // ns_cap/ns_coef/ns_exp, disc-seeded coarsest level, per-level frame
        // rotation. Composes with the component splitter just like the mixer did.
        if (n_components > 1) {
            auto* inner = new PerLevelAdaptiveLayout(config, threads, rng);
            auto* csl = new ogdf::ComponentSplitterLayout();
            csl->setPacker(new ogdf::TileToRowsCCPacker());
            csl->setLayoutModule(inner);

            ogdf::PreprocessorLayout ppl;
            ppl.setLayoutModule(csl);
            ppl.setRandomizePositions(false);
            ppl.call(ga);
        } else {
            PerLevelAdaptiveLayout adaptive(config, threads, rng);
            adaptive.call(ga);
        }
    } else {
        // Legacy single-shot ModularMultilevelMixer path (backward compatible:
        // adaptive=false + untangle=false reproduces the original output).
        ogdf::MultilevelGraph mlg(ga);

        auto* fme = new ogdf::FastMultipoleEmbedder();
        fme->setNumIterations(config.fme_iterations);
        fme->setMultipolePrec(config.fme_precision);
        fme->setDefaultEdgeLength(1);
        fme->setDefaultNodeSize(1);
        fme->setRandomize(false);
        if (config.deterministic) {
            fme->setNumberOfThreads(1);
        }

        auto* sl = new ogdf::ScalingLayout();
        sl->setLayoutRepeats(config.sl_repeats);
        sl->setSecondaryLayout(fme);
        sl->setExtraScalingSteps(config.sl_extra_scaling_steps);
        sl->setScalingType(to_ogdf_scaling(config.sl_scaling_type));
        sl->setScaling(config.sl_scaling_min, config.sl_scaling_max);

        ogdf::InitialPlacer* placer = create_placer(config.placer);
        ogdf::MultilevelBuilder* merger = create_merger(
            config.merger, config.merger_factor, config.merger_adjustment);

        auto* mmm = new ogdf::ModularMultilevelMixer();
        mmm->setLayoutRepeats(config.mmm_repeats);
        mmm->setLevelLayoutModule(sl);
        mmm->setInitialPlacer(placer);
        mmm->setMultilevelBuilder(merger);

        if (n_components > 1) {
            auto* csl = new ogdf::ComponentSplitterLayout();
            auto* packer = new ogdf::TileToRowsCCPacker();
            csl->setPacker(packer);
            csl->setLayoutModule(mmm);

            ogdf::PreprocessorLayout ppl;
            ppl.setLayoutModule(csl);
            ppl.setRandomizePositions(false);
            ppl.call(mlg);
        } else {
            mmm->call(mlg);
        }
        mlg.exportAttributes(ga);
    }

    // Crossing-reduction post-pass (preserves every tree-edge length; see
    // untangle.hpp). Operates on the laid-out coordinates before normalization.
    if (config.untangle) {
        UntangleParams up;
        up.rotate = (config.untangle_mode == UntangleMode::Rotate);
        up.max_sub = config.untangle_max_sub;
        up.passes = config.untangle_passes;
        up.rot_steps = config.untangle_rot_steps;
        up.max_angle = config.untangle_max_angle * 3.14159265358979323846 / 180.0;
        up.slide_eps = config.untangle_slide_eps;
        up.slide_steps = config.untangle_slide_steps;
        untangle_post_pass(g, ga, up, rng);
    }

    result.x.resize(vertex_count);
    result.y.resize(vertex_count);

    int i = 0;
    for (ogdf::node v : g.nodes) {
        result.x[i] = static_cast<float>(ga.x(v));
        result.y[i] = static_cast<float>(ga.y(v));
        i++;
    }

    // Normalize to [-0.5, 0.5]
    if (!result.x.empty()) {
        float min_x = *std::min_element(result.x.begin(), result.x.end());
        float max_x = *std::max_element(result.x.begin(), result.x.end());
        float min_y = *std::min_element(result.y.begin(), result.y.end());
        float max_y = *std::max_element(result.y.begin(), result.y.end());

        float diff_x = max_x - min_x;
        float diff_y = max_y - min_y;

        // Avoid division by zero
        if (diff_x < 1e-10f) diff_x = 1.0f;
        if (diff_y < 1e-10f) diff_y = 1.0f;

        for (size_t j = 0; j < result.x.size(); j++) {
            result.x[j] = (result.x[j] - min_x) / diff_x - 0.5f;
            result.y[j] = (result.y[j] - min_y) / diff_y - 0.5f;
        }
    }

    // Extract edges
    for (ogdf::edge e : g.edges) {
        result.s.push_back(static_cast<uint32_t>(e->source()->index()));
        result.t.push_back(static_cast<uint32_t>(e->target()->index()));
    }

    return result;
}

}
