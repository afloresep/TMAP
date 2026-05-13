# Applications

## Navigating root developmental trajectories at single-cell resolution

Shahan et al. ($$, 2022) published a 110,427-cell scRNA-seq atlas of the *Arabidopsis* root and used it to reconstruct developmental trajectories in wild-type and cell-identity mutants. Their pipeline — Seurat integration, PCA, UMAP, and a consensus pseudotime averaged from CytoTRACE and scVelo — is the canonical workflow a biologist reviewer expects. The authors explicitly declined to use graph-based pseudotime methods on the grounds that such methods *"depend on the selection of dimensional reduction embeddings and parameters"* ($$). TMAP2 removes that dependency: the graph *is* the embedding.

### What we reproduce

We reproduce the ground-tissue sub-atlas — the wild-type integration of 22,600 cortex, endodermis, and quiescent-center (QC) cells that the authors deposited as `GSE152766_Ground_Tissue_Atlas.rds.gz` — with a drop-in replacement for RunUMAP:

```python
from tmap import TMAP
model = TMAP(metric="cosine", n_neighbors=30, seed=42).fit(X_pca)  # 50 integrated PCs
coords = model.embedding_              # (22600, 2) layout
tree   = model.tree_                   # MST over the 30-NN graph
```

`n_neighbors=30` matches the Shahan et al. `RunUMAP` call. The fit runs in under 90 s on a laptop from the 22,600 × 50 integrated-PCA matrix they deposited.

### Figure 1 — UMAP vs TMAP, same cells, same colors

The TMAP tree recovers the same gross topology as the published UMAP — two long branches for cortex and endodermis, meeting at the QC — but represents the branching explicitly as tree structure rather than implicitly through point density (`paper/images/arabidopsis_atlas_umap_vs_tmap.png`).

### Figure 2 — the figure UMAP cannot produce

Between any two cells, the tree defines a unique shortest path; walking the path returns an ordered sequence of intermediate cells. From the QC-centroid cell to the highest-tree-pseudotime endodermal cell, the 50-hop path sweeps through a clean terminal-program signature:

* **CASP1** (Casparian Strip Membrane Domain Protein 1) fires sharply at hops 43–47 — the mature-endodermis signature. Peak expression reaches ~5.2 (log-normalized), from a baseline of 0.
* **MYB36** (the master endodermal transcription factor) rises alongside CASP1 at hops 44–48, consistent with its role in activating the Casparian-strip program.
* **SCR** (SCARECROW) shows its documented dual expression: transient pulses mid-path and a terminal peak at hops 48–49, matching its roles in QC maintenance and late endodermal fate.

The per-edge |Δ consensus_time| distribution is concentrated near zero (median ~0.04), showing that the MST connects cells that are also temporally adjacent in the authors' pipeline — the tree tracks the underlying biology, not only the embedding geometry.

See `paper/images/arabidopsis_path_killshot.png`.

### Quantitative agreement with the published pipeline

Spearman correlation between the TMAP tree pseudotime (geodesic distance from the QC root) and the Shahan consensus pseudotime (CytoTRACE + scVelo average) is **ρ = 0.810** across all 22,600 cells. For context, ρ = 0.810 against a two-method consensus is close to the upper bound achievable without recomputing the consensus itself.

### Full reproduction (one R call, one shell line)

All reproduction scripts live in `examples/`:

```bash
# 1. Fetch + decompose the Seurat RDS into CSVs (once, ~30 min download).
Rscript examples/data/shahan_root/prepare.R

# 2. Build the h5ad used by the example.
python examples/data/shahan_root/build_h5ad.py

# 3. Fit TMAP, write both PNG figures and the interactive HTML, then run
#    the validation gate (ρ >= 0.80, terminal-program marker trend >= 0.40).
python examples/arabidopsis_root_ground_tissue_tmap.py \
    --validate --root-label "Quiescent Center"
```

The R step writes four plain CSVs (PCA embedding, UMAP embedding, cell metadata, marker-gene log-expression), which `build_h5ad.py` assembles into `examples/data/shahan_root/ground_tissue.h5ad`. We deliberately bypass SeuratDisk — it is currently incompatible with SeuratObject v5 — so reproduction depends only on stock Seurat + `R.utils`.

### Interactive exploration

The example also writes a self-contained HTML bundle at `paper/images/arabidopsis_tmap.html` using `model.to_tmapviz().write_html()`. Open it in a browser to toggle between three color layers: categorical cell type, TMAP tree pseudotime (QC-rooted geodesic distance), and the authors' consensus pseudotime. This is the same interface end users get for their own data with a four-line script:

```python
viz = model.to_tmapviz()
viz.add_color_layout("cell_type",       labels,          categorical=True)
viz.add_color_layout("tmap_pseudotime", model.distances_from(root))
viz.write_html("atlas.html")
```

### Caveats and extensions

The ground-tissue sub-atlas that Shahan et al. deposited contains **wild-type cells only** — the *scarecrow-4* and other mutant cells used elsewhere in the paper are deposited under separate accessions. Projecting mutants onto the fitted WT tree with `model.transform()` is a one-line extension but is not part of this reproduction; we report it as a forward-looking capability rather than a result.

## Recovering the seasonal influenza H3N2 antigenic timeline

The Shahan et al. reproduction shows that tree-hops in TMAP track a single biological coordinate (pseudotime). The same property holds for any dataset whose latent ordering is genuinely tree-like. Seasonal influenza H3N2 is the textbook case: hemagglutinin (HA) drifts roughly linearly in time as the virus escapes immune pressure each year, and that drift is the standard validation target in phylodynamics. We use the 12-year HA tree from the Nextstrain Open seasonal-flu build as a public, curated checkpoint against which to grade the TMAP topology.

### What we reproduce

We download the Auspice JSON for `nextstrain.org/flu/seasonal/h3n2/ha/12y` (1,598 tips spanning 2013–2025), reconstruct each tip's HA1 amino-acid sequence by walking the mutations from root to tip, and fit TMAP on amino-acid 5-mer Jaccard distance:

```python
from tmap import TMAP
model = TMAP(metric="jaccard", n_neighbors=60, n_permutations=512, kc=80, seed=42).fit(kmers)
```

The full pipeline lives in `examples/flu_h3n2_ha_tmap.py` and runs end-to-end (download + fit + figure + report) in roughly two minutes on a laptop. No FASTA, no alignment — we reconstruct from the published mutation annotations and never need the raw sequences on disk.

### Figure 1 — TMAP tree colored by collection date

The TMAP layout shows a clean temporal gradient flowing along the tree backbone, with the clade splits (3C.2a, 3C.3a, J.2, …) emerging as side-branches at the years where the literature places them (`paper/images/flu_h3n2_ha_tmap.png`).

### Tree-hops as a date proxy

We pick the oldest tip in the atlas as the root, walk the tree by BFS, and correlate the per-tip hop count with the tip's collection date. The result: **Spearman ρ = 0.890** across all 1,598 tips. Tree topology alone — no dating algorithm, no molecular clock — recovers the temporal order of HA evolution at this Spearman level.

### Quantitative agreement with the Nextstrain ML phylogeny

A second pass (`examples/flu_h3n2_ha_phylogeny_compare.py`) takes the same strains and compares per-pair hop distances in the TMAP tree to per-pair hop distances in the published Nextstrain maximum-likelihood phylogeny. On 199,859 sampled strain pairs, **Spearman ρ = 0.909**. The TMAP tree and the ML phylogeny agree on pairwise distances at the same level the TMAP tree internally agrees with collection date.

### Full reproduction (one shell line)

```bash
# Downloads the Auspice JSON + sidecar root-sequence, fits TMAP,
# writes the figure and the report. Roughly 90 s including download.
python examples/flu_h3n2_ha_tmap.py

# Pass B — pairwise-distance agreement against the Nextstrain ML tree.
python examples/flu_h3n2_ha_phylogeny_compare.py
```

## Indo-European language family recovery

The seasonal-flu case is biology that already had a published reference tree. To show that the tree-hops-as-ordering property is *not* a biology-specific coincidence, we run TMAP on a non-biological sequence problem: the Indo-European Cognate-coded Lexical (IE-CoR) dataset of Heggarty et al. (2023). The dataset encodes cognate-class membership across 200 Swadesh-list concepts for 160 Indo-European languages — the same input that the IE-CoR Bayesian phylogenetic analysis consumed.

### What we reproduce

Each language becomes a Jaccard-shingled set of `"<concept_id>::<cognate_class>"` tokens — one token per (concept, cognate-class) it participates in. Two languages with shared inherited vocabulary share tokens; two unrelated languages share none. We fit TMAP with `n_neighbors=15` since there are only 160 points:

```python
from tmap import TMAP
model = TMAP(metric="jaccard", n_neighbors=15, n_permutations=512, kc=50, seed=42).fit(cognate_sets)
```

The script lives in `examples/ie_languages_tmap.py` and runs in under 30 s including the IE-CoR download.

### Figure — TMAP tree colored by top-level family

The IE-CoR `Clade` field gives hierarchical labels like `"Germanic;North-West;West"`. We flatten to the top-level family (`Germanic`, `Italic`, `Balto-Slavic`, `Celtic`, `Indo-Iranic`, `Hellenic`, `Albanian`, `Anatolian`, `Armenian`, `Tocharian`) for the color legend. The figure (`paper/images/ie_languages_tmap.png`) shows each family forming a clear cluster, with the major IE splits — Anatolian and Tocharian peeling off early, Indo-Iranic vs European, Italic vs Celtic — recovered in the tree backbone.

### Edge purity as a family-recovery test

For every edge in the TMAP tree we check whether the two endpoint languages belong to the same top-level family. The result: **94.3% of edges connect within-family**, versus a **19.4% chance baseline** (the probability that two languages drawn uniformly at random share a top-level family given the family-size distribution). That is a **4.9× lift over chance** — the tree backbone overwhelmingly tracks linguistic descent rather than incidental similarity.

### What this demonstrates

The IE-CoR result has two purposes in the paper. First, it shows that TMAP recovers a known hierarchy without any phylogenetic-inference machinery — no Markov-chain Monte Carlo, no substitution model, no calibrated clock. Second, it broadens the applicability claim beyond bioinformatics: any dataset whose underlying structure is genuinely tree-like (descent with modification, hierarchical clustering, latent ordering) is in scope for TMAP, regardless of whether the modification process is biological. We use the same Jaccard + MinHash + LSHForest + tree-layout pipeline that drove the SARS-CoV-2 spike, scRNA-seq, and seasonal-flu cases.

A posterior phylogenetic tree comparison (analogous to Pass B for the flu case) would be the natural next step. The IE-CoR v1.2 release we fetch does not ship its posterior tree alongside the cognate tables — adding that comparison would require pulling the tree from a separate Supplementary Materials archive and is deferred as a forward-looking extension.

### Full reproduction (one shell line)

```bash
# Downloads IE-CoR CLDF (~5 MB), parses cognate sets, fits TMAP,
# writes the figure and the report. Roughly 20 s on a laptop.
python examples/ie_languages_tmap.py
```
