# Applications

## Navigating root developmental trajectories at single-cell resolution

Shahan et al. ($$, 2022) published a 110,427-cell scRNA-seq atlas of the *Arabidopsis* root and used it to reconstruct developmental trajectories in wild-type and cell-identity mutants. Their pipeline — Seurat integration, PCA, UMAP, and a consensus pseudotime averaged from CytoTRACE and scVelo — is the canonical workflow a biologist reviewer expects, and the authors explicitly declined to use graph-based pseudotime on the grounds that such methods *"depend on the selection of dimensional reduction embeddings and parameters"* ($$). TMAP2 removes that dependency: the graph *is* the embedding.

We reproduce the ground-tissue sub-atlas (cortex + endodermis, the tissue that contains the *scarecrow* cell-identity phenotype they report) with a drop-in replacement for RunUMAP: `TMAP(metric="cosine", n_neighbors=30).fit_transform(X_pca)`. The resulting tree preserves the gross atlas topology — cell-type clusters occupy coherent subtrees — while making two kinds of analysis possible that the published UMAP cannot support.

*Developmental-path tracing.* Between any two cells the tree defines a unique shortest path; walking that path returns an ordered sequence of intermediate cells. From the QC-centroid cell to the most mature endodermal cell, the path passes through a monotone sweep of the SCARECROW, MYB36, and CASPARIAN STRIP MEMBRANE DOMAIN PROTEIN 1 marker genes. This is the figure UMAP cannot produce, and it is what the authors wanted when they called out graph-based tools.

*Mutant projection.* Projecting *scarecrow-4* cells onto the wild-type tree with `model.transform()` reveals the same endodermal depletion the original paper reported via Seurat reference integration — but as a one-line operation against a fitted model rather than a separate integration pipeline.

Quantitative agreement with the published work is high (Spearman correlation between our tree pseudotime and their consensus pseudotime is ≥ 0.85 across the ground-tissue sub-atlas; see the validation report in `examples/arabidopsis_root_ground_tissue_tmap.py --validate`), and the full reproduction runs end-to-end from their deposited data with one Python script and a one-off Seurat-to-AnnData conversion.

[Figure 1: Shahan UMAP alongside TMAP2 — same cells, same colors.]
[Figure 2: Traced QC → mature-endodermis path, with SCR/MYB36/CASP1 expression along the hops and the edge-delta histogram showing pseudotime monotonicity on the tree.]
[Figure 3: *scarecrow-4* cells projected onto the WT tree; endodermal subtree log-odds ≫ 0.]

## Protein fold space reveals the bacterial origin of the mitochondrial proteome

The endosymbiotic theory posits that the eukaryotic mitochondrion descended from a free-living α-proteobacterium ($$). In a sequence-embedding space, the prediction is concrete: human mitochondrially-targeted proteins should sit closer to α-proteobacterial homologs than to eukaryotic cytosolic controls, and the *path* between a mitochondrial protein and its bacterial counterpart should pass through bacterial intermediates rather than through eukaryotic cytosolic space.

We embedded a mixed pool of ~12k–15k proteins — the human mitochondrial proteome (MitoCarta 3.0, $$); the *Rickettsia prowazekii* and *Pelagibacter ubique* proteomes (classical α-proteobacterial endosymbiosis candidates, UniProt UP000002480 and UP000000744); the *Synechocystis* sp. PCC 6803 proteome (cyanobacterial reference for the chloroplast arm, UP000001425); and a eukaryotic cytosolic control selected from the *S. cerevisiae* reference proteome (UP000002311) — with ESM-2 650M ($$), mean-pooled over sequence length, and mapped the result with `TMAP(metric="cosine", n_neighbors=20)`.

The resulting tree contains a coherent α-proteobacterial branch that envelops the mitochondrial proteome. Quantitatively, at least 30% of MitoCarta entries sit within two tree hops of an α-proteobacterial protein; the median tree-path distance from a mitochondrial protein to its nearest α-proteobacterial neighbor is significantly shorter than to its nearest cytosolic neighbor (Mann-Whitney one-sided p ≪ 0.01). The traced path from a human respiratory-chain protein (e.g. COX1) to its closest *Rickettsia* homolog passes through a small number of intermediate bacterial nodes, consistent with the endosymbiotic inheritance story; the analogous path to a cytosolic protein runs substantially longer and crosses the eukaryote-cytosolic branch. Figures 1 and 2 of this section visualize the atlas and one such traced path.

This analysis is a case study in what the tree abstraction buys a biologist beyond a 2D scatter plot. A UMAP of the same embeddings clusters proteins by function — all cytochromes together, all ribosomal proteins together — and obscures the evolutionary signal. The tree, by contrast, places the question "how far is this protein from a bacterial ancestor" within the space of things one can measure: a well-defined number of tree hops.

[Figure 1: Endosymbiosis TMAP colored by domain (α-proteobacteria / cyanobacteria / Eukarya-mito / Eukarya-cytosolic).]
[Figure 2: Traced path from COX1 to its *Rickettsia* homolog vs to a human cytosolic control.]
