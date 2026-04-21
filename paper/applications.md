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
