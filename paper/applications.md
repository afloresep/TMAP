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

## A mixed-proteome atlas spans bacterial kingdoms and eukaryotic compartments

We constructed a ~7,000-protein atlas spanning four taxonomically and functionally distinct classes — the *Rickettsia prowazekii* and *Pelagibacter ubique* proteomes as α-proteobacterial references (UniProt UP000002480, UP000000744); the *Synechocystis* sp. PCC 6803 proteome as a cyanobacterial reference (UP000001425); the human mitochondrial proteome (MitoCarta 3.0, $$); and a eukaryotic cytosolic control selected from the *S. cerevisiae* reference proteome (UP000002311) — embedded each protein with ESM-2 650M ($$) (mean-pooled over sequence length), and mapped the result with `TMAP(metric="cosine", n_neighbors=20).fit_transform(X)`.

The fit takes a few seconds on a laptop, produces a tree in which each of the four classes occupies a coherent subtree, and keeps domain-purity high: more than 90% of tree edges connect two nodes from the same class, and each of the four classes contributes at least one dense, pure ($>95\%$) subtree of $\geq 100$ nodes ([Figure 1]). The interactive HTML export lets a biologist explore the tree by any of the metadata columns — organism, source proteome, MitoCarta compartment — and follow the unique shortest path between any two proteins.

What the tree does *not* show is equally useful as a demonstration of what ESM-2 sequence embeddings carry. The mitochondrial proteome forms a coherent subtree of its own, but does not sit inside the α-proteobacterial branch: the median tree-path from a human mitochondrial protein to its nearest α-proteobacterial neighbor is longer than to its nearest yeast cytosolic neighbor. After two billion years of independent adaptation, the residual α-proteobacterial signal in mitochondrially-targeted eukaryotic proteins is outweighed, at the resolution of a whole-proteome ESM-2 embedding, by modern eukaryote-vs-bacterium sequence statistics. The tree is faithfully showing what the embedding captures; that the embedding does not recapitulate 2-Gy-old endosymbiotic homology at this scale is a property of ESM-2, not of TMAP2.

[Figure 1: Mixed-proteome TMAP colored by taxonomic-compartment class (α-proteobacteria / cyanobacteria / Eukarya-mito / Eukarya-cytosolic). Four coherent subtrees, $>90\%$ same-class edges.]
