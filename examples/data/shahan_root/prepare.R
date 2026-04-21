#!/usr/bin/env Rscript
# Download GSE152766 ground-tissue sub-atlas and convert Seurat RDS → h5ad.
#
# Output: examples/data/shahan_root/ground_tissue.h5ad
#
# Preserves: X_pca (50 dims), X_umap, cell_type, celltype.anno, consensus_time,
#            sample, orig.ident, and any *_mutant columns.
#
# Requirements: R >= 4.1, Seurat >= 4.0, SeuratDisk
#   install.packages("Seurat")
#   remotes::install_github("mojaveazure/seurat-disk")

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratDisk)
})

HERE <- dirname(normalizePath(sys.frame(1)$ofile))
setwd(HERE)

URL  <- "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE152766&format=file&file=GSE152766%5FGround%5FTissue%5FAtlas%2Erds%2Egz"
RDS  <- "Ground_Tissue_Atlas.rds.gz"
UNGZ <- "Ground_Tissue_Atlas.rds"
H5A  <- "ground_tissue.h5ad"
H5S  <- "ground_tissue.h5seurat"

if (!file.exists(RDS) && !file.exists(UNGZ)) {
  message("Downloading ", URL, "  (~10 GB, be patient)")
  options(timeout = 3600)
  download.file(URL, RDS, mode = "wb")
}
if (!file.exists(UNGZ)) {
  message("Decompressing ", RDS)
  R.utils::gunzip(RDS, destname = UNGZ, remove = FALSE)
}

message("Reading Seurat object …")
obj <- readRDS(UNGZ)

# Sanity-print what's inside.
message("Cells: ",       ncol(obj))
message("Assays: ",      paste(Assays(obj), collapse = ", "))
message("Reductions: ",  paste(Reductions(obj), collapse = ", "))
message("Obs columns: ", paste(colnames(obj@meta.data), collapse = ", "))

# Drop everything except the active assay's data slot to keep h5ad small.
DefaultAssay(obj) <- if ("integrated" %in% Assays(obj)) "integrated" else "SCT"
obj <- DietSeurat(obj,
  counts     = FALSE,
  data       = TRUE,
  scale.data = FALSE,
  dimreducs  = c("pca", "umap"))

message("Writing h5Seurat …")
SaveH5Seurat(obj, filename = H5S, overwrite = TRUE)
message("Converting to h5ad …")
Convert(H5S, dest = "h5ad", overwrite = TRUE)
file.rename(sub("\\.h5seurat$", ".h5ad", H5S), H5A)
unlink(H5S)

message("Done. Output: ", file.path(HERE, H5A))
