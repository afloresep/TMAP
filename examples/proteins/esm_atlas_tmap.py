"""ESM Atlas → TMAP: two views of ESMC's protein space.

Visualize a subset of metagenomic proteins as **two** TMAP trees over the
*same* proteins:

  A. raw **ESMC** embeddings (mean-pooled last hidden state), and
  B. **16,384-d SAE feature** profiles (sparse-autoencoder features, the basis
     the ESM Atlas uses to organize protein space).

Each point's pinned card shows an **ESMFold2-Fast** predicted 3D structure plus
metadata: structural confidence (pLDDT / pTM), sequence properties, the
dominant SAE feature (with a link to its public biohub feature page), and 
when sourced from MGnify an "annotated vs. unknown" flag (does the protein
hit any InterPro/Pfam signature), mirroring the atlas's bright/dark coloring.

Everything is recomputed locally from open-weight models: the new ESM Atlas has
no public bulk download, but ESMC, the SAEs, and ESMFold2 are all open weights.

Pipeline (each stage caches to ``examples/data/esm_atlas/``; re-runs are cheap,
folding is resumable):

    fetch  -> proteins.parquet (+ seqs.fasta)
    encode -> esmc_emb.npy  + sae_feat.npy  + sae_top.npy   (one ESMC pass)
    fold   -> esm_atlas_out/structures/<id>.cif  + fold.parquet
    enrich -> meta.parquet
    maps   -> emb.tmap / sae.tmap
    viz    -> esm_atlas_out/{emb,sae}/index.html

Outputs
-------
    examples/esm_atlas_out/emb/index.html    Map A: raw ESMC embeddings
    examples/esm_atlas_out/sae/index.html    Map B: 16,384-d SAE features
    examples/esm_atlas_out/structures/*.cif  ESMFold2-Fast structures

Usage
-----
    # Smoke-test the whole chain end-to-end on a handful of proteins first:
    python examples/proteins/esm_atlas_tmap.py --n 50 --fold-loops 1

    # Full ship-fast demo (~8k proteins) on a >=24 GB GPU:
    python examples/proteins/esm_atlas_tmap.py --n 8000

    # Bring your own sequences (skips fetching):
    python examples/proteins/esm_atlas_tmap.py --fasta my_proteins.fasta

    # Richer metadata via MGnify (annotated-vs-unknown coloring):
    python examples/proteins/esm_atlas_tmap.py --source mgnify --mgya MGYA00585528

    # Serve the result (structures lazy-load over HTTP; needed for the 3D view):
    python examples/proteins/esm_atlas_tmap.py --serve

Requirements
------------
    pip install torch transformers pandas pyarrow
    pip install "esm @ git+https://github.com/Biohub/esm.git@c94ed8d"   # ESMC + ESMFold2
    pip install biopython          # sequence properties (optional but recommended)

Notes
-----
* Defaults to ESMC-600M (1152-d) + the 600M SAE, comfortable on >=24 GB. Pass
  ``--esmc-model Biohub/ESMC-6B --sae-model Biohub/ESMC-6B-sae-k64-codebook16384
  --sae-layer 60`` for the atlas-exact 6B layer-60 features (heavier).
* A few external details are based on current docs and may need a one-line
  tweak when you run: the MGnify download labels, the SAE layer name, and the
  ESMFold2 ``fold(...)`` kwargs. The ``esmatlas`` source and ``--fasta`` path
  are verified to work and need no configuration.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

from tmap import TMAP

# Scripts live in a subfolder; data, caches and outputs stay at examples/.
EXAMPLES_DIR = Path(__file__).resolve().parents[1]


# Config

CACHE_DIR = EXAMPLES_DIR / "data" / "esm_atlas"
OUTPUT_DIR = EXAMPLES_DIR / "esm_atlas_out"
STRUCT_DIR = OUTPUT_DIR / "structures"

# Open-weight model IDs (HuggingFace). Swap to the 6B variants for the
# atlas-exact features (see module docstring).
DEFAULT_ESMC_MODEL = "Biohub/ESMC-600M"
DEFAULT_SAE_MODEL = "Biohub/ESMC-600M-sae-k64-codebook16384"
DEFAULT_ESMFOLD_MODEL = "biohub/ESMFold2-Fast"

# Verified, no-config sequence source: high-quality cluster representatives from
# the (old) ESM Metagenomic Atlas. Amino-acid FASTA, streamable, MGYP ids.
ESMATLAS_FASTA_URL = (
    "https://dl.fbaipublicfiles.com/esmatlas/v0/highquality_clust30/highquality_clust30.fasta"
)
# Public, unauthenticated SAE-feature metadata API (human-readable labels).
FEATURE_API = "https://biohub.ai/esm/protein/api/v1alpha1/features/{idx}"
MGNIFY_API = "https://www.ebi.ac.uk/metagenomics/api/v1"

STAGES = ["fetch", "encode", "fold", "enrich", "maps", "viz"]


def safe_id(raw: str) -> str:
    """Filesystem-safe token for a protein id (SPIRE ids contain '|')."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", raw)[:120]


def _lazy_pandas():
    import pandas as pd  # local import keeps the CLI snappy

    return pd


# 1. Fetch sequences + (optional) metadata


def _iter_fasta_subset(url: str, n: int, min_len: int, max_len: int):
    """Stream a remote FASTA and yield the first ``n`` length-filtered records.

    Streaming + early-stop means we never download the whole (multi-GB) file.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "tmap-esm-atlas-demo"})
    kept = 0
    header: str | None = None
    seq_parts: list[str] = []

    def _flush():
        nonlocal header, seq_parts
        rec = None
        if header is not None:
            seq = "".join(seq_parts).strip().upper()
            if min_len <= len(seq) <= max_len:
                rec = (header.split()[0], seq)
        header, seq_parts = None, []
        return rec

    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted host)
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith(">"):
                rec = _flush()
                if rec is not None:
                    yield rec
                    kept += 1
                    if kept >= n:
                        return
                header = line[1:]
            elif line:
                seq_parts.append(line)
    rec = _flush()
    if rec is not None and kept < n:
        yield rec


def _fetch_esmatlas(n: int, min_len: int, max_len: int) -> tuple[list[str], list[str], dict]:
    print(f"  Streaming up to {n:,} sequences from the ESM Atlas FASTA ...")
    ids, seqs = [], []
    for pid, seq in _iter_fasta_subset(ESMATLAS_FASTA_URL, n, min_len, max_len):
        ids.append(pid)
        seqs.append(seq)
    print(f"  Kept {len(ids):,} sequences (length {min_len}-{max_len}).")
    return ids, seqs, {}


def _fetch_mgnify(mgya: str, n: int, min_len: int, max_len: int) -> tuple[list[str], list[str], dict]:
    """Best-effort MGnify fetch: 'Predicted CDS' FASTA + 'InterPro matches'.

    Gives the authentic annotated-vs-unknown signal. The exact download labels
    are based on MGnify docs; if this fails, fall back to --fasta / --source
    esmatlas. ``extra`` returns {'annotated': {id: bool}} when InterPro is found.
    """
    print(f"  Querying MGnify analysis {mgya} ...")

    def _get_json(url: str) -> dict:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            return json.load(r)

    dl = _get_json(f"{MGNIFY_API}/analyses/{mgya}/downloads")
    cds_url = ipr_url = None
    for item in dl.get("data", []):
        attr = item.get("attributes", {})
        label = (attr.get("description", {}) or {}).get("label", "") or attr.get("alias", "")
        link = (item.get("links", {}) or {}).get("self")
        if not link:
            continue
        low = label.lower()
        if "predicted cds" in low or low.endswith("_cds.faa.gz"):
            cds_url = link
        elif "interpro" in low:
            ipr_url = link
    if cds_url is None:
        raise RuntimeError("Could not find a 'Predicted CDS' download for this analysis.")

    # Predicted CDS is gzipped FASTA; stream + gunzip the first n records.
    import gzip
    import io

    req = urllib.request.Request(cds_url, headers={"User-Agent": "tmap-esm-atlas-demo"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        text = gzip.GzipFile(fileobj=io.BytesIO(resp.read())).read().decode("utf-8", "replace")
    ids, seqs = [], []
    header, parts = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                s = "".join(parts).upper()
                if min_len <= len(s) <= max_len:
                    ids.append(header.split()[0])
                    seqs.append(s)
                    if len(ids) >= n:
                        break
            header, parts = line[1:], []
        elif line:
            parts.append(line)

    annotated: dict[str, bool] = {}
    if ipr_url is not None:
        try:
            req = urllib.request.Request(ipr_url, headers={"User-Agent": "tmap-esm-atlas-demo"})
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
                blob = resp.read()
            if ipr_url.endswith(".gz"):
                import gzip as _gz

                blob = _gz.decompress(blob)
            hit_ids = {ln.split("\t", 1)[0] for ln in blob.decode("utf-8", "replace").splitlines() if ln}
            annotated = {pid: (pid in hit_ids) for pid in ids}
            print(f"  InterPro: {sum(annotated.values()):,}/{len(ids):,} proteins annotated.")
        except Exception as exc:  # pragma: no cover - network best-effort
            print(f"  (InterPro matches unavailable: {exc})")
    print(f"  Kept {len(ids):,} MGnify proteins.")
    return ids, seqs, {"annotated": annotated}


def fetch_sequences(args) -> "object":
    """Resolve sequences + optional metadata into proteins.parquet."""
    pd = _lazy_pandas()
    cache = CACHE_DIR / "proteins.parquet"
    if cache.exists() and not args.force:
        print(f"  Loading cached proteins from {cache}")
        return pd.read_parquet(cache)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    extra: dict = {}
    if args.fasta:
        from tmap.utils.proteins import read_fasta

        print(f"  Reading sequences from {args.fasta}")
        ids, seqs = read_fasta(args.fasta, max_seqs=args.n)
    elif args.source == "mgnify":
        if not args.mgya:
            raise SystemExit("--source mgnify requires --mgya MGYA######## (an assembly analysis id)")
        ids, seqs, extra = _fetch_mgnify(args.mgya, args.n, args.min_len, args.max_len)
    else:  # esmatlas (default, verified, no-config)
        ids, seqs, extra = _fetch_esmatlas(args.n, args.min_len, args.max_len)

    # length filter (read_fasta path) + de-dup ids
    keep = [(i, s) for i, s in zip(ids, seqs) if args.min_len <= len(s) <= args.max_len]
    seen: set[str] = set()
    rows = []
    for pid, seq in keep:
        if pid in seen:
            continue
        seen.add(pid)
        rows.append(
            {
                "id": pid,
                "sequence": seq,
                "source": args.source if not args.fasta else "fasta",
                "annotated": extra.get("annotated", {}).get(pid),
            }
        )
    df = pd.DataFrame(rows)
    # Persist a FASTA too (handy for re-runs / external tools).
    fasta_path = CACHE_DIR / "seqs.fasta"
    fasta_path.write_text("".join(f">{r.id}\n{r.sequence}\n" for r in df.itertuples()), encoding="utf-8")
    df.to_parquet(cache)
    print(f"  {len(df):,} proteins -> {cache}")
    return df


# 2. Encode: ESMC embeddings + SAE features (one forward pass)


def encode_esmc_and_sae(df, args) -> None:
    """Mean-pooled ESMC embeddings and max-pooled SAE feature profiles."""
    emb_path = CACHE_DIR / "esmc_emb.npy"
    sae_path = CACHE_DIR / "sae_feat.npy"
    top_path = CACHE_DIR / "sae_top.npy"
    if all(p.exists() for p in (emb_path, sae_path, top_path)) and not args.force:
        print("  Cached embeddings + SAE features found.")
        return

    import torch
    from transformers import AutoModel, AutoTokenizer

    seqs = df["sequence"].tolist()
    print(f"  Loading {args.esmc_model} + SAE {args.sae_model} ...")
    tok = AutoTokenizer.from_pretrained(args.esmc_model)
    model = AutoModel.from_pretrained(
        args.esmc_model, device_map="auto", dtype=torch.float16
    ).eval()

    # Attach the SAE for the chosen layer. The per-layer file is named
    # layer_<N>.safetensors; if the layer is wrong the load will tell you which
    # layers exist (catch + print and pass --sae-layer).
    layer = args.sae_layer
    sae = AutoModel.from_pretrained(
        args.sae_model,
        allow_patterns=["config.json", f"layer_{layer}.safetensors"],
        device=model.device,
    )
    sae.initialize_layers([layer])
    if str(layer) not in getattr(sae, "layers", {}):
        raise SystemExit(
            f"SAE has no layer {layer}. Available: {sorted(getattr(sae, 'layers', {}))}. "
            f"Re-run with --sae-layer set to one of these."
        )
    model.add_sae_models([sae.layers[str(layer)]])
    sae_key = f"layer{layer}"

    n = len(seqs)
    embeddings: np.ndarray | None = None  # allocated after first pass (dim from output)
    sae_feats = np.zeros((n, 16384), dtype=np.float16)

    # One sequence at a time: the SAE returns a per-residue (L, 16384) sparse
    # tensor with no batch axis, so batching would mix residues across proteins.
    t0 = time.time()
    for i, seq in enumerate(seqs):
        inputs = tok([seq], return_tensors="pt", truncation=True, max_length=args.max_len + 2)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True)

        # Mean-pool the last hidden state over residues (dropping BOS/EOS).
        hs = out["last_hidden_state"][0].float()  # (L, D)
        core = hs[1:-1] if hs.shape[0] > 2 else hs
        emb = core.mean(0).cpu().numpy()
        if embeddings is None:
            embeddings = np.zeros((n, emb.shape[0]), dtype=np.float32)
        embeddings[i] = emb

        # Max-pool SAE activations over residues -> per-protein feature profile.
        sae_out = out["sae_outputs"][sae_key]  # sparse (L, 16384)
        dense = sae_out.to_dense().float() if hasattr(sae_out, "to_dense") else sae_out.float()
        dcore = dense[1:-1] if dense.shape[0] > 2 else dense
        sae_feats[i] = dcore.amax(0).cpu().numpy().astype(np.float16)

        if (i + 1) % 50 == 0 or i + 1 == n:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"  encoded {i + 1:,}/{n:,}  ({rate:.1f} prot/s)", end="\r")
    print()

    top_feature = sae_feats.astype(np.float32).argmax(axis=1).astype(np.int32)
    np.save(emb_path, embeddings)
    np.save(sae_path, sae_feats)
    np.save(top_path, top_feature)
    print(f"  Saved embeddings {embeddings.shape} and SAE features {sae_feats.shape}.")


# 3. Fold structures (ESMFold2-Fast) is  resumable


def fold_structures(df, args) -> None:
    """Predict 3D structures with ESMFold2-Fast; write mmCIF + pLDDT/pTM."""
    pd = _lazy_pandas()
    fold_path = CACHE_DIR / "fold.parquet"
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict] = {}
    if fold_path.exists() and not args.force:
        done = {r.id: {"plddt": r.plddt, "ptm": r.ptm} for r in pd.read_parquet(fold_path).itertuples()}

    todo = [
        (r.id, r.sequence)
        for r in df.itertuples()
        if r.id not in done or not (STRUCT_DIR / f"{safe_id(r.id)}.cif").exists()
    ]
    if not todo:
        print(f"  All {len(df):,} structures already folded.")
        return
    print(f"  Folding {len(todo):,} structures ({len(done):,} cached) with {args.esmfold_model} ...")

    import torch  # noqa: F401  (ensures CUDA is initialised / clear error if absent)
    from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    model = ESMFold2Model.from_pretrained(args.esmfold_model).cuda().eval()
    builder = ESMFold2InputBuilder()

    t0 = time.time()
    for i, (pid, seq) in enumerate(todo, 1):
        try:
            spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)])
            result = builder.fold(
                model, spi,
                num_loops=args.fold_loops,
                num_sampling_steps=args.fold_steps,
                num_diffusion_samples=1,
                seed=0,
            )
            cif = result.complex.to_mmcif() if hasattr(result, "complex") else result.to_mmcif()
            (STRUCT_DIR / f"{safe_id(pid)}.cif").write_text(cif, encoding="utf-8")
            done[pid] = {"plddt": float(result.plddt.float().mean()), "ptm": float(result.ptm)}
        except Exception as exc:  # keep going; one bad sequence shouldn't kill the run
            print(f"\n  ! fold failed for {pid}: {exc}")
            done[pid] = {"plddt": float("nan"), "ptm": float("nan")}

        if i % 25 == 0 or i == len(todo):
            # checkpoint so the run is resumable
            pd.DataFrame(
                [{"id": k, "plddt": v["plddt"], "ptm": v["ptm"]} for k, v in done.items()]
            ).to_parquet(fold_path)
            rate = i / max(time.time() - t0, 1e-6)
            print(f"  folded {i:,}/{len(todo):,}  ({rate:.2f} prot/s)", end="\r")
    print(f"\n  Structures in {STRUCT_DIR}")


# 4. Enrich: properties, confidence, SAE feature labels


def _feature_labels(feature_ids: list[int]) -> dict[int, dict]:
    """Fetch human-readable SAE feature labels from the public biohub API."""
    catalog_path = CACHE_DIR / "feature_catalog.json"
    catalog: dict[str, dict] = {}
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text())
    for idx in sorted(set(int(i) for i in feature_ids)):
        key = str(idx)
        if key in catalog and "summary" in catalog[key]:
            continue
        try:
            req = urllib.request.Request(
                FEATURE_API.format(idx=idx), headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310
                data = json.load(r)
            catalog[key] = {
                "label": data.get("label") or f"feature {idx}",
                "category": data.get("category") or "uncategorized",
                "summary": data.get("summary") or "",
            }
        except Exception:
            catalog[key] = {"label": f"feature {idx}", "category": "unknown", "summary": ""}
    catalog_path.write_text(json.dumps(catalog))
    return {int(k): v for k, v in catalog.items()}


def enrich(df, args) -> "object":
    """Join structural confidence, sequence properties, and SAE labels."""
    pd = _lazy_pandas()
    meta_path = CACHE_DIR / "meta.parquet"
    if meta_path.exists() and not args.force:
        print(f"  Loading cached metadata from {meta_path}")
        return pd.read_parquet(meta_path)

    meta = df.copy()

    # Structural confidence from folding.
    fold_path = CACHE_DIR / "fold.parquet"
    if fold_path.exists():
        meta = meta.merge(pd.read_parquet(fold_path), on="id", how="left")
    else:
        meta["plddt"] = np.nan
        meta["ptm"] = np.nan

    # Physicochemical sequence properties (BioPython-backed where available).
    from tmap.utils.proteins import sequence_properties

    props = sequence_properties(meta["sequence"].tolist(), properties=["length", "molecular_weight", "gravy"])
    for name, arr in props.items():
        meta[name] = arr

    # Dominant SAE feature -> label, category, and a per-point biohub link.
    top = np.load(CACHE_DIR / "sae_top.npy")
    meta["feature_idx"] = top[: len(meta)]
    labels = _feature_labels(meta["feature_idx"].tolist())
    meta["feature_label"] = [labels[int(i)]["label"] for i in meta["feature_idx"]]
    meta["feature_category"] = [labels[int(i)]["category"] for i in meta["feature_idx"]]
    meta["feature_summary"] = [
        (labels[int(i)].get("summary") or labels[int(i)]["label"]).replace("‑", "-")
        for i in meta["feature_idx"]
    ]

    # "Annotated vs. unknown": real InterPro hit (MGnify) else SAE-strength proxy.
    if meta["annotated"].notna().any():
        meta["characterization"] = np.where(meta["annotated"].fillna(False), "annotated", "unknown")
    else:
        sae = np.load(CACHE_DIR / "sae_feat.npy").astype(np.float32)
        strength = sae[np.arange(len(meta)), meta["feature_idx"].to_numpy()]
        thresh = np.nanmedian(strength)
        meta["characterization"] = np.where(strength >= thresh, "characterized", "novel")
        meta["feature_strength"] = strength

    meta.to_parquet(meta_path)
    print(f"  Metadata for {len(meta):,} proteins -> {meta_path}")
    return meta


# 5. Build the two TMAPs

def build_maps(meta, args) -> tuple[TMAP, TMAP]:
    """Fit one TMAP from ESMC embeddings and one from SAE features."""
    n = len(meta)
    emb = np.load(CACHE_DIR / "esmc_emb.npy")[:n].astype(np.float32)
    sae = np.load(CACHE_DIR / "sae_feat.npy")[:n].astype(np.float32)

    def _fit(X: np.ndarray, name: str) -> TMAP:
        print(f"  Fitting TMAP ({name}, metric='cosine', k={args.k}, n={n:,}, dim={X.shape[1]}) ...")
        t0 = time.time()
        model = TMAP(metric="cosine", n_neighbors=args.k, layout_iterations=1000, seed=42).fit(X)
        print(f"    done in {time.time() - t0:.1f}s")
        return model

    return _fit(emb, "ESMC embeddings"), _fit(sae, "SAE features")


# 6. Visualize


def _make_viz(model: TMAP, meta, title: str, args):
    viz = model.to_tmapviz()
    viz.title = title

    viz.add_label("Dominant feature", meta["feature_label"].tolist())
    viz.add_label("Accession", meta["id"].tolist())

    viz.add_color_layout("pLDDT", meta["plddt"].tolist(), color="viridis")
    viz.add_color_layout("pTM", meta["ptm"].tolist(), color="cividis")
    viz.add_color_layout("Length", meta["length"].tolist(), color="magma")
    viz.add_color_layout("Characterization", meta["characterization"].tolist(), categorical=True, color="tab10")
    viz.add_color_layout("SAE feature category", meta["feature_category"].tolist(), categorical=True, color="tab20")

    # Extra card / tooltip labels 
    viz.add_label("Function", meta["feature_summary"].tolist())
    viz.add_label("Feature ID", [str(i) for i in meta["feature_idx"].tolist()])
    if "source" in meta:
        viz.add_label("Source", meta["source"].tolist())

    # 3D structures: shared dir, referenced relative to each map's index.html
    # (served from the output root). Empty string for any protein that failed.
    urls = []
    for r in meta.itertuples():
        cif = STRUCT_DIR / f"{safe_id(r.id)}.cif"
        urls.append(f"../structures/{safe_id(r.id)}.cif" if cif.exists() else "")
    viz.add_3d_structures(urls, source="url", fmt="cif")

    viz.configure_card(
        title_column="Accession",
        subtitle_column="Dominant feature",
        fields=["Function", "SAE feature category", "pLDDT", "pTM", "Length"],
    )
    return viz


def visualize(emb_model: TMAP, sae_model: TMAP, meta, args) -> Path:
    """Write the two interactive TMAP maps (``emb/`` and ``sae/``) under OUTPUT_DIR.

    Each is a self-contained page from ``write_static`` (open it directly, or
    serve the output dir over HTTP so the 3D structure cards can fetch their
    ``.cif`` files). No landing page is generated.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(meta)
    print("  Writing map A (ESMC embeddings) ...")
    _make_viz(emb_model, meta, f"ESM Atlas — ESMC embeddings ({n:,})", args).write_static(OUTPUT_DIR / "emb")
    print("  Writing map B (SAE features) ...")
    _make_viz(sae_model, meta, f"ESM Atlas — SAE features ({n:,})", args).write_static(OUTPUT_DIR / "sae")
    print(f"  Done. Two interactive maps written under {OUTPUT_DIR}:")
    print("    emb/index.html  (raw ESMC embeddings)")
    print("    sae/index.html  (16,384-d SAE features)")
    return OUTPUT_DIR


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ESM Atlas → TMAP demo (ESMC vs SAE).")
    p.add_argument("--n", type=int, default=8000, help="Number of proteins.")
    p.add_argument("--source", choices=["esmatlas", "mgnify"], default="esmatlas")
    p.add_argument("--fasta", type=Path, default=None, help="Use a local FASTA instead of fetching.")
    p.add_argument("--mgya", type=str, default=None, help="MGnify assembly analysis id (for --source mgnify).")
    p.add_argument("--min-len", type=int, default=50)
    p.add_argument("--max-len", type=int, default=400)
    p.add_argument("--k", type=int, default=20, help="TMAP n_neighbors.")
    p.add_argument("--esmc-model", default=DEFAULT_ESMC_MODEL)
    p.add_argument("--sae-model", default=DEFAULT_SAE_MODEL)
    p.add_argument("--sae-layer", type=int, default=30, help="SAE layer index (e.g. 60 for the 6B model).")
    p.add_argument("--esmfold-model", default=DEFAULT_ESMFOLD_MODEL)
    p.add_argument("--fold-loops", type=int, default=3)
    p.add_argument("--fold-steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8, help="ESMC encode batch size.")
    p.add_argument("--stages", default="all", help="Comma list: " + ",".join(STAGES))
    p.add_argument("--force", action="store_true", help="Ignore caches and recompute.")
    p.add_argument("--serve", action="store_true", help="Serve the output (structures need HTTP).")
    p.add_argument("--port", type=int, default=8050)
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Key caches + outputs by (source, n) so different runs (e.g. an --n 50
    # smoke test and an --n 8000 full run) never clobber each other's data.
    global CACHE_DIR, OUTPUT_DIR, STRUCT_DIR
    tag = "fasta" if args.fasta else args.source
    CACHE_DIR = EXAMPLES_DIR / "data" / "esm_atlas" / f"{tag}_n{args.n}"
    OUTPUT_DIR = EXAMPLES_DIR / "esm_atlas_out" / f"{tag}_n{args.n}"
    STRUCT_DIR = OUTPUT_DIR / "structures"

    stages = STAGES if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    pd = _lazy_pandas()

    df = None
    if "fetch" in stages:
        print("[fetch] sequences + metadata")
        df = fetch_sequences(args)
    if df is None:
        df = pd.read_parquet(CACHE_DIR / "proteins.parquet")

    if "encode" in stages:
        print("[encode] ESMC embeddings + SAE features")
        encode_esmc_and_sae(df, args)
    if "fold" in stages:
        print("[fold] ESMFold2-Fast structures")
        fold_structures(df, args)

    meta = None
    if "enrich" in stages:
        print("[enrich] properties + confidence + feature labels")
        meta = enrich(df, args)
    if meta is None and (CACHE_DIR / "meta.parquet").exists():
        meta = pd.read_parquet(CACHE_DIR / "meta.parquet")

    if "maps" in stages or "viz" in stages:
        if meta is None:
            raise SystemExit("Run the 'enrich' stage first (need meta.parquet).")
        emb_model, sae_model = build_maps(meta, args)
        if "viz" in stages:
            print("[viz] writing the two TMAP maps")
            out_dir = visualize(emb_model, sae_model, meta, args)
            if args.serve:
                import http.server
                import os
                import socketserver

                os.chdir(out_dir)
                print(f"Serving {out_dir} at http://127.0.0.1:{args.port}  (Ctrl-C to stop)")
                print(f"  open  http://127.0.0.1:{args.port}/emb/index.html  or  /sae/index.html")
                with socketserver.TCPServer(("", args.port), http.server.SimpleHTTPRequestHandler) as httpd:
                    httpd.serve_forever()


if __name__ == "__main__":
    main()
