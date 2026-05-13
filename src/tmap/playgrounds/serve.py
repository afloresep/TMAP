"""FastAPI entry point. Wires playground modules into HTTP routes."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .base import Playground

STATIC_DIR = Path(__file__).parent / "static"


def build_app(registry: dict[str, Playground]) -> FastAPI:
    app = FastAPI(title="TMAP Playgrounds")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def _resolve(slug: str) -> Playground:
        if slug not in registry:
            raise HTTPException(status_code=404, detail=f"unknown playground: {slug}")
        return registry[slug]

    @app.get("/health")
    def health():
        return {"ok": True, "playgrounds": sorted(registry.keys())}

    @app.get("/playgrounds/{slug}/query")
    def query(slug: str, q: str, k: int = 20):
        pg = _resolve(slug)
        try:
            results = pg.query(q, k=k)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        return {"results": [asdict(r) for r in results]}

    @app.get("/playgrounds/{slug}/path")
    def path(slug: str, a: str, b: str):
        pg = _resolve(slug)
        try:
            res = pg.path(a, b)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        return asdict(res)

    @app.get("/playgrounds/{slug}/add")
    def add(slug: str, q: str):
        pg = _resolve(slug)
        try:
            return asdict(pg.add(q))
        except NotImplementedError as e:
            raise HTTPException(status_code=501, detail=str(e)) from None
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

    @app.get("/playgrounds/{slug}/gallery")
    def gallery(slug: str):
        pg = _resolve(slug)
        return {"items": pg.gallery()}

    return app


def default_registry() -> dict[str, Playground]:
    """Return the registry of available playgrounds, populated from cached data.

    Numba-backed playgrounds (spike) are registered before playgrounds that load
    PyTorch (word). PyTorch's OpenMP initialisation conflicts with numba's parallel
    runtime when both are active in the same process, so numba must be warmed up
    first. The warmup query also ensures the LSH forest JIT code is compiled before
    uvicorn's async worker loop begins; without it, the first HTTP query triggers
    numba JIT compilation inside an async worker thread and segfaults.
    """
    reg: dict[str, Playground] = {}

    # --- Spike (numba / LSH Forest) -- must be registered BEFORE word/ST loads PyTorch ---
    spike_root = (
        Path(__file__).resolve().parents[3] / "examples" / "data" / "sars_cov2_spike" / "playground"
    )
    if (spike_root / "spike.tmap").exists():
        from .spike import SpikePlayground, _load_df, _load_model, _norm_cached
        pg = SpikePlayground(spike_root / "spike.tmap", spike_root / "spike_meta.parquet")
        model = _load_model(str(spike_root / "spike.tmap"))
        _load_df(str(spike_root / "spike_meta.parquet"))
        _norm_cached(str(spike_root / "spike.tmap"))
        try:
            df = _load_df(str(spike_root / "spike_meta.parquet"))
            first_seq = str(df.iloc[0]["sequence"])
            k = getattr(pg, "_k", 6)
            warmup_kmer = [[first_seq[i:i + k] for i in range(len(first_seq) - k + 1)]]
            model.kneighbors(warmup_kmer)
        except Exception:
            pass  # Warmup is best-effort; a failure here should not block serving
        reg["spike"] = pg

    # --- ChEMBL (USearch / no numba) ---
    chembl_root = Path(__file__).resolve().parents[3] / "data" / "chembl"
    if (chembl_root / "chembl_full.tmap").exists():
        from .chembl import ChemblPlayground
        reg["chembl"] = ChemblPlayground(
            chembl_root / "chembl_full.tmap",
            chembl_root / "chembl_full_meta.parquet",
        )

    # --- Words (sentence-transformers / PyTorch) -- keep last to avoid OMP conflict ---
    root = Path(__file__).resolve().parents[3] / "examples" / "data" / "word50k_cache"
    model_path = root / "word_tmap.model"
    if model_path.exists():
        from .word import WordPlayground
        embed_fn = _make_word_embed_fn()
        reg["words"] = WordPlayground(
            model_path,
            root / "word_list.npy",
            root / "word_categories.npy",
            embed_fn,
        )

    # --- Proteins (ESM-2 / PyTorch) -- registered last; ESM-2 also uses PyTorch ---
    prot_root = (
        Path(__file__).resolve().parents[3]
        / "examples" / "data" / "endosymbiosis" / "playground"
    )
    if (prot_root / "proteins.tmap").exists():
        from .protein import ProteinPlayground, make_esm2_encoder
        try:
            encode = make_esm2_encoder()  # eager: loads ESM-2 once at startup
        except Exception as e:
            print(f"[playgrounds] skipping proteins playground (ESM-2 failed to load: {e})")
        else:
            reg["proteins"] = ProteinPlayground(
                prot_root / "proteins.tmap",
                prot_root / "proteins_meta.parquet",
                encode_fn=encode,
                gallery_items=_load_gallery(),
            )

    return reg


def _load_gallery() -> list[dict[str, str]]:
    """Load curated 'famous proteins' from gallery.json if present."""
    import json
    path = (
        Path(__file__).resolve().parents[3]
        / "examples" / "data" / "endosymbiosis" / "playground" / "gallery.json"
    )
    if path.exists():
        return json.loads(path.read_text())
    return []


def _make_word_embed_fn():
    """Return a callable str -> np.ndarray encoding a word with the TMAP's sentence model."""
    from functools import lru_cache

    import numpy as np
    from sentence_transformers import SentenceTransformer

    @lru_cache(maxsize=1)
    def _model():
        return SentenceTransformer("all-MiniLM-L6-v2")

    def _encode(word: str) -> np.ndarray:
        return _model().encode([word], normalize_embeddings=True)[0].astype("float32")
    return _encode


def main():
    import argparse

    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    uvicorn.run(build_app(default_registry()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
