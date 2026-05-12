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
    """Populated by individual playground modules in subsequent tasks."""
    return {}


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
