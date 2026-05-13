"""PG2 -- ChEMBL molecules (Morgan FP + MinHash/USearch Jaccard)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from tmap import TMAP
from tmap.utils import fingerprints_from_smiles

from .base import PathNode, PathResult, Playground, QueryResult, normalize_coords


class ChemblPlayground(Playground):
    slug = "chembl"
    title = "ChEMBL molecules"

    def __init__(
        self,
        model_path: Path,
        meta_path: Path,
        *,
        fp_radius: int = 3,
        fp_n_bits: int = 2048,
    ):
        self._model_path = Path(model_path)
        self._meta_path = Path(meta_path)
        self._fp_radius = fp_radius
        self._fp_n_bits = fp_n_bits

    @property
    def _model(self) -> TMAP:
        return _load_model(str(self._model_path))

    @property
    def _df(self):
        return _load_df(str(self._meta_path))

    @property
    def _norm(self) -> np.ndarray:
        return _norm_cached(str(self._model_path))

    def _encode(self, smiles: str) -> np.ndarray:
        """Compute Morgan fingerprint for a single SMILES string.

        Raises ValueError if RDKit cannot parse the SMILES.
        """
        fps = fingerprints_from_smiles(
            [smiles],
            fp_type="morgan",
            radius=self._fp_radius,
            n_bits=self._fp_n_bits,
        )
        if len(fps) == 0:
            raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
        return fps

    def _find_idx(self, smiles: str) -> tuple[int, str]:
        """Return (index, resolved_label) for a SMILES.

        Tries an exact match in the metadata first, then falls back to
        nearest-neighbor lookup via the stored fingerprint index.
        """
        df = self._df
        match = df.index[df["canonical_smiles"] == smiles]
        if len(match):
            i = int(match[0])
            return i, str(df.iloc[i]["chembl_id"])
        fps = self._encode(smiles)
        indices, _ = self._model.kneighbors(fps)
        i = int(indices[0][0])
        return i, str(df.iloc[i]["chembl_id"])

    def query(self, q: str, k: int = 20) -> list[QueryResult]:
        """Return up to k nearest molecules for a query SMILES.

        Args:
            q: Query SMILES string.
            k: Number of neighbors to return.

        Returns:
            List of QueryResult ordered by distance (ascending).

        Raises:
            ValueError: If the SMILES cannot be parsed by RDKit.
        """
        fps = self._encode(q)
        indices, distances = self._model.kneighbors(fps)
        df = self._df
        out: list[QueryResult] = []
        for i, d in zip(indices[0], distances[0]):
            if i < 0:
                break
            row = df.iloc[int(i)]
            out.append(QueryResult(
                idx=int(i),
                distance=round(float(d), 4),
                label=str(row["chembl_id"]),
                extra={
                    "chembl_id": str(row["chembl_id"]),
                    "smiles": str(row["canonical_smiles"]),
                    "scaffold": str(row.get("scaffold", "")),
                    "mw": round(float(row.get("mw", 0)), 1),
                    "logp": round(float(row.get("logp", 0)), 2),
                    "qed": round(float(row.get("qed", 0)), 3),
                },
            ))
            if len(out) >= k:
                break
        return out

    def path(self, a: str, b: str) -> PathResult:
        """Trace the tree path between two molecules.

        Args:
            a: Start SMILES (exact match or nearest-neighbor lookup).
            b: End SMILES (exact match or nearest-neighbor lookup).

        Returns:
            PathResult with the sequence of tree nodes between a and b.
        """
        ia, ra = self._find_idx(a)
        ib, rb = self._find_idx(b)
        ids = self._model.tree_.path(ia, ib)
        emb = self._norm
        df = self._df
        nodes = [
            PathNode(
                idx=int(n),
                nx=float(emb[int(n), 0]),
                ny=float(emb[int(n), 1]),
                label=str(df.iloc[int(n)]["chembl_id"]),
            )
            for n in ids
        ]
        return PathResult(nodes=nodes, resolved_a=ra, resolved_b=rb)

    def add(self, item: str) -> QueryResult:
        fps = self._encode(item)
        model = self._model
        model.add_points(fps)
        # add_points appends to model.embedding_; the new row is the last one
        new_idx = len(model.embedding_) - 1
        # Invalidate the cached normalized embedding so the new point is included
        _norm_cached.cache_clear()
        norm = normalize_coords(model.embedding_)
        return QueryResult(
            idx=new_idx,
            distance=0.0,
            label=item,
            extra={
                "nx": float(norm[new_idx, 0]),
                "ny": float(norm[new_idx, 1]),
                "is_new_point": True,
            },
        )


@lru_cache(maxsize=2)
def _load_model(path: str) -> TMAP:
    return TMAP.load(path)


@lru_cache(maxsize=2)
def _load_df(path: str):
    import pandas as pd
    return pd.read_parquet(path)


@lru_cache(maxsize=2)
def _norm_cached(model_path: str) -> np.ndarray:
    return normalize_coords(_load_model(model_path).embedding_)
