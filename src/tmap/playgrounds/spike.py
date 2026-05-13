"""PG3 -- SARS-CoV-2 spike (AA k-mer shingles + MinHash)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from tmap import TMAP

from .base import PathNode, PathResult, Playground, QueryResult, normalize_coords


def _shingle(seq: str, k: int) -> list[str]:
    """Return all overlapping k-mers from an amino-acid sequence."""
    seq = seq.strip().upper()
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


class SpikePlayground(Playground):
    slug = "spike"
    title = "SARS-CoV-2 spike"

    def __init__(self, model_path: Path, meta_path: Path, k: int = 6, n_perm: int = 128):
        self._model_path = Path(model_path)
        self._meta_path = Path(meta_path)
        self._k = k
        self._n_perm = n_perm

    @property
    def _model(self) -> TMAP:
        return _load_model(str(self._model_path))

    @property
    def _df(self):
        return _load_df(str(self._meta_path))

    @property
    def _norm(self) -> np.ndarray:
        return _norm_cached(str(self._model_path))

    def _looks_like_sequence(self, q: str) -> bool:
        """Return True when the query looks like an amino-acid sequence rather than a strain name.

        A sequence is at least 10 characters long and consists entirely of
        standard single-letter amino-acid codes (plus X, *, and gap -).
        Strain names typically contain slashes, digits, or other characters
        outside the AA alphabet, so this threshold is intentionally low.
        """
        s = q.strip().upper()
        return len(s) >= 10 and set(s).issubset(set("ACDEFGHIKLMNPQRSTVWYX*-"))

    def _encode(self, seq: str) -> list[list[str]]:
        """Shingle a single sequence into a list-of-sets format for kneighbors()."""
        if not seq.strip():
            raise ValueError("empty sequence")
        return [_shingle(seq, self._k)]

    def _find_idx(self, q: str) -> tuple[int, str]:
        """Return (row index, strain label) for a query that is a strain name or sequence.

        If q matches a strain name exactly, use that row directly. Otherwise
        treat q as a sequence and find its nearest neighbor.
        """
        df = self._df
        m = df.index[df["strain"] == q]
        if len(m):
            i = int(m[0])
            return i, str(df.iloc[i]["strain"])
        if self._looks_like_sequence(q):
            indices, _ = self._model.kneighbors(self._encode(q))
            i = int(indices[0][0])
            return i, str(df.iloc[i]["strain"])
        raise ValueError(f"unknown strain and not a valid sequence: {q[:30]}...")

    def query(self, q: str, k: int = 20) -> list[QueryResult]:
        """Return up to k nearest spike sequences for a query.

        Args:
            q: Either a strain name (exact match) or an amino-acid sequence
               (at least 20 chars drawn from the standard AA alphabet).
            k: Number of neighbors to return.

        Returns:
            List of QueryResult ordered by distance (ascending).

        Raises:
            ValueError: If q is neither a known strain name nor a valid sequence.
        """
        if self._looks_like_sequence(q):
            kmer_sets = self._encode(q)
        else:
            i, _ = self._find_idx(q)
            seq = str(self._df.iloc[i]["sequence"])
            kmer_sets = self._encode(seq)
        indices, distances = self._model.kneighbors(kmer_sets)
        df = self._df
        out: list[QueryResult] = []
        for i, d in zip(indices[0], distances[0]):
            if i < 0:
                break
            row = df.iloc[int(i)]
            out.append(QueryResult(
                idx=int(i),
                distance=round(float(d), 4),
                label=str(row["strain"]),
                extra={
                    "clade": str(row["clade"]),
                    "country": str(row.get("country", "")),
                    "date": float(row.get("date_numeric", 0.0)),
                },
            ))
            if len(out) >= k:
                break
        return out

    def path(self, a: str, b: str) -> PathResult:
        """Trace the tree path between two spike sequences.

        Args:
            a: Start strain name or sequence.
            b: End strain name or sequence.

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
                label=str(df.iloc[int(n)]["strain"]),
            )
            for n in ids
        ]
        return PathResult(nodes=nodes, resolved_a=ra, resolved_b=rb)


@lru_cache(maxsize=2)
def _load_model(p: str) -> TMAP:
    return TMAP.load(p)


@lru_cache(maxsize=2)
def _load_df(p: str):
    import pandas as pd
    return pd.read_parquet(p)


@lru_cache(maxsize=2)
def _norm_cached(p: str) -> np.ndarray:
    return normalize_coords(_load_model(p).embedding_)
