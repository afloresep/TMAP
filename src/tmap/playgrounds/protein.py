"""PG4 -- cross-kingdom proteins (ESM-2 embeddings, cosine)."""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np

from tmap import TMAP

from .base import PathNode, PathResult, Playground, QueryResult, normalize_coords

_AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYX*-")


class ProteinPlayground(Playground):
    slug = "proteins"
    title = "Cross-kingdom proteins"

    def __init__(
        self,
        model_path: Path,
        meta_path: Path,
        encode_fn: Callable[[str], np.ndarray],
        gallery_items: list[dict[str, str]] | None = None,
    ):
        self._model_path = Path(model_path)
        self._meta_path = Path(meta_path)
        self._encode_fn = encode_fn
        self._gallery = gallery_items or []

    @property
    def _model(self) -> TMAP:
        return _load_model(str(self._model_path))

    @property
    def _df(self):
        return _load_df(str(self._meta_path))

    @property
    def _norm(self) -> np.ndarray:
        return _norm_cached(str(self._model_path))

    @property
    def _input_emb(self) -> np.ndarray:
        return _load_input(str(self._model_path.parent / "embeddings.npy"))

    def _looks_like_sequence(self, q: str) -> bool:
        """Return True when q looks like an amino-acid sequence (>=20 chars, AA alphabet)."""
        s = q.strip().upper()
        return len(s) >= 20 and set(s).issubset(_AA_ALPHABET)

    def _accession_idx(self, acc: str) -> int | None:
        """Return the DataFrame row index for an accession, or None if not found."""
        df = self._df
        m = df.index[df["accession"] == acc]
        return int(m[0]) if len(m) else None

    def _vec_for_query(self, q: str) -> np.ndarray:
        """Return a (1, D) float32 array for q, reusing stored embeddings when possible."""
        idx = self._accession_idx(q)
        if idx is not None:
            return self._input_emb[idx : idx + 1]
        if self._looks_like_sequence(q):
            return self._encode_fn(q).reshape(1, -1)
        raise ValueError(f"unknown accession and not a valid sequence: {q[:30]}...")

    def _find_idx(self, q: str) -> tuple[int, str]:
        """Return (row index, accession label) for a query that is an accession or sequence."""
        idx = self._accession_idx(q)
        if idx is not None:
            return idx, str(self._df.iloc[idx]["accession"])
        vec = self._vec_for_query(q)
        indices, _ = self._model.kneighbors(vec)
        i = int(indices[0][0])
        return i, str(self._df.iloc[i]["accession"])

    def query(self, q: str, k: int = 20) -> list[QueryResult]:
        """Return up to k nearest proteins for a query.

        Args:
            q: Either a UniProt accession (exact match, reuses stored embedding)
               or an amino-acid sequence of at least 20 characters (ESM-2 encoded).
            k: Number of neighbors to return.

        Returns:
            List of QueryResult ordered by distance (ascending).

        Raises:
            ValueError: If q is neither a known accession nor a valid sequence.
        """
        vec = self._vec_for_query(q)
        indices, distances = self._model.kneighbors(vec)
        df = self._df
        out: list[QueryResult] = []
        for i, d in zip(indices[0], distances[0]):
            if i < 0:
                break
            row = df.iloc[int(i)]
            out.append(QueryResult(
                idx=int(i),
                distance=round(float(d), 4),
                label=str(row["accession"]),
                extra={
                    "organism": str(row.get("organism", "")),
                    "domain": str(row.get("domain", "")),
                    "compartment": str(row.get("compartment", "")),
                    "alphafold_url": (
                        f"https://alphafold.ebi.ac.uk/files/"
                        f"AF-{row['accession']}-F1-model_v4.pdb"
                    ),
                },
            ))
            if len(out) >= k:
                break
        return out

    def path(self, a: str, b: str) -> PathResult:
        """Trace the tree path between two proteins.

        Args:
            a: Start accession or sequence.
            b: End accession or sequence.

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
                label=str(df.iloc[int(n)]["accession"]),
            )
            for n in ids
        ]
        return PathResult(nodes=nodes, resolved_a=ra, resolved_b=rb)

    def gallery(self) -> list[dict[str, str]]:
        """Return the curated list of famous proteins (empty if no gallery.json was loaded)."""
        return list(self._gallery)


@lru_cache(maxsize=2)
def _load_model(p: str) -> TMAP:
    return TMAP.load(p)


@lru_cache(maxsize=2)
def _load_df(p: str):
    import pandas as pd
    return pd.read_parquet(p)


@lru_cache(maxsize=2)
def _load_input(p: str) -> np.ndarray:
    return np.load(p).astype("float32")


@lru_cache(maxsize=2)
def _norm_cached(p: str) -> np.ndarray:
    return normalize_coords(_load_model(p).embedding_)


def make_esm2_encoder() -> Callable[[str], np.ndarray]:
    """Load ESM-2 650M once and return a mean-pooled (1280,) float32 encoder.

    The model is loaded eagerly when this factory is called. The returned
    callable is suitable for passing to ProteinPlayground as encode_fn.
    """
    import esm  # fair-esm
    import torch

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.eval().to(device)

    def _encode(seq: str) -> np.ndarray:
        seq = seq.strip().upper().replace("*", "").replace("-", "")
        if not seq:
            raise ValueError("empty sequence")
        if len(seq) > 1022:
            seq = seq[:1022]
        _, _, toks = batch_converter([("q", seq)])
        toks = toks.to(device)
        with torch.no_grad():
            out = model(toks, repr_layers=[33], return_contacts=False)
        rep = out["representations"][33][0, 1 : 1 + len(seq), :].mean(0)
        return rep.detach().to("cpu").float().numpy()

    return _encode
