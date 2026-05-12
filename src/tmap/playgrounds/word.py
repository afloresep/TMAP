"""PG1 -- word embeddings."""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np

from tmap import TMAP

from .base import PathNode, PathResult, Playground, QueryResult, normalize_coords


class WordPlayground(Playground):
    slug = "words"
    title = "Word embeddings"

    def __init__(
        self,
        model_path: Path,
        words_path: Path,
        categories_path: Path,
        embed_fn: Callable[[str], np.ndarray],
    ):
        self._model_path = Path(model_path)
        self._words_path = Path(words_path)
        self._cats_path = Path(categories_path)
        self._embed_fn = embed_fn

    @property
    def _model(self) -> TMAP:
        return _load_model(str(self._model_path))

    @property
    def _words(self) -> np.ndarray:
        return _load_words(str(self._words_path))

    @property
    def _categories(self) -> np.ndarray:
        return _load_cats(str(self._cats_path))

    @property
    def _norm_emb(self) -> np.ndarray:
        return _norm_cached(str(self._model_path))

    def _find_idx(self, word: str) -> tuple[int, str]:
        """Return (index, resolved_label) for word. Falls back to nearest neighbor."""
        matches = np.where(self._words == word)[0]
        if len(matches):
            return int(matches[0]), word
        emb = self._embed_fn(word).reshape(1, -1)
        idx, _ = self._model.kneighbors(emb)
        best = int(idx[0][0])
        return best, str(self._words[best])

    def query(self, q: str, k: int = 20) -> list[QueryResult]:
        if not q.strip():
            raise ValueError("empty query")
        emb = self._embed_fn(q).reshape(1, -1)
        indices, distances = self._model.kneighbors(emb)
        out: list[QueryResult] = []
        for i, d in zip(indices[0], distances[0]):
            if i < 0:
                break
            out.append(QueryResult(
                idx=int(i),
                distance=round(float(d), 4),
                label=str(self._words[int(i)]),
                extra={"category": str(self._categories[int(i)])},
            ))
            if len(out) >= k:
                break
        return out

    def path(self, a: str, b: str) -> PathResult:
        ia, ra = self._find_idx(a)
        ib, rb = self._find_idx(b)
        ids = self._model.tree_.path(ia, ib)
        emb = self._norm_emb
        nodes = [
            PathNode(
                idx=int(n),
                nx=float(emb[int(n), 0]),
                ny=float(emb[int(n), 1]),
                label=str(self._words[int(n)]),
            )
            for n in ids
        ]
        return PathResult(nodes=nodes, resolved_a=ra, resolved_b=rb)


@lru_cache(maxsize=4)
def _load_model(path: str) -> TMAP:
    return TMAP.load(path)


@lru_cache(maxsize=4)
def _load_words(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=True)


@lru_cache(maxsize=4)
def _load_cats(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=True)


@lru_cache(maxsize=4)
def _norm_cached(model_path: str) -> np.ndarray:
    return normalize_coords(_load_model(model_path).embedding_)
