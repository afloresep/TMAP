"""Shared types and helpers for playground modules."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class QueryResult:
    idx: int
    distance: float
    label: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PathNode:
    idx: int
    nx: float   # normalized [0,1] x coord (for overlay canvas)
    ny: float
    label: str


@dataclass
class PathResult:
    nodes: list[PathNode]
    resolved_a: str
    resolved_b: str


def normalize_coords(emb: np.ndarray) -> np.ndarray:
    """Map an (N, 2) embedding into [0, 1] x [0, 1] for the path overlay."""
    out = emb.astype(float).copy()
    for j in range(out.shape[1]):
        lo, hi = out[:, j].min(), out[:, j].max()
        span = hi - lo or 1.0
        out[:, j] = (out[:, j] - lo) / span
    return out


class Playground(ABC):
    """Each playground is an instance of this class registered in serve.py."""

    slug: str = ""
    title: str = ""

    @abstractmethod
    def query(self, q: str, k: int = 20) -> list[QueryResult]: ...

    @abstractmethod
    def path(self, a: str, b: str) -> PathResult: ...

    def add(self, item: str) -> QueryResult:
        raise NotImplementedError(f"{self.slug} does not support add_points yet")

    def gallery(self) -> list[dict[str, str]]:
        return []
