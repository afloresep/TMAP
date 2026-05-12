import numpy as np
import pytest

from tmap.playgrounds.base import Playground, normalize_coords


def test_normalize_coords_unit_square():
    emb = np.array([[0.0, 0.0], [10.0, 5.0], [5.0, 2.5]], dtype=float)
    norm = normalize_coords(emb)
    assert norm.shape == emb.shape
    assert norm.min() == pytest.approx(0.0)
    assert norm.max() == pytest.approx(1.0)


def test_playground_abc_requires_methods():
    with pytest.raises(TypeError):
        Playground()
