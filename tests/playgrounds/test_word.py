import numpy as np
import pytest

from tmap.playgrounds.word import WordPlayground


@pytest.fixture
def tiny_word_pg(tmp_path):
    from tmap import TMAP
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 8)).astype("float32")
    model = TMAP(metric="cosine", n_neighbors=3, seed=0, store_index=True).fit(X)
    model_path = tmp_path / "model.tmap"
    model.save(model_path)
    words_path = tmp_path / "words.npy"
    cats_path = tmp_path / "cats.npy"
    np.save(words_path, np.array(["cat", "dog", "car", "truck"], dtype=object))
    np.save(cats_path, np.array(["animal", "animal", "vehicle", "vehicle"], dtype=object))
    # Use a deterministic embed_fn — return one of the stored input vectors
    def embed_fn(w: str) -> np.ndarray:
        return rng.normal(size=8).astype("float32")
    return WordPlayground(model_path, words_path, cats_path, embed_fn=embed_fn)


def test_query_returns_topk(tiny_word_pg):
    results = tiny_word_pg.query("cat", k=3)
    assert len(results) == 3
    assert all(0 <= r.idx < 4 for r in results)


def test_path_exact_match(tiny_word_pg):
    pr = tiny_word_pg.path("cat", "truck")
    assert pr.resolved_a == "cat" and pr.resolved_b == "truck"
    assert len(pr.nodes) >= 2
    assert pr.nodes[0].label == "cat"
    assert pr.nodes[-1].label == "truck"
