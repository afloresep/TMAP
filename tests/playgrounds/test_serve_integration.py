from fastapi.testclient import TestClient

from tmap.playgrounds.base import PathNode, PathResult, Playground, QueryResult
from tmap.playgrounds.serve import build_app


class _Fake(Playground):
    slug = "fake"
    title = "Fake"

    def query(self, q, k=20):
        return [QueryResult(idx=1, distance=0.1, label=q, extra={})]

    def path(self, a, b):
        return PathResult(
            nodes=[PathNode(idx=0, nx=0.0, ny=0.0, label=a),
                   PathNode(idx=1, nx=1.0, ny=1.0, label=b)],
            resolved_a=a, resolved_b=b,
        )


def test_health():
    client = TestClient(build_app({}))
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "playgrounds": []}


def test_query_route_dispatches_to_playground():
    client = TestClient(build_app({"fake": _Fake()}))
    r = client.get("/playgrounds/fake/query?q=hello")
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["label"] == "hello"


def test_path_route():
    client = TestClient(build_app({"fake": _Fake()}))
    r = client.get("/playgrounds/fake/path?a=x&b=y")
    assert r.status_code == 200
    body = r.json()
    assert body["resolved_a"] == "x" and body["resolved_b"] == "y"
    assert len(body["nodes"]) == 2


def test_unknown_slug_404():
    client = TestClient(build_app({}))
    assert client.get("/playgrounds/nope/query?q=x").status_code == 404
