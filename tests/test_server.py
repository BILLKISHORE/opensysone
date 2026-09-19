import pytest
from fastapi.testclient import TestClient

from opensysone.engine import MODEL_VERSION
from opensysone.schema import SchemaError
from opensysone.server import create_app


class FakeEngine:
    model_id = "fake"
    ready = True

    def decide(self, req):
        if "boom" in req.questions:
            raise SchemaError(["body", "questions", "boom"], "no")
        if "crash" in req.questions:
            raise RuntimeError("gpu fell over")
        return {"model": MODEL_VERSION,
                "answers": {q: {"type": "noul", "noul": 0.9} for q in req.questions},
                "usage": {"input_tokens": 3, "output_tokens": 0}, "stamp": {}, "cached": False,
                "warnings": [], "timing": {}}

    def decide_bulk(self, reqs):
        return [self.decide(r) for r in reqs]


BODY = {"model": "jev-latest", "state": "x", "questions": {"q": {"type": "noul", "instructions": "y"}}}


@pytest.fixture
def client():
    return TestClient(create_app(FakeEngine()), raise_server_exceptions=False)


def test_systemone_happy_path_and_request_id(client):
    r = client.post("/v1/systemone", json=BODY)
    assert r.status_code == 200
    assert r.json()["answers"]["q"]["noul"] == 0.9
    assert r.headers["x-request-id"]


def test_unknown_model_is_404_jev_shape(client):
    r = client.post("/v1/systemone", json={**BODY, "model": "gpt-9"})
    assert r.status_code == 404
    assert r.json()["detail"]["error_type"] == "not_found_error"


def test_bad_body_is_422_list(client):
    r = client.post("/v1/systemone", json={"state": "x"})
    assert r.status_code == 422 and isinstance(r.json()["detail"], list)


def test_schema_error_is_422_with_location(client):
    r = client.post("/v1/systemone", json={**BODY, "questions": {"boom": {"type": "noul"}}})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "questions", "boom"]


def test_engine_crash_is_500_jev_shape(client):
    r = client.post("/v1/systemone", json={**BODY, "questions": {"crash": {"type": "noul"}}})
    assert r.status_code == 500
    assert r.json()["detail"]["error_type"] == "api_error"


def test_bulk(client):
    r = client.post("/v1/bulk", json={"items": [BODY, BODY]})
    assert r.status_code == 200 and len(r.json()["items"]) == 2


def test_models_and_health(client):
    names = [m["name"] for m in client.get("/v1/models").json()["models"]]
    assert "opensysone-latest" in names and MODEL_VERSION in names
    h = client.get("/health").json()
    assert h["ok"] is True and h["model"] == "fake" and h["requests_served"] >= 0


def test_api_key_required_when_configured():
    c = TestClient(create_app(FakeEngine(), api_key="sk-test"))
    assert c.post("/v1/systemone", json=BODY).status_code == 401
    assert c.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer sk-test"}).status_code == 200
    assert c.get("/health").status_code == 200  # health stays open


def test_queue_full_is_529():
    c = TestClient(create_app(FakeEngine(), max_queue=0))
    r = c.post("/v1/systemone", json=BODY)
    assert r.status_code == 529
    assert r.headers["retry-after"] == "1"
