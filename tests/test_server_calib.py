import pytest
from fastapi.testclient import TestClient

from opensysone.calib.calibrators import Calibrator, KindCalibration
from opensysone.calib.store import OutcomeStore
from opensysone.server import create_app

BODY = {"model": "opensysone-latest", "state": "Charged twice.",
        "questions": {"team": {"type": "choice", "instructions": "Which team?",
                               "criteria": {"billing": "b", "ops": "o"}}}}


class FakeEngine:
    model_id = "fake"
    ready = True

    def __init__(self):
        self.calibrator = None

    def set_calibrator(self, c):
        self.calibrator = c

    def decide(self, req):
        return {"model": "opensysone-test",
                "answers": {"team": {"type": "choice", "choice": "billing",
                                     "probabilities": {"billing": 0.55, "ops": 0.45}, "confidence": 0.01}},
                "usage": {"input_tokens": 5, "output_tokens": 0}, "stamp": {"calibration": None},
                "cached": False, "warnings": [], "timing": {},
                "_raw": {"prompt_tokens": 5, "questions": {
                    "team": {"kind": "choice", "options": ["billing", "ops"], "probs": [0.6, 0.4],
                             "source": "model"}}}}

    def decide_bulk(self, reqs):
        return [self.decide(r) for r in reqs]


@pytest.fixture
def client(tmp_path):
    store = OutcomeStore(str(tmp_path / "o.sqlite"), feedback_out=str(tmp_path / "fb.jsonl"))
    cal_dir = tmp_path / "cal"
    Calibrator({"choice": KindCalibration("temperature", 2.0)}).save(cal_dir)
    engine = FakeEngine()
    c = TestClient(create_app(engine, store=store, calibration_dir=str(cal_dir)))
    c.engine = engine
    return c


def test_request_id_in_body_and_feedback_round_trip(client):
    body = client.post("/v1/systemone", json=BODY).json()
    rid = body["request_id"]
    assert rid and "_raw" not in body
    r = client.post("/v1/feedback", json={"request_id": rid, "question": "team", "label": "ops"})
    assert r.status_code == 200 and r.json() == {"ok": True, "correct": False, "label": "ops"}


def test_feedback_errors(client):
    r = client.post("/v1/feedback", json={"request_id": "nope", "question": "team", "label": "ops"})
    assert r.status_code == 404
    rid = client.post("/v1/systemone", json=BODY).json()["request_id"]
    r = client.post("/v1/feedback", json={"request_id": rid, "question": "team", "label": "sales"})
    assert r.status_code == 422


def test_audit_and_queue(client):
    rid = client.post("/v1/systemone", json=BODY).json()["request_id"]
    items = client.get("/v1/queue", params={"threshold": 0.6}).json()["items"]
    assert [i["request_id"] for i in items] == [rid]
    client.post("/v1/feedback", json={"request_id": rid, "question": "team", "label": "billing"})
    assert client.get("/v1/queue").json()["items"] == []
    audit = client.get("/v1/audit").json()
    assert audit["n"] == 1 and audit["by_kind"]["choice"]["accuracy"] == 1.0


def test_bulk_items_get_distinct_request_ids(client):
    items = client.post("/v1/bulk", json={"items": [BODY, BODY]}).json()["items"]
    assert items[0]["request_id"] != items[1]["request_id"]
    assert items[0]["request_id"].endswith("-0")


def test_reload_swaps_calibrator(client):
    r = client.post("/v1/calibration/reload")
    assert r.status_code == 200 and len(r.json()["calibration"]) == 16
    assert client.engine.calibrator is not None


def test_store_routes_404_without_store():
    c = TestClient(create_app(FakeEngine()))
    for method, path in (("post", "/v1/feedback"), ("get", "/v1/audit"), ("get", "/v1/queue")):
        kwargs = {"json": {"request_id": "x", "question": "q", "label": "a"}} if method == "post" else {}
        r = getattr(c, method)(path, **kwargs)
        assert r.status_code == 404 and "--outcomes" in r.json()["detail"]["message"]
    r = c.post("/v1/calibration/reload")
    assert r.status_code == 404 and "--calibration" in r.json()["detail"]["message"]
