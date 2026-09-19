import socket
import threading
import time

import pytest
import uvicorn

from opensysone.engine import MODEL_VERSION
from opensysone.server import create_app

typesafe_sdk = pytest.importorskip("typesafe_sdk")


class JevShapedEngine:
    model_id = "fake"
    ready = True

    def decide(self, req):
        answers = {}
        for qid, q in req.questions.items():
            if q.type == "noul":
                answers[qid] = {"type": "noul", "noul": 1.0}
            elif q.type == "choice":
                names = list(q.criteria)
                answers[qid] = {"type": "choice", "choice": names[0],
                                "probabilities": {n: (1.0 if i == 0 else 0.0) for i, n in enumerate(names)},
                                "confidence": 1.0}
            elif q.type == "score":
                n = len(q.criteria)
                answers[qid] = {"type": "score", "score": float(n - 1),
                                "legend": {str(i): str(c) for i, c in enumerate(q.criteria)},
                                "probabilities": {str(i): (1.0 if i == n - 1 else 0.0) for i in range(n)},
                                "confidence": 1.0}
        return {"model": MODEL_VERSION, "answers": answers,
                "usage": {"input_tokens": 10, "output_tokens": 0}, "stamp": {}, "cached": False,
                "warnings": [], "timing": {}}

    def decide_bulk(self, reqs):
        return [self.decide(r) for r in reqs]


@pytest.fixture(scope="module")
def base_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(create_app(JevShapedEngine(), api_key="sk-test"),
                            host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(timeout=5)


def test_sdk_round_trip(base_url, monkeypatch):
    monkeypatch.setenv("TYPESAFE_BASE_URL", base_url)
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
    client = typesafe_sdk.TypeSafeClient()
    r = client.system_one(
        "Everything is down and we have a demo with our biggest client at noon.",
        {
            "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
            "team": {"type": "choice", "instructions": "Which team should handle it?",
                     "criteria": {"outage": "service down", "billing": "charges, refunds",
                                  "feature": "requests, how-to"}},
            "tone": {"type": "score", "instructions": "How upset is the customer?",
                     "criteria": ["calm", "annoyed", "furious"]},
        },
    )
    assert r.nouls["urgent"].noul == 1.0
    assert r.choices["team"].choice == "outage"
    assert r.scores["tone"].score == 2.0
