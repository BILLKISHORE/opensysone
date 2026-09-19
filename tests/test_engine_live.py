import pytest

from opensysone.engine import Engine
from opensysone.wire import SystemOneRequest

pytestmark = pytest.mark.live
MODEL = "mlx-community/Qwen3.8-27B-4bit"


@pytest.fixture(scope="module")
def engine():
    e = Engine(MODEL)
    e.load()
    return e


def test_live_decide_matches_openjev_readme_example(engine):
    req = SystemOneRequest.model_validate({
        "state": "Everything is down and we have a demo with our biggest client at noon.",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
            "team": {"type": "choice", "instructions": "Which team should handle it?",
                     "criteria": {"outage": "service down", "billing": "charges, refunds",
                                  "feature": "requests, how-to"}},
            "tone": {"type": "score", "instructions": "How upset is the customer?",
                     "criteria": ["calm", "annoyed", "furious"]},
        }})
    out = engine.decide(req)
    assert out["answers"]["urgent"]["noul"] > 0.5
    assert out["answers"]["team"]["choice"] == "outage"
    assert out["answers"]["tone"]["score"] > 1.0
    assert out["usage"]["input_tokens"] > 50
    assert out["timing"]["max_rows"] in (None, 1)
