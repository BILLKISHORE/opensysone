# opensysone

An open System One model for Apple Silicon. Send it a state and a set of typed
questions, get back a decision for each question with a calibrated probability,
from one forward pass over a local model. No text is generated, so the answer
can never fall outside the schema you asked for.

The interface target is the wire format of TypeSafe's Jev, so their SDK works
against it unchanged. The model, the training, and the code are our own.
opensysone is an independent project and is not affiliated with TypeSafe AI.

## Status

Pre-release `0.1.0a1`. The Jev-compatible server runs locally with the M1
feature set: constraints between questions, abstain, multi-select, ranked
choices, pairwise judging, score spread, position-bias averaging, prior
correction, rules-first, a decision cache, policy bands, schema lint, and a
version stamp on every response. Calibration training (M2) is next.

Measured on a MacBook Pro M5 Pro 48 GB with Qwen3.8-27B at 4-bit, one
question over the state "my card was charged twice":

| Options | Latency | Correct | Peak memory |
|---|---|---|---|
| 16 | 3.6 s | yes | 16.3 GB |
| 64 | 6.8 s | yes | 17.5 GB |
| 128 | 16.4 s | yes | 19.0 GB |
| 255 | 47.3 s | yes | 21.8 GB |

Raw numbers and the script are in `benchmarks/`. This backbone is dense, so
every pass costs all 27B parameters; a 3B-active backbone would be several
times faster and is on the list to measure. Batched scoring is disabled on
this backbone because its batched and unbatched probabilities disagree
(jevmlx's parity check), so questions are scored one row per pass.

## Install

```bash
pip install "opensysone @ git+https://github.com/BILLKISHORE/opensysone@v0.1.0a1"
opensysone serve --model mlx-community/Qwen3.8-27B-4bit --port 8080
```

The first start downloads the 16 GB checkpoint and runs a parity check;
allow a few minutes.

## Use

With TypeSafe's SDK, unchanged:

```bash
pip install typesafe-sdk
export TYPESAFE_BASE_URL=http://127.0.0.1:8080
export TYPESAFE_API_KEY=anything
```

```python
from typesafe_sdk import TypeSafeClient

r = TypeSafeClient().system_one(
    "Everything is down and we have a demo with our biggest client at noon.",
    {
        "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
        "team":   {"type": "choice", "instructions": "Which team should handle it?",
                   "criteria": {"outage": "service down", "billing": "charges, refunds", "feature": "requests, how-to"}},
        "tone":   {"type": "score", "instructions": "How upset is the customer?",
                   "criteria": ["calm", "annoyed", "furious"]},
    },
)
print(r.nouls["urgent"].noul, r.choices["team"].choice, r.scores["tone"].score)
```

The SDK's `extra_body` argument carries the extensions below.

With curl, using extensions Jev does not have:

```bash
curl -s localhost:8080/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice this month and I want it fixed today.",
  "samples": 4,
  "rules": [{"question": "refund", "pattern": "charged twice", "answer": true}],
  "questions": {
    "refund": {"type": "noul", "instructions": "Does the customer want money back?",
               "policy": {"act": 0.9, "review": 0.6}},
    "team":   {"type": "choice", "instructions": "Which team?", "abstain": true,
               "criteria": {"billing": "charges", "outage": "service down"}},
    "tags":   {"type": "multi", "instructions": "Which apply?",
               "criteria": {"angry": "", "deadline": "", "duplicate_charge": ""}}
  }
}'
```

Request extensions: `samples` (shuffle option order and average, removes
position bias), `prior_correction`, `rules` (regex answers that skip the
model), `constraints` (implies, excludes, exclusivity across questions),
`strict` (lint findings become errors), `cache: false`. Question extensions:
`abstain`, `policy`, type `multi`, type `pairwise`. Every response carries a
`stamp` (backbone revision, prompt version, package version), `warnings`
from the schema lint, and `timing`.

Other endpoints: `POST /v1/bulk`, `GET /v1/models`, `GET /health`. From the
terminal: `opensysone decide`, `opensysone lint`.

## What it will do

```
state:      "Everything is down and we have a demo with our biggest client at noon."
questions:  urgent  (yes/no)   Does the customer need a reply within the hour?
            team    (choice)   outage | billing | feature
            tone    (score)    calm | annoyed | furious

answer:     urgent = true      p = 0.97
            team   = outage    p = 0.94
            tone   = furious   p = 0.81
```

All three answers come from a single pass. Your software reads the
probabilities and decides when to act on its own and when to ask a person.

## Plan

1. Backbone benchmark on a 48 GB Mac against TypeSafe's public evaluation set. Done for one backbone; more to come.
2. Jev-compatible HTTP server over the jevmlx engine. Done (`0.1.0a1`).
3. Calibration trained into a decision head, so a probability of 0.9 is right
   about nine times in ten.
4. Published benchmark results and a release you can `pip install`.

## Hardware

Apple Silicon with 48 GB of unified memory or more. Developed and measured on
a MacBook Pro M5 Pro, 48 GB.

## Built on

- [jevmlx](https://github.com/bnsd55/jevmlx) (MIT): parallel constrained
  decisions for MLX models. The engine and the evaluation tooling.
- [mlx-lm](https://github.com/ml-explore/mlx-lm): model loading and inference
  on Apple Silicon.
- Default backbone: Qwen3.8-27B (Apache-2.0), 4-bit. Any mlx-lm model can be passed to `--model`.

## License

MIT. See [LICENSE](LICENSE).
