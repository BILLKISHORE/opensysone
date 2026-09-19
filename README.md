# opensysone

An open System One model for Apple Silicon. Send it a state and a set of typed
questions, get back a decision for each question with a calibrated probability,
from one forward pass over a local model. No text is generated, so the answer
can never fall outside the schema you asked for.

The interface target is the wire format of TypeSafe's Jev, so their SDK works
against it unchanged. The model, the training, and the code are our own.
opensysone is an independent project and is not affiliated with TypeSafe AI.

## Status

Early. Phase 0 (backbone benchmark) is running. There is nothing to install
yet. Watch the repo or check back in a few weeks.

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

1. Backbone benchmark on a 48 GB Mac against TypeSafe's public evaluation set.
2. Jev-compatible HTTP server (`POST /v1/systemone`, question types `noul`,
   `choice`, `score`) over the jevmlx engine.
3. Calibration trained into a decision head, so a probability of 0.9 is right
   about nine times in ten.
4. Published benchmark results and a release you can `pip install`.

## Hardware

Apple Silicon with 48 GB of unified memory or more. Developed and measured on
a MacBook Pro M5 Pro, 48 GB.

## Built on

- [jevmlx](https://github.com/bnsd55/jevmlx) (MIT): parallel constrained
  decisions for MLX models. The engine and the evaluation harness.
- [mlx-lm](https://github.com/ml-explore/mlx-lm): model loading and inference
  on Apple Silicon.
- Backbone under evaluation: Qwen3.8-27B (Apache-2.0), 4-bit.

## License

MIT. See [LICENSE](LICENSE).
