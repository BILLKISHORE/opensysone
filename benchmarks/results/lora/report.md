# opensysone LoRA on typed-decisions test

Labels in typed-decisions are frontier-model consensus, not human gold. A calibrated probability here is the chance the consensus agrees.

| Question type | n | Accuracy raw | Accuracy calibrated | ECE raw | ECE calibrated | ECE equal-mass raw | ECE equal-mass calibrated | NLL winner raw | NLL winner calibrated | NLL vote shares raw | NLL vote shares calibrated | Brier raw | Brier calibrated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| choice | 600 | 0.720 | 0.720 | 0.054 | 0.054 | 0.055 | 0.055 | 0.639 | 0.639 | 1.353 | 1.353 | 0.146 | 0.146 |
| noul | 600 | 0.845 | 0.845 | 0.012 | 0.022 | 0.031 | 0.033 | 0.332 | 0.333 | 0.637 | 0.587 | 0.084 | 0.072 |
| score | 800 | 0.743 | 0.743 | 0.045 | 0.037 | 0.055 | 0.049 | 0.601 | 0.598 | 1.589 | 1.550 | 0.164 | 0.158 |
| all | 2000 | 0.766 | 0.766 | 0.037 | 0.027 | 0.036 | 0.032 | 0.531 | 0.531 | 1.233 | 1.202 | 0.135 | 0.129 |

95% interval over all questions, resampling whole cases:

| | Accuracy | ECE | NLL winner |
|---|---|---|---|
| raw | 0.745 to 0.788 | 0.022 to 0.056 | 0.490 to 0.573 |
| calibrated | 0.745 to 0.788 | 0.015 to 0.046 | 0.490 to 0.572 |

## choice

![choice reliability](reliability-choice.svg)

| Confidence bin | n raw | confidence raw | accuracy raw | n calibrated | confidence calibrated | accuracy calibrated |
|---|---|---|---|---|---|---|
| 0.3 to 0.4 | 13 | 0.375 | 0.462 | 13 | 0.375 | 0.462 |
| 0.4 to 0.5 | 30 | 0.460 | 0.367 | 30 | 0.460 | 0.367 |
| 0.5 to 0.6 | 101 | 0.552 | 0.505 | 101 | 0.552 | 0.505 |
| 0.6 to 0.7 | 90 | 0.651 | 0.556 | 90 | 0.651 | 0.556 |
| 0.7 to 0.8 | 71 | 0.755 | 0.620 | 71 | 0.755 | 0.620 |
| 0.8 to 0.9 | 94 | 0.853 | 0.840 | 94 | 0.853 | 0.840 |
| 0.9 to 1.0 | 201 | 0.972 | 0.950 | 201 | 0.972 | 0.950 |

## noul

![noul reliability](reliability-noul.svg)

| Confidence bin | n raw | confidence raw | accuracy raw | n calibrated | confidence calibrated | accuracy calibrated |
|---|---|---|---|---|---|---|
| 0.5 to 0.6 | 56 | 0.549 | 0.500 | 65 | 0.548 | 0.508 |
| 0.6 to 0.7 | 69 | 0.658 | 0.667 | 80 | 0.650 | 0.713 |
| 0.7 to 0.8 | 67 | 0.751 | 0.746 | 79 | 0.749 | 0.759 |
| 0.8 to 0.9 | 87 | 0.853 | 0.816 | 105 | 0.853 | 0.867 |
| 0.9 to 1.0 | 321 | 0.973 | 0.972 | 271 | 0.970 | 0.982 |

## score

![score reliability](reliability-score.svg)

| Confidence bin | n raw | confidence raw | accuracy raw | n calibrated | confidence calibrated | accuracy calibrated |
|---|---|---|---|---|---|---|
| 0.3 to 0.4 | 3 | 0.352 | 0.333 | 3 | 0.352 | 0.333 |
| 0.4 to 0.5 | 18 | 0.469 | 0.500 | 26 | 0.472 | 0.423 |
| 0.5 to 0.6 | 116 | 0.555 | 0.534 | 118 | 0.554 | 0.542 |
| 0.6 to 0.7 | 117 | 0.653 | 0.547 | 121 | 0.649 | 0.570 |
| 0.7 to 0.8 | 158 | 0.753 | 0.728 | 150 | 0.752 | 0.740 |
| 0.8 to 0.9 | 137 | 0.851 | 0.803 | 134 | 0.849 | 0.799 |
| 0.9 to 1.0 | 251 | 0.969 | 0.928 | 248 | 0.966 | 0.931 |

## all

![all reliability](reliability-all.svg)

| Confidence bin | n raw | confidence raw | accuracy raw | n calibrated | confidence calibrated | accuracy calibrated |
|---|---|---|---|---|---|---|
| 0.3 to 0.4 | 16 | 0.370 | 0.438 | 16 | 0.370 | 0.438 |
| 0.4 to 0.5 | 48 | 0.463 | 0.417 | 56 | 0.465 | 0.393 |
| 0.5 to 0.6 | 273 | 0.553 | 0.516 | 284 | 0.552 | 0.521 |
| 0.6 to 0.7 | 276 | 0.654 | 0.580 | 291 | 0.650 | 0.605 |
| 0.7 to 0.8 | 296 | 0.753 | 0.706 | 300 | 0.752 | 0.717 |
| 0.8 to 0.9 | 318 | 0.852 | 0.818 | 333 | 0.851 | 0.832 |
| 0.9 to 1.0 | 773 | 0.971 | 0.952 | 720 | 0.969 | 0.956 |
