# Calibration of Qwen3.8-27B on typed-decisions (test split)

Labels in typed-decisions are frontier-model consensus, not human gold. A calibrated probability here is the chance the consensus agrees.

| Question type | n | Accuracy raw | Accuracy calibrated | ECE raw | ECE calibrated | ECE equal-mass raw | ECE equal-mass calibrated | NLL winner raw | NLL winner calibrated | NLL vote shares raw | NLL vote shares calibrated | Brier raw | Brier calibrated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| choice | 600 | 0.595 | 0.740 | 0.097 | 0.051 | 0.100 | 0.059 | 1.084 | 0.627 | 1.253 | 1.325 | 0.175 | 0.134 |
| noul | 600 | 0.810 | 0.832 | 0.147 | 0.020 | 0.149 | 0.033 | 0.494 | 0.351 | 0.565 | 0.612 | 0.080 | 0.081 |
| score | 800 | 0.635 | 0.724 | 0.045 | 0.047 | 0.058 | 0.060 | 0.888 | 0.620 | 1.222 | 1.673 | 0.132 | 0.160 |
| all | 2000 | 0.675 | 0.761 | 0.079 | 0.030 | 0.079 | 0.037 | 0.828 | 0.541 | 1.034 | 1.250 | 0.129 | 0.129 |

95% interval over all questions, resampling whole cases:

| | Accuracy | ECE | NLL winner |
|---|---|---|---|
| raw | 0.653 to 0.699 | 0.063 to 0.099 | 0.796 to 0.861 |
| calibrated | 0.740 to 0.783 | 0.020 to 0.049 | 0.502 to 0.579 |

## choice

![choice reliability](reliability-choice.svg)

| Confidence bin | n raw | confidence raw | accuracy raw | n calibrated | confidence calibrated | accuracy calibrated |
|---|---|---|---|---|---|---|
| 0.2 to 0.3 | 25 | 0.271 | 0.400 | 0 | n/a | n/a |
| 0.3 to 0.4 | 165 | 0.359 | 0.400 | 14 | 0.361 | 0.357 |
| 0.4 to 0.5 | 172 | 0.450 | 0.570 | 50 | 0.462 | 0.600 |
| 0.5 to 0.6 | 101 | 0.549 | 0.624 | 98 | 0.546 | 0.571 |
| 0.6 to 0.7 | 59 | 0.650 | 0.814 | 92 | 0.653 | 0.576 |
| 0.7 to 0.8 | 44 | 0.741 | 0.886 | 85 | 0.755 | 0.647 |
| 0.8 to 0.9 | 34 | 0.845 | 0.971 | 92 | 0.853 | 0.891 |
| 0.9 to 1.0 | 0 | n/a | n/a | 169 | 0.957 | 0.964 |

## noul

![noul reliability](reliability-noul.svg)

| Confidence bin | n raw | confidence raw | accuracy raw | n calibrated | confidence calibrated | accuracy calibrated |
|---|---|---|---|---|---|---|
| 0.5 to 0.6 | 156 | 0.552 | 0.590 | 47 | 0.546 | 0.511 |
| 0.6 to 0.7 | 232 | 0.651 | 0.802 | 73 | 0.655 | 0.630 |
| 0.7 to 0.8 | 176 | 0.742 | 0.977 | 98 | 0.752 | 0.694 |
| 0.8 to 0.9 | 36 | 0.833 | 1.000 | 100 | 0.848 | 0.850 |
| 0.9 to 1.0 | 0 | n/a | n/a | 282 | 0.968 | 0.979 |

## score

![score reliability](reliability-score.svg)

| Confidence bin | n raw | confidence raw | accuracy raw | n calibrated | confidence calibrated | accuracy calibrated |
|---|---|---|---|---|---|---|
| 0.2 to 0.3 | 1 | 0.295 | 0.000 | 0 | n/a | n/a |
| 0.3 to 0.4 | 58 | 0.367 | 0.414 | 5 | 0.375 | 0.400 |
| 0.4 to 0.5 | 152 | 0.451 | 0.507 | 41 | 0.468 | 0.512 |
| 0.5 to 0.6 | 178 | 0.552 | 0.567 | 115 | 0.549 | 0.504 |
| 0.6 to 0.7 | 149 | 0.643 | 0.564 | 133 | 0.650 | 0.594 |
| 0.7 to 0.8 | 121 | 0.747 | 0.769 | 121 | 0.750 | 0.702 |
| 0.8 to 0.9 | 110 | 0.845 | 0.891 | 160 | 0.846 | 0.800 |
| 0.9 to 1.0 | 31 | 0.920 | 1.000 | 225 | 0.958 | 0.916 |

## all

![all reliability](reliability-all.svg)

| Confidence bin | n raw | confidence raw | accuracy raw | n calibrated | confidence calibrated | accuracy calibrated |
|---|---|---|---|---|---|---|
| 0.2 to 0.3 | 26 | 0.272 | 0.385 | 0 | n/a | n/a |
| 0.3 to 0.4 | 223 | 0.361 | 0.404 | 19 | 0.364 | 0.368 |
| 0.4 to 0.5 | 324 | 0.451 | 0.540 | 91 | 0.465 | 0.560 |
| 0.5 to 0.6 | 435 | 0.551 | 0.589 | 260 | 0.547 | 0.531 |
| 0.6 to 0.7 | 440 | 0.648 | 0.723 | 298 | 0.652 | 0.597 |
| 0.7 to 0.8 | 341 | 0.744 | 0.891 | 304 | 0.752 | 0.684 |
| 0.8 to 0.9 | 180 | 0.842 | 0.928 | 352 | 0.849 | 0.838 |
| 0.9 to 1.0 | 31 | 0.920 | 1.000 | 676 | 0.962 | 0.954 |
