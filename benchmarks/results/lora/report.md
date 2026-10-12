# opensysone LoRA on typed-decisions test

Labels in typed-decisions are frontier-model consensus, not human gold. A calibrated probability here is the chance the consensus agrees.

| Question type | n | Accuracy raw | Accuracy calibrated | ECE raw | ECE calibrated | ECE equal-mass raw | ECE equal-mass calibrated | NLL raw | NLL calibrated | Brier raw | Brier calibrated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| choice | 600 | 0.732 | 0.732 | 0.056 | 0.056 | 0.054 | 0.054 | 1.273 | 1.273 | 0.138 | 0.138 |
| noul | 600 | 0.852 | 0.852 | 0.022 | 0.022 | 0.021 | 0.021 | 0.602 | 0.602 | 0.074 | 0.074 |
| score | 800 | 0.726 | 0.726 | 0.058 | 0.058 | 0.065 | 0.065 | 1.514 | 1.514 | 0.161 | 0.161 |
| all | 2000 | 0.765 | 0.765 | 0.030 | 0.030 | 0.034 | 0.034 | 1.168 | 1.168 | 0.128 | 0.128 |

95% interval over all questions, resampling whole cases:

| | Accuracy | ECE | NLL |
|---|---|---|---|
| raw | 0.743 to 0.787 | 0.017 to 0.049 | 1.133 to 1.205 |
| calibrated | 0.743 to 0.787 | 0.017 to 0.049 | 1.133 to 1.205 |

## choice

![choice reliability](reliability-choice.svg)

| Confidence bin | n raw | confidence raw | accuracy raw | n calibrated | confidence calibrated | accuracy calibrated |
|---|---|---|---|---|---|---|
| 0.3 to 0.4 | 12 | 0.373 | 0.500 | 12 | 0.373 | 0.500 |
| 0.4 to 0.5 | 45 | 0.458 | 0.533 | 45 | 0.458 | 0.533 |
| 0.5 to 0.6 | 77 | 0.549 | 0.532 | 77 | 0.549 | 0.532 |
| 0.6 to 0.7 | 96 | 0.650 | 0.573 | 96 | 0.650 | 0.573 |
| 0.7 to 0.8 | 84 | 0.747 | 0.548 | 84 | 0.747 | 0.548 |
| 0.8 to 0.9 | 109 | 0.847 | 0.872 | 109 | 0.847 | 0.872 |
| 0.9 to 1.0 | 177 | 0.970 | 0.972 | 177 | 0.970 | 0.972 |

## noul

![noul reliability](reliability-noul.svg)

| Confidence bin | n raw | confidence raw | accuracy raw | n calibrated | confidence calibrated | accuracy calibrated |
|---|---|---|---|---|---|---|
| 0.5 to 0.6 | 72 | 0.550 | 0.569 | 72 | 0.550 | 0.569 |
| 0.6 to 0.7 | 77 | 0.659 | 0.688 | 77 | 0.659 | 0.688 |
| 0.7 to 0.8 | 76 | 0.750 | 0.803 | 76 | 0.750 | 0.803 |
| 0.8 to 0.9 | 81 | 0.853 | 0.827 | 81 | 0.853 | 0.827 |
| 0.9 to 1.0 | 294 | 0.971 | 0.983 | 294 | 0.971 | 0.983 |

## score

![score reliability](reliability-score.svg)

| Confidence bin | n raw | confidence raw | accuracy raw | n calibrated | confidence calibrated | accuracy calibrated |
|---|---|---|---|---|---|---|
| 0.3 to 0.4 | 7 | 0.382 | 0.714 | 7 | 0.382 | 0.714 |
| 0.4 to 0.5 | 40 | 0.470 | 0.500 | 40 | 0.470 | 0.500 |
| 0.5 to 0.6 | 101 | 0.548 | 0.455 | 101 | 0.548 | 0.455 |
| 0.6 to 0.7 | 138 | 0.657 | 0.565 | 138 | 0.657 | 0.565 |
| 0.7 to 0.8 | 141 | 0.750 | 0.766 | 141 | 0.750 | 0.766 |
| 0.8 to 0.9 | 148 | 0.852 | 0.797 | 148 | 0.852 | 0.797 |
| 0.9 to 1.0 | 225 | 0.964 | 0.916 | 225 | 0.964 | 0.916 |

## all

![all reliability](reliability-all.svg)

| Confidence bin | n raw | confidence raw | accuracy raw | n calibrated | confidence calibrated | accuracy calibrated |
|---|---|---|---|---|---|---|
| 0.3 to 0.4 | 19 | 0.376 | 0.579 | 19 | 0.376 | 0.579 |
| 0.4 to 0.5 | 85 | 0.463 | 0.518 | 85 | 0.463 | 0.518 |
| 0.5 to 0.6 | 250 | 0.549 | 0.512 | 250 | 0.549 | 0.512 |
| 0.6 to 0.7 | 311 | 0.656 | 0.598 | 311 | 0.656 | 0.598 |
| 0.7 to 0.8 | 301 | 0.750 | 0.714 | 301 | 0.750 | 0.714 |
| 0.8 to 0.9 | 338 | 0.851 | 0.828 | 338 | 0.851 | 0.828 |
| 0.9 to 1.0 | 696 | 0.968 | 0.958 | 696 | 0.968 | 0.958 |
