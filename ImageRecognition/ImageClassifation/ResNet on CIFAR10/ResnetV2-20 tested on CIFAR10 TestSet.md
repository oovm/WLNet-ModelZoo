# ResnetV2-20 tested on CIFAR10 TestSet
![Task](https://img.shields.io/badge/Task-Classifation-Orange.svg)
![Size](https://img.shields.io/badge/Size-1.0946%20MB-blue.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-92.390%25-brightgreen.svg)
![Speed](https://img.shields.io/badge/Speed-5.680%20ms-ff69b4.svg)

Automatically generated on Sat 17 Nov 2018 22:08:43

## Network structure:
- Network Size: **1.09468 MB**
- Parameters: **273 670**
- Nodes Count: **72**
- Speed: **5.680 ms/sample**
- Layer Statistics
  - BatchNormalizationLayer: **20**
  - ConvolutionLayer: **21**
  - ElementwiseLayer: **19**
  - ThreadingLayer: **9**
  - AggregationLayer: **1**
  - LinearLayer: **1**
  - SoftmaxLayer: **1**


## Accuracy Curve
![Classification Curve.png](https://i.loli.net/2018/11/17/5bf021480f2eb.png)

![High Precision Classification Curve.png](https://i.loli.net/2018/11/17/5bf02148da6f8.png)

## Main Indicator
  - Top-1: **92.3900%**
  - Top-2: **97.6900%**
  - Top-3: **99.0200%**
  - Top-5: **99.7900%**
  - LogLikelihood: **-2438.72**
  - CrossEntropyLoss: **0.243872**
  - ProbabilityLoss: **0.0338326**
  - MeanProbability: **90.4426%**
  - GeometricMeanProbability: **78.3588%**
  - VarianceProbability: **0.0555114**
  - ScottPi: **0.915443**
  - CohenKappa: **0.915444**
  - RejectionRate: **0.00000%**

![Accuracy Rejection Curve.png](https://i.loli.net/2018/11/17/5bf0214b8b2b2.png)

## Class Indicator
| Class | Count | TPRate | TNRate | FPRate | FNRate | F1Score |
|-------|-------|--------|--------|--------|--------|---------|
| airplane | 1000 | 94.6000% | 98.8778% | 1.12222% | 5.40000% | 0.92427 |
| automobile | 1000 | 96.5000% | 99.6333% | 0.36666% | 3.50000% | 0.96596 |
| bird | 1000 | 89.8000% | 98.8889% | 1.11111% | 10.2000% | 0.89889 |
| cat | 1000 | 83.3999% | 98.3778% | 1.62222% | 16.6000% | 0.84242 |
| deer | 1000 | 95.1000% | 98.9778% | 1.02222% | 4.90000% | 0.93098 |
| dog | 1000 | 87.2000% | 98.6666% | 1.33333% | 12.8000% | 0.87550 |
| frog | 1000 | 91.7000% | 99.6222% | 0.37777% | 8.30000% | 0.94003 |
| horse | 1000 | 93.8999% | 99.5778% | 0.42222% | 6.10000% | 0.94992 |
| ship | 1000 | 96.6000% | 99.3667% | 0.63333% | 3.40000% | 0.95501 |
| truck | 1000 | 95.1000% | 99.5556% | 0.44444% | 4.90000% | 0.95529 |

## Hard Class
![ConfusionMatrix.png](https://i.loli.net/2018/11/17/5bf0214b8cf8b.png)

## Evaluation Report
| Index | TestID | Result | Time | MemoryChange |
|-------|--------|--------|------|--------------|
| 1 | Pre-define | Success | 0.00000 s | +0.00204 MB |
| 2 | GPU Warm-Up | Success | 4.97705 s | +65.1176 MB |
| 3 | Loading Model | Success | 0.03690 s | +1.45081 MB |
| 4 | Loading Data | Success | 0.39401 s | +42.3258 MB |
| 5 | Benchmark Test | Success | 38.7599 s | +15.7559 MB |
| 6 | Result Dump | Success | 0.21043 s | -1.18253 MB |
| 7 | Analyzing | Success | 8.26185 s | +23.1480 MB |
