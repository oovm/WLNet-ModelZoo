# WRN16-10 tested on CIFAR10 TestSet
![Task](https://img.shields.io/badge/Task-Classifation-Orange.svg)
![Size](https://img.shields.io/badge/Size-68.502%20MB-blue.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-96.500%25-brightgreen.svg)
![Speed](https://img.shields.io/badge/Speed-41.81%20ms-ff69b4.svg)

Automatically generated on Sat 17 Nov 2018 23:33:14

## Network structure:
- Network Size: **68.5028 MB**
- Parameters: **17 125 702**
- Nodes Count: **53**
- Speed: **41.81 ms/sample**
- Layers:
  - AggregationLayer: **1**
  - BatchNormalizationLayer: **15**
  - ConvolutionLayer: **16**
  - ElementwiseLayer: **13**
  - LinearLayer: **1**
  - SoftmaxLayer: **1**
  - ThreadingLayer: **6**


## Accuracy Curve
![Classification Curve.png](https://i.loli.net/2018/11/17/5bf03625035c0.png)

![High Precision Classification Curve.png](https://i.loli.net/2018/11/17/5bf036245e592.png)

## Main Indicator
  - Top-1: **96.5000%**
  - Top-2: **99.0099%**
  - Top-3: **99.6300%**
  - Top-5: **99.8900%**
  - LogLikelihood: **-1152.98**
  - CrossEntropyLoss: **0.115298**
  - ProbabilityLoss: **0.0171053**
  - MeanProbability: **95.3702%**
  - GeometricMeanProbability: **89.1101%**
  - VarianceProbability: **0.0274391**
  - ScottPi: **0.961111**
  - CohenKappa: **0.961111**
  - RejectionRate: **0.00000%**

![Accuracy Rejection Curve.png](https://i.loli.net/2018/11/17/5bf0362505198.png)

## Class Indicator
| Class | Count | TPRate | TNRate | FPRate | FNRate | F1Score |
|-------|-------|--------|--------|--------|--------|---------|
| airplane | 1000 | 97.8000% | 99.5111% | 0.48888% | 2.19999% | 0.96735 |
| automobile | 1000 | 97.7000% | 99.8222% | 0.17777% | 2.30000% | 0.98043 |
| bird | 1000 | 94.8000% | 99.5222% | 0.47777% | 5.20000% | 0.95228 |
| cat | 1000 | 93.0000% | 99.1222% | 0.87777% | 7.00000% | 0.92583 |
| deer | 1000 | 97.7000% | 99.6111% | 0.38888% | 2.30000% | 0.97117 |
| dog | 1000 | 92.7000% | 99.3889% | 0.61111% | 7.30000% | 0.93541 |
| frog | 1000 | 98.2000% | 99.7667% | 0.23333% | 1.79999% | 0.98052 |
| horse | 1000 | 97.3999% | 99.9222% | 0.07777% | 2.60000% | 0.98334 |
| ship | 1000 | 98.2000% | 99.6666% | 0.33333% | 1.79999% | 0.97614 |
| truck | 1000 | 97.5000% | 99.7778% | 0.22222% | 2.50000% | 0.97744 |

## Hard Class
![ConfusionMatrix.png](https://i.loli.net/2018/11/17/5bf0362506ce3.png)

## Evaluation Report
| Index | TestID | Result | Time | MemoryChange |
|-------|--------|--------|------|--------------|
| 1 | Pre-define | Success | 0.00000 s | +0.00204 MB |
| 2 | GPU Warm-Up | Success | 4.95182 s | +52.0761 MB |
| 3 | Loading Model | Success | 0.76698 s | +68.7395 MB |
| 4 | Loading Data | Success | 0.35405 s | +42.3256 MB |
| 5 | Benchmark Test | Success | 402.325 s | +15.3367 MB |
| 6 | Result Dump | Success | 0.23836 s | -1.16490 MB |
| 7 | Analyzing | MessagesFailure | 8.04391 s | +22.1369 MB |
