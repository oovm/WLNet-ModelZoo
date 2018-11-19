# WRN16-10 trained on CIFAR10
![Task](https://img.shields.io/badge/Task-Classifation-Orange.svg)
![Size](https://img.shields.io/badge/Size-68.502%20MB-blue.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-96.500%25-brightgreen.svg)
![Speed](https://img.shields.io/badge/Speed-1.616%20ms-ff69b4.svg)

Automatically generated on Mon 19 Nov 2018 12:46:10

## Network structure:
- Network Size: **68.5028 MB**
- Parameters: **17 125 702**
- Nodes Count: **53**
- Speed: **1.616 ms/sample**
- Layers:
  - AggregationLayer: **1**
  - BatchNormalizationLayer: **15**
  - ConvolutionLayer: **16**
  - ElementwiseLayer: **13**
  - LinearLayer: **1**
  - SoftmaxLayer: **1**
  - ThreadingLayer: **6**


## Accuracy Curve
![Classification Curve.png](https://i.loli.net/2018/11/19/5bf2402dc95e8.png)

![High Precision Classification Curve.png](https://i.loli.net/2018/11/19/5bf2402decafc.png)

## Main Indicator
  - Top-1: **96.5000%**
  - Top-2: **99.0099%**
  - Top-3: **99.6300%**
  - Top-5: **99.8900%**
  - LogLikelihood: **-1152.98**
  - CrossEntropyLoss: **0.115298**
  - ProbabilityLoss: **0.0171053**
  - MeanProbability: **95.3702%**
  - GeometricMeanProbability: **89.1100%**
  - VarianceProbability: **0.0274391**
  - ScottPi: **0.961111**
  - CohenKappa: **0.961111**
  - RejectionRate: **0.00000%**

![Accuracy Rejection Curve.png](https://i.loli.net/2018/11/19/5bf2402e13374.png)

## Class Indicator
| Class | Count | TPRate | TNRate | FPRate | FNRate | F1Score |
|-------|-------|--------|--------|--------|--------|---------|
| airplane | 1000 | 97.8000% | 99.5111% | 0.48888% | 2.19999% | 0.96735 |
| automobile | 1000 | 97.7000% | 99.8222% | 0.17777% | 2.30000% | 0.98043 |
| bird | 1000 | 94.8000% | 99.5222% | 0.47777% | 5.20000% | 0.95228 |
| cat | 1000 | 93.0000% | 99.1222% | 0.87777% | 7.00000% | 0.92583 |
| deer | 1000 | 97.7000% | 99.6111% | 0.38888% | 2.30000% | 0.97117 |
| dog | 1000 | 92.7000% | 99.3888% | 0.61111% | 7.30000% | 0.93541 |
| frog | 1000 | 98.2000% | 99.7666% | 0.23333% | 1.79999% | 0.98052 |
| horse | 1000 | 97.3999% | 99.9222% | 0.07777% | 2.60000% | 0.98334 |
| ship | 1000 | 98.2000% | 99.6666% | 0.33333% | 1.79999% | 0.97614 |
| truck | 1000 | 97.5000% | 99.7777% | 0.22222% | 2.50000% | 0.97744 |

## Hard Class
![ConfusionMatrix.png](https://i.loli.net/2018/11/19/5bf2402e14fdc.png)

## Evaluation Report
| Index | TestID | Result | Time | MemoryChange |
|-------|--------|--------|------|--------------|
| 1 | Dependency Check | Success | 2.15999 s | +5.22214 MB |
| 2 | Pre-define | Success | 0.00000 s | +0.00177 MB |
| 3 | GPU Warm-Up | Success | 5.75713 s | +52.9652 MB |
| 4 | Loading Model | Success | 1.24082 s | +68.7620 MB |
| 5 | Loading Data | Success | 0.37262 s | +42.3238 MB |
| 6 | Benchmark Test | Success | 16.7268 s | +1.40838 MB |
| 7 | Result Dump | Success | 0.09674 s | -0.04493 MB |
| 8 | Analyzing | Success | 9.13358 s | +24.0519 MB |
