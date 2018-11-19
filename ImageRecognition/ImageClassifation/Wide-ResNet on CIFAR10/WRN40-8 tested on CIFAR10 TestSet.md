# WRN40-8 trained on CIFAR10
![Task](https://img.shields.io/badge/Task-Classifation-Orange.svg)
![Size](https://img.shields.io/badge/Size-143.07%20MB-blue.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-97.170%25-brightgreen.svg)
![Speed](https://img.shields.io/badge/Speed-3.631%20ms-ff69b4.svg)

Automatically generated on Mon 19 Nov 2018 12:52:01

## Network structure:
- Network Size: **143.08 MB**
- Parameters: **35 769 926**
- Nodes Count: **137**
- Speed: **3.631 ms/sample**
- Layers:
  - AggregationLayer: **1**
  - BatchNormalizationLayer: **39**
  - ConvolutionLayer: **40**
  - ElementwiseLayer: **37**
  - LinearLayer: **1**
  - SoftmaxLayer: **1**
  - ThreadingLayer: **18**


## Accuracy Curve
![Classification Curve.png](https://i.loli.net/2018/11/19/5bf24181d8906.png)

![High Precision Classification Curve.png](https://i.loli.net/2018/11/19/5bf24182219a3.png)

## Main Indicator
  - Top-1: **97.1700%**
  - Top-2: **99.1500%**
  - Top-3: **99.6700%**
  - Top-5: **99.8900%**
  - LogLikelihood: **-1256.93**
  - CrossEntropyLoss: **0.125693**
  - ProbabilityLoss: **0.00630461**
  - MeanProbability: **96.8052%**
  - GeometricMeanProbability: **88.1885%**
  - VarianceProbability: **0.0241112**
  - ScottPi: **0.968555**
  - CohenKappa: **0.968556**
  - RejectionRate: **0.00000%**

![Accuracy Rejection Curve.png](https://i.loli.net/2018/11/19/5bf24182612a1.png)

## Class Indicator
| Class | Count | TPRate | TNRate | FPRate | FNRate | F1Score |
|-------|-------|--------|--------|--------|--------|---------|
| airplane | 1000 | 98.3000% | 99.6666% | 0.33333% | 1.70000% | 0.97665 |
| automobile | 1000 | 97.8999% | 99.8888% | 0.11111% | 2.10000% | 0.98441 |
| bird | 1000 | 95.6000% | 99.7333% | 0.26666% | 4.39999% | 0.96565 |
| cat | 1000 | 94.3000% | 99.2777% | 0.72222% | 5.70000% | 0.93924 |
| deer | 1000 | 97.8999% | 99.5666% | 0.43333% | 2.10000% | 0.97026 |
| dog | 1000 | 93.8999% | 99.4777% | 0.52222% | 6.10000% | 0.94561 |
| frog | 1000 | 98.3000% | 99.8555% | 0.14444% | 1.70000% | 0.98496 |
| horse | 1000 | 98.2000% | 99.9000% | 0.10000% | 1.79999% | 0.98643 |
| ship | 1000 | 99.1000% | 99.7000% | 0.30000% | 0.89999% | 0.98216 |
| truck | 1000 | 98.2000% | 99.7888% | 0.21111% | 1.79999% | 0.98150 |

## Hard Class
![ConfusionMatrix.png](https://i.loli.net/2018/11/19/5bf2418262dda.png)

## Evaluation Report
| Index | TestID | Result | Time | MemoryChange |
|-------|--------|--------|------|--------------|
| 1 | Dependency Check | Success | 2.27033 s | +5.10068 MB |
| 2 | Pre-define | Success | 0.00000 s | +0.00159 MB |
| 3 | GPU Warm-Up | Success | 4.71256 s | +65.3660 MB |
| 4 | Loading Model | Success | 1.77886 s | +143.682 MB |
| 5 | Loading Data | Success | 0.34904 s | +42.3258 MB |
| 6 | Benchmark Test | Success | 36.8359 s | +0.77343 MB |
| 7 | Result Dump | Success | 0.07583 s | -0.12036 MB |
| 8 | Analyzing | Success | 8.56566 s | +26.2763 MB |
