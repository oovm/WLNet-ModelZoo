# WRN28-10 trained on CIFAR10
![Task](https://img.shields.io/badge/Task-Classifation-Orange.svg)
![Size](https://img.shields.io/badge/Size-145.98%20MB-blue.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-97.119%25-brightgreen.svg)
![Speed](https://img.shields.io/badge/Speed-3.713%20ms-ff69b4.svg)

Automatically generated on Mon 19 Nov 2018 12:49:42

## Network structure:
- Network Size: **145.989 MB**
- Parameters: **36 497 222**
- Nodes Count: **95**
- Speed: **3.713 ms/sample**
- Layers:
  - AggregationLayer: **1**
  - BatchNormalizationLayer: **27**
  - ConvolutionLayer: **28**
  - ElementwiseLayer: **25**
  - LinearLayer: **1**
  - SoftmaxLayer: **1**
  - ThreadingLayer: **12**


## Accuracy Curve
![Classification Curve.png](https://i.loli.net/2018/11/19/5bf240f955722.png)

![High Precision Classification Curve.png](https://i.loli.net/2018/11/19/5bf240f9cf50e.png)

## Main Indicator
  - Top-1: **97.1199%**
  - Top-2: **99.1600%**
  - Top-3: **99.6900%**
  - Top-5: **99.9400%**
  - LogLikelihood: **-1246.3**
  - CrossEntropyLoss: **0.12463**
  - ProbabilityLoss: **0.00663905**
  - MeanProbability: **96.7439%**
  - GeometricMeanProbability: **88.2823%**
  - VarianceProbability: **0.0246335**
  - ScottPi: **0.968**
  - CohenKappa: **0.968**
  - RejectionRate: **0.00000%**

![Accuracy Rejection Curve.png](https://i.loli.net/2018/11/19/5bf240f9d0c64.png)

## Class Indicator
| Class | Count | TPRate | TNRate | FPRate | FNRate | F1Score |
|-------|-------|--------|--------|--------|--------|---------|
| airplane | 1000 | 97.5000% | 99.6666% | 0.33333% | 2.50000% | 0.97256 |
| automobile | 1000 | 98.7000% | 99.8666% | 0.13333% | 1.30000% | 0.98749 |
| bird | 1000 | 96.8999% | 99.6777% | 0.32222% | 3.10000% | 0.96996 |
| cat | 1000 | 92.5000% | 99.3222% | 0.67777% | 7.50000% | 0.93152 |
| deer | 1000 | 98.3000% | 99.6222% | 0.37777% | 1.70000% | 0.97471 |
| dog | 1000 | 93.7000% | 99.3888% | 0.61111% | 6.30000% | 0.94076 |
| frog | 1000 | 98.5000% | 99.8555% | 0.14444% | 1.50000% | 0.98598 |
| horse | 1000 | 98.4000% | 99.8555% | 0.14444% | 1.60000% | 0.98547 |
| ship | 1000 | 98.6000% | 99.6777% | 0.32222% | 1.40000% | 0.97866 |
| truck | 1000 | 98.1000% | 99.8666% | 0.13333% | 1.90000% | 0.98444 |

## Hard Class
![ConfusionMatrix.png](https://i.loli.net/2018/11/19/5bf240f9d0399.png)

## Evaluation Report
| Index | TestID | Result | Time | MemoryChange |
|-------|--------|--------|------|--------------|
| 1 | Dependency Check | Success | 2.13078 s | +5.09124 MB |
| 2 | Pre-define | Success | 0.00000 s | +0.00159 MB |
| 3 | GPU Warm-Up | Success | 4.23514 s | +54.1124 MB |
| 4 | Loading Model | Success | 1.82257 s | +146.422 MB |
| 5 | Loading Data | Success | 0.35257 s | +42.3258 MB |
| 6 | Benchmark Test | Success | 37.7018 s | +1.54736 MB |
| 7 | Result Dump | Success | 0.10576 s | -0.08278 MB |
| 8 | Analyzing | Success | 9.34498 s | +26.2562 MB |
