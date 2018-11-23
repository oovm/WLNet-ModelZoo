# ResnetV2-110 trained on CIFAR10
![Task](https://img.shields.io/badge/Task-Classifation-Orange.svg)
![Size](https://img.shields.io/badge/Size-6.9545%20MB-blue.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-95.270%25-brightgreen.svg)
![Speed](https://img.shields.io/badge/Speed-0.532%20ms-ff69b4.svg)

Automatically generated on Sun 18 Nov 2018 20:31:59

## Network structure:
- Network Size: **6.95452 MB**
- Parameters: **1 738 630**
- Nodes Count: **387**
- Speed: **0.532 ms/sample**
- Layers:
  - AggregationLayer: **1**
  - BatchNormalizationLayer: **110**
  - ConvolutionLayer: **111**
  - ElementwiseLayer: **109**
  - LinearLayer: **1**
  - SoftmaxLayer: **1**
  - ThreadingLayer: **54**


## Accuracy Curve
![Classification Curve.png](https://i.loli.net/2018/11/18/5bf15f5518d08.png)

![High Precision Classification Curve.png](https://i.loli.net/2018/11/18/5bf15f555eaa4.png)

## Main Indicator
  - Top-1: **95.2700%**
  - Top-2: **98.5300%**
  - Top-3: **99.5100%**
  - Top-5: **99.8900%**
  - LogLikelihood: **-1911.91**
  - CrossEntropyLoss: **0.191191**
  - ProbabilityLoss: **0.0111677**
  - MeanProbability: **94.6614%**
  - GeometricMeanProbability: **82.5974%**
  - VarianceProbability: **0.0387726**
  - ScottPi: **0.947444**
  - CohenKappa: **0.947444**
  - RejectionRate: **0.00000%**

![Accuracy Rejection Curve.png](https://i.loli.net/2018/11/18/5bf15f55a304a.png)

## Class Indicator
| Class | Count | TPRate | TNRate | FPRate | FNRate | F1Score |
|-------|-------|--------|--------|--------|--------|---------|
| airplane | 1000 | 95.8000% | 99.3444% | 0.65555% | 4.20000% | 0.94992 |
| automobile | 1000 | 98.2000% | 99.7444% | 0.25555% | 1.79999% | 0.97955 |
| bird | 1000 | 92.0000% | 99.3777% | 0.62222% | 8.00000% | 0.93117 |
| cat | 1000 | 90.7000% | 98.8000% | 1.20000% | 9.30000% | 0.90024 |
| deer | 1000 | 97.1000% | 99.3666% | 0.63333% | 2.90000% | 0.95759 |
| dog | 1000 | 92.0000% | 99.1111% | 0.88888% | 8.00000% | 0.92000 |
| frog | 1000 | 96.3999% | 99.8111% | 0.18888% | 3.59999% | 0.97324 |
| horse | 1000 | 96.3999% | 99.8111% | 0.18888% | 3.59999% | 0.97324 |
| ship | 1000 | 97.7000% | 99.6222% | 0.37777% | 2.30000% | 0.97165 |
| truck | 1000 | 96.3999% | 99.7555% | 0.24444% | 3.59999% | 0.97079 |

## Hard Class
![ConfusionMatrix.png](https://i.loli.net/2018/11/18/5bf15f570503b.png)

## Evaluation Report
| Index | TestID | Result | Time | MemoryChange |
|-------|--------|--------|------|--------------|
| 1 | Dependency Check | Success | 2.23602 s | +0.03504 MB |
| 2 | Pre-define | Success | 0.00099 s | +0.00159 MB |
| 3 | GPU Warm-Up | Success | 4.35339 s | +14.6607 MB |
| 4 | Loading Model | Success | 0.25036 s | +8.71920 MB |
| 5 | Loading Data | Success | 0.34408 s | +42.3246 MB |
| 6 | Benchmark Test | Success | 5.79950 s | +4.05528 MB |
| 7 | Result Dump | Success | 0.02194 s | -0.34230 MB |
| 8 | Analyzing | Success | 8.34576 s | +26.2853 MB |
