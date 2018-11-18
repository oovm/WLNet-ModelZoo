# LeNet trained on MNIST
![Task](https://img.shields.io/badge/Task-Classifation-Orange.svg)
![Size](https://img.shields.io/badge/Size-1.7243%20MB-blue.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-98.480%25-brightgreen.svg)
![Speed](https://img.shields.io/badge/Speed-0.355%20ms-ff69b4.svg)

Automatically generated on Sun 18 Nov 2018 20:07:43

## Network structure:
- Network Size: **1.72432 MB**
- Parameters: **431 080**
- Nodes Count: **11**
- Speed: **0.355 ms/sample**
- Layers:
  - ConvolutionLayer: **2**
  - ElementwiseLayer: **3**
  - FlattenLayer: **1**
  - LinearLayer: **2**
  - PoolingLayer: **2**
  - SoftmaxLayer: **1**


## Accuracy Curve
![Classification Curve.png](https://i.loli.net/2018/11/17/5bf004151b12a.png)

![High Precision Classification Curve.png](https://i.loli.net/2018/11/17/5bf0041535eab.png)

## Main Indicator
  - Top-1: **98.4800%**
  - Top-2: **99.7299%**
  - Top-3: **99.9100%**
  - Top-5: **99.9900%**
  - LogLikelihood: **-659.657**
  - CrossEntropyLoss: **0.0659657**
  - ProbabilityLoss: **0.00522322**
  - MeanProbability: **98.1506%**
  - GeometricMeanProbability: **93.6163%**
  - VarianceProbability: **0.0130536**
  - ScottPi: **0.983105**
  - CohenKappa: **0.983105**
  - RejectionRate: **0.00000%**

![Accuracy Rejection Curve.png](https://i.loli.net/2018/11/17/5bf00415503a1.png)

## Class Indicator
| Class | Count | TPRate | TNRate | FPRate | FNRate | F1Score |
|-------|-------|--------|--------|--------|--------|---------|
| 0 | 980 | 99.3877% | 99.8558% | 0.14412% | 0.61224% | 0.99034 |
| 1 | 1135 | 99.4713% | 99.9097% | 0.09024% | 0.52863% | 0.99383 |
| 2 | 1032 | 97.4806% | 99.8996% | 0.10035% | 2.51937% | 0.98290 |
| 3 | 1010 | 98.9108% | 99.8220% | 0.17797% | 1.08910% | 0.98666 |
| 4 | 982 | 97.7596% | 99.9445% | 0.05544% | 2.24032% | 0.98613 |
| 5 | 892 | 98.8789% | 99.7145% | 0.28546% | 1.12107% | 0.98000 |
| 6 | 958 | 97.4947% | 99.9336% | 0.06635% | 2.50521% | 0.98419 |
| 7 | 1028 | 98.3463% | 99.6879% | 0.31208% | 1.65369% | 0.97822 |
| 8 | 974 | 98.4599% | 99.8448% | 0.15510% | 1.54004% | 0.98510 |
| 9 | 1009 | 98.5133% | 99.6996% | 0.30030% | 1.48662% | 0.97931 |

## Hard Class
![ConfusionMatrix.png](https://i.loli.net/2018/11/17/5bf0041545c4e.png)

## Evaluation Report
| Index | TestID | Result | Time | MemoryChange |
|-------|--------|--------|------|--------------|
| 1 | Dependency Check | Success | 2.18416 s | +5.10072 MB |
| 2 | Pre-define | Success | 0.00000 s | +0.00170 MB |
| 3 | GPU Warm-Up | Success | 4.97171 s | +65.3663 MB |
| 4 | Loading Model | Success | 0.08876 s | +7.45142 MB |
| 5 | Loading Data | Success | 2.02858 s | +36.3034 MB |
| 6 | Benchmark Test | Success | 3.89352 s | +1.05823 MB |
| 7 | Result Dump | Success | 0.08679 s | -0.05690 MB |
| 8 | Analyzing | Success | 17.6138 s | +19.0384 MB |
