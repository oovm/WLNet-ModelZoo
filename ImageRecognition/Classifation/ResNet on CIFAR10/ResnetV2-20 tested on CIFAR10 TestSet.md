# ResnetV2-20 trained on CIFAR10
![Task](https://img.shields.io/badge/Task-Classifation-Orange.svg)
![Size](https://img.shields.io/badge/Size-1.0946%20MB-blue.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-92.390%25-brightgreen.svg)
![Speed](https://img.shields.io/badge/Speed-0.210%20ms-ff69b4.svg)

Automatically generated on Sun 18 Nov 2018 20:20:15

## Network structure:
- Network Size: **1.09468 MB**
- Parameters: **273 670**
- Nodes Count: **72**
- Speed: **0.210 ms/sample**
- Layers:
  - AggregationLayer: **1**
  - BatchNormalizationLayer: **20**
  - ConvolutionLayer: **21**
  - ElementwiseLayer: **19**
  - LinearLayer: **1**
  - SoftmaxLayer: **1**
  - ThreadingLayer: **9**


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
  - MeanProbability: **90.4425%**
  - GeometricMeanProbability: **78.3588%**
  - VarianceProbability: **0.0555114**
  - ScottPi: **0.915443**
  - CohenKappa: **0.915444**
  - RejectionRate: **0.00000%**

![Accuracy Rejection Curve.png](https://i.loli.net/2018/11/17/5bf0214b8b2b2.png)

## Class Indicator
| Class | Count | TPRate | TNRate | FPRate | FNRate | F1Score |
|-------|-------|--------|--------|--------|--------|---------|
| airplane | 1000 | 94.6000% | 98.8777% | 1.12222% | 5.40000% | 0.92427 |
| automobile | 1000 | 96.5000% | 99.6333% | 0.36666% | 3.50000% | 0.96596 |
| bird | 1000 | 89.8000% | 98.8888% | 1.11111% | 10.2000% | 0.89889 |
| cat | 1000 | 83.3999% | 98.3777% | 1.62222% | 16.6000% | 0.84242 |
| deer | 1000 | 95.1000% | 98.9777% | 1.02222% | 4.90000% | 0.93098 |
| dog | 1000 | 87.2000% | 98.6666% | 1.33333% | 12.8000% | 0.87550 |
| frog | 1000 | 91.7000% | 99.6222% | 0.37777% | 8.30000% | 0.94003 |
| horse | 1000 | 93.8999% | 99.5777% | 0.42222% | 6.10000% | 0.94992 |
| ship | 1000 | 96.6000% | 99.3666% | 0.63333% | 3.40000% | 0.95501 |
| truck | 1000 | 95.1000% | 99.5555% | 0.44444% | 4.90000% | 0.95529 |

## Hard Class
![ConfusionMatrix.png](https://i.loli.net/2018/11/17/5bf0214b8cf8b.png)

## Evaluation Report
| Index | TestID | Result | Time | MemoryChange |
|-------|--------|--------|------|--------------|
| 1 | Dependency Check | Success | 2.24699 s | +5.10072 MB |
| 2 | Pre-define | Success | 0.00000 s | +0.00170 MB |
| 3 | GPU Warm-Up | Success | 4.95176 s | +65.3663 MB |
| 4 | Loading Model | Success | 0.03194 s | +1.45080 MB |
| 5 | Loading Data | Success | 0.34408 s | +42.3258 MB |
| 6 | Benchmark Test | Success | 2.58708 s | +0.60913 MB |
| 7 | Result Dump | Success | 0.01900 s | -0.06256 MB |
| 8 | Analyzing | Success | 8.74064 s | +25.9704 MB |
